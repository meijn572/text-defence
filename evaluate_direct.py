# -*- coding: utf-8 -*-
"""直接评测脚本（复刻 run_all.py 的实验04流程）
包含：
1) 读取 test_full.csv
2) 生成强攻击 J/K/L 并保存到 data/adversarial/
3) 评测三个主模型 + 融合模型消融
4) 保存 results/eval_results.csv 和 results/figures/compare_f1.png
"""

import os
import sys
import traceback

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
if BASE not in sys.path:
    sys.path.insert(0, BASE)

try:
    # ================================================================
    # Step 1: 先读 CSV（遵守项目约束）
    # ================================================================
    print("Step 1: Reading test data...", flush=True)
    import pandas as pd

    test_path = os.path.join(BASE, 'data', 'adversarial', 'test_full.csv')
    if not os.path.exists(test_path):
        print(f"[ERROR] 测试数据不存在: {test_path}")
        print("  请先运行: python experiments/generate_adv.py")
        sys.exit(1)

    test_df = pd.read_csv(test_path)
    print(f"  Loaded test set: {len(test_df)} rows", flush=True)

    # ================================================================
    # Step 2: 导入 ML / 可视化模块
    # ================================================================
    print("Step 2: Importing ML modules...", flush=True)
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import f1_score

    from utils import set_seed, DATA_ADV, RESULTS_DIR, FIGURES_DIR
    from defense.text_channel import BertClassifier
    from defense.fusion_model import FusionClassifier, create_data_loader
    from attack.char_shuffle import attack_adjacent_swap
    from attack.homophone_chinese import attack_homophone

    print("  Imports OK", flush=True)

    set_seed(42)
    DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  DEVICE={DEV}", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ================================================================
    # Step 3: 工具函数
    # ================================================================
    def load_model(cls, path):
        pp = os.path.join(BASE, 'data', 'processed', path)
        if not os.path.exists(pp):
            print(f"  [WARN] 模型不存在，跳过: {pp}")
            return None

        model = (FusionClassifier(freeze_channels=True) if cls == FusionClassifier else cls()).to(DEV)
        ck = torch.load(pp, map_location=DEV)
        state = ck['model_state'] if isinstance(ck, dict) and 'model_state' in ck else ck
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"  [WARN] {path}: missing={len(missing)}, unexpected={len(unexpected)}")
        return model

    def eval_subset(model, texts, labels, ablation_channel=None):
        loader = create_data_loader(texts, labels, batch_size=16, shuffle=False)
        preds_all, labels_all = [], []
        model.eval()
        with torch.no_grad():
            for batch_texts, batch_labels in loader:
                if isinstance(model, FusionClassifier) and ablation_channel:
                    logits = model(batch_texts, ablation=[ablation_channel])
                else:
                    logits = model(batch_texts)
                preds_all.extend(torch.argmax(logits, dim=1).cpu().tolist())
                labels_all.extend(batch_labels.cpu().tolist())

        acc = sum(1 for p, y in zip(preds_all, labels_all) if p == y) / len(labels_all)
        f1 = f1_score(labels_all, preds_all, zero_division=0)
        return {'accuracy': acc, 'f1': f1}

    # ================================================================
    # Step 4: 生成强攻击 J/K/L（与 A-I 口径一致：混入全部正常样本）
    # ================================================================
    print("Step 4: Generating strong attacks J/K/L...", flush=True)
    spam_mask = (test_df['label'] == 1) & (test_df['attack_type'].isna())
    normal_mask = (test_df['label'] == 0) & (test_df['attack_type'].isna())
    spam_texts = test_df[spam_mask]['text'].tolist()
    normal_texts = test_df[normal_mask]['text'].tolist()

    normal_for_adv = pd.DataFrame({
        'text': normal_texts,
        'label': [0] * len(normal_texts),
        'attack_type': ['normal'] * len(normal_texts),
        'original_text': normal_texts,
    })

    df_J = pd.DataFrame({
        'text': [attack_adjacent_swap(t, swap_ratio=0.8) for t in spam_texts],
        'label': 1,
        'attack_type': 'J',
        'original_text': spam_texts,
    })
    df_K = pd.DataFrame({
        'text': [attack_homophone(t, replace_ratio=0.8) for t in spam_texts],
        'label': 1,
        'attack_type': 'K',
        'original_text': spam_texts,
    })
    df_L = pd.DataFrame({
        'text': [
            attack_adjacent_swap(
                attack_homophone(t, replace_ratio=0.8),
                swap_ratio=0.8,
            )
            for t in spam_texts
        ],
        'label': 1,
        'attack_type': 'L',
        'original_text': spam_texts,
    })

    # J/K/L 单文件评测口径：强攻击垃圾 + 全部正常样本
    df_J_mixed = pd.concat([normal_for_adv, df_J], ignore_index=True).sample(frac=1, random_state=42)
    df_K_mixed = pd.concat([normal_for_adv, df_K], ignore_index=True).sample(frac=1, random_state=42)
    df_L_mixed = pd.concat([normal_for_adv, df_L], ignore_index=True).sample(frac=1, random_state=42)

    for df_adv, fname in [
        (df_J_mixed, 'adv_J_strong_shuffle.csv'),
        (df_K_mixed, 'adv_K_strong_homophone.csv'),
        (df_L_mixed, 'adv_L_combined.csv'),
    ]:
        df_adv.to_csv(os.path.join(DATA_ADV, fname), index=False)

    print(f"  Strong attacks generated (mixed): J={len(df_J_mixed)}, K={len(df_K_mixed)}, L={len(df_L_mixed)}", flush=True)

    # ================================================================
    # Step 5: 构建评测子集
    # ================================================================
    ATTACK_NAMES = {
        'A': '字符删除', 'B': '字符插入', 'C': '跨语种同形', 'D': '零宽注入',
        'E': '同义词', 'F': '音近字', 'G': '形近字', 'H': '繁简混用', 'I': '字符乱序',
        'J': '★强乱序', 'K': '★强音近', 'L': '★混合攻击',
    }

    subsets = {}
    # 原始样本: 仅 clean normal + clean spam
    orig_mask = test_df['attack_type'].isna()
    subsets['原始样本'] = (
        test_df[orig_mask]['text'].tolist(),
        test_df[orig_mask]['label'].tolist(),
    )

    # A-I: 从 generate_adv.py 生成的单攻击混合集文件读取
    adv_file_map = {
        'A': 'adv_A_char_delete.csv',
        'B': 'adv_B_char_insert.csv',
        'C': 'adv_C_homoglyph_unicode.csv',
        'D': 'adv_D_zero_width.csv',
        'E': 'adv_E_synonym.csv',
        'F': 'adv_F_homophone_cn.csv',
        'G': 'adv_G_homoglyph_cn.csv',
        'H': 'adv_H_fanjian_split.csv',
        'I': 'adv_I_char_shuffle.csv',
        'J': 'adv_J_strong_shuffle.csv',
        'K': 'adv_K_strong_homophone.csv',
        'L': 'adv_L_combined.csv',
    }

    for aid in 'ABCDEFGHIJKL':
        file_name = adv_file_map[aid]
        file_path = os.path.join(DATA_ADV, file_name)
        if not os.path.exists(file_path):
            print(f"  [WARN] 子集文件不存在，跳过: {file_path}")
            continue
        sub_df = pd.read_csv(file_path)
        subsets[f"对抗_{aid} ({ATTACK_NAMES[aid]})"] = (
            sub_df['text'].tolist(),
            sub_df['label'].tolist(),
        )

    print(f"Step 5: Built {len(subsets)} subsets", flush=True)

    # ================================================================
    # Step 6: 加载模型并评测
    # ================================================================
    print("Step 6: Loading models...", flush=True)
    eval_models = {
        '朴素 BERT': load_model(BertClassifier, 'baseline_bert.pth'),
        'BERT + 正规化': load_model(BertClassifier, 'baseline_bert_aug.pth'),
        '四通道融合 (本文)': load_model(FusionClassifier, 'fusion_model.pth'),
    }
    eval_models = {k: v for k, v in eval_models.items() if v is not None}

    if not eval_models:
        print('[ERROR] 没有可用模型，请先训练模型')
        sys.exit(1)

    print("  Available models:", ', '.join(eval_models.keys()), flush=True)

    results = []
    print("\n" + "=" * 55)
    print("  正常评测")
    print("=" * 55)
    for model_name, model in eval_models.items():
        print(f"\n  {model_name}:")
        for subset_name, (texts, labels) in subsets.items():
            m = eval_subset(model, texts, labels)
            results.append({'model': model_name, 'subset': subset_name, **m})
            print(f"    {subset_name:30s} | F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}")

    print("\n" + "=" * 55)
    print("  消融实验 (置零近似，结果供参考)")
    print("=" * 55)
    fusion_eval = eval_models.get('四通道融合 (本文)')
    if fusion_eval is not None:
        for channel in ['text', 'phonetic', 'visual', 'bow']:
            print(f"\n  置零 {channel} 通道:")
            for subset_name, (texts, labels) in subsets.items():
                m = eval_subset(fusion_eval, texts, labels, ablation_channel=channel)
                results.append({'model': f'融合-置零{channel}', 'subset': subset_name, **m})
                print(f"    {subset_name:30s} | F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}")

    # ================================================================
    # Step 7: 保存结果与图表
    # ================================================================
    rdf = pd.DataFrame(results)
    result_csv = os.path.join(RESULTS_DIR, 'eval_results.csv')
    rdf.to_csv(result_csv, index=False)

    main_models = ['朴素 BERT', 'BERT + 正规化', '四通道融合 (本文)']
    plot_df = rdf[rdf['model'].isin(main_models)]
    if not plot_df.empty:
        pivot = plot_df.pivot(index='subset', columns='model', values='f1')
        fig, ax = plt.subplots(figsize=(14, 6))
        pivot.plot(kind='bar', ax=ax)
        ax.set_title('各攻击类型下不同模型的 F1 对比')
        ax.set_ylabel('F1 Score')
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc='lower right')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, 'compare_f1.png')
        fig.savefig(fig_path, dpi=150)
        print(f"\n图表已保存: {fig_path}")

    print("\n" + "=" * 55)
    print("  ✅ 评测完成")
    print(f"  结果: {result_csv}")
    print("=" * 55)

except Exception as e:
    print(f"\n\nERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
