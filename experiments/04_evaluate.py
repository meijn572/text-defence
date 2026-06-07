# -*- coding: utf-8 -*-
"""
实验 04: 全面评测与消融实验

评测内容:
  1. 在所有测试子集上评估各模型性能
  2. 消融实验: 逐个去掉通道, 量化每个通道的贡献
  3. 生成可视化结果图表

输出:
  - results/eval_results.csv        详细评测结果
  - results/figures/ablation.png    消融实验柱状图
  - results/figures/compare.png     各攻击类型 F1 对比图
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无 GUI 后端
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from utils import (
    set_seed, compute_metrics, print_metrics,
    DATA_ADV, RESULTS_DIR, FIGURES_DIR, DEVICE
)
from attack import ATTACK_REGISTRY
from defense.text_channel import BertClassifier
from defense.fusion_model import FusionClassifier, create_data_loader

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def evaluate_on_subset(model, texts, labels, batch_size=16):
    """在指定子集上评估模型"""
    loader = create_data_loader(texts, labels, batch_size=batch_size, shuffle=False)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_texts, batch_labels in loader:
            if isinstance(model, BertClassifier):
                logits = model(batch_texts)
            else:
                logits = model(batch_texts)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch_labels.cpu().tolist())

    return compute_metrics(np.array(all_labels), np.array(all_preds))


def load_model(model_type, device):
    """
    加载指定类型的模型

    model_type: 'baseline', 'baseline_aug', 'fusion'
    """
    from defense.text_channel import BertClassifier
    from defense.fusion_model import FusionClassifier

    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

    if model_type == 'baseline':
        model = BertClassifier().to(device)
        path = os.path.join(base_dir, 'baseline_bert.pth')
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
        else:
            print(f"[WARN] 未找到模型: {path}, 使用随机初始化")
    elif model_type == 'baseline_aug':
        model = BertClassifier().to(device)
        path = os.path.join(base_dir, 'baseline_bert_aug.pth')
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
        else:
            print(f"[WARN] 未找到模型: {path}, 使用随机初始化")
    elif model_type == 'fusion':
        model = FusionClassifier(freeze_channels=False).to(device)
        path = os.path.join(base_dir, 'fusion_model.pth')
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint['model_state'])
        else:
            print(f"[WARN] 未找到模型: {path}, 使用随机初始化")
    else:
        raise ValueError(f"未知模型类型: {model_type}")

    return model


def main():
    set_seed(42)
    print("=" * 60)
    print("  实验 04: 全面评测与消融实验")
    print("=" * 60)

    # ================================================================
    # 1. 加载测试数据
    # ================================================================
    test_path = os.path.join(DATA_ADV, 'test_full.csv')
    if not os.path.exists(test_path):
        print(f"[ERROR] 测试数据不存在: {test_path}")
        return

    test_df = pd.read_csv(test_path)
    print(f"\n测试数据: {len(test_df)} 条")

    # 按攻击类型分组
    subsets = {}
    # 原始子集
    orig_mask = test_df['attack_type'].isna() | (test_df['attack_type'] == '')
    subsets['原始样本'] = (
        test_df[orig_mask]['text'].tolist(),
        test_df[orig_mask]['label'].tolist()
    )

    # 各类对抗子集
    for attack_id in 'ABCDEFGHI':
        mask = test_df['attack_type'] == attack_id
        if mask.sum() > 0:
            name = ATTACK_REGISTRY[attack_id][2]
            subsets[f'对抗_{attack_id} ({name})'] = (
                test_df[mask]['text'].tolist(),
                test_df[mask]['label'].tolist()
            )

    print(f"测试子集: {list(subsets.keys())}")

    # ================================================================
    # 2. 加载模型并评测
    # ================================================================
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    results = []
    models_to_eval = []

    # 尝试加载各模型
    for model_type, display_name in [
        ('baseline', '朴素 BERT'),
        ('baseline_aug', 'BERT + 正规化'),
    ]:
        try:
            model = load_model(model_type, DEVICE)
            models_to_eval.append((display_name, model, model_type))
        except Exception as e:
            print(f"[SKIP] {display_name}: {e}")

    # 融合模型 (先加载全模型, 后续做消融)
    try:
        fusion = load_model('fusion', DEVICE)
        models_to_eval.append(('四通道融合 (本文)', fusion, 'fusion'))
    except Exception as e:
        print(f"[SKIP] 四通道融合: {e}")

    # ================================================================
    # 3. 逐模型 × 逐子集评测
    # ================================================================
    print(f"\n{'='*50}")
    print(f"  开始评测 ({len(models_to_eval)} 个模型 × {len(subsets)} 个子集)")
    print(f"{'='*50}")

    for model_name, model, model_type in models_to_eval:
        print(f"\n{'─'*40}")
        print(f"  模型: {model_name}")
        print(f"{'─'*40}")

        for subset_name, (texts, labels) in subsets.items():
            metrics = evaluate_on_subset(model, texts, labels)
            results.append({
                'model': model_name,
                'subset': subset_name,
                **metrics,
            })
            print(f"  {subset_name:30s} | F1: {metrics['f1']:.4f} | "
                  f"Acc: {metrics['accuracy']:.4f}")

    # 保存详细结果
    results_df = pd.DataFrame(results)
    results_path = os.path.join(RESULTS_DIR, 'eval_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n详细结果已保存: {results_path}")

    # ================================================================
    # 4. 可视化 1: 各攻击类型 F1 对比柱状图
    # ================================================================
    print(f"\n[可视化] 生成对比图表...")

    fig, ax = plt.subplots(figsize=(14, 6))
    pivot = results_df.pivot(index='subset', columns='model', values='f1')

    # 按攻击类型排序
    subset_order = [k for k in subsets.keys() if k in pivot.index]
    pivot = pivot.reindex(subset_order)

    pivot.plot(kind='bar', ax=ax, width=0.7)
    ax.set_title('各攻击类型下 F1-Score 对比', fontsize=14, fontweight='bold')
    ax.set_xlabel('测试子集', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=10)
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% 基准线')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    fig_path1 = os.path.join(FIGURES_DIR, 'compare_f1.png')
    plt.savefig(fig_path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {fig_path1}")

    # ================================================================
    # 5. 可视化 2: 指定攻击下的消融效果
    # ================================================================
    # 提取关键攻击类型的 F1
    key_attacks = ['原始样本', '对抗_F (中文音近字替换)', '对抗_I (字符乱序)']

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    plot_data = results_df[results_df['subset'].isin(key_attacks)]
    sns.barplot(data=plot_data, x='subset', y='f1', hue='model', ax=ax2)
    ax2.set_title('关键攻击类型 F1-Score 对比', fontsize=14, fontweight='bold')
    ax2.set_ylabel('F1-Score', fontsize=12)
    ax2.set_ylim(0, 1.05)
    plt.xticks(rotation=15, ha='right')
    plt.legend(loc='lower right')
    plt.tight_layout()

    fig_path2 = os.path.join(FIGURES_DIR, 'key_attacks.png')
    plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {fig_path2}")

    # ================================================================
    # 6. 打印汇总报告
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  评测报告汇总")
    print(f"{'='*60}")

    # 按模型汇总
    for model_name in results_df['model'].unique():
        model_results = results_df[results_df['model'] == model_name]
        avg_f1 = model_results['f1'].mean()
        worst_f1 = model_results['f1'].min()
        worst_subset = model_results.loc[model_results['f1'].idxmin(), 'subset']
        print(f"\n  [{model_name}]")
        print(f"    平均 F1: {avg_f1:.4f}")
        print(f"    最低 F1: {worst_f1:.4f} (子集: {worst_subset})")

    print(f"\n{'='*60}")
    print(f"  ✓ 评测完成!")
    print(f"  结果: {results_path}")
    print(f"  图表: {FIGURES_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
