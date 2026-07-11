# -*- coding: utf-8 -*-
"""生成 3 种加强攻击变体 + 重新评测"""
import sys, os, random, pandas as pd
import torch, numpy as np
from sklearn.metrics import f1_score

BASE = os.path.dirname(os.path.abspath(__file__))  # 自动获取路径
os.chdir(BASE)
sys.path.insert(0, BASE)

# ⚠️ CSV 必须在 torch 导入前
df = pd.read_csv(os.path.join(BASE, 'data', 'adversarial', 'test_full.csv'))
spam_mask = (df['label'] == 1) & (df['attack_type'].isna())
spam_texts = df[spam_mask]['text'].tolist()
normal_df = df[(df['label'] == 0) & (df['attack_type'].isna())][['text', 'label', 'attack_type', 'original_text']].copy()
print(f"垃圾文本数: {len(spam_texts)}")


def mix_with_matched_normals(attack_df):
    normal_count = round(len(attack_df) * len(normal_df) / len(spam_texts))
    return pd.concat([
        normal_df.sample(n=normal_count, random_state=42),
        attack_df,
    ], ignore_index=True).sample(frac=1, random_state=42)

from utils import set_seed, DATA_ADV, RESULTS_DIR
from attack.char_shuffle import attack_shuffle
from attack.homophone_chinese import attack_homophone
from attack import is_attack_applicable
from defense.text_channel import BertClassifier
from defense.fusion_model import FusionClassifier, create_data_loader

set_seed(42)
random.seed(42)

# ============================================
# 生成 3 种加强攻击
# ============================================
print("\n生成加强攻击...")

# 仅保留适用且实际发生改写的强攻击样本。
def strong_attack_df(attack_id, transform):
    rows = []
    for original_text in spam_texts:
        if not is_attack_applicable(original_text, attack_id):
            continue
        attacked_text = transform(original_text)
        if attacked_text != original_text:
            rows.append({
                'text': attacked_text,
                'label': 1,
                'attack_type': attack_id,
                'original_text': original_text,
            })
    return pd.DataFrame(rows, columns=['text', 'label', 'attack_type', 'original_text'])

# J: 强乱序 (大窗口)
df_J = strong_attack_df('J', lambda text: attack_shuffle(text, window_size=7, shuffle_ratio=0.8))
df_J_mixed = mix_with_matched_normals(df_J)
df_J_mixed.to_csv(os.path.join(DATA_ADV, 'adv_J_strong_shuffle.csv'), index=False)
print(f"  J.强乱序: {len(df_J)} 条")

# K: 强音近 (高替换率)
df_K = strong_attack_df('K', lambda text: attack_homophone(text, replace_ratio=0.8))
df_K_mixed = mix_with_matched_normals(df_K)
df_K_mixed.to_csv(os.path.join(DATA_ADV, 'adv_K_strong_homophone.csv'), index=False)
print(f"  K.强音近: {len(df_K)} 条")

# L: 混合攻击 (音近 + 乱序)
df_L = strong_attack_df(
    'L',
    lambda text: attack_shuffle(attack_homophone(text, replace_ratio=0.8), window_size=5, shuffle_ratio=0.8),
)
df_L_mixed = mix_with_matched_normals(df_L)
df_L_mixed.to_csv(os.path.join(DATA_ADV, 'adv_L_combined.csv'), index=False)
print(f"  L.混合(音近+乱序): {len(df_L)} 条")

# 构建扩展测试集（原测试集 + 3种强攻击）
test_ext = pd.concat([df, df_J, df_K, df_L], ignore_index=True)
print(f"\n扩展测试集: {len(test_ext)} 条 (原{len(df)} + 强攻击{len(df_J)+len(df_K)+len(df_L)})")

# ============================================
# 重新评测：朴素 BERT vs 四通道融合
# ============================================
print("\n" + "="*50)
print("  评测: 朴素 BERT vs 四通道融合")
print("  测试子集: 原始 + A~I (原) + J/K/L (强)")
print("="*50)

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
print("\n加载模型...")
model_bert = BertClassifier().to(DEV)
model_bert.load_state_dict(torch.load(os.path.join(BASE, 'data', 'processed', 'baseline_bert.pth'), map_location=DEV))
print("  朴素 BERT 加载 OK")

model_fusion = FusionClassifier(freeze_channels=False).to(DEV)
ck = torch.load(os.path.join(BASE, 'data', 'processed', 'fusion_model.pth'), map_location=DEV)
model_fusion.load_state_dict(ck['model_state'])
print("  四通道融合 加载 OK")

ATTACK_NAMES = {
    'A': '字符删除', 'B': '字符插入', 'C': '跨语种同形', 'D': '零宽注入',
    'E': '同义词', 'F': '音近字', 'G': '形近字', 'H': '繁简混用', 'I': '字符乱序',
    'J': '★强乱序', 'K': '★强音近', 'L': '★混合攻击'
}

# 构建子集
subsets = {}
orig_mask = test_ext['attack_type'].isna()
subsets['原始样本'] = (test_ext[orig_mask]['text'].tolist(), test_ext[orig_mask]['label'].tolist())

for aid in list('ABCDEFGHI') + list('JKL'):
    file_map = {
        'A': 'adv_A_char_delete.csv', 'B': 'adv_B_char_insert.csv',
        'C': 'adv_C_homoglyph_unicode.csv', 'D': 'adv_D_zero_width.csv',
        'E': 'adv_E_synonym.csv', 'F': 'adv_F_homophone_cn.csv',
        'G': 'adv_G_homoglyph_cn.csv', 'H': 'adv_H_fanjian_split.csv',
        'I': 'adv_I_char_shuffle.csv', 'J': 'adv_J_strong_shuffle.csv',
        'K': 'adv_K_strong_homophone.csv', 'L': 'adv_L_combined.csv',
    }
    path = os.path.join(DATA_ADV, file_map[aid])
    if os.path.exists(path):
        group = pd.read_csv(path)
        name = ATTACK_NAMES.get(aid, aid)
        subsets[f'对抗_{aid} ({name})'] = (group['text'].astype(str).tolist(), group['label'].astype(int).tolist())

# 评测
results = []
def eval_model(model, texts, labels, ablation=None):
    """评测函数，返回 F1 和 Accuracy"""
    loader = create_data_loader(texts, labels, batch_size=16, shuffle=False)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_texts, batch_labels in loader:
            if isinstance(model, FusionClassifier) and ablation:
                logits = model(batch_texts, ablation=ablation)
            else:
                logits = model(batch_texts)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch_labels.cpu().tolist())
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    accuracy = sum(1 for p,l in zip(all_preds, all_labels) if p==l) / len(all_labels)
    return f1, accuracy

for model_name, model in [('朴素 BERT', model_bert), ('四通道融合', model_fusion)]:
    print(f"\n  {model_name}:")
    model.eval()
    for sname, (txts, lbs) in subsets.items():
        f1, acc = eval_model(model, txts, lbs)
        results.append({'model': model_name, 'subset': sname, 'f1': f1, 'accuracy': acc})
        marker = ''
        if 'J' in sname or 'K' in sname or 'L' in sname:
            marker = ' ★'
        print(f"    {sname:30s} | F1={f1:.4f}  Acc={acc:.4f}{marker}")

# 消融
print(f"\n{('='*50)}")
print(f"  消融实验")
print(f"{'='*50}")
for channel in ['text', 'phonetic', 'visual', 'bow']:
    print(f"\n  四通道融合 - 消融: 置零 {channel} 通道")
    model_fusion.eval()
    for sname, (txts, lbs) in subsets.items():
        f1, acc = eval_model(model_fusion, txts, lbs, ablation=[channel])
        results.append({'model': f"四通道融合: N{channel}", 'subset': sname, 'f1': f1, 'accuracy': acc})
        marker = ''
        if 'J' in sname or 'K' in sname or 'L' in sname:
            marker = ' ★'
        print(f"    {sname:30s} | F1={f1:.4f}  Acc={acc:.4f}{marker}")


# 对比表
rdf = pd.DataFrame(results)
rdf.to_csv(os.path.join(RESULTS_DIR, 'strong_attack_results.csv'), index=False)

print(f"\n{'='*50}")
print(f"  对比：强攻击下的 F1 差异")
print(f"{'='*50}")
title = f"{'攻击类型':25s} | {'朴素BERT':>8s} | {'四通道融合':>8s}"
for channel in ['text', 'phonetic', 'visual', 'bow']:
    title += f" | {'融合: N'+channel:>8s}"
print(title)
print("-" * 58)
for aid in list('ABCDEFGHI') + list('JKL'):
    name = ATTACK_NAMES.get(aid, aid)
    subset_key = f'对抗_{aid} ({name})'
    bert_row = rdf[(rdf.model=='朴素 BERT') & (rdf.subset==subset_key)]
    fusion_row = rdf[(rdf.model=='四通道融合') & (rdf.subset==subset_key)]
    ablation_rows = {channel: rdf[(rdf.model==f'四通道融合: N{channel}') & (rdf.subset==subset_key)] for channel in ['text', 'phonetic', 'visual', 'bow']}
    if len(bert_row) > 0 and len(fusion_row) > 0:
        bf1 = bert_row.iloc[0]['f1']
        ff1 = fusion_row.iloc[0]['f1']
        abl = {channel: ablation_rows[channel].iloc[0]['f1'] if len(ablation_rows[channel]) > 0 else None 
            for channel in ['text', 'phonetic', 'visual', 'bow']}
        print(f"{name:25s} | {bf1:8.4f} | {ff1:8.4f} | {abl['text']:8.4f} | {abl['phonetic']:8.4f} | {abl['visual']:8.4f} | {abl['bow']:8.4f}")
