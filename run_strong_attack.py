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
print(f"垃圾文本数: {len(spam_texts)}")

from utils import set_seed, DATA_ADV, RESULTS_DIR
from attack.char_shuffle import attack_shuffle
from attack.homophone_chinese import attack_homophone
from defense.text_channel import BertClassifier
from defense.fusion_model import FusionClassifier, create_data_loader

set_seed(42)
random.seed(42)

# ============================================
# 生成 3 种加强攻击
# ============================================
print("\n生成加强攻击...")

# J: 强乱序 (大窗口)
adv_J = []
for t in spam_texts:
    adv_J.append(attack_shuffle(t, window_size=7, shuffle_ratio=0.8))
df_J = pd.DataFrame({'text': adv_J, 'label': [1]*len(adv_J), 'attack_type': 'J', 'original_text': spam_texts})
df_J.to_csv(os.path.join(DATA_ADV, 'adv_J_strong_shuffle.csv'), index=False)
print(f"  J.强乱序: {len(df_J)} 条")

# K: 强音近 (高替换率)
adv_K = []
for t in spam_texts:
    adv_K.append(attack_homophone(t, replace_ratio=0.8))
df_K = pd.DataFrame({'text': adv_K, 'label': [1]*len(adv_K), 'attack_type': 'K', 'original_text': spam_texts})
df_K.to_csv(os.path.join(DATA_ADV, 'adv_K_strong_homophone.csv'), index=False)
print(f"  K.强音近: {len(df_K)} 条")

# L: 混合攻击 (音近 + 乱序)
adv_L = []
for t in spam_texts:
    t2 = attack_homophone(t, replace_ratio=0.8)
    adv_L.append(attack_shuffle(t2, window_size=5, shuffle_ratio=0.8))
df_L = pd.DataFrame({'text': adv_L, 'label': [1]*len(adv_L), 'attack_type': 'L', 'original_text': spam_texts})
df_L.to_csv(os.path.join(DATA_ADV, 'adv_L_combined.csv'), index=False)
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
    mask = test_ext['attack_type'] == aid
    if mask.sum() > 0:
        name = ATTACK_NAMES.get(aid, aid)
        subsets[f'对抗_{aid} ({name})'] = (test_ext[mask]['text'].tolist(), test_ext[mask]['label'].tolist())

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
    model.eval()
    for sname, (txts, lbs) in subsets.items():
        f1, acc = eval_model(model_fusion, txts, lbs, ablation=channel)
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
