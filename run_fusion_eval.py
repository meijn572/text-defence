# -*- coding: utf-8 -*-
"""实验03+04: 融合模型训练 + 评测 (CSV先于torch加载)"""
import sys, os, time, traceback

BASE = r'd:\3_second\big_data\work\text-defense'
os.chdir(BASE)
sys.path.insert(0, BASE)

# ⚠️ 关键：CSV必须在torch之前读取！
import pandas as pd
train_df = pd.read_csv(os.path.join(BASE, 'data', 'adversarial', 'train.csv'))
print(f"[数据] 训练集: {len(train_df)} 条")
test_df = pd.read_csv(os.path.join(BASE, 'data', 'adversarial', 'test_full.csv'))
print(f"[数据] 测试集: {len(test_df)} 条")

# 现在可以安全导入torch
import torch, torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

from utils import set_seed, DATA_ADV, DEVICE
from attack import ATTACK_REGISTRY
from defense.text_channel import BertClassifier
from defense.fusion_model import FusionClassifier, create_data_loader

set_seed(42)
DEV = torch.device('cpu')

# ========================================
# 实验03: 训练融合模型
# ========================================
print("\n" + "=" * 60)
print("  实验 03: 训练四通道融合模型")
print("=" * 60)

texts = train_df['text'].tolist()
labels = train_df['label'].tolist()
X_tr, X_val, y_tr, y_val = train_test_split(
    texts, labels, test_size=0.1, random_state=42, stratify=labels)
print(f"训练: {len(X_tr)} / 验证: {len(X_val)}")

train_loader = create_data_loader(X_tr, y_tr, batch_size=4, shuffle=True)
val_loader = create_data_loader(X_val, y_val, batch_size=4, shuffle=False)

model = FusionClassifier(freeze_channels=False).to(DEV)
pt_path = os.path.join(BASE, 'data', 'processed', 'baseline_bert.pth')
pretrained = torch.load(pt_path, map_location=DEV)
bert_keys = {k: v for k, v in pretrained.items() if 'text_channel.bert' in k}
model.load_state_dict(bert_keys, strict=False)
print(f"BERT权重已加载 ({len(bert_keys)} keys)")

for n, p in model.named_parameters():
    if 'text_channel.bert' in n:
        p.requires_grad = False
trainable = [p for p in model.parameters() if p.requires_grad]
print(f"可训练参数: {sum(p.numel() for p in trainable):,}")

opt = torch.optim.AdamW(trainable, lr=1e-4)
crit = nn.CrossEntropyLoss()
best_f1 = 0.0
best_state = None

t0 = time.time()
for epoch in range(2):
    model.train(); total_loss = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/2')
    for bt, bl in pbar:
        bl = bl.to(DEV); opt.zero_grad()
        logits = model(bt)
        loss = crit(logits, bl); loss.backward(); opt.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    train_loss = total_loss / len(train_loader)

    model.eval(); ap, al = [], []
    with torch.no_grad():
        for bt, bl in val_loader:
            bl = bl.to(DEV)
            logits = model(bt)
            ap.extend(torch.argmax(logits, dim=1).tolist())
            al.extend(bl.tolist())
    val_f1 = f1_score(al, ap, zero_division=0)
    val_acc = sum(1 for a,b in zip(ap,al) if a==b)/len(al)
    print(f"  Epoch {epoch+1}/2 | TL={train_loss:.4f} | VA={val_acc:.4f} | VF1={val_f1:.4f} | T={time.time()-t0:.0f}s")
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

if best_state:
    model.load_state_dict(best_state)
os.makedirs(os.path.join(BASE, 'data', 'processed'), exist_ok=True)
torch.save({'model_state': model.state_dict(), 'best_f1': best_f1},
           os.path.join(BASE, 'data', 'processed', 'fusion_model.pth'))
print(f"✅ 融合模型已保存 (best F1={best_f1:.4f})")

# ========================================
# 实验04: 全面评测
# ========================================
print("\n" + "=" * 60)
print("  实验 04: 全面评测与消融实验")
print("=" * 60)

def eval_model(model, texts, labels):
    loader = create_data_loader(texts, labels, batch_size=16, shuffle=False)
    model.eval(); ap, al = [], []
    with torch.no_grad():
        for bt, bl in loader:
            if isinstance(model, BertClassifier):
                logits = model(bt)
            else:
                logits = model(bt)
            ap.extend(torch.argmax(logits, dim=1).cpu().tolist())
            al.extend(bl.cpu().tolist())
    return {
        'f1': f1_score(al, ap, zero_division=0),
        'accuracy': sum(1 for a,b in zip(ap,al) if a==b)/len(al)
    }

# 构建测试子集
subsets = {}
orig_mask = test_df['attack_type'].isna()
subsets['原始样本'] = (test_df[orig_mask]['text'].tolist(), test_df[orig_mask]['label'].tolist())
for aid in 'ABCDEFGHI':
    mask = test_df['attack_type'] == aid
    if mask.sum() > 0:
        subsets[f'对抗_{aid}'] = (test_df[mask]['text'].tolist(), test_df[mask]['label'].tolist())

# 加载模型
from utils import RESULTS_DIR, FIGURES_DIR
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)

models = {}
for name, cls, path in [
    ('朴素 BERT', BertClassifier, 'baseline_bert.pth'),
    ('BERT + 正规化', BertClassifier, 'baseline_bert_aug.pth'),
    ('四通道融合 (本文)', FusionClassifier, 'fusion_model.pth'),
]:
    m = cls().to(DEV)
    pp = os.path.join(BASE, 'data', 'processed', path)
    if os.path.exists(pp):
        ck = torch.load(pp, map_location=DEV)
        if 'model_state' in ck:
            m.load_state_dict(ck['model_state'])
        else:
            m.load_state_dict(ck)
        print(f"[模型] {name} 加载成功")
    else:
        print(f"[SKIP] {name}: 模型文件不存在")
        continue
    models[name] = m

# 评测
results = []
print(f"\n评测: {len(models)} 模型 x {len(subsets)} 子集\n")
for mname, mdl in models.items():
    print(f"  {mname}:")
    for sname, (txts, lbs) in subsets.items():
        m = eval_model(mdl, txts, lbs)
        results.append({'model': mname, 'subset': sname, **m})
        print(f"    {sname:12s} | F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}")

rdf = pd.DataFrame(results)
rdf.to_csv(os.path.join(RESULTS_DIR, 'eval_results.csv'), index=False)

# 可视化
print("\n[可视化] 生成图表...")
fig, ax = plt.subplots(figsize=(14, 6))
pivot = rdf.pivot(index='subset', columns='model', values='f1')
pivot.plot(kind='bar', ax=ax)
ax.set_title('各攻击类型下不同模型的 F1 对比'); ax.set_ylabel('F1 Score')
ax.legend(loc='lower right'); plt.xticks(rotation=45, ha='right')
plt.tight_layout(); fig.savefig(os.path.join(FIGURES_DIR, 'compare_f1.png'), dpi=150)
print(f"  图表已保存: {FIGURES_DIR}")

print("\n" + "=" * 60)
print("  ✅ 实验03+04 全部完成!")
print(f"  结果: {RESULTS_DIR}/eval_results.csv")
print(f"  图表: {FIGURES_DIR}/")
print("=" * 60)
