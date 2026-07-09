# -*- coding: utf-8 -*-
"""一键运行全部实验（无 subprocess，单进程顺序执行）"""
import sys, os, time

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
sys.path.insert(0, BASE)

# ⚠️ 所有 CSV 必须在 torch 之前读取
import pandas as pd
from sklearn.model_selection import train_test_split

print("[数据] 读取原始数据...")
from utils import load_raw_data, DATA_ADV, RESULTS_DIR, FIGURES_DIR, set_seed
raw_df = load_raw_data()
print(f"[数据] 原始数据: {len(raw_df)} 条")

# 检查对抗样本是否已生成，没有才生成
train_path = os.path.join(DATA_ADV, 'train.csv')
test_path  = os.path.join(DATA_ADV, 'test_full.csv')
if os.path.exists(train_path) and os.path.exists(test_path):
    print("[数据] 对抗样本已存在，跳过生成")
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
else:
    print("[数据] 生成对抗样本...")
    # 01 的逻辑：纯 pandas，直接调
    from experiments.generate_adv import main as generate_adv
    generate_adv()
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

print(f"[数据] 训练集: {len(train_df)} 条，测试集: {len(test_df)} 条")

# 现在才导入 torch
import torch, torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from defense.text_channel import BertClassifier
from defense.preprocess import preprocess_text
from defense.fusion_model import FusionClassifier, create_data_loader
from attack import ATTACK_REGISTRY
from attack.char_shuffle import attack_adjacent_swap
from attack.homophone_chinese import attack_homophone

set_seed(42)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] 使用设备: {DEV}")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE, 'data', 'processed'), exist_ok=True)

# ============================================================
# 通用工具
# ============================================================
def train_model(model, train_loader, val_loader, epochs, lr, name, metric='acc'):
    """通用训练函数，metric='acc' 或 'f1'"""
    model = model.to(DEV)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_score, best_state = 0.0, None

    print(f"\n{'='*55}")
    print(f"  训练: {name} | epochs={epochs} lr={lr} device={DEV}")
    print(f"{'='*55}")

    t0 = time.time()
    for epoch in range(epochs):
        model.train(); total_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for bt, bl in pbar:
            bl = bl.to(DEV); optimizer.zero_grad()
            loss = criterion(model(bt), bl)
            loss.backward(); optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        model.eval(); ap, al = [], []
        with torch.no_grad():
            for bt, bl in val_loader:
                bl = bl.to(DEV)
                ap.extend(torch.argmax(model(bt), dim=1).cpu().tolist())
                al.extend(bl.cpu().tolist())

        acc = sum(1 for a, b in zip(ap, al) if a == b) / len(al)
        vf1 = f1_score(al, ap, zero_division=0)
        score = vf1 if metric == 'f1' else acc
        print(f"  Epoch {epoch+1}/{epochs} | Loss={total_loss/len(train_loader):.4f} "
              f"| Acc={acc:.4f} | F1={vf1:.4f} | T={time.time()-t0:.0f}s")

        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model


def eval_subset(model, texts, labels, ablation_channel=None):
    """评测单个子集，返回 {'f1': ..., 'accuracy': ...}"""
    loader = create_data_loader(texts, labels, batch_size=16, shuffle=False)
    ap, al = [], []
    model.eval()
    with torch.no_grad():
        for bt, bl in loader:
            if isinstance(model, FusionClassifier) and ablation_channel:
                logits = model(bt, ablation=[ablation_channel])
            else:
                logits = model(bt)
            ap.extend(torch.argmax(logits, dim=1).cpu().tolist())
            al.extend(bl.cpu().tolist())
    return {
        'f1':       f1_score(al, ap, zero_division=0),
        'accuracy': sum(1 for a, b in zip(ap, al) if a == b) / len(al)
    }


def load_model(cls, path):
    """加载模型，自动判断保存格式"""
    pp = os.path.join(BASE, 'data', 'processed', path)
    if not os.path.exists(pp):
        return None
    m = (FusionClassifier(freeze_channels=True) if cls == FusionClassifier
         else cls()).to(DEV)
    ck = torch.load(pp, map_location=DEV)
    state = ck['model_state'] if 'model_state' in ck else ck
    missing, unexpected = m.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  [警告] missing={len(missing)}, unexpected={len(unexpected)}")
    return m


# ============================================================
# 实验 02: 训练基线模型
# ============================================================
print("\n" + "=" * 55)
print("  实验 02: 训练基线模型")
print("=" * 55)

train_texts  = train_df['text'].tolist()
train_labels = train_df['label'].tolist()
train_texts_clean = [preprocess_text(t) for t in train_texts]

X_tr, X_val, y_tr, y_val = train_test_split(
    train_texts, train_labels, test_size=0.1, random_state=42, stratify=train_labels)
X_tr_c, X_val_c, _, _ = train_test_split(
    train_texts_clean, train_labels, test_size=0.1, random_state=42, stratify=train_labels)

# 基线1：朴素 BERT
model_bert = train_model(
    BertClassifier(freeze_bert=False),
    create_data_loader(X_tr,   y_tr,  batch_size=4, shuffle=True),
    create_data_loader(X_val,  y_val, batch_size=8, shuffle=False),
    epochs=3, lr=2e-5, name='朴素 BERT'
)
torch.save(model_bert.state_dict(),
           os.path.join(BASE, 'data', 'processed', 'baseline_bert.pth'))
print("✅ baseline_bert.pth 已保存")

# 基线2：BERT + 正规化
model_bert_aug = train_model(
    BertClassifier(freeze_bert=False),
    create_data_loader(X_tr_c,  y_tr,  batch_size=4, shuffle=True),
    create_data_loader(X_val_c, y_val, batch_size=8, shuffle=False),
    epochs=3, lr=2e-5, name='BERT + 正规化'
)
torch.save(model_bert_aug.state_dict(),
           os.path.join(BASE, 'data', 'processed', 'baseline_bert_aug.pth'))
print("✅ baseline_bert_aug.pth 已保存")

# ============================================================
# 实验 03: 训练融合模型
# ============================================================
print("\n" + "=" * 55)
print("  实验 03: 训练四通道融合模型")
print("=" * 55)

model_fusion = FusionClassifier(freeze_channels=False).to(DEV)

# 加载 BERT 预训练权重
pretrained = torch.load(
    os.path.join(BASE, 'data', 'processed', 'baseline_bert.pth'), map_location=DEV)
bert_keys = {k: v for k, v in pretrained.items() if 'text_channel.bert' in k}
missing, _ = model_fusion.load_state_dict(bert_keys, strict=False)
print(f"BERT权重已加载 ({len(bert_keys)} keys)")

# 冻结 BERT，只训练其余通道和融合头
for n, p in model_fusion.named_parameters():
    if 'text_channel.bert' in n:
        p.requires_grad = False
trainable = [p for p in model_fusion.parameters() if p.requires_grad]
print(f"可训练参数: {sum(p.numel() for p in trainable):,}")

model_fusion = train_model(
    model_fusion,
    create_data_loader(X_tr,  y_tr,  batch_size=4, shuffle=True),
    create_data_loader(X_val, y_val, batch_size=4, shuffle=False),
    epochs=5, lr=1e-4, name='四通道融合', metric='f1'
)
torch.save({'model_state': model_fusion.state_dict()},
           os.path.join(BASE, 'data', 'processed', 'fusion_model.pth'))
print("✅ fusion_model.pth 已保存")

# ============================================================
# 实验 04: 全面评测（含强攻击 + 消融）
# ============================================================
print("\n" + "=" * 55)
print("  实验 04: 全面评测")
print("=" * 55)

# 生成强攻击子集 J/K/L
spam_mask  = (test_df['label'] == 1) & (test_df['attack_type'].isna())
spam_texts = test_df[spam_mask]['text'].tolist()

df_J = pd.DataFrame({'text': [attack_adjacent_swap(t, swap_ratio=0.8) for t in spam_texts],
                     'label': 1, 'attack_type': 'J', 'original_text': spam_texts})
df_K = pd.DataFrame({'text': [attack_homophone(t, replace_ratio=0.8) for t in spam_texts],
                     'label': 1, 'attack_type': 'K', 'original_text': spam_texts})
df_L = pd.DataFrame({
    'text': [
        attack_adjacent_swap(
            attack_homophone(t, replace_ratio=0.8),
            swap_ratio=0.8
        )
        for t in spam_texts
    ],
    'label': 1,
    'attack_type': 'L',
    'original_text': spam_texts
})
for df_adv, fname in [(df_J, 'adv_J_strong_shuffle.csv'),
                      (df_K, 'adv_K_strong_homophone.csv'),
                      (df_L, 'adv_L_combined.csv')]:
    df_adv.to_csv(os.path.join(DATA_ADV, fname), index=False)
print(f"强攻击样本生成: J={len(df_J)}, K={len(df_K)}, L={len(df_L)} 条")

test_ext = pd.concat([test_df, df_J, df_K, df_L], ignore_index=True)

# 构建子集
ATTACK_NAMES = {
    'A':'字符删除','B':'字符插入','C':'跨语种同形','D':'零宽注入',
    'E':'同义词','F':'音近字','G':'形近字','H':'繁简混用','I':'字符乱序',
    'J':'★强乱序','K':'★强音近','L':'★混合攻击'
}
subsets = {}
orig_mask = test_ext['attack_type'].isna()
subsets['原始样本'] = (test_ext[orig_mask]['text'].tolist(),
                       test_ext[orig_mask]['label'].tolist())
for aid in 'ABCDEFGHIJKL':
    mask = test_ext['attack_type'] == aid
    if mask.sum() > 0:
        subsets[f'对抗_{aid} ({ATTACK_NAMES[aid]})'] = (
            test_ext[mask]['text'].tolist(),
            test_ext[mask]['label'].tolist())

# 加载所有模型（评测用，freeze_channels=True）
eval_models = {
    '朴素 BERT':        load_model(BertClassifier,   'baseline_bert.pth'),
    'BERT + 正规化':    load_model(BertClassifier,   'baseline_bert_aug.pth'),
    '四通道融合 (本文)': load_model(FusionClassifier, 'fusion_model.pth'),
}
eval_models = {k: v for k, v in eval_models.items() if v is not None}

# 正常评测
results = []
for mname, mdl in eval_models.items():
    print(f"\n  {mname}:")
    for sname, (txts, lbs) in subsets.items():
        m = eval_subset(mdl, txts, lbs)
        results.append({'model': mname, 'subset': sname, **m})
        print(f"    {sname:30s} | F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}")

# 消融实验
print("\n消融实验 (置零近似，存在分布偏移，结果供参考):")
fusion_eval = eval_models.get('四通道融合 (本文)')
if fusion_eval:
    for channel in ['text', 'phonetic', 'visual', 'bow']:
        print(f"\n  置零 {channel} 通道:")
        for sname, (txts, lbs) in subsets.items():
            m = eval_subset(fusion_eval, txts, lbs, ablation_channel=channel)
            results.append({'model': f'融合-置零{channel}', 'subset': sname, **m})
            print(f"    {sname:30s} | F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}")

# 保存结果
rdf = pd.DataFrame(results)
rdf.to_csv(os.path.join(RESULTS_DIR, 'eval_results.csv'), index=False)

# 可视化（只画三个主模型）
main_models = ['朴素 BERT', 'BERT + 正规化', '四通道融合 (本文)']
plot_df = rdf[rdf['model'].isin(main_models)]
pivot   = plot_df.pivot(index='subset', columns='model', values='f1')
fig, ax = plt.subplots(figsize=(14, 6))
pivot.plot(kind='bar', ax=ax)
ax.set_title('各攻击类型下不同模型的 F1 对比')
ax.set_ylabel('F1 Score'); ax.set_ylim(0.0, 1.05)
ax.legend(loc='lower right')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'compare_f1.png'), dpi=150)
print(f"\n图表已保存: {FIGURES_DIR}/compare_f1.png")

print("\n" + "=" * 55)
print("  ✅ 全部实验完成!")
print(f"  结果: {RESULTS_DIR}/eval_results.csv")
print(f"  图表: {FIGURES_DIR}/compare_f1.png")
print("=" * 55)