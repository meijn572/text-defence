# -*- coding: utf-8 -*-
"""演示脚本 —— 加载已有模型，快速推理（无需训练）"""
import sys, os

BASE = r"d:\3_second\big_data\work\text-defense"
os.chdir(BASE)
sys.path.insert(0, BASE)

import pandas as pd
test_df = pd.read_csv(os.path.join(BASE, "data", "adversarial", "test_full.csv"))
from attack import ATTACK_REGISTRY

import torch, numpy as np
from sklearn.metrics import f1_score
from defense.text_channel import BertClassifier
from defense.fusion_model import FusionClassifier, create_data_loader

DEV = torch.device("cpu")
PROCESSED = os.path.join(BASE, "data", "processed")

print("=" * 50)
print("  加载模型参数（无需训练）")
print("=" * 50)

bert = BertClassifier().to(DEV)
bert.load_state_dict(torch.load(os.path.join(PROCESSED, "baseline_bert.pth"), map_location=DEV))
bert.eval()
print("  [OK] 朴素 BERT (391MB)")

fusion = FusionClassifier(freeze_channels=False).to(DEV)
ck = torch.load(os.path.join(PROCESSED, "fusion_model.pth"), map_location=DEV)
fusion.load_state_dict(ck["model_state"])
fusion.eval()
print("  [OK] 四通道融合 (438MB)")

print("\n" + "=" * 50)
print("  BERT 全量评测")
print("=" * 50)

for aid in ["原始"] + list("ABCDEFGHI"):
    if aid == "原始":
        mask = test_df["attack_type"].isna()
        name = "原始样本"
    else:
        mask = test_df["attack_type"] == aid
        name = f"{aid}.{ATTACK_REGISTRY[aid][2]}"
    txts = test_df[mask]["text"].tolist()
    lbs = test_df[mask]["label"].tolist()

    loader = create_data_loader(txts, lbs, batch_size=16, shuffle=False)
    ap = []
    with torch.no_grad():
        for bt, bl in loader:
            ap.extend(torch.argmax(bert(bt), dim=1).cpu().tolist())
    f1 = f1_score(lbs, ap, zero_division=0)
    print(f"  {name:18s}  F1={f1:.4f}  ({len(txts)}条)")

print("\n" + "=" * 50)
print("  单条推理演示")
print("=" * 50)

examples = [
    ("明天下午三点开会请准时参加", "正常短信"),
    ("恭喜您获得iPhone大奖点击领取", "垃圾短信"),
    ("免废领取优惠卷先到先得", "形近字攻击"),
    ("佳薇芯加我好友带你赚大前", "音近字攻击"),
    ("代辦證件质量保证快速出證", "繁简混用"),
]

for text, desc in examples:
    with torch.no_grad():
        bp = torch.softmax(bert([text]), dim=1)[0]
        fp = torch.softmax(fusion([text]), dim=1)[0]
    print(f"\n  [{desc}] {text}")
    print(f"    BERT:   正常={bp[0]:.3f} 垃圾={bp[1]:.3f}")
    print(f"    Fusion: 正常={fp[0]:.3f} 垃圾={fp[1]:.3f}")

print("\n" + "=" * 50)
print("  完整评测结果（实验记录）")
print("=" * 50)
results = [
    ("原始样本", 0.9055, 0.9043), ("A 字符删除", 0.9779, 0.9796),
    ("B 字符插入", 0.9967, 0.9967), ("C 跨语种同形", 0.9510, 0.9583),
    ("D 零宽注入", 0.9529, 0.9547), ("E 同义词", 0.9529, 0.9565),
    ("F 音近字", 0.9779, 0.9761), ("G 形近字", 0.9565, 0.9583),
    ("H 繁简混用", 0.9583, 0.9529), ("I 字符乱序", 0.9865, 0.9831),
]
print(f"  {'攻击类型':14s} | {'BERT':>8s} | {'Fusion':>8s}")
print("  " + "-" * 36)
for name, bf1, ff1 in results:
    print(f"  {name:14s} | {bf1:8.4f} | {ff1:8.4f}")

print("\nDone. No training needed.")
