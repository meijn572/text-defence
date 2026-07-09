# -*- coding: utf-8 -*-
"""根据当前数据集与已训练模型计算完整分类指标。
输出：results/eval_results_pr.csv
"""

import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from defense.text_channel import BertClassifier
from defense.fusion_model import FusionClassifier, create_data_loader


BASE = os.path.dirname(os.path.abspath(__file__))
DATA_ADV = os.path.join(BASE, "data", "adversarial")
DATA_PROCESSED = os.path.join(BASE, "data", "processed")
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(cls, ckpt_name):
    ckpt_path = os.path.join(DATA_PROCESSED, ckpt_name)
    if not os.path.exists(ckpt_path):
        return None

    model = (FusionClassifier(freeze_channels=True) if cls == FusionClassifier else cls()).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def eval_subset(model, texts, labels):
    loader = create_data_loader(texts, labels, batch_size=16, shuffle=False)
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch_texts, batch_labels in loader:
            logits = model(batch_texts)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(batch_labels.cpu().tolist())

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "support": len(y_true),
    }


def build_subsets():
    test_full = pd.read_csv(os.path.join(DATA_ADV, "test_full.csv"))

    subsets = {}
    orig_mask = test_full["attack_type"].isna()
    subsets["原始样本"] = (
        test_full[orig_mask]["text"].tolist(),
        test_full[orig_mask]["label"].tolist(),
    )

    file_map = {
        "A": "adv_A_char_delete.csv",
        "B": "adv_B_char_insert.csv",
        "C": "adv_C_homoglyph_unicode.csv",
        "D": "adv_D_zero_width.csv",
        "E": "adv_E_synonym.csv",
        "F": "adv_F_homophone_cn.csv",
        "G": "adv_G_homoglyph_cn.csv",
        "H": "adv_H_fanjian_split.csv",
        "I": "adv_I_char_shuffle.csv",
        "J": "adv_J_strong_shuffle.csv",
        "K": "adv_K_strong_homophone.csv",
        "L": "adv_L_combined.csv",
    }
    name_map = {
        "A": "字符删除",
        "B": "字符插入",
        "C": "跨语种同形",
        "D": "零宽注入",
        "E": "同义词",
        "F": "音近字",
        "G": "形近字",
        "H": "繁简混用",
        "I": "字符乱序",
        "J": "强乱序",
        "K": "强音近",
        "L": "混合攻击",
    }

    for key, filename in file_map.items():
        path = os.path.join(DATA_ADV, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            subsets[f"对抗_{key} ({name_map[key]})"] = (df["text"].tolist(), df["label"].tolist())

    return subsets


def main():
    subsets = build_subsets()

    models = {
        "朴素 BERT": load_model(BertClassifier, "baseline_bert.pth"),
        "BERT + 正规化": load_model(BertClassifier, "baseline_bert_aug.pth"),
        "四通道融合 (本文)": load_model(FusionClassifier, "fusion_model.pth"),
    }
    models = {k: v for k, v in models.items() if v is not None}

    if not models:
        raise RuntimeError("没有可用模型，请先训练 baseline_bert.pth / baseline_bert_aug.pth / fusion_model.pth")

    rows = []
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        for subset_name, (texts, labels) in subsets.items():
            m = eval_subset(model, texts, labels)
            rows.append({"model": model_name, "subset": subset_name, **m})
            print(
                f"  {subset_name:22s} "
                f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} Acc={m['accuracy']:.4f}"
            )

    result_df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "eval_results_pr.csv")
    result_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n已保存: {out_csv}")
    print("\n模型平均指标（按子集简单平均）:")
    print(result_df.groupby("model")[["precision", "recall", "f1", "accuracy"]].mean().round(4))


if __name__ == "__main__":
    main()
