# -*- coding: utf-8 -*-
"""Evaluate BERT/Fusion recall by weakly supervised spam content type.

Outputs:
- results/content_type_model_predictions.csv
- results/content_type_model_recall.csv

This script evaluates only spam samples (label=1) for content-type recall,
because the content type labels are defined for spam categories.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.metrics import recall_score

BASE = Path(__file__).resolve().parent
os.chdir(BASE)
sys.path.insert(0, str(BASE))

from content_type_analysis import assign_content_type  # noqa: E402
from defense.fusion_model import FusionClassifier  # noqa: E402
from defense.text_channel import BertClassifier  # noqa: E402

DATA_DIR = BASE / "data" / "adversarial"
PROCESSED = BASE / "data" / "processed"
RESULT_DIR = BASE / "results"
RESULT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16 if torch.cuda.is_available() else 4
MAX_PER_CONTENT_TYPE_CPU = 30


def load_test_spam(full: bool = False) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "test_full.csv")
    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(int)
    df = df[df["label"] == 1].copy().reset_index(drop=True)
    df["content_source_text"] = df.apply(
        lambda row: row["original_text"] if pd.notna(row.get("original_text")) else row["text"],
        axis=1,
    )
    df["content_type"] = df["content_source_text"].apply(assign_content_type)
    df["attack_type"] = df["attack_type"].fillna("Original")

    if not full and DEVICE.type == "cpu":
        df = (
            df.groupby("content_type", group_keys=False)
            .apply(lambda part: part.sample(min(len(part), MAX_PER_CONTENT_TYPE_CPU), random_state=42))
            .reset_index(drop=True)
        )
        print(f"[INFO] CPU quick mode: sampled up to {MAX_PER_CONTENT_TYPE_CPU} spam samples per content type")
    return df


def load_models() -> tuple[BertClassifier, FusionClassifier]:
    bert_path = PROCESSED / "baseline_bert.pth"
    fusion_path = PROCESSED / "fusion_model.pth"
    if not bert_path.exists():
        raise FileNotFoundError(f"Missing model file: {bert_path}")
    if not fusion_path.exists():
        raise FileNotFoundError(f"Missing model file: {fusion_path}")

    print(f"[INFO] device={DEVICE}, batch_size={BATCH_SIZE}")

    bert = BertClassifier().to(DEVICE)
    bert.load_state_dict(torch.load(bert_path, map_location=DEVICE))
    bert.eval()

    fusion = FusionClassifier(freeze_channels=False, device=DEVICE).to(DEVICE)
    checkpoint = torch.load(fusion_path, map_location=DEVICE)
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    fusion.load_state_dict(state_dict)
    fusion.eval()
    return bert, fusion


@torch.inference_mode()
def predict_batches(model: torch.nn.Module, texts: list[str], model_name: str) -> tuple[list[int], list[float]]:
    preds: list[int] = []
    spam_probs: list[float] = []
    start = time.perf_counter()
    for start_idx in range(0, len(texts), BATCH_SIZE):
        batch = texts[start_idx:start_idx + BATCH_SIZE]
        logits = model(batch)
        probs = torch.softmax(logits, dim=1).detach().cpu()
        preds.extend(torch.argmax(probs, dim=1).tolist())
        spam_probs.extend(probs[:, 1].tolist())
    elapsed = time.perf_counter() - start
    print(f"[{model_name}] predicted {len(texts)} samples in {elapsed:.1f}s")
    return preds, spam_probs


def summarize_recall(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    models = ["BERT", "Fusion"]
    for content_type, part in df.groupby("content_type"):
        y_true = part["label"].tolist()
        row: dict[str, Any] = {
            "content_type": content_type,
            "n_samples": len(part),
        }
        for model in models:
            pred_col = f"{model}_pred"
            recall = recall_score(y_true, part[pred_col].tolist(), zero_division=0)
            missed = int((part[pred_col] == 0).sum())
            row[f"{model}_recall"] = recall
            row[f"{model}_missed"] = missed
        row["Fusion_minus_BERT"] = row["Fusion_recall"] - row["BERT_recall"]
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("n_samples", ascending=False)

    total_row: dict[str, Any] = {
        "content_type": "ALL_SPAM",
        "n_samples": len(df),
    }
    y_true_all = df["label"].tolist()
    for model in models:
        pred_col = f"{model}_pred"
        total_row[f"{model}_recall"] = recall_score(y_true_all, df[pred_col].tolist(), zero_division=0)
        total_row[f"{model}_missed"] = int((df[pred_col] == 0).sum())
    total_row["Fusion_minus_BERT"] = total_row["Fusion_recall"] - total_row["BERT_recall"]
    summary = pd.concat([pd.DataFrame([total_row]), summary], ignore_index=True)
    return summary


def main() -> None:
    df = load_test_spam()
    bert, fusion = load_models()
    texts = df["text"].astype(str).tolist()

    bert_preds, bert_probs = predict_batches(bert, texts, "BERT")
    fusion_preds, fusion_probs = predict_batches(fusion, texts, "Fusion")

    df["BERT_pred"] = bert_preds
    df["BERT_spam_prob"] = bert_probs
    df["Fusion_pred"] = fusion_preds
    df["Fusion_spam_prob"] = fusion_probs

    predictions_path = RESULT_DIR / "content_type_model_predictions.csv"
    df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    summary = summarize_recall(df)
    recall_path = RESULT_DIR / "content_type_model_recall.csv"
    summary.to_csv(recall_path, index=False, encoding="utf-8-sig")

    print("\nRecall by content type:")
    display_cols = [
        "content_type", "n_samples", "BERT_recall", "Fusion_recall",
        "Fusion_minus_BERT", "BERT_missed", "Fusion_missed",
    ]
    print(summary[display_cols].round(4).to_string(index=False))
    print(f"\nSaved: {predictions_path}")
    print(f"Saved: {recall_path}")


if __name__ == "__main__":
    main()
