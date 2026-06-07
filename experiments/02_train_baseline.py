# -*- coding: utf-8 -*-
"""
实验 02: 训练基线模型

训练两个基线模型:
  1. 朴素 BERT (无任何增强)
  2. BERT + 正规化 + A/B增强 (鲁棒基线)

两者都用于后续消融实验的对比
"""

import os
import sys
# 确保项目根目录在 sys.path 中（兼容各种运行方式）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# 切换到项目根目录
if os.getcwd() != PROJECT_ROOT:
    os.chdir(PROJECT_ROOT)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils import (
    set_seed, compute_metrics, print_metrics,
    DATA_ADV, DEVICE
)
from defense.text_channel import BertClassifier
from defense.preprocess import preprocess_text
from defense.fusion_model import create_data_loader


def train_model(model, train_loader, val_loader, epochs: int,
                lr: float, model_name: str, device: str):
    """
    通用训练函数

    返回: (训练好的模型, 历史记录)
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    print(f"\n{'='*50}")
    print(f"  训练 {model_name}")
    print(f"  Epochs: {epochs}, LR: {lr}, Device: {device}")
    print(f"{'='*50}")

    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        # -- 训练 --
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_texts, batch_labels in pbar:
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(batch_texts)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)

        # -- 验证 --
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_texts, batch_labels in val_loader:
                batch_labels = batch_labels.to(device)
                logits = model(batch_texts)
                loss = criterion(logits, batch_labels)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch_labels.cpu().tolist())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        print(f"  Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 恢复最佳模型
    if best_state:
        model.load_state_dict(best_state)

    return model, history


def main():
    set_seed(42)
    print("=" * 60)
    print("  实验 02: 训练基线模型")
    print("=" * 60)

    # ================================================================
    # 1. 加载训练数据
    # ================================================================
    train_path = os.path.join(DATA_ADV, 'train.csv')
    if not os.path.exists(train_path):
        print(f"[ERROR] 训练数据不存在: {train_path}")
        print("  请先运行: python experiments/01_generate_adv.py")
        return

    train_df = pd.read_csv(train_path)
    print(f"\n训练数据: {len(train_df)} 条")
    print(f"  正常: {(train_df['label']==0).sum()}, "
          f"垃圾: {(train_df['label']==1).sum()}")

    # ================================================================
    # 2. 划分训练/验证集 (90/10)
    # ================================================================
    train_texts = train_df['text'].tolist()
    train_labels = train_df['label'].tolist()

    # 为基线2: 应用正规化预处理
    train_texts_clean = [preprocess_text(t) for t in train_texts]

    X_tr, X_val, y_tr, y_val = train_test_split(
        train_texts, train_labels, test_size=0.1,
        random_state=42, stratify=train_labels
    )
    X_tr_clean, X_val_clean, _, _ = train_test_split(
        train_texts_clean, train_labels, test_size=0.1,
        random_state=42, stratify=train_labels
    )

    # ================================================================
    # 3. 训练基线 1: 朴素 BERT
    # ================================================================
    print(f"\n{'#'*50}")
    print(f"#  基线 1: 朴素 BERT (无增强)")
    print(f"{'#'*50}")

    model1 = BertClassifier(freeze_bert=False)

    # 使用未经增强的数据训练
    train_loader1 = create_data_loader(X_tr, y_tr, batch_size=4, shuffle=True)
    val_loader1 = create_data_loader(X_val, y_val, batch_size=8, shuffle=False)

    model1, history1 = train_model(
        model1, train_loader1, val_loader1,
        epochs=1, lr=2e-5, model_name="BERT基线", device=DEVICE
    )

    # 保存模型
    save_path1 = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed',
                              'baseline_bert.pth')
    os.makedirs(os.path.dirname(save_path1), exist_ok=True)
    torch.save(model1.state_dict(), save_path1)
    print(f"\n基线1模型已保存: {save_path1}")

    # ================================================================
    # 4. 训练基线 2: BERT + 正规化 + A/B增强
    # ================================================================
    print(f"\n{'#'*50}")
    print(f"#  基线 2: BERT + 正规化 + A/B增强")
    print(f"{'#'*50}")

    model2 = BertClassifier(freeze_bert=False)

    # 使用经正规化 + A/B增强的数据训练
    train_loader2 = create_data_loader(X_tr_clean, y_tr, batch_size=4, shuffle=True)
    val_loader2 = create_data_loader(X_val_clean, y_val, batch_size=8, shuffle=False)

    model2, history2 = train_model(
        model2, train_loader2, val_loader2,
        epochs=1, lr=2e-5, model_name="BERT+正规化+增强", device=DEVICE
    )

    save_path2 = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed',
                              'baseline_bert_aug.pth')
    torch.save(model2.state_dict(), save_path2)
    print(f"\n基线2模型已保存: {save_path2}")

    print(f"\n{'='*60}")
    print(f"  ✓ 基线模型训练完成!")
    print(f"{'='*60}")


if __name__ == '__main__':
    import sys, traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
