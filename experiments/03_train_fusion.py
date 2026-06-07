# -*- coding: utf-8 -*-
"""
实验 03: 训练四通道融合模型

训练策略:
  1. 先单独训练各子通道 (可选, 或直接用预训练权重)
  2. 冻结子通道参数, 只训练融合头
  3. (可选) 解冻后微调全模型

由于课程项目资源有限, 这里采用简化策略:
  - BERT: 加载基线2预训练权重
  - 拼音CNN/视觉CNN/字袋MLP: 随机初始化, 与融合头一起训练
"""

import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils import set_seed, compute_metrics, print_metrics, DATA_ADV, DEVICE
from defense.preprocess import preprocess_text
from defense.fusion_model import (
    FusionClassifier, create_data_loader, train_epoch, evaluate
)


def main():
    set_seed(42)
    print("=" * 60)
    print("  实验 03: 训练四通道融合模型")
    print("=" * 60)

    # ================================================================
    # 1. 加载训练数据
    # ================================================================
    train_path = os.path.join(DATA_ADV, 'train.csv')
    if not os.path.exists(train_path):
        print(f"[ERROR] 训练数据不存在: {train_path}")
        return

    train_df = pd.read_csv(train_path)
    texts = train_df['text'].tolist()
    labels = train_df['label'].tolist()
    print(f"\n训练数据: {len(texts)} 条")

    # 划分训练/验证集
    X_tr, X_val, y_tr, y_val = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )
    print(f"  训练: {len(X_tr)} 条, 验证: {len(X_val)} 条")

    train_loader = create_data_loader(X_tr, y_tr, batch_size=2, shuffle=True)
    val_loader = create_data_loader(X_val, y_val, batch_size=4, shuffle=False)

    # ================================================================
    # 2. 创建融合模型（CPU 训练，避免 GPU 驱动兼容性问题）
    # ================================================================
    print(f"\n[模型] 创建四通道融合模型...")
    model = FusionClassifier(freeze_channels=False)
    TRAIN_DEVICE = torch.device('cpu')  # 用 CPU 训练（GPU 训练有驱动兼容性bug）
    model = model.to(TRAIN_DEVICE)

    # 尝试加载预训练的 BERT 权重
    pretrained_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'processed', 'baseline_bert.pth'
    )
    if os.path.exists(pretrained_path):
        print(f"[模型] 加载预训练 BERT 权重: {pretrained_path}")
        pretrained = torch.load(pretrained_path, map_location=TRAIN_DEVICE)
        bert_state = {k: v for k, v in pretrained.items()
                      if k.startswith('text_channel.bert')}
        model.load_state_dict(bert_state, strict=False)
        print("[模型] BERT 权重加载成功")

        # 冻结 BERT
        for n, p in model.named_parameters():
            if 'text_channel.bert' in n:
                p.requires_grad = False
        print("[模型] BERT 已冻结")

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"[模型] 可训练: {sum(p.numel() for p in trainable_params):,} "
              f"/ 总计: {sum(p.numel() for p in model.parameters()):,}")
    else:
        trainable_params = model.parameters()

    # ================================================================
    # 3. CPU 训练
    # ================================================================
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    epochs = 2

    print(f"\n{'='*50}")
    print(f"  训练四通道融合模型 (CPU, epochs={epochs})")
    print(f"{'='*50}")

    best_f1 = 0.0
    best_state = None

    for epoch in range(epochs):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, TRAIN_DEVICE)

        # 验证
        val_loss, val_preds, val_labels = evaluate(
            model, val_loader, criterion, TRAIN_DEVICE
        )
        metrics = compute_metrics(
            np.array(val_labels), np.array(val_preds)
        )

        print(f"  Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Acc: {metrics['accuracy']:.4f} | "
              f"F1: {metrics['f1']:.4f}")

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 恢复最佳模型
    if best_state:
        model.load_state_dict(best_state)
    print(f"\n最佳验证 F1: {best_f1:.4f}")

    # ================================================================
    # 4. 保存模型
    # ================================================================
    save_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed',
                             'fusion_model.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'best_f1': best_f1,
    }, save_path)
    print(f"\n融合模型已保存: {save_path}")

    print(f"\n{'='*60}")
    print(f"  ✓ 四通道融合模型训练完成!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
