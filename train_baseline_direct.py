# -*- coding: utf-8 -*-
"""直接训练脚本 - 绕过模块加载问题"""
import sys, os, traceback

# 设置项目路径
BASE = r'd:\3_second\big_data\work\text-defense'
os.chdir(BASE)
if BASE not in sys.path:
    sys.path.insert(0, BASE)

try:
    print("Step 1: Reading data first (before heavy imports)...", flush=True)
    import pandas as pd
    import os, sys
    from utils import DATA_ADV
    
    train_path = os.path.join(DATA_ADV, 'train.csv')
    train_df = pd.read_csv(train_path)
    print(f"  Loaded {len(train_df)} records", flush=True)

    print("Step 2: Importing ML modules...", flush=True)
    import torch
    import torch.nn as nn
    import numpy as np
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm
    print("  Basic imports OK", flush=True)

    from utils import set_seed, DEVICE
    print(f"  Utils OK, DEVICE={DEVICE}", flush=True)

    from defense.text_channel import BertClassifier
    print("  BertClassifier OK", flush=True)

    from defense.preprocess import preprocess_text
    print("  preprocess_text OK", flush=True)

    from defense.fusion_model import create_data_loader
    print("  create_data_loader OK", flush=True)

    # 训练函数
    def train_model(model, train_loader, val_loader, epochs, lr, model_name, device):
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0
        best_state = None

        print(f"\n{'='*50}")
        print(f"  训练 {model_name}")
        print(f"  Epochs: {epochs}, LR: {lr}, Device: {device}")
        print(f"{'='*50}")

        for epoch in range(epochs):
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
            print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)
        return model

    # ================================================================
    # 主流程 - 数据已在前面加载
    # ================================================================
    set_seed(42)
    print("=" * 60, flush=True)
    print("  实验 02: 训练基线模型 (CPU)", flush=True)
    print(f"  训练数据: {len(train_df)} 条", flush=True)
    print(f"    正常: {(train_df['label']==0).sum()}, 垃圾: {(train_df['label']==1).sum()}", flush=True)
    print("=" * 60, flush=True)

    # 预处理 + 划分
    print("Step 3: Preprocessing...", flush=True)
    train_texts = train_df['text'].tolist()
    train_labels = train_df['label'].tolist()
    
    print("  Cleaning texts...", flush=True)
    train_texts_clean = [preprocess_text(t) for t in train_texts]
    print(f"  Cleaned {len(train_texts_clean)} texts", flush=True)

    X_tr, X_val, y_tr, y_val = train_test_split(
        train_texts, train_labels, test_size=0.1,
        random_state=42, stratify=train_labels
    )
    X_tr_clean, X_val_clean, _, _ = train_test_split(
        train_texts_clean, train_labels, test_size=0.1,
        random_state=42, stratify=train_labels
    )
    print(f"  Train: {len(X_tr)}, Val: {len(X_val)}", flush=True)

    # 训练基线1: 朴素BERT
    print("Step 4: Training baseline 1 (BERT plain)...", flush=True)
    model1 = BertClassifier(freeze_bert=False)
    train_loader1 = create_data_loader(X_tr, y_tr, batch_size=4, shuffle=True)
    val_loader1 = create_data_loader(X_val, y_val, batch_size=8, shuffle=False)
    model1 = train_model(model1, train_loader1, val_loader1,
                         epochs=1, lr=2e-5, model_name="BERT基线", device=DEVICE)

    save_path1 = os.path.join(BASE, 'data', 'processed', 'baseline_bert.pth')
    os.makedirs(os.path.dirname(save_path1), exist_ok=True)
    torch.save(model1.state_dict(), save_path1)
    print(f"\n基线1模型已保存: {save_path1}")

    # 训练基线2: BERT + 正规化 + 增强
    print("Step 5: Training baseline 2 (BERT + norm + aug)...", flush=True)
    model2 = BertClassifier(freeze_bert=False)
    train_loader2 = create_data_loader(X_tr_clean, y_tr, batch_size=4, shuffle=True)
    val_loader2 = create_data_loader(X_val_clean, y_val, batch_size=8, shuffle=False)
    model2 = train_model(model2, train_loader2, val_loader2,
                         epochs=1, lr=2e-5, model_name="BERT+正规化+增强", device=DEVICE)

    save_path2 = os.path.join(BASE, 'data', 'processed', 'baseline_bert_aug.pth')
    torch.save(model2.state_dict(), save_path2)
    print(f"\n基线2模型已保存: {save_path2}")

    print(f"\n{'='*60}")
    print(f"  ✓ 基线模型训练完成!")
    print(f"{'='*60}")

except Exception as e:
    print(f"\n\nERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
