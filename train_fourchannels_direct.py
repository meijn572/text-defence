# -*- coding: utf-8 -*-
"""四通道融合模型训练脚本
依赖: data/adversarial/train.csv 已存在（run generate_adv.py 先生成）
      data/processed/baseline_bert.pth 已存在（run train_baseline_direct.py 先训练）
"""
import sys, os, traceback

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
if BASE not in sys.path:
    sys.path.insert(0, BASE)

try:
    # ================================================================
    # Step 1: 读数据（必须在 torch 之前）
    # ================================================================
    print("Step 1: Reading train data...", flush=True)
    import pandas as pd
    from utils import DATA_ADV, DATA_PROCESSED

    train_path = os.path.join(DATA_ADV, 'train.csv')
    if not os.path.exists(train_path):
        print(f"[ERROR] 训练数据不存在: {train_path}")
        print("  请先运行: python experiments/generate_adv.py")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    print(f"  Loaded {len(train_df)} records "
          f"(normal={int((train_df.label==0).sum())}, "
          f"spam={int((train_df.label==1).sum())})", flush=True)

    bert_path = os.path.join(DATA_PROCESSED, 'baseline_bert.pth')
    if not os.path.exists(bert_path):
        print(f"[ERROR] 基线 BERT 权重不存在: {bert_path}")
        print("  请先运行: python train_baseline_direct.py")
        sys.exit(1)

    # ================================================================
    # Step 2: 导入 ML 模块
    # ================================================================
    print("Step 2: Importing ML modules...", flush=True)
    import torch
    import torch.nn as nn
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    from tqdm import tqdm
    print("  Basic imports OK", flush=True)

    from utils import set_seed, DEVICE
    print(f"  Utils OK, DEVICE={DEVICE}", flush=True)

    from defense.fusion_model import FusionClassifier, create_data_loader
    print("  FusionClassifier OK", flush=True)

    # ================================================================
    # Step 3: 训练函数（以验证 F1 选最优 checkpoint）
    # ================================================================
    def train_model(model, train_loader, val_loader, epochs, lr, device):
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)
        criterion = nn.CrossEntropyLoss()
        best_f1 = 0.0
        best_state = None

        total_params = sum(p.numel() for p in model.parameters())
        trainable_n = sum(p.numel() for p in trainable_params)
        print(f"\n{'='*55}")
        print(f"  训练四通道融合模型")
        print(f"  Epochs: {epochs}, LR: {lr}, Device: {device}")
        print(f"  可训练参数: {trainable_n:,} / 总参数: {total_params:,}")
        print(f"{'='*55}")

        for epoch in range(epochs):
            # ---- 训练 ----
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

            # ---- 验证 ----
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
            val_f1 = f1_score(all_labels, all_preds, zero_division=0)
            marker = ' ★' if val_f1 > best_f1 else ''
            print(f"  Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Acc: {val_acc:.4f} | F1: {val_f1:.4f}{marker}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)
        print(f"  最优验证 F1: {best_f1:.4f}")
        return model, best_f1

    # ================================================================
    # Step 4: 数据预处理与划分
    # ================================================================
    set_seed(42)
    print(f"\n{'='*55}", flush=True)
    print(f"  实验 03: 训练四通道融合模型 ({DEVICE})", flush=True)
    print(f"  训练数据: {len(train_df)} 条", flush=True)
    print(f"{'='*55}", flush=True)

    print("Step 4: Splitting train/val...", flush=True)
    texts = train_df['text'].tolist()
    labels = train_df['label'].tolist()

    X_tr, X_val, y_tr, y_val = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )
    print(f"  Train: {len(X_tr)}, Val: {len(X_val)}", flush=True)

    train_loader = create_data_loader(X_tr,  y_tr,  batch_size=4, shuffle=True)
    val_loader   = create_data_loader(X_val, y_val, batch_size=8, shuffle=False)

    # ================================================================
    # Step 5: 构建模型并加载 BERT 预训练权重
    # ================================================================
    print("Step 5: Building FusionClassifier...", flush=True)
    model = FusionClassifier(freeze_channels=False).to(DEVICE)

    print(f"  Loading BERT weights from: {bert_path}", flush=True)
    pretrained = torch.load(bert_path, map_location=DEVICE)
    bert_keys = {k: v for k, v in pretrained.items() if 'text_channel.bert' in k}
    missing, unexpected = model.load_state_dict(bert_keys, strict=False)
    print(f"  BERT weights loaded: {len(bert_keys)} keys "
          f"(missing={len(missing)}, unexpected={len(unexpected)})", flush=True)

    # 冻结 BERT，只训练拼音CNN / 视觉CNN / BoW MLP / projection / gate / classifier
    for name, param in model.named_parameters():
        if 'text_channel.bert' in name:
            param.requires_grad = False

    frozen_n  = sum(p.numel() for n, p in model.named_parameters() if not p.requires_grad)
    train_n   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  冻结 BERT: {frozen_n:,} 参数；可训练: {train_n:,} 参数", flush=True)

    # ================================================================
    # Step 6: 训练
    # ================================================================
    print("Step 6: Training...", flush=True)
    model, best_f1 = train_model(
        model, train_loader, val_loader,
        epochs=3, lr=1e-4, device=DEVICE
    )

    # ================================================================
    # Step 7: 保存模型
    # ================================================================
    save_path = os.path.join(DATA_PROCESSED, 'fusion_model.pth')
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'best_f1': best_f1}, save_path)
    print(f"\n融合模型已保存: {save_path}")

    print(f"\n{'='*55}")
    print(f"  ✓ 四通道融合模型训练完成!  最优验证 F1: {best_f1:.4f}")
    print(f"{'='*55}")

except Exception as e:
    print(f"\n\nERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
