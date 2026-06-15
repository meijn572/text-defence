# -*- coding: utf-8 -*-
"""
防御 ②: 文本通道 (BERT)

使用 BERT-base-chinese 对正规化后的文本进行编码
输出 [CLS] token 的 768 维向量作为文本语义特征

训练方式:
  - 在垃圾文本分类任务上微调 BERT
  - 可单独作为基线模型 (baseline)
  - 也可作为融合模型的一部分 (冻结或微调)
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class TextChannel(nn.Module):
    """
    文本通道 —— BERT 编码器

    输入: 正规化后的文本字符串
    输出: 768维语义特征向量
    """

    def __init__(self, model_name: str = 'bert-base-chinese',
                 freeze_bert: bool = False, dropout: float = 0.1):
        """
        参数:
            model_name:  BERT 预训练模型名称
            freeze_bert: 是否冻结 BERT 参数 (用于融合模型)
            dropout:     dropout 比例
        """
        super().__init__()

        print(f"[文本通道] 加载 BERT 模型: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(
            model_name, local_files_only=False
        )
        self.bert = BertModel.from_pretrained(
            model_name, local_files_only=False
        )
        self.dropout = nn.Dropout(dropout)
        self.feature_dim = 768  # BERT-base 的隐藏维度

        # 是否冻结 BERT 参数
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print("[文本通道] BERT 参数已冻结")

    def forward(self, texts: list) -> torch.Tensor:
        """
        前向传播

        参数:
            texts: 文本列表, 如 ["免费领取", "明天开会"]

        返回:
            text_features: (batch_size, 768) 的语义特征向量
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,           # 自动 padding 到 batch 中最长
            truncation=True,        # 截断超长文本
            max_length=64,           # 垃圾短信通常很短, 64 足够 (加快训练)
            return_tensors='pt'    # 返回 PyTorch tensor
        )

        # 移动到设备
        device = next(self.bert.parameters()).device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        # BERT 编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 取 [CLS] token 的最后一层隐藏状态作为句子表示
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        cls_output = self.dropout(cls_output)

        return cls_output

    def encode(self, texts: list) -> torch.Tensor:
        """
        编码文本 (不计算梯度, 用于提取特征)

        返回: (batch_size, 768) 特征向量
        """
        self.eval()
        with torch.no_grad():
            return self.forward(texts)


class BertClassifier(nn.Module):
    """
    基于 BERT 的二分类器 (基线模型)

    结构: BERT → [CLS] → Dropout → FC(2) → Softmax
    """

    def __init__(self, model_name: str = 'bert-base-chinese',
                 freeze_bert: bool = False):
        super().__init__()
        self.text_channel = TextChannel(model_name, freeze_bert)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),  # 二分类: [正常, 垃圾]
        )

    def forward(self, texts: list) -> torch.Tensor:
        """返回 logits: (batch, 2)"""
        features = self.text_channel(texts)
        return self.classifier(features)

    def predict(self, texts: list) -> torch.Tensor:
        """返回预测类别: (batch,) 0=正常, 1=垃圾"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(texts)
            return torch.argmax(logits, dim=1)


# ============================================================
# 训练工具函数
# ============================================================

def train_bert_classifier(model, train_loader, val_loader,
                          epochs: int = 3, lr: float = 2e-5,
                          device: str = 'cuda'):
    """
    训练 BERT 分类器

    返回: 训练好的模型
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\n{'='*50}")
    print(f"  训练 BERT 基线分类器")
    print(f"  Epochs: {epochs}, LR: {lr}")
    print(f"{'='*50}")

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_texts, batch_labels in train_loader:
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(batch_texts)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_texts, batch_labels in val_loader:
                batch_labels = batch_labels.to(device)
                logits = model(batch_texts)
                loss = criterion(logits, batch_labels)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == batch_labels).sum().item()
                val_total += len(batch_labels)

        val_acc = val_correct / val_total if val_total > 0 else 0
        print(f"  Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | "
              f"Val Acc: {val_acc:.4f}")

    return model


if __name__ == '__main__':
    print("=" * 50)
    print("  文本通道模块测试")
    print("=" * 50)

    # 测试编码
    channel = TextChannel()
    test_texts = ["免费领取优惠券", "明天上午十点开会", "加微信领红包"]
    with torch.no_grad():
        features = channel(test_texts)
    print(f"\n输入: {len(test_texts)} 条文本")
    print(f"输出特征维度: {features.shape}")  # 应为 (3, 768)
    print("✓ 文本通道测试通过")
