# -*- coding: utf-8 -*-
"""
防御 ⑥: 四通道融合模型

将 4 个通道的特征拼接后通过 MLP 进行二分类

架构:
  text(768) + phonetic(256) + visual(512) + bow(128)
  → Concat(1664) → FC(512) → FC(128) → Softmax(2)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .text_channel import TextChannel
from .phonetic_channel import PhoneticChannel
from .visual_channel import VisualChannel
from .bow_channel import BowChannel
from .preprocess import preprocess_text


class FusionClassifier(nn.Module):
    """
    四通道融合分类器

    四种特征互补:
      - 文本通道: 语义理解 (音近字失效时, 视觉/Bow 补上)
      - 语音通道: 发音模式 (音近字攻击克星)
      - 视觉通道: 字形特征 (跨语种同形字克星)
      - 字袋通道: 字符集合 (字符乱序攻击克星)
    """

    def __init__(self, freeze_channels: bool = True,
             text_model_name: str = 'bert-base-chinese',
             device: torch.device = None):  # 新增device参数,决定是否使用gpu
        super().__init__()
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.text_channel = TextChannel(model_name=text_model_name, freeze_bert=freeze_channels)
        self.phonetic_channel = PhoneticChannel()
        self.visual_channel = VisualChannel(freeze_cnn=freeze_channels)
        self.bow_channel = BowChannel()
        
        self.text_channel = self.text_channel.to(self.device)
        self.phonetic_channel = self.phonetic_channel.to(self.device)
        self.visual_channel = self.visual_channel.to(self.device)
        self.bow_channel = self.bow_channel.to(self.device)
        
        total_dim = (self.text_channel.feature_dim +     # 768
                    self.phonetic_channel.feature_dim + # 256
                    self.visual_channel.feature_dim +   # 512
                    self.bow_channel.feature_dim)       # 128
        print(f"[融合模型] 总输入维度: {total_dim}, 设备: {self.device}")
        
        self.fusion_head = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, texts: list, return_features: bool = False, ablation: list = None):
        """
        前向传播

        参数:
            texts:          文本列表
            return_features: 是否返回特征
            ablation:       要屏蔽的通道列表 (用于消融实验)

        返回:
            logits:         (batch, 2) 分类输出
            ablation_feats: (可选) 消融特征字典
        """
        # 预处理: 正规化
        clean_texts = [preprocess_text(t) for t in texts]

        # 提取四通道特征
        text_feat = self.text_channel(clean_texts)          # (batch, 768)
        phonetic_feat = self.phonetic_channel(clean_texts)  # (batch, 256)
        visual_feat = self.visual_channel(clean_texts)      # (batch, 512)
        bow_feat = self.bow_channel(clean_texts)            # (batch, 128)

        # 消融实验: 将指定通道特征置零
        if ablation:
            if 'text' in ablation:
                text_feat = torch.zeros_like(text_feat)
            if 'phonetic' in ablation:
                phonetic_feat = torch.zeros_like(phonetic_feat)
            if 'visual' in ablation:
                visual_feat = torch.zeros_like(visual_feat)
            if 'bow' in ablation:
                bow_feat = torch.zeros_like(bow_feat)

        # 拼接
        combined = torch.cat([text_feat, phonetic_feat,
                              visual_feat, bow_feat], dim=1)  # (batch, 1664)

        # 融合分类
        logits = self.fusion_head(combined)  # (batch, 2)

        if return_features:
            return logits, {
                'text': text_feat,
                'phonetic': phonetic_feat,
                'visual': visual_feat,
                'bow': bow_feat,
                'combined': combined,
            }
        return logits

    def predict(self, texts: list) -> torch.Tensor:
        """返回预测类别: (batch,) 0=正常, 1=垃圾"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(texts)
            return torch.argmax(logits, dim=1)

    def predict_proba(self, texts: list) -> torch.Tensor:
        """返回概率: (batch, 2) [正常概率, 垃圾概率]"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(texts)
            return torch.softmax(logits, dim=1)


class FourChannelDataset(Dataset):
    """
    四通道数据集

    只存储文本和标签, 预处理在各通道内部进行
    """

    def __init__(self, texts: list, labels: list):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def create_data_loader(texts: list, labels: list,
                       batch_size: int = 16, shuffle: bool = True):
    """
    创建 DataLoader

    注意: 此 DataLoader 返回原始文本而非 tensor,
          因为各通道内部会做 tokenize
    """
    dataset = FourChannelDataset(texts, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(model, loader, optimizer, criterion, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    for batch_texts, batch_labels in loader:
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        logits = model(batch_texts)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_texts, batch_labels in loader:
            batch_labels = batch_labels.to(device)
            logits = model(batch_texts)
            loss = criterion(logits, batch_labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch_labels.cpu().tolist())
    return total_loss / len(loader), all_preds, all_labels


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = FusionClassifier(device=device)
    model = model.to(device)  # fusion_head 也移到 GPU
    
    test_texts = [
        "免费领取优惠券",
        "明天上午十点开会",
        "佳薇芯免废戴理",
    ]
    
    model.eval()
    with torch.no_grad():
        logits = model(test_texts)
        probs = torch.softmax(logits, dim=1)
    
    print(f"\n测试文本数: {len(test_texts)}")
    print(f"Logits shape: {logits.shape}")
    for i, t in enumerate(test_texts):
        print(f"  [{i}] {t}")
        print(f"      正常: {probs[i][0]:.4f}, 垃圾: {probs[i][1]:.4f}")
    print("✓ 融合模型测试通过")
