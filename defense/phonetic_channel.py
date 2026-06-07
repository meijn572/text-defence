# -*- coding: utf-8 -*-
"""
防御 ③: 语音通道 (拼音 + TextCNN) ★ 核心创新

原理:
  将中文文本转为拼音序列, 在拼音上做 TextCNN 卷积
  音近字替换后汉字不同但拼音相同 → 语音通道免疫音近字攻击

示例:
  "加微信" → pinyin: "jia wei xin"
  "佳薇芯" → pinyin: "jia wei xin"  ← 完全一致!

架构:
  拼音序列 → Char Embedding → 1D Conv (2,3,4) → GlobalMaxPool → FC → 256维
"""

import torch
import torch.nn as nn
from pypinyin import pinyin, Style


# 拼音字符集: 26个小写字母 + 空格 + 数字声调 + padding
PINYIN_VOCAB = sorted(set('abcdefghijklmnopqrstuvwxyz 012345'))
PINYIN_CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(PINYIN_VOCAB)}  # 0 留给 padding
PINYIN_VOCAB_SIZE = len(PINYIN_CHAR_TO_IDX) + 1  # +1 for padding idx 0


def text_to_pinyin(text: str, with_tone: bool = False) -> str:
    """
    将中文文本转换为拼音字符串

    参数:
        text:      中文文本, 如 "加微信免费代理"
        with_tone: 是否保留声调数字

    返回:
        拼音字符串, 如 "jia wei xin mian fei dai li"

    示例:
        >>> text_to_pinyin("佳薇芯免废代理")
        "jia wei xin mian fei dai li"
        >>> text_to_pinyin("加微信免费代理")
        "jia wei xin mian fei dai li"
        # ↑ 音近字替换后拼音完全一致!
    """
    style = Style.TONE3 if with_tone else Style.NORMAL
    py_list = pinyin(text, style=style)
    # 提取拼音, 非中文部分保留原字符
    result = []
    for item in py_list:
        p = item[0]
        result.append(p if p else ' ')
    return ' '.join(result)


def pinyin_to_indices(pinyin_str: str, max_len: int = 200) -> torch.Tensor:
    """
    将拼音字符串转为字符索引序列

    参数:
        pinyin_str: 拼音字符串, 如 "jia wei xin"
        max_len:    最大长度 (padding/截断)

    返回:
        LongTensor, shape: (max_len,)
    """
    indices = []
    for c in pinyin_str.lower():
        if c in PINYIN_CHAR_TO_IDX:
            indices.append(PINYIN_CHAR_TO_IDX[c])
        else:
            indices.append(0)  # 未知字符用 padding

    # Padding 或截断
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    else:
        indices = indices[:max_len]

    return torch.tensor(indices, dtype=torch.long)


class PhoneticChannel(nn.Module):
    """
    语音通道 —— 拼音 TextCNN

    输入: 正规化后的中文文本
    输出: 256维发音特征向量
    """

    def __init__(self, vocab_size: int = PINYIN_VOCAB_SIZE,
                 embed_dim: int = 64,
                 num_filters: int = 128,
                 filter_sizes: tuple = (2, 3, 4),
                 dropout: float = 0.3,
                 max_pinyin_len: int = 200):
        """
        参数:
            vocab_size:     拼音字符集大小 (~30)
            embed_dim:      字符嵌入维度
            num_filters:    每种卷积核的输出通道数
            filter_sizes:   卷积核大小 (在拼音上捕捉 2-gram, 3-gram, 4-gram)
            dropout:        dropout 比例
            max_pinyin_len: 拼音序列最大长度
        """
        super().__init__()

        self.max_pinyin_len = max_pinyin_len
        conv_output_dim = num_filters * len(filter_sizes)  # 例如 128*3 = 384
        self.feature_dim = 256  # FC 压缩后的输出维度

        # 拼音字符嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 多个 1D 卷积核 (捕捉不同粒度的拼音 n-gram 模式)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=num_filters,
                      kernel_size=k)
            for k in filter_sizes
        ])

        # 输出层: 压缩到 256 维
        self.fc = nn.Sequential(
            nn.Linear(conv_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, texts: list) -> torch.Tensor:
        """
        前向传播

        参数:
            texts: 中文文本列表

        返回:
            phonetic_features: (batch_size, 256) 发音特征向量
        """
        device = self.embedding.weight.device

        # 步骤1: 文本 → 拼音字符串
        pinyin_strs = [text_to_pinyin(t) for t in texts]

        # 步骤2: 拼音字符串 → 字符索引序列
        indices_list = [pinyin_to_indices(p, self.max_pinyin_len) for p in pinyin_strs]
        input_tensor = torch.stack(indices_list).to(device)  # (batch, max_len)

        # 步骤3: 字符嵌入
        embedded = self.embedding(input_tensor)  # (batch, max_len, embed_dim)
        embedded = embedded.transpose(1, 2)      # (batch, embed_dim, max_len) → 适配 Conv1D

        # 步骤4: 多尺度卷积 + 池化
        conv_outputs = []
        for conv in self.convs:
            x = conv(embedded)              # (batch, num_filters, L)
            x = torch.relu(x)
            x = torch.max_pool1d(x, x.size(2))  # GlobalMaxPool → (batch, num_filters, 1)
            conv_outputs.append(x.squeeze(2))    # (batch, num_filters)

        # 步骤5: 拼接所有卷积输出 → FC → 256维
        combined = torch.cat(conv_outputs, dim=1)  # (batch, num_filters * len(filter_sizes))
        output = self.fc(combined)                  # (batch, 256)

        return output


class PhoneticClassifier(nn.Module):
    """
    独立的语音通道分类器 (用于消融实验)

    结构: TextCNN → FC(256) → FC(2)
    """

    def __init__(self):
        super().__init__()
        self.phonetic_channel = PhoneticChannel()
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, texts: list) -> torch.Tensor:
        features = self.phonetic_channel(texts)
        return self.classifier(features)


if __name__ == '__main__':
    print("=" * 50)
    print("  语音通道模块测试")
    print("=" * 50)

    # 测试拼音转换
    test_pairs = [
        ("加微信免费代理", "正常文本"),
        ("佳薇芯免废戴理", "音近字攻击"),
    ]
    print("\n拼音对比测试:")
    for text, desc in test_pairs:
        py = text_to_pinyin(text)
        print(f"  [{desc}] {text}")
        print(f"    拼音: {py}")

    # 测试模型
    channel = PhoneticChannel()
    test_texts = ["加微信领红包", "明天开会记得带文件", "代办证件"]
    with torch.no_grad():
        features = channel(test_texts)
    print(f"\n输入: {len(test_texts)} 条文本")
    print(f"输出特征维度: {features.shape}")  # 应为 (3, 256)
    print(f"拼音字符集大小: {PINYIN_VOCAB_SIZE}")
    print("✓ 语音通道测试通过")
