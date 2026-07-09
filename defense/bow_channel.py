# -*- coding: utf-8 -*-
"""
防御 ⑤: 字袋通道 (Char-BoW + Pinyin-BoW + MLP)

原理:
  提取字符级 Unigram + 拼音Unigram 特征
  完全顺序无关，对字符乱序攻击天然鲁棒

注意:
  本通道不引入人为排序结构（如 sorted bigram）
  保持纯 set / frequency invariance
"""

import torch
import torch.nn as nn


# ============================================================
# 字符集定义（保持不变）
# ============================================================

COMMON_CHARS = (
    "的一是在了不和有大这主中人上为们地个用工时要动国产以我到"
    "他会作来分生对于学下级就年阶义发成部民可出能方进同行面说"
    "种过命度革而多子后自社加小机也经力线本电高量长党得实家定"
    "深法表着水理化争现所二起政三好十战无农使性前等反体合斗路"
    "图把结第里正新开论之物从当两些还天资事队批如应形想制心样"
    "干都向变关点育重其思与间内去因件日利相由压员气业代全组数"
    "免费领取优惠红包大奖点击代理贷款小姐证件办理微信加群"
    "免废薇芯佳戴理洁晓待款棉面威信嘉家号码码链连"
    "赚现金牌秒杀卡套特价折扣快送件品质保证"
)

CHAR_TO_IDX = {c: i for i, c in enumerate(dict.fromkeys(COMMON_CHARS))}
CHAR_VOCAB_SIZE = len(CHAR_TO_IDX)


COMMON_PINYIN_SYLLABLES = [
    'a', 'ai', 'an', 'ang', 'ao',
    'ba', 'bai', 'ban', 'bang', 'bao', 'bei', 'ben', 'beng', 'bi', 'bian',
    'biao', 'bie', 'bin', 'bing', 'bo', 'bu',
    'ca', 'cai', 'can', 'cang', 'cao', 'ce', 'cen', 'ceng',
    # ...（省略其余拼音，保持你原版本即可）
]
PINYIN_SYL_TO_IDX = {p: i for i, p in enumerate(COMMON_PINYIN_SYLLABLES)}
PINYIN_SYL_VOCAB_SIZE = len(COMMON_PINYIN_SYLLABLES)


# ============================================================
# Feature extraction
# ============================================================

def extract_char_bow(text: str) -> torch.Tensor:
    """字符 unigram (set + frequency)"""
    bow = torch.zeros(CHAR_VOCAB_SIZE, dtype=torch.float32)
    for c in text:
        if c in CHAR_TO_IDX:
            bow[CHAR_TO_IDX[c]] += 1
    return bow


def extract_pinyin_bow(text: str) -> torch.Tensor:
    """拼音 unigram"""
    from pypinyin import pinyin, Style

    bow = torch.zeros(PINYIN_SYL_VOCAB_SIZE, dtype=torch.float32)

    try:
        py_list = pinyin(text, style=Style.NORMAL)
        for item in py_list:
            p = item[0]
            if p in PINYIN_SYL_TO_IDX:
                bow[PINYIN_SYL_TO_IDX[p]] += 1
    except Exception:
        pass

    return bow


# ============================================================
# BOW Channel
# ============================================================

class BowChannel(nn.Module):
    """
    字袋通道（严格版）

    设计目标:
      - 对字符顺序攻击 invariant
      - 不引入人为排序结构
      - 保留真实统计信息（frequency + presence）
    """

    def __init__(self, hidden_dim: int = 256, output_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()

        self.char_dim = CHAR_VOCAB_SIZE
        self.pinyin_dim = PINYIN_SYL_VOCAB_SIZE
        self.total_input_dim = self.char_dim + self.pinyin_dim

        self.feature_dim = output_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        print(f"[BowChannel] input={self.total_input_dim}, output={output_dim}")

    def forward(self, texts: list) -> torch.Tensor:
        device = next(self.encoder.parameters()).device

        batch = []
        for text in texts:
            char_bow = extract_char_bow(text)
            pinyin_bow = extract_pinyin_bow(text)

            combined = torch.cat([char_bow, pinyin_bow], dim=0)
            batch.append(combined)

        x = torch.stack(batch).to(device)
        return self.encoder(x)