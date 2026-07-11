# -*- coding: utf-8 -*-
"""
攻击 I: 随机相邻字符交换（严格局部扰动版本）

特点:
- 每个字符对独立决定是否交换
- 交换后跳步，避免扰动扩散
- 不存在滑窗重叠污染
"""

import random
from collections import Counter


def attack_adjacent_swap(text: str, swap_ratio: float = 0.4) -> str:
    """
    严格随机相邻交换（non-cascading version）

    参数:
        text: 原始文本
        swap_ratio: 相邻交换概率

    返回:
        扰动后的文本
    """

    if not text or len(text) < 2:
        return text

    chars = list(text)
    i = 0

    while i < len(chars) - 1:
        if random.random() < swap_ratio:
            # 交换相邻字符
            chars[i], chars[i + 1] = chars[i + 1], chars[i]

            # 跳过下一个字符，防止连锁扰动
            i += 2
        else:
            i += 1

    return ''.join(chars)


def attack_shuffle(text: str, window_size: int = 7, shuffle_ratio: float = 0.8) -> str:
    """在局部窗口内打乱字符，保持长度和字符多重集合不变。"""
    if not text or len(text) < 2:
        return text

    chars = list(text)
    window_size = max(2, min(window_size, len(chars)))
    index = 0
    while index < len(chars) - 1:
        end = min(len(chars), index + window_size)
        if random.random() < shuffle_ratio and end - index >= 2:
            window = chars[index:end]
            random.shuffle(window)
            chars[index:end] = window
        index += window_size
    return ''.join(chars)


# ============================================================
# 保持原 verify（但建议修正 set→Counter，这里先不动结构）
# ============================================================

def verify_shuffle(original: str, shuffled: str) -> dict:
    return {
        'same_chars': Counter(original) == Counter(shuffled),
        'same_length': len(original) == len(shuffled),
        'diff_count': sum(1 for a, b in zip(original, shuffled) if a != b),
    }


# ============================================================
# 测试
# ============================================================

if __name__ == '__main__':
    samples = [
        "免费领取优惠券",
        "加微信领取红包大奖",
        "代办各类证件质量保证",
    ]

    print("=== 严格相邻交换攻击 I ===")

    for s in samples:
        result = attack_adjacent_swap(s)
        info = verify_shuffle(s, result)

        print(f"原文: {s}")
        print(f"攻击: {result}")
        print(f"diff: {info['diff_count']}/{len(s)}")
        print()