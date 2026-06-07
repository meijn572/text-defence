# -*- coding: utf-8 -*-
"""
攻击 I: 字符乱序 ★ 最难防御的攻击
原理: 打乱文本中汉字的顺序, 利用人脑的"自动纠错"能力
例如: "免费领取优惠券" → "费免取领惠优券"

人脑可以自动纠正并正确理解, 但 BERT/拼音CNN/视觉CNN 都无法处理
只有字袋通道 (BoW) 对此攻击天然免疫
"""

import random


def attack_shuffle(text: str, window_size: int = 3,
                   shuffle_ratio: float = 0.5) -> str:
    """
    滑动窗口内随机打乱字符顺序

    参数:
        text:         原始文本
        window_size:  乱序窗口大小 (2~4), 默认 3
        shuffle_ratio: 每个窗口被打乱的概率, 默认 50%

    返回:
        字符乱序后的文本

    示例:
        >>> attack_shuffle("免费领取优惠券")
        "费免取领惠优券"   # 每3字窗口内随机打乱

    原理:
        人在阅读中文时是并行处理的, 尤其对于短句,
        只要字符集合不变, 大脑会自动"脑补"正确顺序。
        这个攻击对 BERT 是致命的——位置编码全部错位。
    """
    if not text or len(text) < window_size:
        return text

    chars = list(text)

    # 滑动窗口, 每个窗口以 shuffle_ratio 概率打乱
    for i in range(len(chars) - window_size + 1):
        if random.random() < shuffle_ratio:
            window = chars[i:i + window_size]
            random.shuffle(window)
            chars[i:i + window_size] = window

    return ''.join(chars)


def attack_adjacent_swap(text: str, swap_ratio: float = 0.4) -> str:
    """
    随机交换相邻字符 (更温和的乱序方式)

    参数:
        text:       原始文本
        swap_ratio: 每对相邻字符被交换的概率, 默认 40%

    返回:
        相邻交换后的文本

    示例:
        >>> attack_adjacent_swap("免费领取优惠券")
        "费免取领惠优券"
    """
    if not text:
        return text

    chars = list(text)
    for i in range(len(chars) - 1):
        if random.random() < swap_ratio:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]

    return ''.join(chars)


def verify_shuffle(original: str, shuffled: str) -> dict:
    """
    验证乱序效果: 检查字符集合是否一致

    返回: {'same_chars': bool, 'same_length': bool, 'diff_count': int}
    """
    return {
        'same_chars': set(original) == set(shuffled),
        'same_length': len(original) == len(shuffled),
        'diff_count': sum(1 for a, b in zip(original, shuffled) if a != b),
    }


if __name__ == '__main__':
    samples = [
        "免费领取优惠券",
        "加微信领取红包大奖",
        "代办各类证件质量保证",
    ]

    print("=== 滑动窗口乱序 ===")
    for s in samples:
        result = attack_shuffle(s)
        info = verify_shuffle(s, result)
        print(f"  原文: {s}")
        print(f"  攻击: {result}")
        print(f"  字符集一致: {info['same_chars']}, "
              f"不同位置: {info['diff_count']}/{len(s)}")
        print()

    print("=== 相邻交换 ===")
    for s in samples:
        result = attack_adjacent_swap(s)
        info = verify_shuffle(s, result)
        print(f"  原文: {s}")
        print(f"  攻击: {result}")
        print()
