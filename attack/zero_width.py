# -*- coding: utf-8 -*-
"""
攻击 D: 零宽字符注入
原理: 在文本中插入肉眼不可见的 Unicode 零宽字符
人看不到这些字符, 但它们会破坏 BERT 的 token 边界
"""

import random

# 零宽字符池
ZERO_WIDTH_CHARS = [
    '\u200B',  # 零宽空格 (Zero Width Space)
    '\u200C',  # 零宽非连接符 (Zero Width Non-Joiner)
    '\u200D',  # 零宽连接符 (Zero Width Joiner)
    '\uFEFF',  # 字节顺序标记 (BOM, 也常被用作零宽空格)
]


def attack_zero_width(text: str, inject_ratio: float = 0.3) -> str:
    """
    在文本中注入零宽字符

    参数:
        text:         原始文本
        inject_ratio: 注入比例, 默认 30% (每3个字插1个零宽字符)

    返回:
        含零宽字符的文本 (肉眼不可见!)

    示例:
        >>> attack_zero_width("代办证件")
        "代\u200B办证\u200C件"   # 中间藏了零宽字符
        >>> # 在 Python 中打印出来看起来还是 "代办证件"
    """
    if not text:
        return text

    chars = list(text)
    n_inject = max(1, int(len(chars) * inject_ratio))

    for _ in range(n_inject):
        pos = random.randint(0, len(chars))
        zw_char = random.choice(ZERO_WIDTH_CHARS)
        chars.insert(pos, zw_char)

    return ''.join(chars)


def detect_zero_width(text: str) -> int:
    """
    检测文本中零宽字符的数量 (用于验证攻击效果)

    返回: 零宽字符个数
    """
    count = 0
    for c in text:
        if c in ZERO_WIDTH_CHARS:
            count += 1
    return count


if __name__ == '__main__':
    samples = ["代办证件", "免费领取", "加微信"]
    for s in samples:
        result = attack_zero_width(s)
        visible = ''.join(c for c in result if c not in ZERO_WIDTH_CHARS)
        zw_count = detect_zero_width(result)
        print(f"  原文: {s}")
        print(f"  攻击后可见: {visible}")
        print(f"  注入零宽字符数: {zw_count}")
        print(f"  攻击后总长度: {len(result)} (可见 {len(visible)} + 隐藏 {zw_count})")
        print()
