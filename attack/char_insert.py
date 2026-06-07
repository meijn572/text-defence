# -*- coding: utf-8 -*-
"""
攻击 B: 字符插入
原理: 在文本中随机插入无意义特殊符号, 干扰 token 化
人脑自动忽略这些符号, 但模型会将其当作有效 token
"""

import random

# 插入符号池 —— 常见无意义分隔符
INSERT_SYMBOLS = ['*', '/', '|', '~', '_', '.', '^', '#']


def attack_char_insert(text: str, insert_ratio: float = 0.15) -> str:
    """
    在文本随机位置插入无意义符号

    参数:
        text:        原始文本
        insert_ratio: 插入比例, 默认 15%

    返回:
        插入符号后的文本

    示例:
        >>> attack_char_insert("代开发票")
        "代*开|发~票"   # 随机插入符号
    """
    if not text:
        return text

    chars = list(text)
    # 插入数量 = 文本长度 × 比例, 至少 1 个
    n_insert = max(1, int(len(chars) * insert_ratio))

    for _ in range(n_insert):
        pos = random.randint(0, len(chars))
        symbol = random.choice(INSERT_SYMBOLS)
        chars.insert(pos, symbol)

    return ''.join(chars)


if __name__ == '__main__':
    samples = [
        "代开发票请联系我",
        "免费领取优惠券",
        "小额贷款当天放款",
    ]
    for s in samples:
        result = attack_char_insert(s)
        print(f"  原文: {s}")
        print(f"  攻击: {result}\n")
