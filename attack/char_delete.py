# -*- coding: utf-8 -*-
"""
攻击 A: 字符删除
原理: 随机删除文本中一定比例的中文字符, 破坏分词效果
人仍能通过上下文补全理解, 但 BERT 的 token 序列被破坏
"""

import random


def attack_char_delete(text: str, delete_ratio: float = 0.2) -> str:
    """
    随机删除中文字符

    参数:
        text:         原始文本
        delete_ratio: 删除比例 (0.0~1.0), 默认 20%

    返回:
        删除字符后的文本

    示例:
        >>> attack_char_delete("免费领取优惠券")
        "免费领取券"   # 可能删除 "优" 和 "惠"
    """
    if not text:
        return text

    chars = list(text)
    # 只对中文字符做删除, 保留标点/数字/英文
    chinese_indices = [i for i, c in enumerate(chars)
                       if '\u4e00' <= c <= '\u9fff']

    if not chinese_indices:
        return text

    # 至少删除 1 个字符
    n_delete = max(1, int(len(chinese_indices) * delete_ratio))
    n_delete = min(n_delete, len(chinese_indices))  # 不能超过总数

    delete_indices = set(random.sample(chinese_indices, n_delete))

    return ''.join(c for i, c in enumerate(chars) if i not in delete_indices)


if __name__ == '__main__':
    # 测试
    samples = [
        "免费领取优惠券，数量有限先到先得",
        "代办各类证件，质量保证",
        "加微信领取红包大奖",
    ]
    for s in samples:
        result = attack_char_delete(s)
        print(f"  原文: {s}")
        print(f"  攻击: {result}")
        print(f"  删除了 {len(s)-len(result)} 个字符\n")
