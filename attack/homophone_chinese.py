# -*- coding: utf-8 -*-
"""
攻击 F: 中文音近字替换 ★ 核心攻击
原理: 将中文字符替换为同音/近音字, 利用拼音相同的特性绕过关键词匹配
例如: "加微信" → "佳薇芯" (拼音都是 "jia wei xin")

这是中文垃圾文本最常用的混淆手段之一, 如 "胖鸡的猫巴"
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import random
from utils import load_homophone_map


def attack_homophone(text: str, replace_ratio: float = 0.5) -> str:
    """
    将中文字符随机替换为同音字

    参数:
        text:          原始文本
        replace_ratio: 替换比例 (0.0~1.0), 默认 50%

    返回:
        音近字替换后的文本

    示例:
        >>> attack_homophone("加微信免费代理")
        "佳薇芯免废戴理"   # 每个字的发音不变, 汉字全变了

    注意:
        这个攻击对 BERT 非常致命——token 完全不同
        但语音通道能检测到——拼音序列完全一致
    """
    if not text:
        return text

    homophone_map = load_homophone_map()  # 加载同音字映射表

    chars = list(text)
    # 找出所有可替换的中文字符 (必须有非空同音候选)
    candidates = [(i, c) for i, c in enumerate(chars)
                  if c in homophone_map and homophone_map[c]]

    if not candidates:
        return text

    n_replace = max(1, int(len(candidates) * replace_ratio))
    n_replace = min(n_replace, len(candidates))

    for i, c in random.sample(candidates, n_replace):
        # 从同音候选字中随机选一个 (如果候选列表为空则跳过)
        if c in homophone_map and homophone_map[c]:
            chars[i] = random.choice(homophone_map[c])

    return ''.join(chars)


if __name__ == '__main__':
    samples = [
        "加微信领取红包",
        "免费办理各类证件",
        "小额贷款当天放款",
        "代办信用卡无需面签",
    ]
    for s in samples:
        result = attack_homophone(s)
        # 显示拼音对比
        from pypinyin import pinyin, Style
        orig_py = ' '.join(x[0] for x in pinyin(s, style=Style.TONE3))
        adv_py = ' '.join(x[0] for x in pinyin(result, style=Style.TONE3))
        print(f"  原文: {s}")
        print(f"  攻击: {result}")
        print(f"  原拼音: {orig_py}")
        print(f"  攻拼音: {adv_py}")
        print(f"  拼音一致: {orig_py == adv_py}")
        print()
