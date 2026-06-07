# -*- coding: utf-8 -*-
"""
攻击 G: 中文形近字替换
原理: 将汉字替换为形状相似的其他汉字
例如: "免费" → "兔费", "已" → "己"
人脑扫一眼会忽略细微差异, 但字符编码完全不同
"""

import random

# 形近字映射表 —— 人工整理的中文形近字对
HOMOGLYPH_CN_MAP = {
    # 原字: [形近字候选列表]
    '免': ['兔', '勉'],
    '已': ['己', '巳'],
    '未': ['末'],
    '人': ['入', '八'],
    '日': ['曰'],
    '干': ['千', '于'],
    '大': ['太', '犬'],
    '王': ['玉', '主'],
    '片': ['⽚'],   # 右侧是 Unicode 兼容区的 ⽚ (U+2F5A)
    '户': ['戸', '⼾'],
    '士': ['土'],
    '刀': ['刃', '刁'],
    '天': ['夭', '夫'],
    '贝': ['见'],
    '名': ['各'],
    '问': ['间', '闻'],
    '午': ['牛'],
    '斤': ['斥'],
    '爪': ['瓜'],
    '乒': ['乓'],
    '冶': ['治'],
    '准': ['淮'],
    '晴': ['睛'],
    '狠': ['狼'],
    '侍': ['待'],
    '贷': ['货'],
    '侯': ['候'],
    '钧': ['钓'],
    '竞': ['竟'],
    '茶': ['荼'],
    '崇': ['祟'],
    '折': ['拆'],
    '析': ['折', '拆'],
    '栗': ['粟'],
    '拔': ['拨'],
    '券': ['卷'],
    '微': ['徽'],
}

# 某些垃圾文本中的"定制"形近字
SPECIAL_HOMOGLYPH = {
    '费': ['废'],   # "免费" → "免废" (常见垃圾文本变形)
    '信': ['伩'],   # "微信" → "微伩"
    '看': ['着'],   # "看片" → "着片"
}


def attack_homoglyph_cn(text: str, replace_ratio: float = 0.3) -> str:
    """
    将汉字随机替换为形近字

    参数:
        text:          原始文本
        replace_ratio: 替换比例, 默认 30%

    返回:
        形近字替换后的文本

    示例:
        >>> attack_homoglyph_cn("免费看片")
        "兔费看⽚"   # 两个字被替换为形近字
    """
    if not text:
        return text

    # 合并通用形近字和特殊形近字
    full_map = {**HOMOGLYPH_CN_MAP, **SPECIAL_HOMOGLYPH}

    chars = list(text)
    candidates = [(i, c) for i, c in enumerate(chars) if c in full_map]

    if not candidates:
        return text

    n_replace = max(1, int(len(candidates) * replace_ratio))
    n_replace = min(n_replace, len(candidates))

    for i, c in random.sample(candidates, n_replace):
        chars[i] = random.choice(full_map[c])

    return ''.join(chars)


if __name__ == '__main__':
    samples = [
        "免费看片点击观看",
        "已办理的证件已寄出",
        "大王叫我来巡山",
        "微信支付优惠",
    ]
    for s in samples:
        result = attack_homoglyph_cn(s)
        print(f"  原文: {s}")
        print(f"  攻击: {result}")
        # 显示哪些字被替换了
        diffs = [(a, b) for a, b in zip(s, result) if a != b]
        if diffs:
            print(f"  替换: {diffs}")
        print()
