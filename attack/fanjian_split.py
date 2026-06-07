# -*- coding: utf-8 -*-
"""
攻击 H: 繁简混用 / 拆字混淆
原理:
  1. 繁简混用: 将简体字随机转为繁体, 制造字符级混淆
     例如: "代办证件" → "代辦證件"
  2. 拆字: 将合体字拆为偏旁部首
     例如: "枪" → "木仓" (枪 = 木 + 仓)
"""

import random

# 拆字映射表 —— 将合体字拆为两个常见部件
SPLIT_CHAR_MAP = {
    '枪': '木仓',
    '好': '女子',
    '明': '日月',
    '林': '木木',
    '从': '人人',
    '众': '人人人',
    '品': '口口口',
    '轰': '车又车又',
    '想': '相心',
    '怒': '奴心',
    '歪': '不正',
    '孬': '不好',
    '甭': '不用',
    '覅': '勿要',
}


def attack_fanjian_mix(text: str, convert_ratio: float = 0.5) -> str:
    """
    随机将简体字转为繁体字 (繁简混用)

    参数:
        text:          原始文本
        convert_ratio: 转换比例, 默认 50%

    返回:
        繁简混用的文本

    示例:
        >>> attack_fanjian_mix("代办证件")
        "代辦證件"

    注意: 需要安装 OpenCC: pip install OpenCC
    """
    if not text:
        return text

    try:
        from opencc import OpenCC
        cc = OpenCC('s2t')  # 简体 → 繁体
    except ImportError:
        print("[WARN] OpenCC 未安装, 使用内置简繁映射表")
        return _fallback_fanjian(text, convert_ratio)

    chars = list(text)
    # 只转换中文字符
    chinese_indices = [i for i, c in enumerate(chars)
                       if '\u4e00' <= c <= '\u9fff']

    if not chinese_indices:
        return text

    n_convert = max(1, int(len(chinese_indices) * convert_ratio))

    for idx in random.sample(chinese_indices, n_convert):
        traditional = cc.convert(chars[idx])
        if traditional and traditional != chars[idx]:
            chars[idx] = traditional

    return ''.join(chars)


def _fallback_fanjian(text: str, convert_ratio: float) -> str:
    """
    内置简繁映射表 (当 OpenCC 不可用时)
    覆盖最常用的繁简差异字
    """
    FANJIAN_MAP = {
        '办': '辦', '证': '證', '件': '件', '代': '代',
        '为': '爲', '会': '會', '发': '發', '开': '開',
        '关': '關', '门': '門', '车': '車', '马': '馬',
        '风': '風', '电': '電', '长': '長', '时': '時',
        '问': '問', '对': '對', '动': '動', '实': '實',
        '学': '學', '国': '國', '个': '個', '们': '們',
    }
    chars = list(text)
    candidates = [(i, c) for i, c in enumerate(chars) if c in FANJIAN_MAP]
    if not candidates:
        return text
    n_convert = max(1, int(len(candidates) * convert_ratio))
    for i, c in random.sample(candidates, min(n_convert, len(candidates))):
        chars[i] = FANJIAN_MAP[c]
    return ''.join(chars)


def attack_split_char(text: str, replace_ratio: float = 0.2) -> str:
    """
    将合体字拆为部件 (拆字攻击)

    参数:
        text:          原始文本
        replace_ratio: 替换比例, 默认 20%

    返回:
        含拆字的文本

    示例:
        >>> attack_split_char("打枪的不要")
        "打木仓的不要"   # 枪 → 木仓
    """
    if not text:
        return text

    chars = list(text)
    candidates = [(i, c) for i, c in enumerate(chars) if c in SPLIT_CHAR_MAP]

    if not candidates:
        return text

    n_replace = max(1, int(len(candidates) * replace_ratio))
    n_replace = min(n_replace, len(candidates))

    for i, c in random.sample(candidates, n_replace):
        chars[i] = SPLIT_CHAR_MAP[c]

    return ''.join(chars)


if __name__ == '__main__':
    samples = ["代办证件", "免费领取优惠券", "打枪不要", "你好明天"]
    print("=== 繁简混用 ===")
    for s in samples:
        result = attack_fanjian_mix(s)
        print(f"  原文: {s}")
        print(f"  攻击: {result}\n")

    print("=== 拆字攻击 ===")
    for s in ["打枪", "你好吗", "想不想"]:
        result = attack_split_char(s)
        print(f"  原文: {s}")
        print(f"  攻击: {result}")
