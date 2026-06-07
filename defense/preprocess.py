# -*- coding: utf-8 -*-
"""
防御 ①: 中文深度正规化 (前置预处理)

处理内容:
  - NFKC Unicode 正规化: 统一全角/半角, 删除线等装饰字符
  - 零宽字符剔除: 移除 U+200B/U+200C/U+200D/U+FEFF
  - 繁简统一: 繁体转简体 (使用 OpenCC)
  - 跨语种同形字映射: 西里尔 'а' → 拉丁 'a'

注意: 正规化不还原音近字/形近字, 留给后级通道处理
"""

import re
import unicodedata


# 零宽字符正则
ZERO_WIDTH_RE = re.compile(r'[\u200B\u200C\u200D\uFEFF\u200E\u200F\u2060\u180E]')

# 跨语种同形字映射表
UNICODE_HOMOGLYPH_MAP = {
    # 西里尔 → 拉丁
    '\u0430': 'a', '\u0410': 'A',  # а А
    '\u0435': 'e', '\u0415': 'E',  # е Е
    '\u043E': 'o', '\u041E': 'O',  # о О
    '\u0440': 'p', '\u0420': 'P',  # р Р
    '\u0441': 'c', '\u0421': 'C',  # с С
    '\u0443': 'y',                  # у
    '\u0445': 'x', '\u0425': 'X',  # х Х
    '\u0412': 'B',                  # В
    '\u041C': 'M',                  # М
    '\u041D': 'H',                  # Н
    '\u0422': 'T',                  # Т
    '\u0406': 'I', '\u0456': 'i',  # І і
    '\u0408': 'J', '\u0458': 'j',  # Ј ј
    '\u041A': 'K', '\u043A': 'k',  # К к
    # 希腊 → 拉丁
    '\u0391': 'A',                  # Α
    '\u0395': 'E',                  # Ε
    '\u0397': 'H',                  # Η
    '\u0399': 'I',                  # Ι
    '\u039A': 'K',                  # Κ
    '\u039C': 'M',                  # Μ
    '\u039D': 'N',                  # Ν
    '\u039F': 'O',                  # Ο
    '\u03A1': 'P',                  # Ρ
    '\u03A4': 'T',                  # Τ
    '\u03A5': 'Y',                  # Υ
    '\u03A7': 'X',                  # Χ
}


def remove_zero_width(text: str) -> str:
    """
    移除零宽字符

    这些字符肉眼不可见但占据字符位置, 会破坏 BERT tokenizer 的分词边界
    """
    return ZERO_WIDTH_RE.sub('', text)


def normalize_unicode(text: str) -> str:
    """
    NFKC Unicode 正规化

    效果:
      - 全角英数字 → 半角: "ＡＢＣ" → "ABC"
      - 全角假名 → 半角
      - 删除线组合字符 → 去除: "x\u0336" → "x"
      - 连字字符 → 拆分
    """
    return unicodedata.normalize('NFKC', text)


def map_homoglyph_unicode(text: str) -> str:
    """
    将跨语种同形 Unicode 字符映射回标准拉丁字母

    例如: 西里尔 'а' (U+0430) → 拉丁 'a' (U+0061)
    """
    result = []
    for c in text:
        result.append(UNICODE_HOMOGLYPH_MAP.get(c, c))
    return ''.join(result)


def fanjian_to_simplified(text: str) -> str:
    """
    繁体转简体

    优先使用 OpenCC, 不可用时使用内置映射表
    """
    try:
        from opencc import OpenCC
        cc = OpenCC('t2s')
        return cc.convert(text)
    except ImportError:
        # OpenCC 不可用, 使用简单映射
        FANJIAN_SIMPLE = {
            '辦': '办', '證': '证', '開': '开', '關': '关',
            '門': '门', '車': '车', '馬': '马', '風': '风',
            '電': '电', '長': '长', '時': '时', '問': '问',
            '對': '对', '動': '动', '實': '实', '學': '学',
            '國': '国', '個': '个', '們': '们', '爲': '为',
            '會': '会', '發': '发',
        }
        result = []
        for c in text:
            result.append(FANJIAN_SIMPLE.get(c, c))
        return ''.join(result)


def preprocess_text(text: str, verbose: bool = False) -> str:
    """
    中文深度正规化 —— 完整预处理管线

    ┌─────────────────────────────────────────┐
    │ 输入: 原始文本 (可能含各类混淆)            │
    │   ↓                                     │
    │ 步骤1: NFKC Unicode 正规化               │
    │ 步骤2: 零宽字符剔除                       │
    │ 步骤3: 跨语种同形字映射                   │
    │ 步骤4: 繁简统一 (繁体→简体)               │
    │   ↓                                     │
    │ 输出: 正规化文本                          │
    └─────────────────────────────────────────┘

    注意: 不还原音近字/形近字! 例如:
      "佳薇芯" → 仍然是 "佳薇芯" (不强行改"微信")
      因为强行还原会误伤正常人名, 交给语音通道处理
    """
    original = text

    # 步骤1: NFKC 正规化
    text = normalize_unicode(text)

    # 步骤2: 零宽字符剔除
    text = remove_zero_width(text)

    # 步骤3: 跨语种同形字映射
    text = map_homoglyph_unicode(text)

    # 步骤4: 繁简统一
    text = fanjian_to_simplified(text)

    if verbose and text != original:
        print(f"[正规化] 原: {original!r} → 现: {text!r}")

    return text


if __name__ == '__main__':
    # 测试各种正规化场景
    test_cases = [
        # (输入, 描述)
        ("代\u200B办\u200C证件", "零宽字符"),
        ("аррlе", "西里尔同形字"),
        ("代辦證件", "繁体字"),
        ("x\u0336y\u0336", "删除线组合字符"),
        ("ＡＢＣ123", "全角英数字"),
        ("佳薇芯加您", "音近字(不应被还原)"),
    ]

    print("=" * 60)
    print("  中文深度正规化测试")
    print("=" * 60)
    for text, desc in test_cases:
        result = preprocess_text(text)
        changed = "✓" if text != result else "✗ 无变化"
        print(f"\n  [{desc}]")
        print(f"  输入: {text!r} (长度: {len(text)})")
        print(f"  输出: {result!r} (长度: {len(result)})")
        print(f"  变化: {changed}")
