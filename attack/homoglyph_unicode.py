# -*- coding: utf-8 -*-
"""
攻击 C: 跨语种同形字替换
原理: 将英文字母替换为西里尔/希腊字母中形状完全相同的字符
例如: 拉丁 'a' → 西里尔 'а' (U+0430), 肉眼无法区分
"""

import random

# 同形字映射表: 拉丁字母 → 同形 Unicode 字符
HOMOGLYPH_UNICODE_MAP = {
    'a': 'а',   # U+0430 西里尔小写 a
    'A': 'А',   # U+0410 西里尔大写 A
    'b': 'Ь',   # U+042C 西里尔大写软音符号 (视觉同形)
    'B': 'В',   # U+0412 西里尔大写 Ve
    'c': 'с',   # U+0441 西里尔小写 es
    'C': 'С',   # U+0421 西里尔大写 Es
    'e': 'е',   # U+0435 西里尔小写 ie
    'E': 'Е',   # U+0415 西里尔大写 Ie
    'h': 'һ',   # U+04BB 西里尔小写 shha
    'H': 'Н',   # U+041D 西里尔大写 En
    'i': 'і',   # U+0456 西里尔小写 Byelorussian i
    'I': 'І',   # U+0406 西里尔大写 Byelorussian I
    'j': 'ј',   # U+0458 西里尔小写 je
    'J': 'Ј',   # U+0408 西里尔大写 Je
    'k': 'к',   # U+043A 西里尔小写 ka
    'K': 'К',   # U+041A 西里尔大写 Ka
    'm': 'м',   # U+043C 西里尔小写 em
    'M': 'М',   # U+041C 西里尔大写 Em
    'o': 'о',   # U+043E 西里尔小写 o
    'O': 'О',   # U+041E 西里尔大写 O
    'p': 'р',   # U+0440 西里尔小写 er
    'P': 'Р',   # U+0420 西里尔大写 Er
    's': 'ѕ',   # U+0455 西里尔小写 dze
    'T': 'Т',   # U+0422 西里尔大写 Te
    'x': 'х',   # U+0445 西里尔小写 ha
    'X': 'Х',   # U+0425 西里尔大写 Ha
    'y': 'у',   # U+0443 西里尔小写 u
    'Y': 'Υ',   # U+03A5 希腊大写 Upsilon
}


def attack_unicode_homoglyph(text: str, replace_ratio: float = 0.5) -> str:
    """
    将英文字母替换为同形 Unicode 字符

    参数:
        text:          原始文本
        replace_ratio: 替换比例, 默认 50%

    返回:
        含同形 Unicode 字符的文本

    示例:
        >>> attack_unicode_homoglyph("apple")
        "аррlе"   # 前三个字母被替换为西里尔同形字
    """
    if not text:
        return text

    chars = list(text)
    # 找出所有可替换的英文字母位置
    candidates = [(i, c) for i, c in enumerate(chars)
                  if c in HOMOGLYPH_UNICODE_MAP]

    if not candidates:
        return text

    n_replace = max(1, int(len(candidates) * replace_ratio))
    n_replace = min(n_replace, len(candidates))

    for i, c in random.sample(candidates, n_replace):
        chars[i] = HOMOGLYPH_UNICODE_MAP[c]

    return ''.join(chars)


if __name__ == '__main__':
    samples = ["apple", "PayPal", "iPhone", "Google"]
    for s in samples:
        result = attack_unicode_homoglyph(s)
        print(f"  原文: {s}")
        print(f"  攻击: {result}")
        # 验证：打印每个字符的 Unicode 码点
        print(f"  码点: {' '.join(f'U+{ord(c):04X}' for c in result)}")
        print()
