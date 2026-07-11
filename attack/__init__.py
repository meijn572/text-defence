# -*- coding: utf-8 -*-
"""
攻击方模块 —— 生成 9 种对抗样本

设计原则：
  1. 只攻击垃圾样本（label=1），正常短信保持不变
  2. 保持可读性 —— 控制扰动强度, 人仍能读懂
  3. 随机性 —— 每次运行结果不同
  4. 独立生成 —— 每种攻击输出独立 CSV
"""

from .char_delete import attack_char_delete
from .char_insert import attack_char_insert
from .homoglyph_unicode import attack_unicode_homoglyph
from .zero_width import attack_zero_width
from .synonym_replace import attack_synonym
from .homophone_chinese import attack_homophone
from .homoglyph_chinese import attack_homoglyph_cn
from .fanjian_split import attack_fanjian_mix, attack_split_char
from .char_shuffle import attack_adjacent_swap
from .pinyin_abbrev import attack_pinyin_abbrev
from .homoglyph_unicode import HOMOGLYPH_UNICODE_MAP
from .synonym_replace import SYNONYM_MAP

# 攻击函数注册表 —— 方便批量调用
ATTACK_REGISTRY = {
    'A': ('char_delete',       attack_char_delete,       '字符删除'),
    'B': ('char_insert',       attack_char_insert,       '字符插入'),
    'C': ('homoglyph_unicode', attack_unicode_homoglyph,  '跨语种同形字'),
    'D': ('zero_width',        attack_zero_width,        '零宽字符注入'),
    'E': ('synonym',           attack_synonym,           '同义词替换'),
    'F': ('homophone_cn',      attack_homophone,         '中文音近字替换'),
    'G': ('homoglyph_cn',      attack_homoglyph_cn,      '中文形近字替换'),
    'H': ('fanjian_split',     attack_fanjian_mix,       '繁简混用'),
    'I': ('char_shuffle',      attack_adjacent_swap,     '字符乱序'),
    'M': ('pinyin_abbrev',     attack_pinyin_abbrev,     '拼音首字母混淆'),
}


def get_attack_names() -> list:
    """返回所有攻击类型标识符列表"""
    return list(ATTACK_REGISTRY.keys())


def contains_chinese(text: str) -> bool:
    """判断文本是否至少包含一个常用汉字。"""
    return any('\u4e00' <= char <= '\u9fff' for char in str(text))


def is_attack_applicable(text: str, attack_id: str) -> bool:
    """判断攻击是否适用于文本的语言与可替换字符。

    中文专用攻击不应用于纯英文短信；跨语种同形字和同义词攻击还需包含
    对应的可替换对象。调用方仍应丢弃攻击后未发生实际改变的样本。
    """
    if not text:
        return False

    if attack_id in {'A', 'F', 'G', 'H', 'K', 'L', 'M'}:
        return contains_chinese(text)
    if attack_id == 'C':
        return any(char in HOMOGLYPH_UNICODE_MAP for char in text)
    if attack_id == 'E':
        return any(word in text for word in SYNONYM_MAP)
    if attack_id in {'I', 'J'}:
        return len(text) >= 2
    return True


def apply_attack(text: str, attack_id: str) -> str:
    """对单条文本应用指定攻击"""
    if attack_id not in ATTACK_REGISTRY:
        raise ValueError(f"未知攻击类型: {attack_id}, 可选: {list(ATTACK_REGISTRY.keys())}")
    if not is_attack_applicable(text, attack_id):
        return text
    _, func, _ = ATTACK_REGISTRY[attack_id]
    return func(text)
