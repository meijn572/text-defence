"""攻击 M: 中文拼音首字母镶嵌。"""

from pypinyin import lazy_pinyin


def attack_pinyin_abbrev(text: str) -> str:
    """交替保留汉字并以拼音首字母替换下一汉字，维持可读性。

    例如 ``免费领取优惠券`` 变为 ``免f领q优h券``。非中文字符不计入交替位置，
    保持数字、链接、英文和标点不变。
    """
    if not text:
        return text

    transformed = []
    chinese_index = 0
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            if chinese_index % 2 == 1:
                syllable = lazy_pinyin(char, errors='default')[0]
                transformed.append(syllable[0] if syllable else char)
            else:
                transformed.append(char)
            chinese_index += 1
        else:
            transformed.append(char)
    return ''.join(transformed)
