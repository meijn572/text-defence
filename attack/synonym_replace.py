# -*- coding: utf-8 -*-
"""
攻击 E: 同义词替换
原理: 将文本中的关键词替换为同义表达, 意思不变但用词不同
例如: "免费" → "白送", "领取" → "带走"
"""

import random
import jieba

# 同义词映射表 (可自行扩充)
SYNONYM_MAP = {
    '免费': ['不收费', '白送', '零元', '无偿', '免单'],
    '领取': ['获得', '拿到', '带走', '收取'],
    '办理': ['代办', '处理', '搞定', '操作'],
    '优惠': ['折扣', '特价', '实惠', '便宜'],
    '点击': ['访问', '打开', '进入', '查看'],
    '加群': ['进群', '入群', '添加群聊'],
    '大奖': ['豪礼', '厚礼', '超级奖励'],
    '专业': ['资深', '靠谱', '一流'],
    '贷款': ['借款', '借钱', '融资'],
    '红包': ['现金', '赏金', '奖励金'],
    '代理': ['代办', '中介', '渠道'],
    '兼职': ['副业', '零工', '散活'],
    '注册': ['登记', '开户', '申请'],
    '限量': ['限时', '名额有限', '不多'],
    '秒杀': ['抢购', '特卖', '限时抢'],
}


def attack_synonym(text: str, replace_ratio: float = 0.3) -> str:
    """
    将文本中的关键词随机替换为同义表达

    参数:
        text:          原始文本
        replace_ratio: 替换比例, 默认 30%

    返回:
        同义词替换后的文本

    示例:
        >>> attack_synonym("免费领取优惠券")
        "白送获得折扣券"   # 三个词被替换
    """
    if not text:
        return text

    # 用 jieba 分词
    words = list(jieba.cut(text))
    # 找出可替换的词
    candidates = [(i, w) for i, w in enumerate(words) if w in SYNONYM_MAP]

    if not candidates:
        return text

    n_replace = max(1, int(len(candidates) * replace_ratio))
    n_replace = min(n_replace, len(candidates))

    for i, w in random.sample(candidates, n_replace):
        words[i] = random.choice(SYNONYM_MAP[w])

    return ''.join(words)


if __name__ == '__main__':
    samples = [
        "免费领取优惠券",
        "专业代理记账",
        "点击链接领取红包大奖",
    ]
    for s in samples:
        result = attack_synonym(s)
        print(f"  原文: {s}")
        print(f"  攻击: {result}\n")
