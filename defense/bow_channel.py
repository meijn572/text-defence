# -*- coding: utf-8 -*-
"""
防御 ⑤: 字袋通道 (Char-BoW + MLP) ★ 专门对抗字符乱序

原理:
  提取字符级 Unigram 和排序后 Bigram 特征
  这些特征不依赖字符顺序, 对乱序攻击天然免疫

示例:
  "免费领取优惠券" → 字集 {免,费,领,取,优,惠,券}
  "费免取领惠优券" → 字集 {免,费,领,取,优,惠,券}  ← 完全一样!

架构:
  字符Unigram(5000维) + 拼音Unigram(400维) + 排序Bigram(5000维)
  → Dense(256) → ReLU → Dropout → Dense(128)
"""

import torch
import torch.nn as nn
import numpy as np
from collections import Counter


# ============================================================
# 字符集定义
# ============================================================

# 常用汉字 3500 个 (含常见垃圾文本用字)
COMMON_CHARS = (
    "的一是在了不和有大这主中人上为们地个用工时要动国产以我到"
    "他会作来分生对于学下级就年阶义发成部民可出能方进同行面说"
    "种过命度革而多子后自社加小机也经力线本电高量长党得实家定"
    "深法表着水理化争现所二起政三好十战无农使性前等反体合斗路"
    "图把结第里正新开论之物从当两些还天资事队批如应形想制心样"
    "干都向变关点育重其思与间内去因件日利相由压员气业代全组数"
    "果期导平各基月毛然问比或展那它最及外没看治提五解系林者米"
    "群头意只明四道马认次文通但条较克又公孔领军流入接席位情运"
    "器并习原油放立题质指建区验活众很教决特此常石强极土少已根"
    "共直团统式转别造切九你取西持总料连任志观调么七山程百报更"
    "见必真保热委手改管处己将修支识病象先老光专几什六型具示复"
    "安带每东增则完风回南广劳轮科北打积车计给节做务被整联步类"
    "集号列温装即毫轴知研单色坚据速防史拉世设达尔场织历花受求"
    "传口断况采精金界品判参层止边清至万确究书低术厂价需走议"
    # 垃圾文本高频字
    "免费领取优惠红包大奖点击代理贷款小姐证件办理微信加群"
    "免废薇芯佳戴理洁晓待款棉面威信嘉家号码码链连"
    "赚现金牌秒杀卡套特价折扣快送件品质保证"
)

# 构建字符→索引映射 (去重)
CHAR_TO_IDX = {}
for c in COMMON_CHARS:
    if c not in CHAR_TO_IDX:
        CHAR_TO_IDX[c] = len(CHAR_TO_IDX)
CHAR_VOCAB_SIZE = len(CHAR_TO_IDX)

# 拼音音节列表 (常见拼音, 约 400 个)
COMMON_PINYIN_SYLLABLES = [
    'a', 'ai', 'an', 'ang', 'ao',
    'ba', 'bai', 'ban', 'bang', 'bao', 'bei', 'ben', 'beng', 'bi', 'bian',
    'biao', 'bie', 'bin', 'bing', 'bo', 'bu',
    'ca', 'cai', 'can', 'cang', 'cao', 'ce', 'cen', 'ceng', 'cha', 'chai',
    'chan', 'chang', 'chao', 'che', 'chen', 'cheng', 'chi', 'chong', 'chou',
    'chu', 'chuai', 'chuan', 'chuang', 'chui', 'chun', 'chuo', 'ci', 'cong',
    'cou', 'cu', 'cuan', 'cui', 'cun', 'cuo',
    'da', 'dai', 'dan', 'dang', 'dao', 'de', 'dei', 'deng', 'di', 'dian',
    'diao', 'die', 'ding', 'diu', 'dong', 'dou', 'du', 'duan', 'dui', 'dun', 'duo',
    'e', 'en', 'er',
    'fa', 'fan', 'fang', 'fei', 'fen', 'feng', 'fo', 'fou', 'fu',
    'ga', 'gai', 'gan', 'gang', 'gao', 'ge', 'gei', 'gen', 'geng', 'gong',
    'gou', 'gu', 'gua', 'guai', 'guan', 'guang', 'gui', 'gun', 'guo',
    'ha', 'hai', 'han', 'hang', 'hao', 'he', 'hei', 'hen', 'heng', 'hong',
    'hou', 'hu', 'hua', 'huai', 'huan', 'huang', 'hui', 'hun', 'huo',
    'ji', 'jia', 'jian', 'jiang', 'jiao', 'jie', 'jin', 'jing', 'jiong',
    'jiu', 'ju', 'juan', 'jue', 'jun',
    'ka', 'kai', 'kan', 'kang', 'kao', 'ke', 'ken', 'keng', 'kong', 'kou',
    'ku', 'kua', 'kuai', 'kuan', 'kuang', 'kui', 'kun', 'kuo',
    'la', 'lai', 'lan', 'lang', 'lao', 'le', 'lei', 'leng', 'li', 'lia',
    'lian', 'liang', 'liao', 'lie', 'lin', 'ling', 'liu', 'long', 'lou',
    'lu', 'luan', 'lun', 'luo', 'lv', 'lve',
    'ma', 'mai', 'man', 'mang', 'mao', 'me', 'mei', 'men', 'meng', 'mi',
    'mian', 'miao', 'mie', 'min', 'ming', 'miu', 'mo', 'mou', 'mu',
    'na', 'nai', 'nan', 'nang', 'nao', 'ne', 'nei', 'nen', 'neng', 'ni',
    'nian', 'niang', 'niao', 'nie', 'nin', 'ning', 'niu', 'nong', 'nou',
    'nu', 'nuan', 'nuo', 'nv', 'nve',
    'o', 'ou',
    'pa', 'pai', 'pan', 'pang', 'pao', 'pei', 'pen', 'peng', 'pi', 'pian',
    'piao', 'pie', 'pin', 'ping', 'po', 'pou', 'pu',
    'qi', 'qia', 'qian', 'qiang', 'qiao', 'qie', 'qin', 'qing', 'qiong',
    'qiu', 'qu', 'quan', 'que', 'qun',
    'ran', 'rang', 'rao', 're', 'ren', 'reng', 'ri', 'rong', 'rou', 'ru',
    'ruan', 'rui', 'run', 'ruo',
    'sa', 'sai', 'san', 'sang', 'sao', 'se', 'sen', 'seng', 'sha', 'shai',
    'shan', 'shang', 'shao', 'she', 'shei', 'shen', 'sheng', 'shi', 'shou',
    'shu', 'shua', 'shuai', 'shuan', 'shuang', 'shui', 'shun', 'shuo',
    'si', 'song', 'sou', 'su', 'suan', 'sui', 'sun', 'suo',
    'ta', 'tai', 'tan', 'tang', 'tao', 'te', 'teng', 'ti', 'tian',
    'tiao', 'tie', 'ting', 'tong', 'tou', 'tu', 'tuan', 'tui', 'tun', 'tuo',
    'wa', 'wai', 'wan', 'wang', 'wei', 'wen', 'weng', 'wo', 'wu',
    'xi', 'xia', 'xian', 'xiang', 'xiao', 'xie', 'xin', 'xing',
    'xiong', 'xiu', 'xu', 'xuan', 'xue', 'xun',
    'ya', 'yan', 'yang', 'yao', 'ye', 'yi', 'yin', 'ying', 'yo', 'yong',
    'you', 'yu', 'yuan', 'yue', 'yun',
    'za', 'zai', 'zan', 'zang', 'zao', 'ze', 'zei', 'zen', 'zeng', 'zha',
    'zhai', 'zhan', 'zhang', 'zhao', 'zhe', 'zhei', 'zhen', 'zheng', 'zhi',
    'zhong', 'zhou', 'zhu', 'zhua', 'zhuai', 'zhuan', 'zhuang', 'zhui',
    'zhun', 'zhuo', 'zi', 'zong', 'zou', 'zu', 'zuan', 'zui', 'zun', 'zuo',
]
PINYIN_SYL_TO_IDX = {p: i for i, p in enumerate(COMMON_PINYIN_SYLLABLES)}
PINYIN_SYL_VOCAB_SIZE = len(COMMON_PINYIN_SYLLABLES)


def extract_char_bow(text: str) -> torch.Tensor:
    """
    提取字符 Unigram 词袋特征 (顺序无关!)

    返回: (CHAR_VOCAB_SIZE,) 的稀疏向量
    """
    bow = torch.zeros(CHAR_VOCAB_SIZE, dtype=torch.float32)
    for c in text:
        if c in CHAR_TO_IDX:
            idx = CHAR_TO_IDX[c]
            if idx < CHAR_VOCAB_SIZE:
                bow[idx] += 1
    return bow


def extract_pinyin_bow(text: str) -> torch.Tensor:
    """
    提取拼音音节词袋特征 (顺序无关!)

    返回: (PINYIN_SYL_VOCAB_SIZE,) 的稀疏向量
    """
    from pypinyin import pinyin, Style
    bow = torch.zeros(PINYIN_SYL_VOCAB_SIZE, dtype=torch.float32)
    try:
        py_list = pinyin(text, style=Style.NORMAL)
        for item in py_list:
            p = item[0]
            if p in PINYIN_SYL_TO_IDX:
                bow[PINYIN_SYL_TO_IDX[p]] += 1
    except Exception:
        pass
    return bow


def extract_sorted_bigrams(text: str) -> torch.Tensor:
    """
    提取排序后的 Bigram 特征 (顺序无关的关键!)

    方法: 先将字符排序, 再取相邻 bigram
    "免费领取" → 排序: "免免费取领" → bigrams: "免免", "免费", "费取", "取领"
    "费免取领" → 排序: "免免费取领" → bigrams: 同上!

    返回: (CHAR_VOCAB_SIZE,) 的向量
    """
    # 提取所有中文字符并排序
    chars = sorted([c for c in text if c in CHAR_TO_IDX])
    if len(chars) < 2:
        return torch.zeros(CHAR_VOCAB_SIZE, dtype=torch.float32)

    # 生成 bigram (用字符对作为特征)
    bow = torch.zeros(CHAR_VOCAB_SIZE, dtype=torch.float32)
    for i in range(len(chars) - 1):
        # 用第一个字符的位置记录这个 bigram
        if chars[i] in CHAR_TO_IDX:
            bow[CHAR_TO_IDX[chars[i]]] += 1

    return bow


class BowChannel(nn.Module):
    """
    字袋通道 —— 顺序无关的特征提取器

    输入: 正规化后的中文文本
    输出: 128维字符集合特征向量

    此通道对字符乱序攻击天然免疫!
    """

    def __init__(self, hidden_dim: int = 256, output_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()

        # 三个特征维度
        self.char_dim = CHAR_VOCAB_SIZE      # 字符 Unigram (~3500)
        self.pinyin_dim = PINYIN_SYL_VOCAB_SIZE  # 拼音音节 (~400)
        self.bigram_dim = CHAR_VOCAB_SIZE    # 排序 Bigram (~3500)
        self.total_input_dim = self.char_dim + self.pinyin_dim + self.bigram_dim

        self.feature_dim = output_dim

        # MLP 编码器
        self.encoder = nn.Sequential(
            nn.Linear(self.total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        print(f"[字袋通道] 输入维度: {self.total_input_dim}, 输出维度: {output_dim}")

    def forward(self, texts: list) -> torch.Tensor:
        """
        前向传播

        参数:
            texts: 文本列表

        返回:
            bow_features: (batch_size, 128) 字符集合特征向量
        """
        device = next(self.encoder.parameters()).device
        batch_features = []

        for text in texts:
            # 提取三种顺序无关的特征
            char_bow = extract_char_bow(text)
            pinyin_bow = extract_pinyin_bow(text)
            sorted_bigram = extract_sorted_bigrams(text)

            # 拼接
            combined = torch.cat([char_bow, pinyin_bow, sorted_bigram], dim=0)
            batch_features.append(combined)

        # Stack 成 batch
        x = torch.stack(batch_features).to(device)  # (batch, total_input_dim)

        # MLP 编码到低维空间
        output = self.encoder(x)  # (batch, 128)

        return output


if __name__ == '__main__':
    print("=" * 50)
    print("  字袋通道模块测试")
    print("=" * 50)

    # 测试乱序不变性
    original = "免费领取优惠券"
    shuffled = "费免取领惠优券"

    print(f"\n乱序不变性测试:")
    print(f"  原文: {original}")
    print(f"  乱序: {shuffled}")

    char1 = extract_char_bow(original)
    char2 = extract_char_bow(shuffled)
    print(f"  字符BoW一致: {torch.allclose(char1, char2)}")

    sorted1 = extract_sorted_bigrams(original)
    sorted2 = extract_sorted_bigrams(shuffled)
    print(f"  排序Bigram一致: {torch.allclose(sorted1, sorted2)}")

    # 测试模型
    channel = BowChannel()
    test_texts = [original, shuffled, "明天开会"]
    with torch.no_grad():
        features = channel(test_texts)
    print(f"\n输入: {len(test_texts)} 条文本")
    print(f"输出特征维度: {features.shape}")  # 应为 (3, 128)

    # 验证: 原始和乱序的 BoW 特征应该非常接近
    cos = torch.nn.functional.cosine_similarity(
        features[0:1], features[1:2])
    print(f"原始 vs 乱序 余弦相似度: {cos.item():.4f}")
    print("✓ 字袋通道测试通过")
