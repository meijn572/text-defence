# -*- coding: utf-8 -*-
"""
工具模块 —— 提供整个项目共用的辅助函数
包括：数据加载、分词映射表构建、目录管理等
"""

import os
import json
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


# ============================================================
# 0. 全局配置
# ============================================================

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据目录
DATA_RAW = os.path.join(ROOT_DIR, 'data', 'raw')
DATA_ADV = os.path.join(ROOT_DIR, 'data', 'adversarial')
DATA_PROCESSED = os.path.join(ROOT_DIR, 'data', 'processed')
DATA_DICT = os.path.join(ROOT_DIR, 'data', 'dict')

# 结果目录
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')

# 随机种子（保证实验可复现）
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 设备选择（用 CPU 避免脚本模式下的 CUDA 驱动兼容性问题）
import torch
_USE_GPU = torch.cuda.is_available() and os.environ.get('USE_GPU', '0') == '1'
DEVICE = torch.device('cuda' if _USE_GPU else 'cpu')
print(f"[INFO] 使用设备: {DEVICE} (GPU={'ON' if _USE_GPU else 'OFF'})")


# ============================================================
# 1. 数据加载与保存
# ============================================================

def load_raw_data(filepath: str = None) -> pd.DataFrame:
    """
    加载原始标注数据
    要求格式: CSV 文件，至少包含 'text' 和 'label' 两列
    label: 0=正常, 1=垃圾

    如果文件不存在，自动生成示例数据集用于测试
    """
    if filepath is None:
        filepath = os.path.join(DATA_RAW, 'spam_data.csv')

    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f"[INFO] 加载数据: {filepath}, 共 {len(df)} 条")
        return df
    else:
        print(f"[WARN] 数据文件不存在: {filepath}")
        print("[INFO] 自动生成示例数据集用于测试...")
        return _generate_demo_data()


def _generate_demo_data() -> pd.DataFrame:
    """
    生成演示用数据集（当真实数据不可用时）
    包含常见垃圾短信模板和正常短信模板
    """
    # 垃圾短信模板
    spam_templates = [
        "加微信领取红包大奖",
        "恭喜您获得大奖，点击链接领取",
        "免费领取优惠券，数量有限先到先得",
        "代办各类证件，质量保证，快速出证",
        "专业代理记账，价格优惠",
        "小额贷款，无需抵押，当天放款",
        "您好，您的快递已到达，请点击链接查收",
        "性感美女在线直播，点击观看",
        "恭喜您成为今日幸运用户，领取苹果手机",
        "内部股票推荐，稳赚不赔，加群了解",
        "您的银行卡异常，请点击链接验证",
        "招聘兼职，日入千元，在家办公",
        "免费办理大额信用卡，无需面签",
        "学位证毕业证快速办理，官网可查",
        "澳门首家线上赌场上线啦，注册送彩金",
        "刷单返利，一单50，多劳多得",
        "您有一个包裹待签收，点击确认地址",
        "积分即将过期，点击兑换好礼",
        "特价机票内部渠道，比官网便宜一半",
        "您已欠费，请尽快缴费以免停机",
        "加薇芯看更多精彩内容",
        "免废领取优惠卷，先到先得",
        "加我威信，带你赚大前",
        "代幵发票，诚心合作",
        "棋牌游戏，真人视讯，体验刺激",
    ]

    # 正常短信模板
    normal_templates = [
        "明天上午十点开会，请准时参加",
        "妈妈我今天晚点回家吃饭",
        "你的快递放门卫了记得拿",
        "老师今天的作业是什么",
        "周末一起去爬山吗",
        "生日快乐祝你天天开心",
        "下周二部门团建大家做好准备",
        "请把上周的报告发给我一下",
        "好的没问题我明天处理",
        "今天晚上吃什么我来做",
        "你到哪了我在地铁站等你",
        "记得明天带身份证去办理",
        "下午三点面试别忘了",
        "爸我今天加班不回去吃了",
        "你的体检报告出来了指标都正常",
        "这周末有没有空聚一下",
        "帮我看看这个文件格式对不对",
        "天气冷了多穿点衣服",
        "图书馆的书到期了记得还",
        "下个月的预算表做好了吗",
    ]

    # 生成数据：每个模板做少量随机变体
    data = []
    for template in spam_templates:
        data.append({'text': template, 'label': 1})
        # 加一些简单变体
        for _ in range(3):
            variant = template
            data.append({'text': variant, 'label': 1})

    for template in normal_templates:
        data.append({'text': template, 'label': 0})
        for _ in range(3):
            variant = template
            data.append({'text': variant, 'label': 0})

    df = pd.DataFrame(data)
    # 保存到 raw 目录
    os.makedirs(DATA_RAW, exist_ok=True)
    save_path = os.path.join(DATA_RAW, 'spam_data.csv')
    df.to_csv(save_path, index=False)
    print(f"[INFO] 示例数据已保存至: {save_path}, 共 {len(df)} 条")
    return df


def save_adv_data(df: pd.DataFrame, attack_type: str):
    """保存对抗样本到 adversarial 目录"""
    os.makedirs(DATA_ADV, exist_ok=True)
    path = os.path.join(DATA_ADV, f'adv_{attack_type}.csv')
    df.to_csv(path, index=False)


def load_adv_data(attack_type: str) -> pd.DataFrame:
    """加载指定类型的对抗样本"""
    path = os.path.join(DATA_ADV, f'adv_{attack_type}.csv')
    return pd.read_csv(path)


# ============================================================
# 2. 映射表构建
# ============================================================

def build_homophone_map() -> Dict[str, List[str]]:
    """
    构建音近字映射表
    使用 pypinyin 按拼音分组，同音字互为替换候选

    返回: {'微': ['威', '薇', '危', ...], '信': ['芯', '辛', '新', ...], ...}
    """
    from pypinyin import pinyin, Style

    # 常用3500字（取最常用的一批）
    common_chars = (
        # 一级常用字 2500
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
        "县兵虫固除般引齿千胜细影济白格效置推空配刀叶率今选养德话"
        "查差半敌始片施响收华觉备名红续均药标记难存测士身紧液派准"
        "斤角降维板许破述技消底床田势端感往神便圆村构照容非搞亚磨"
        "族火段算适讲按值美态黄易彪服早班麦削信排台声该击素张密害"
        "候草何树肥继右属市严径螺检左页抗苏显苦英快称坏移约巴材省"
        "黑武培著河帝仅针怎植京助升王眼她抓含苗副杂普谈围食射源例"
        "致酸旧却充足短划剂宣环落首尺波承粉践府鱼随考刻靠够满夫失"
        "包住促枝局菌杆周护岩师举曲春元超负砂封换太模贫减阳扬江析"
        "亩木言球朝医校古呢稻宋听唯输滑站另卫字鼓刚写刘微略范供阿"
        "块某功套友限项余倒卷创律雨让骨远帮初皮播优占死毒圈伟季训"
        "控激找叫云互跟裂粮粒母练塞钢顶策双留误础吸阻故寸盾晚丝女"
        "散焊攻株亲院冷彻弹错散尼盾商视艺灭版烈零室轻血倍缺厘泵察"
        "绝富城冲喷壤简否柱李望盘磁雄似困巩益洲脱投送奴侧润盖挥距"
        "触星松获独官混纪依未突架宽冬兴章湿偏纹吃执阀矿寨责熟稳夺"
        "硬价努翻奇甲预职评读背协损棉侵灰虽矛厚罗泥辟告卵箱掌氧恩"
        "爱停曾溶营终纲孟钱待尽俄缩沙退陈讨奋械载胞幼哪剥迫旋征槽"
        "殖握担仍呀载鲜吧卡粗介钻逐弱脚怕盐末阴丰编印蜂急拿扩伤飞"
        "露核缘游振操央伍域甚迅辉异序免纸夜乡久隶缸夹念兰映沟乙吗"
        "儒杀汽磷艰晶插埃燃欢铁补咱芽永瓦倾阵碳演威附牙芽永瓦斜灌"
        "欧献顺猪洋腐请透司危括脉笑宜若尾束壮暴企菜穗楚汉愈绿拖牛"
        "份染既秋遍锻玉夏疗尖殖井费州访吹荣铜沿替滚客召旱悟刺脑措"
        "贯藏敢令隙炉壳硫煤迎铸粘探临薄旬善福纵择礼愿伏残雷延烟句"
        "纯渐耕跑泽慢栽鲁赤繁境潮横掉锥希池败船假亮谓托伙哲怀割摆"
        "贡呈劲财仪沉炼麻罪祖息辅穿贷销齐鼠抽画饲龙库守筑房歌寒困"
        "哥吃钱喝酒茶烟睡醒累忙闲笑哭闹买卖借还偷抢骗偷"
        # 垃圾文本高频字
        "免费领取优惠红包大奖点击代理贷款小姐证件办理微信加群赚现金"
        "牌秒杀卡套特价折扣快送件品质保证号码链连性感美女直播在线"
        "观看苹果手机内部股票推荐稳赚不赔招聘兼职日入千元在家办公"
        "学位证毕业证快速办理官网可查澳门线上赌场注册送彩金刷单返利"
        "多劳多得包裹签收积分过期兑换好礼内部渠道便宜欠费缴费停机"
        "薇芯佳戴理洁晓待款废棉面费威信嘉家"
    )

    # 按拼音分组
    pinyin_groups = defaultdict(list)
    for char in common_chars:
        try:
            py = pinyin(char, style=Style.TONE3)[0][0]
            pinyin_groups[py].append(char)
        except Exception:
            pass

    # 每组内互为替换候选
    homophone_map = {}
    for py, chars in pinyin_groups.items():
        if len(chars) >= 2:
            for c in chars:
                homophone_map[c] = [x for x in chars if x != c]

    # 保存到文件
    os.makedirs(DATA_DICT, exist_ok=True)
    with open(os.path.join(DATA_DICT, 'homophone_map.json'), 'w', encoding='utf-8') as f:
        json.dump(homophone_map, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 音近字映射表已构建: {len(homophone_map)} 个字符有同音替换候选")
    return homophone_map


def load_homophone_map() -> Dict[str, List[str]]:
    """加载音近字映射表，如果不存在则自动构建"""
    path = os.path.join(DATA_DICT, 'homophone_map.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return build_homophone_map()


# ============================================================
# 3. 中文文本工具
# ============================================================

def get_chinese_chars(text: str) -> List[str]:
    """提取文本中的所有中文字符"""
    return [c for c in text if '\u4e00' <= c <= '\u9fff']


def is_chinese_char(c: str) -> bool:
    """判断单个字符是否为中文"""
    return '\u4e00' <= c <= '\u9fff'


# ============================================================
# 4. 评估工具
# ============================================================

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算分类评估指标

    返回: {'accuracy': ..., 'precision': ..., 'recall': ..., 'f1': ...}
    """
    return {
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'f1': round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def print_metrics(metrics: Dict[str, float], title: str = ""):
    """格式化打印评估指标"""
    if title:
        print(f"\n{'='*50}\n  {title}\n{'='*50}")
    print(f"  Accuracy:  {metrics['accuracy']:.2%}")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall:    {metrics['recall']:.2%}")
    print(f"  F1-Score:  {metrics['f1']:.2%}")


# ============================================================
# 5. 固定随机种子
# ============================================================

def set_seed(seed: int = SEED):
    """固定所有随机种子，确保实验可复现"""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # 测试：生成演示数据 + 构建音近字映射表
    df = load_raw_data()
    print(f"\n数据统计:")
    print(f"  正常短信: {(df['label']==0).sum()} 条")
    print(f"  垃圾短信: {(df['label']==1).sum()} 条")

    build_homophone_map()
