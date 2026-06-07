# -*- coding: utf-8 -*-
"""
防御方模块 —— 四通道文本检测

通道分工:
  ① 中文深度正规化 → 前置清洗 (Unicode/繁简/零宽)
  ② 文本通道 (BERT) → 语义理解
  ③ 语音通道 (拼音CNN) → 发音模式
  ④ 视觉通道 (渲染CNN) → 字形特征
  ⑤ 字袋通道 (Char-BoW) → 字符集合
  ⑥ 四通道融合 → 综合判定
"""

from .preprocess import preprocess_text
from .text_channel import TextChannel
from .phonetic_channel import PhoneticChannel
from .bow_channel import BowChannel
# 以下模块需要网络下载权重，按需导入，不自动加载：
# from .visual_channel import VisualChannel      # 需下载 ResNet
# from .fusion_model import FusionClassifier     # 依赖 visual_channel
