# -*- coding: utf-8 -*-
"""
防御 ④: 视觉通道 (文本渲染 + CNN)

原理:
  将文本渲染为图像, 用预训练 CNN 提取视觉字形特征
  跨语种同形字在图片上看起来不同 → CNN 能捕捉字形差异

流程:
  文本 → PIL 渲染白底黑字图片 (224×224) → ResNet18 → 512维特征
"""

import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# 图片渲染参数
# ============================================================
IMAGE_SIZE = 224          # 输出图片尺寸
FONT_SIZE = 20            # 默认字号
FONT_COLOR = (0, 0, 0)    # 黑色文字
BG_COLOR = (255, 255, 255)  # 白色背景

# 尝试加载中文字体, 失败则用默认字体
def _get_chinese_fonts():
    """
    获取系统中可用的中文字体列表
    用于多字体随机增强
    """
    import os
    font_paths = []

    # Windows 常见字体路径
    win_fonts = [
        'C:/Windows/Fonts/simsun.ttc',       # 宋体
        'C:/Windows/Fonts/simhei.ttf',       # 黑体
        'C:/Windows/Fonts/simkai.ttf',       # 楷体
        'C:/Windows/Fonts/msyh.ttc',         # 微软雅黑
        'C:/Windows/Fonts/STKAITI.TTF',       # 华文楷体
    ]

    for fp in win_fonts:
        if os.path.exists(fp):
            try:
                font_paths.append(fp)
            except Exception:
                pass

    return font_paths if font_paths else [None]  # None 表示用 PIL 默认字体


FONT_PATHS = _get_chinese_fonts()
print(f"[视觉通道] 可用中文字体: {len([f for f in FONT_PATHS if f])} 个")


# 图像预处理 (ImageNet 标准化)
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),  # 转为 [0,1] 的 tensor, (C, H, W)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def render_text_to_image(text: str, font_path: str = None,
                         font_size: int = None) -> Image.Image:
    """
    将文本渲染为白底黑字图片

    参数:
        text:      要渲染的文本
        font_path: 字体路径 (None 用默认)
        font_size: 字号 (None 用随机 16~24)

    返回:
        PIL Image, RGB 模式, 尺寸 224×224
    """
    import random
    if font_size is None:
        font_size = random.randint(16, 24)

    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    # 创建画布
    img = Image.new('RGB', (IMAGE_SIZE * 2, IMAGE_SIZE * 2), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # 文字换行处理: 每行最多放 15 个字
    max_chars_per_line = 15
    lines = []
    for i in range(0, len(text), max_chars_per_line):
        lines.append(text[i:i + max_chars_per_line])

    # 计算文字总高度和起始位置
    line_height = font_size + 4
    total_height = len(lines) * line_height
    start_y = max(10, (IMAGE_SIZE * 2 - total_height) // 2)

    for idx, line in enumerate(lines):
        # 获取该行文字宽度
        try:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
        except Exception:
            line_width = len(line) * font_size

        x = max(10, (IMAGE_SIZE * 2 - line_width) // 2)
        y = start_y + idx * line_height
        draw.text((x, y), line, fill=FONT_COLOR, font=font)

    # 缩放到标准尺寸
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

    return img


class VisualChannel(nn.Module):
    """
    视觉通道 —— ResNet18 图像编码器

    输入: 正规化后的中文文本 (内部自动渲染为图片)
    输出: 512维视觉字形特征向量
    """

    def __init__(self, freeze_cnn: bool = True, dropout: float = 0.2):
        """
        参数:
            freeze_cnn: 是否冻结 ResNet 预训练权重
            dropout:    dropout 比例
        """
        super().__init__()

        print("[视觉通道] 加载 ResNet18 预训练模型")
        # 尝试下载预训练权重，网络不可用时使用随机初始化
        try:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception as e:
            print(f"[视觉通道] 下载失败({e})，使用随机初始化")
            resnet = models.resnet18(weights=None)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])  # 去掉 FC
        self.feature_dim = 512

        # 冻结 CNN 参数
        if freeze_cnn:
            for param in self.cnn_backbone.parameters():
                param.requires_grad = False
            print("[视觉通道] ResNet 参数已冻结")

        self.dropout = nn.Dropout(dropout)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _texts_to_images(self, texts: list) -> torch.Tensor:
        """
        将文本批量渲染为图片 tensor

        返回: (batch, 3, 224, 224)
        """
        import random
        device = next(self.cnn_backbone.parameters()).device
        image_tensors = []

        for text in texts:
            # 随机选字体 (多字体增强)
            if FONT_PATHS and random.random() < 0.8:
                font_path = random.choice(FONT_PATHS)
            else:
                font_path = None

            # 渲染图片
            img = render_text_to_image(text, font_path=font_path)
            # 转为 tensor 并标准化
            img_tensor = IMAGE_TRANSFORM(img)
            image_tensors.append(img_tensor)

        return torch.stack(image_tensors).to(device)

    def forward(self, texts: list) -> torch.Tensor:
        """
        前向传播

        参数:
            texts: 文本列表

        返回:
            visual_features: (batch_size, 512) 视觉特征向量
        """
        # 文本 → 图片 tensor
        images = self._texts_to_images(texts)  # (batch, 3, 224, 224)

        # CNN 提取特征
        features = self.cnn_backbone(images)    # (batch, 512, 1, 1)
        features = features.view(features.size(0), -1)  # (batch, 512)
        features = self.dropout(features)

        return features


if __name__ == '__main__':
    print("=" * 50)
    print("  视觉通道模块测试")
    print("=" * 50)

    # 测试渲染
    test_text = "免费领取优惠券"
    img = render_text_to_image(test_text)
    print(f"\n文本: {test_text}")
    print(f"渲染图片尺寸: {img.size}")
    # 保存测试图片
    import os
    os.makedirs('data/processed', exist_ok=True)
    img.save('data/processed/test_render.png')
    print("测试图片已保存: data/processed/test_render.png")

    # 测试模型
    channel = VisualChannel()
    test_texts = ["免费领取", "аррlе", "明天开会"]
    with torch.no_grad():
        features = channel(test_texts)
    print(f"\n输入: {len(test_texts)} 条文本")
    print(f"输出特征维度: {features.shape}")  # 应为 (3, 512)
    print("✓ 视觉通道测试通过")
