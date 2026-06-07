# 文本检测对抗攻防系统

> Text Defense: Adversarial Attack & Defense for Chinese Spam Detection  
> 大数据原理与技术 · 期末大作业

## 概述

本项目构建了一个完整的**中文垃圾短信对抗攻防系统**，包含：

- **攻击方**：9 种中文特有对抗攻击（音近字、形近字、繁简混用、字符乱序等）
- **防御方**：四通道融合检测架构（BERT + 拼音CNN + 视觉ResNet + 字袋BoW）
- **实验验证**：3 模型 × 12 攻击类型的全面对比评测

### 核心发现

**预训练 BERT 对中文对抗扰动具有意外的高鲁棒性**——12 种攻击下 F1 均 ≥ 0.95。四通道融合设计合理，但在当前规模下未体现显著优势。

详见 [`实验结论.md`](实验结论.md)

## 快速开始

### 环境要求

```
Python >= 3.8
torch >= 2.0
transformers >= 4.30
```

### 安装

```bash
pip install -r requirements.txt
```

### 演示（无需训练，直接看效果）

```bash
python demo.py
```

加载已训练模型（`data/processed/*.pth`），在 3,300 条测试集上评测并展示单条推理效果。

### 完整实验流程

```bash
# 实验01：对抗样本生成（~30秒）
python run_small_exp.py

# 实验02：基线BERT训练（~13分钟，CPU）
python train_baseline_direct.py

# 实验03+04：融合模型训练 + 评测（~12分钟）
python run_fusion_eval.py

# 强攻击加测（~3分钟）
python run_strong_attack.py
```

> ⚠️ 本机 GPU 因 CUDA 上下文冲突不可用，所有实验均在 CPU 完成。详见 [`项目状态.md`](项目状态.md)。

## 项目结构

```
├── demo.py                      演示脚本（加载模型直接推理）
├── run_small_exp.py             实验01：对抗样本生成
├── train_baseline_direct.py     实验02：基线BERT训练
├── run_fusion_eval.py           实验03+04：融合训练+评测
├── run_strong_attack.py         加测：强攻击对比
├── utils.py                     工具模块
├── requirements.txt             依赖清单
│
├── attack/                      攻击方（9种攻击）
│   ├── char_delete.py           字符删除
│   ├── char_insert.py           字符插入
│   ├── homoglyph_unicode.py     跨语种同形字
│   ├── zero_width.py            零宽字符注入
│   ├── synonym_replace.py       同义词替换
│   ├── homophone_chinese.py     中文音近字替换
│   ├── homoglyph_chinese.py     中文形近字替换
│   ├── fanjian_split.py         繁简/拆字混淆
│   └── char_shuffle.py          字符乱序
│
├── defense/                     防御方（4通道）
│   ├── preprocess.py            中文深度正规化
│   ├── text_channel.py          BERT 文本通道
│   ├── phonetic_channel.py      拼音 TextCNN 语音通道
│   ├── visual_channel.py        渲染+ResNet 视觉通道
│   ├── bow_channel.py           字袋 BoW 通道
│   └── fusion_model.py          四通道融合分类器
│
├── data/
│   ├── raw/                     原始数据
│   ├── adversarial/             对抗样本
│   └── processed/               训练好的模型（.pth）
│
├── results/                     评测结果+图表
├── docs/
│   ├── 架构文档.md              完整架构设计
│   ├── 项目状态.md              实验记录
│   └── 实验结论.md              结果分析
└── README.md                    本文件
```

## 攻击方

| 编号 | 攻击方式 | 原理 | 示例 |
|:----:|---------|------|------|
| A | 字符删除 | 随机删除中文字符 | "免费领取" → "免费取" |
| B | 字符插入 | 插入特殊符号 | "代开发票" → "代*开*发*票" |
| C | 跨语种同形 | Unicode 同形字符替换 | "apple" → "аррlе" |
| D | 零宽注入 | 插入不可见字符 | "代办​证件" |
| E | 同义词替换 | 换说法不换意思 | "办证" → "办理证件" |
| F | **音近字** | 同音汉字替换 | "加微信" → "佳薇芯" |
| G | **形近字** | 形似汉字替换 | "免费" → "免废" |
| H | **繁简混用** | 繁简转换+拆字 | "枪" → "木仓" |
| I | **字符乱序** | 打乱汉字顺序 | "免费领取" → "费免取领" |

## 防御方：四通道架构

```
输入文本
  ├─ ① 中文深度正规化（Unicode + 繁简 + 拼音预处理）
  ├─ ② BERT 文本通道 → 768维语义特征
  ├─ ③ 拼音 CNN 通道 → 256维语音特征
  ├─ ④ ResNet 视觉通道 → 512维字形特征
  ├─ ⑤ BoW 字袋通道 → 128维字符集合特征
  └─ ⑥ 融合层 → 分类（正常/垃圾）
```

## 实验结果

| 攻击类型 | 朴素 BERT | 四通道融合 |
|---------|:---------:|:----------:|
| 原始样本 | 0.9055 | 0.9043 |
| A 字符删除 | 0.9779 | 0.9796 |
| B 字符插入 | 0.9967 | 0.9967 |
| C 跨语种同形 | 0.9510 | 0.9583 |
| D 零宽注入 | 0.9529 | 0.9547 |
| E 同义词 | 0.9529 | 0.9565 |
| F 音近字 | 0.9779 | 0.9761 |
| G 形近字 | 0.9565 | 0.9583 |
| H 繁简混用 | 0.9583 | 0.9529 |
| I 字符乱序 | 0.9865 | 0.9831 |

> 完整评测含强攻击加测（J/K/L），详见 [`实验结论.md`](实验结论.md)。

## 模型下载

模型文件较大（共 1.2GB），请从网盘下载后放入 `data/processed/`：

| 文件 | 大小 | 说明 |
|------|------|------|
| `baseline_bert.pth` | 391MB | 朴素 BERT 基线 |
| `baseline_bert_aug.pth` | 391MB | BERT + 正规化 |
| `fusion_model.pth` | 438MB | 四通道融合 |

> 链接待补充

## 已知限制

- GPU 不可用（subprocess CUDA 上下文冲突，详见项目状态文档）
- CSV 必须在 torch 导入前读取
- 模型较大，建议 CPU 推理或仅用 BERT 模型

## License

MIT
