# 数据集划分说明

## 1. 原始数据

**源文件：** `data/raw/spam_data_full.csv`

- **规模：** 12,935 条
- **标签分布：**
  - 正常短信 (label=0)：7,741 条（59.8%）
  - 垃圾短信 (label=1)：5,194 条（40.2%）
- **格式：** CSV，包含 `text` 和 `label` 两列

---

## 2. 训练/测试集划分

基于随机种子 42 的分层抽样，以 7:3 比例划分：

### 训练集 `data/adversarial/train.csv`

- **规模：** 10,144 条（70% 的原始数据）
- **标签分布：**
  - 正常短信：5,418 条
  - 垃圾短信：4,726 条（含对抗增强）
    - 原始垃圾：4,181 条
    - 攻击类型 A (字符删除)：545 条
    - 攻击类型 B (字符插入)：545 条

**用途：** 用于训练朴素 BERT 和四通道融合模型

### 测试集 `data/adversarial/test_full.csv`

- **规模：** 17,903 条（30% 的原始数据 + A~I 对抗样本）
- **标签分布：**
  - 正常短信：2,323 条（全部原始）
  - 垃圾短信：15,580 条
    - 原始垃圾 (attack_type=NaN)：3,881 条
    - 对抗_A 到对抗_I (各攻击类型各 1,558 条)：共 14,022 条
    - 攻击类型：A, B, C, D, E, F, G, H, I

**用途：** 基础（13子集）评测的数据池，也是生成强攻击 J/K/L 的来源

**关键设计：** 正常样本与垃圾样本不重叠，避免数据泄漏

---

## 3. 对抗样本文件 A-L

### 设计原则

**A~I（基础对抗）：** 每个文件均包含"全部正常样本 + 该攻击类型的垃圾样本"

**J~L（强对抗）：** 基于 I 生成，混合强度设置（如乱序率 80%、替换率 80%）

### 文件清单

所有对抗文件位于 `data/adversarial/`，格式均为 CSV（`text`, `label`, `attack_type`, `original_text`）：

| 文件名 | 攻击类型 | 垃圾样本数 | 正常样本数 | 总计 | 描述 |
|--------|--------|----------|----------|------|------|
| `adv_A_char_delete.csv` | A | 1,558 | 2,323 | 3,881 | 字符删除 |
| `adv_B_char_insert.csv` | B | 1,558 | 2,323 | 3,881 | 字符插入 |
| `adv_C_homoglyph_unicode.csv` | C | 1,558 | 2,323 | 3,881 | 跨语种同形字 |
| `adv_D_zero_width.csv` | D | 1,558 | 2,323 | 3,881 | 零宽字符注入 |
| `adv_E_synonym.csv` | E | 1,558 | 2,323 | 3,881 | 同义词替换 |
| `adv_F_homophone_cn.csv` | F | 1,558 | 2,323 | 3,881 | 音近字替换 |
| `adv_G_homoglyph_cn.csv` | G | 1,558 | 2,323 | 3,881 | 形近字替换 |
| `adv_H_fanjian_split.csv` | H | 1,558 | 2,323 | 3,881 | 繁简混用 |
| `adv_I_char_shuffle.csv` | I | 1,558 | 2,323 | 3,881 | 字符乱序 |
| `adv_J_strong_shuffle.csv` | J ★ | 1,558 | 2,323 | 3,881 | 强乱序（80% 乱序率） |
| `adv_K_strong_homophone.csv` | K ★ | 1,558 | 2,323 | 3,881 | 强音近（80% 替换率） |
| `adv_L_combined.csv` | L ★ | 1,558 | 2,323 | 3,881 | 混合攻击（乱序 + 音近） |

**说明：**
- ★ 标记表示强攻击，强度参数（如乱序率、替换率）为 80%，相比 A~I 的默认 30% 更强
- A~I 均出自 `test_full.csv` 的对应子集
- J~K~L 在评测时动态生成（见 `evaluate_direct.py` 第 97-128 行）

### 生成脚本位置

- **生成逻辑：** [experiments/generate_adv.py](experiments/generate_adv.py) 第 43-61 行
- **动态生成 J/K/L：** [evaluate_direct.py](evaluate_direct.py) 第 97-128 行

---

## 4. 评测子集划分方式

### 基础评测（正常评测）：13 个子集

从各对应的对抗样本文件读取：

| 子集名称 | 来源文件 | 规模 | 用途 |
|---------|---------|------|------|
| 原始样本 | `test_full.csv`（无攻击行） | 3,881 | baseline性能 |
| 对抗_A (字符删除) | `adv_A_char_delete.csv` | 3,881 | 攻击鲁棒性 |
| 对抗_B (字符插入) | `adv_B_char_insert.csv` | 3,881 | 攻击鲁棒性 |
| ... (C~I) | ... | 3,881 | 攻击鲁棒性 |
| 对抗_J (★强乱序) | `adv_J_strong_shuffle.csv` | 3,881 | 强攻击鲁棒性 |
| 对抗_K (★强音近) | `adv_K_strong_homophone.csv` | 3,881 | 强攻击鲁棒性 |
| 对抗_L (★混合攻击) | `adv_L_combined.csv` | 3,881 | 强攻击鲁棒性 |

**所有子集的标签分布相同：**
- 正常样本：2,323（60.0%）
- 垃圾样本：1,558（40.0%）
- 总计：3,881

### 消融评测

每个对抗子集在四通道融合模型上进行通道置零消融：

- `-文本通道`：屏蔽 BERT（文本理解通道）
- `-拼音通道`：屏蔽拼音音素 CNN（语音相似性通道）
- `-视觉通道`：屏蔽 ResNet（字形相似性通道）
- `-字袋通道`：屏蔽词频直方图（统计特征通道）

---

## 5. 数据流向示意

```
raw/spam_data_full.csv (12,935)
        |
        +--- 70% -----> train.csv (10,144)
        |                    |
        |                    +-- 训练 baseline_bert.pth
        |                    +-- 训练 baseline_bert_aug.pth
        |                    +-- 训练 fusion_model.pth
        |
        +--- 30% -----> test_full.csv (17,903)
                             |
                             +-- 提取原始部分 (3,881)
                             +-- 提取攻击部分 (A-I 各 1,558)
                                  |
                                  +-- adv_A~adv_I（各含全部正常 2,323 + 该攻击垃圾 1,558）
                                  +-- 生成强攻击 J/K/L（各含全部正常 2,323 + 强攻击垃圾 1,558）
                                       |
                                       +-- 13 子集 (原始 + A~L)
                                            |
                                            +-- 正常评测
                                            +-- 消融评测
```

---

## 6. 关键参数配置

| 参数 | 值 | 位置 |
|-----|-----|------|
| 训练/测试比 | 70:30 | [generate_adv.py](experiments/generate_adv.py) 第 15 行 |
| 训练集对抗增强率 | A+B 各 15% | [generate_adv.py](experiments/generate_adv.py) 第 22 行 |
| 测试集对抗样本数 | 每类 1,558 | 由 30% 的原始垃圾数自动确定 |
| 强攻击强度 | 乱序率/替换率 80% | [evaluate_direct.py](evaluate_direct.py) 第 106, 111, 120 行 |
| 随机种子 | 42 | `utils.py` SEED 常量 |

---

## 7. 特殊设计说明

### Q1: 为什么 adv_{A..I} 都包含全部正常样本？

**目的：** 保证单文件可独立用作二分类评测集，不依赖其他文件。

**优势：**
- 每个对抗文件自包含，支持单独评测某种攻击的鲁棒性
- 避免混淆：正常样本只来自 test_full 的原始部分，不重复计算

**代价：** 正常样本被重复 10 次（A-J 各一次），但由于总体规模（3,881 × 10 ≈ 39k）不会造成内存压力。

### Q2: 为什么 test_full.csv 不重复正常样本？

**目的：** 作为统一的"数据池"存储，避免数据冗余。

**用途：** 主要在解析其他 adv_{*}.csv 文件时作为参考，或在特殊评测（如对比原始集 vs 各攻击集）时使用。

### Q3: 为什么训练集也做对抗增强（A+B）？

**目的：** 让模型在训练阶段见过部分对抗扰动，提升泛化鲁棒性。

**设置：** 仅增强 A（删除）、B（插入）两类，占垃圾短信的 15%（各 7.5%），保持数据分布不至过度人工化。

---

## 8. 文件检查清单

确保数据完整性，运行以下检查：

```bash
# 检查文件存在
ls -lh data/adversarial/train.csv data/adversarial/test_full.csv
ls -lh data/adversarial/adv_*.csv | wc -l  # 应显示 12

# 验证行数
wc -l data/adversarial/*.csv

# 验证标签无缺失
python -c "
import pandas as pd
for f in ['train.csv', 'test_full.csv'] + [f'adv_{x}_*.csv' for x in 'ABCDEFGHIJKL']:
    df = pd.read_csv(f'data/adversarial/{f}')
    assert df['label'].notna().all(), f'{f} has null labels'
print('✓ All files pass label integrity check')
"
```

---

## 9. 结果输出位置

### 评测结果

- **详细指标（Precision/Recall/F1/Accuracy + 混淆矩阵）：** [results/eval_results_pr.csv](results/eval_results_pr.csv)
- **图表（F1 对比柱状图）：** [results/figures/compare_f1.png](results/figures/compare_f1.png)

### 日志

- **完整运行日志：** [results/run_all_log.txt](results/run_all_log.txt)
- **指标计算日志：** [results/metrics_run.log](results/metrics_run.log)

---

## 附表：标签分布汇总

| 数据集 | 规模 | 正常 (label=0) | 垃圾 (label=1) | 正常比例 | 垃圾比例 |
|--------|------|--------|--------|---------|---------|
| 原始数据 | 12,935 | 7,741 | 5,194 | 59.8% | 40.2% |
| 训练集 | 10,144 | 5,418 | 4,726 | 53.4% | 46.6% |
| 测试集 | 17,903 | 2,323 | 15,580 | 13.0% | 87.0% |
| adv_* (各文件) | 3,881 | 2,323 | 1,558 | 60.0% | 40.0% |

**注：** 测试集中垃圾比例高是因为包含了 A~I 的所有对抗样本。单个 adv 文件维持 60:40 以保持评测的公平性。
