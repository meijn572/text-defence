# -*- coding: utf-8 -*-
"""
实验 01: 一键生成全部对抗样本

从原始标注数据出发, 对垃圾样本应用 9 种攻击,
输出:
  - data/adversarial/adv_{A..I}.csv   每种攻击的对抗样本
  - data/adversarial/test_full.csv     完整混合测试集
  - data/adversarial/train.csv         训练集 (原始 + A/B 增强)
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from utils import (
    load_raw_data, save_adv_data, set_seed,
    DATA_RAW, DATA_ADV
)
from attack import ATTACK_REGISTRY


def main():
    set_seed(42)
    print("=" * 60)
    print("  实验 01: 对抗样本生成")
    print("=" * 60)

    # ================================================================
    # 1. 加载原始数据
    # ================================================================
    df = load_raw_data()
    print(f"\n原始数据统计:")
    print(f"  正常短信: {(df['label'] == 0).sum()} 条")
    print(f"  垃圾短信: {(df['label'] == 1).sum()} 条")

    # ================================================================
    # 2. 划分训练集和测试集
    # ================================================================
    # 测试集取 30%, 用于评估原始样本性能
    train_df, test_orig_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df['label']
    )
    print(f"\n数据划分:")
    print(f"  训练集: {len(train_df)} 条")
    print(f"  测试集(原始): {len(test_orig_df)} 条")

    # ================================================================
    # 3. 生成训练增强数据 (仅 A/B 两种)
    # ================================================================
    print(f"\n{'='*40}")
    print(f"  训练集增强 (仅 A/B)")
    print(f"{'='*40}")

    spam_train = train_df[train_df['label'] == 1].copy()
    normal_train = train_df[train_df['label'] == 0].copy()

    # A. 字符删除增强
    print(f"\n  [训练增强] A. 字符删除...")
    a_func = ATTACK_REGISTRY['A'][1]
    adv_a_texts = spam_train['text'].apply(a_func)
    adv_a_df = pd.DataFrame({
        'text': adv_a_texts,
        'label': 1,
        'attack_type': 'A',
        'original_text': spam_train['text'].values,
    })

    # B. 字符插入增强
    print(f"  [训练增强] B. 字符插入...")
    b_func = ATTACK_REGISTRY['B'][1]
    adv_b_texts = spam_train['text'].apply(b_func)
    adv_b_df = pd.DataFrame({
        'text': adv_b_texts,
        'label': 1,
        'attack_type': 'B',
        'original_text': spam_train['text'].values,
    })

    # 合并训练集: 原始 + A增强 + B增强
    train_full = pd.concat([
        normal_train,    # 正常短信 (不变)
        spam_train,      # 原始垃圾短信
        adv_a_df,        # 字符删除增强
        adv_b_df,        # 字符插入增强
    ], ignore_index=True)
    train_full = train_full.sample(frac=1, random_state=42).reset_index(drop=True)

    train_path = os.path.join(DATA_ADV, 'train.csv')
    train_full.to_csv(train_path, index=False)
    print(f"\n  训练集已保存: {train_path}")
    print(f"  训练集总量: {len(train_full)} 条")
    print(f"    正常: {(train_full['label']==0).sum()}, "
          f"垃圾: {(train_full['label']==1).sum()}")

    # ================================================================
    # 4. 生成测试集对抗样本 (全部 9 种攻击)
    # ================================================================
    print(f"\n{'='*40}")
    print(f"  测试集对抗样本生成 (A~I, 共 9 种)")
    print(f"{'='*40}")

    # 只对测试集中的垃圾样本做攻击
    spam_test = test_orig_df[test_orig_df['label'] == 1].copy()
    normal_test = test_orig_df[test_orig_df['label'] == 0].copy()

    adv_dfs = {}
    for attack_id, (name, func, desc) in ATTACK_REGISTRY.items():
        print(f"\n  [测试攻击] {attack_id}. {desc}...")
        adv_texts = spam_test['text'].apply(func)
        adv_df = pd.DataFrame({
            'text': adv_texts,
            'label': 1,  # 对抗样本仍然标记为垃圾
            'attack_type': attack_id,
            'original_text': spam_test['text'].values,
        })
        save_adv_data(adv_df, f'{attack_id}_{name}')
        adv_dfs[attack_id] = adv_df
        print(f"    生成 {len(adv_df)} 条对抗样本")

    # ================================================================
    # 5. 合并完整测试集
    # ================================================================
    print(f"\n{'='*40}")
    print(f"  合并完整测试集")
    print(f"{'='*40}")

    test_parts = [normal_test, spam_test]  # 原始正常 + 原始垃圾
    for attack_id in sorted(adv_dfs.keys()):
        test_parts.append(adv_dfs[attack_id])

    test_full = pd.concat(test_parts, ignore_index=True)
    test_full = test_full.sample(frac=1, random_state=42).reset_index(drop=True)

    test_path = os.path.join(DATA_ADV, 'test_full.csv')
    test_full.to_csv(test_path, index=False)

    print(f"  完整测试集已保存: {test_path}")
    print(f"\n  测试集构成:")
    print(f"    原始正常:     {(test_full['label']==0).sum()} 条")
    for attack_id in sorted(adv_dfs.keys()):
        count = (test_full['attack_type'] == attack_id).sum()
        desc = ATTACK_REGISTRY[attack_id][2]
        print(f"    对抗_{attack_id} ({desc}): {count} 条")
    print(f"    ─────────────────────")
    print(f"    测试集总计:   {len(test_full)} 条")

    print(f"\n{'='*60}")
    print(f"  ✓ 对抗样本生成完成!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
