# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from utils import load_raw_data, save_adv_data, set_seed, DATA_ADV
from attack import ATTACK_REGISTRY

def main():
    set_seed(42)
    print("=" * 60)
    print("  实验 01: 对抗样本生成 (定制版)")
    print("=" * 60)

    # 1. 加载与划分
    df = load_raw_data()
    train_df, test_orig_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    
    spam_train = train_df[train_df['label'] == 1].copy()
    normal_train = train_df[train_df['label'] == 0].copy()
    spam_test = test_orig_df[test_orig_df['label'] == 1].copy()
    normal_test = test_orig_df[test_orig_df['label'] == 0].copy()

    # 2. 训练集增强 (A+B 占垃圾短信的 30%，即每种 15%)
    print(f"\n[训练增强] 目标：A+B 占垃圾短信 30%")
    target_train_count = int(len(spam_train) * 0.15)
    
    train_parts = [normal_train, spam_train]
    for aid in ['A', 'B']:
        func = ATTACK_REGISTRY[aid][1]
        adv_df = pd.DataFrame({
            'text': spam_train['text'].apply(func),
            'label': 1, 'attack_type': aid, 'original_text': spam_train['text'].values
        }).sample(n=min(target_train_count, len(spam_train)), random_state=42)
        train_parts.append(adv_df)
    
    train_full = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=42)
    train_full.to_csv(os.path.join(DATA_ADV, 'train.csv'), index=False)
    print(f"  训练集已保存，共 {len(train_full)} 条")

    # 3. 测试集生成
    #   - adv_{A..I}.csv: 该攻击垃圾样本 + 全部正常样本（便于单文件二分类评测）
    #   - test_full.csv: 仍保持 原始正常 + 原始垃圾 + 各攻击垃圾，避免重复叠加正常样本
    print(f"\n[测试增强] 每种攻击对全部 {len(spam_test)} 条垃圾短信生成对抗样本")
    print(f"[测试增强] 每个 adv_{{A..I}} 文件将混入全部 {len(normal_test)} 条正常样本")

    adv_dfs = []
    normal_for_adv = pd.DataFrame({
        'text': normal_test['text'].values,
        'label': normal_test['label'].values,
        'attack_type': 'normal',
        'original_text': normal_test['text'].values,
    })

    for aid, (name, func, desc) in ATTACK_REGISTRY.items():
        adv_df = pd.DataFrame({
            'text': spam_test['text'].apply(func),
            'label': 1, 'attack_type': aid, 'original_text': spam_test['text'].values
        })

        # 单攻击评测文件：攻击垃圾 + 全部正常
        adv_mixed = pd.concat([normal_for_adv, adv_df], ignore_index=True).sample(frac=1, random_state=42)
        save_adv_data(adv_mixed, f'{aid}_{name}')

        # test_full 仅追加攻击垃圾，避免把正常样本重复 9 次
        adv_dfs.append(adv_df)

    # 4. 合并完整测试集
    test_full = pd.concat([normal_test, spam_test] + adv_dfs, ignore_index=True).sample(frac=1, random_state=42)
    test_full.to_csv(os.path.join(DATA_ADV, 'test_full.csv'), index=False)

    # 打印分布统计
    print(f"\n测试集分布统计:")
    print(f"  正常: {len(normal_test)}, 原始垃圾: {len(spam_test)}")
    for aid in sorted(ATTACK_REGISTRY.keys()):
        count = (test_full['attack_type'] == aid).sum()
        print(f"  对抗_{aid}: {count} 条")
    print(f"  总计: {len(test_full)} 条")

if __name__ == '__main__':
    main()