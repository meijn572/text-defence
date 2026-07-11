# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from utils import load_raw_data, save_adv_data, set_seed, DATA_ADV
from attack import ATTACK_REGISTRY, apply_attack


def build_changed_attack_df(source_texts, attack_id):
    """仅保留适用且实际发生改写的攻击样本。"""
    records = []
    for original_text in source_texts:
        attacked_text = apply_attack(original_text, attack_id)
        if attacked_text != original_text:
            records.append({
                'text': attacked_text,
                'label': 1,
                'attack_type': attack_id,
                'original_text': original_text,
            })
    return pd.DataFrame(records, columns=['text', 'label', 'attack_type', 'original_text'])


def mix_with_matched_normals(normal_df, attack_df, original_spam_count):
    """按原始测试集类别比例抽取正常短信，形成可比的二分类子集。"""
    normal_count = round(len(attack_df) * len(normal_df) / original_spam_count)
    sampled_normals = normal_df.sample(n=normal_count, random_state=42)
    return pd.concat([sampled_normals, attack_df], ignore_index=True).sample(
        frac=1, random_state=42
    )

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

    # 保存不含任何攻击增强的训练集，用于检验未见变体的挑战性。
    clean_train = train_df.copy()
    clean_train['attack_type'] = 'clean'
    clean_train['original_text'] = clean_train['text']
    clean_train = clean_train.sample(frac=1, random_state=42)
    clean_train.to_csv(os.path.join(DATA_ADV, 'train_clean.csv'), index=False)
    print(f"  纯净训练集已保存，共 {len(clean_train)} 条")

    # 2. 训练集增强 (A+B 占垃圾短信的 30%，即每种 15%)
    print(f"\n[训练增强] 目标：A+B 占垃圾短信 30%")
    target_train_count = int(len(spam_train) * 0.15)
    
    train_parts = [normal_train, spam_train]
    for aid in ['A', 'B']:
        adv_candidates = build_changed_attack_df(spam_train['text'].tolist(), aid)
        if adv_candidates.empty:
            raise ValueError(f'训练集没有可用于攻击 {aid} 的垃圾短信')
        adv_df = adv_candidates.sample(
            n=min(target_train_count, len(adv_candidates)), random_state=42
        )
        train_parts.append(adv_df)
        print(f"  A/B 增强 {aid}: 候选改写 {len(adv_candidates)}，采样 {len(adv_df)}")
    
    train_full = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=42)
    train_full.to_csv(os.path.join(DATA_ADV, 'train.csv'), index=False)
    print(f"  训练集已保存，共 {len(train_full)} 条")

    # 3. 测试集生成
    #   - adv_{A..M}.csv: 该攻击垃圾样本 + 全部正常样本（便于单文件二分类评测）
    #   - test_full.csv: 仍保持 原始正常 + 原始垃圾 + 各攻击垃圾，避免重复叠加正常样本
    print(f"\n[测试增强] 仅对适用且实际改写的垃圾短信生成对抗样本")
    print("[测试增强] 每个 adv_{A..M} 文件按原始测试集比例混入正常样本")

    adv_dfs = []
    normal_for_adv = pd.DataFrame({
        'text': normal_test['text'].values,
        'label': normal_test['label'].values,
        'attack_type': 'normal',
        'original_text': normal_test['text'].values,
    })

    for aid, (name, func, desc) in ATTACK_REGISTRY.items():
        adv_df = build_changed_attack_df(spam_test['text'].tolist(), aid)
        if adv_df.empty:
            raise ValueError(f'测试集没有可用于攻击 {aid} 的垃圾短信')

        # 单攻击评测文件：攻击垃圾 + 全部正常
        adv_mixed = mix_with_matched_normals(normal_for_adv, adv_df, len(spam_test))
        save_adv_data(adv_mixed, f'{aid}_{name}')

        # test_full 仅追加攻击垃圾，避免把正常样本重复多次
        adv_dfs.append(adv_df)

    # 4. 合并完整测试集
    test_full = pd.concat([normal_test, spam_test] + adv_dfs, ignore_index=True).sample(frac=1, random_state=42)
    test_full.to_csv(os.path.join(DATA_ADV, 'test_full.csv'), index=False)

    # 打印分布统计
    print(f"\n测试集分布统计:")
    print(f"  正常: {len(normal_test)}, 原始垃圾: {len(spam_test)}")
    for aid in sorted(ATTACK_REGISTRY.keys()):
        count = (test_full['attack_type'] == aid).sum()
        print(f"  对抗_{aid}: {count} 条实际改写垃圾")
    print(f"  总计: {len(test_full)} 条")

if __name__ == '__main__':
    main()