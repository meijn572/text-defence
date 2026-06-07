# -*- coding: utf-8 -*-
"""
一键运行精简实验 —— 使用抽样数据集，顺序运行 01→02→03→04
完成后自动恢复原始数据
"""
import os, sys, shutil, subprocess, time

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
RAW_DIR = os.path.join(BASE, 'data', 'raw')
DATA_MAIN = os.path.join(RAW_DIR, 'spam_data.csv')
DATA_FULL = os.path.join(RAW_DIR, 'spam_data_full.csv')
DATA_SMALL = os.path.join(RAW_DIR, 'spam_data_2k.csv')


def run_step(script, desc):
    print(f"\n{'='*55}")
    print(f"  {desc}")
    print(f"{'='*55}")
    t0 = time.time()
    r = subprocess.run([sys.executable, script], cwd=BASE)
    elapsed = time.time() - t0
    if r.returncode != 0:
        print(f"\n[FAIL] {desc} (exit={r.returncode}, {elapsed:.0f}s)")
        return False
    print(f"[OK] {desc} ({elapsed:.0f}s)")
    return True


def main():
    # ---- 0. 准备精简数据集 ----
    print("=" * 55)
    print("  精简数据集实验")
    print("=" * 55)

    # 如果没有全量备份，先备份
    if not os.path.exists(DATA_FULL) and os.path.exists(DATA_MAIN):
        shutil.copy(DATA_MAIN, DATA_FULL)
        print("[备份] spam_data.csv -> spam_data_full.csv")

    # 如果没有精简数据，先生成
    if not os.path.exists(DATA_SMALL):
        print("[生成] 创建 2000 条精简数据集...")
        subprocess.run([sys.executable, 'sample_data.py', '-n', '2000'], cwd=BASE)

    if not os.path.exists(DATA_SMALL):
        print("[ERROR] 精简数据集生成失败")
        return

    # 替换为精简数据
    shutil.copy(DATA_SMALL, DATA_MAIN)
    import pandas as pd
    df = pd.read_csv(DATA_MAIN)
    print(f"[数据] 当前使用: {len(df)} 条 (正常:{sum(df.label==0)}, 垃圾:{sum(df.label==1)})")

    # ---- 运行实验 ----
    ok = True
    ok = run_step('experiments/01_generate_adv.py', '实验01: 生成对抗样本') and ok
    ok = run_step('experiments/02_train_baseline.py', '实验02: 训练基线BERT') and ok
    ok = run_step('experiments/03_train_fusion.py', '实验03: 训练融合模型(CPU)') and ok
    ok = run_step('experiments/04_evaluate.py', '实验04: 评测+可视化') and ok

    # ---- 恢复全量数据 ----
    if os.path.exists(DATA_FULL):
        shutil.copy(DATA_FULL, DATA_MAIN)
        print(f"\n[恢复] 已恢复全量数据")

    # ---- 总结 ----
    print(f"\n{'='*55}")
    print(f"  {'✓ 全部完成!' if ok else '✗ 有步骤失败'}")
    print(f"  评测结果: results/eval_results.csv")
    print(f"  可视化:   results/figures/")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()
