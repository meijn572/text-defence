# -*- coding: utf-8 -*-
"""Weakly supervised content-type analysis for spam texts.

This script assigns coarse content types to spam messages using keyword rules,
then evaluates available model prediction files by content type when possible.
It is intended for report-level analysis rather than replacing manual labels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data" / "adversarial"
RESULT_DIR = BASE / "results"
RESULT_DIR.mkdir(exist_ok=True)

CONTENT_RULES: dict[str, list[str]] = {
    "金融贷款": ["贷款", "信贷", "额度", "放款", "无抵押", "抵押", "借款", "还款", "信用卡", "银行", "利息", "资金"],
    "中奖福利": ["中奖", "大奖", "奖品", "领取", "恭喜", "iphone", "红包", "抽奖", "礼品", "免费", "优惠券"],
    "证件发票": ["办证", "证件", "发票", "代开", "学历", "文凭", "资格证", "身份证", "驾驶证", "刻章"],
    "营销广告": ["优惠", "促销", "折扣", "开盘", "房", "楼盘", "商铺", "vip", "会员", "咨询", "热线", "活动", "报名"],
    "色情交友": ["美女", "成人", "激情", "裸聊", "约会", "交友", "小姐", "上门", "陪聊", "夜总会"],
    "博彩赌博": ["博彩", "赌博", "棋牌", "下注", "彩票", "中奖率", "娱乐城", "赌球", "六合彩", "开奖"],
    "钓鱼链接": ["http", "https", "www", "登录", "验证", "账户", "账号", "密码", "客户端", "下载", "链接", "url"],
    "其他垃圾": [],
}


def assign_content_type(text: object) -> str:
    value = "" if pd.isna(text) else str(text).lower()
    for content_type, keywords in CONTENT_RULES.items():
        if content_type == "其他垃圾":
            continue
        if any(keyword.lower() in value for keyword in keywords):
            return content_type
    return "其他垃圾"


def load_test_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "test_full.csv")
    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(int)
    df["content_type"] = df.apply(
        lambda row: assign_content_type(row["original_text"] if pd.notna(row.get("original_text")) else row["text"])
        if row["label"] == 1 else "正常短信",
        axis=1,
    )
    return df


def summarize_distribution(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["content_type", "label"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["label", "count"], ascending=[False, False])
    )
    return summary


def summarize_spam_attack_distribution(df: pd.DataFrame) -> pd.DataFrame:
    spam_df = df[df["label"] == 1].copy()
    spam_df["attack_type"] = spam_df["attack_type"].fillna("Original")
    pivot = pd.pivot_table(
        spam_df,
        index="content_type",
        columns="attack_type",
        values="text",
        aggfunc="count",
        fill_value=0,
    )
    pivot["total"] = pivot.sum(axis=1)
    return pivot.sort_values("total", ascending=False)


def load_prediction_files() -> list[tuple[str, pd.DataFrame]]:
    candidates = [
        ("classic_baseline_results", RESULT_DIR / "classic_baseline_results.csv"),
    ]
    loaded: list[tuple[str, pd.DataFrame]] = []
    for name, path in candidates:
        if path.exists():
            loaded.append((name, pd.read_csv(path)))
    return loaded


def build_report_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    spam_df = df[df["label"] == 1]
    for content_type, part in spam_df.groupby("content_type"):
        examples = part["text"].dropna().astype(str).head(3).tolist()
        rows.append({
            "content_type": content_type,
            "n_samples": len(part),
            "ratio_in_spam": len(part) / max(len(spam_df), 1),
            "example_1": examples[0] if len(examples) > 0 else "",
            "example_2": examples[1] if len(examples) > 1 else "",
            "example_3": examples[2] if len(examples) > 2 else "",
        })
    return sorted(rows, key=lambda row: row["n_samples"], reverse=True)


def main() -> None:
    df = load_test_data()
    distribution = summarize_distribution(df)
    attack_distribution = summarize_spam_attack_distribution(df)
    examples = pd.DataFrame(build_report_rows(df))

    distribution_path = RESULT_DIR / "content_type_distribution.csv"
    attack_path = RESULT_DIR / "content_type_attack_distribution.csv"
    examples_path = RESULT_DIR / "content_type_examples.csv"

    distribution.to_csv(distribution_path, index=False, encoding="utf-8-sig")
    attack_distribution.to_csv(attack_path, encoding="utf-8-sig")
    examples.to_csv(examples_path, index=False, encoding="utf-8-sig")

    print("Content type distribution:")
    print(examples[["content_type", "n_samples", "ratio_in_spam"]].to_string(index=False))
    print(f"\nSaved: {distribution_path}")
    print(f"Saved: {attack_path}")
    print(f"Saved: {examples_path}")


if __name__ == "__main__":
    main()
