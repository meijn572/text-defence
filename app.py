# -*- coding: utf-8 -*-
"""Gradio 前端展示 —— 中文垃圾短信对抗检测系统。

功能：
- 自动检测 CPU/GPU 并加载已有模型
- 单条短信多模型预测
- 对抗攻击生成与攻击前后预测对比
- 展示已有实验结果 CSV
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
import torch

BASE = Path(__file__).resolve().parent
os.chdir(BASE)
sys.path.insert(0, str(BASE))

from attack import ATTACK_REGISTRY, apply_attack  # noqa: E402
from defense.fusion_model import FusionClassifier  # noqa: E402
from defense.text_channel import BertClassifier  # noqa: E402

PROCESSED = BASE / "data" / "processed"
RESULTS = BASE / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
TORCH_INFO = f"PyTorch {torch.__version__} | CUDA: {torch.version.cuda or 'N/A'}"

MODEL_LOCK = threading.Lock()
BERT_MODEL: BertClassifier | None = None
FUSION_MODEL: FusionClassifier | None = None
LOAD_MESSAGE = "模型尚未加载"

EXAMPLES = [
    "明天下午三点开会请准时参加",
    "恭喜您获得iPhone大奖点击领取",
    "加微信领取免费提现额度",
    "代辦證件质量保证快速出證",
    "佳薇芯加我好友带你赚大前",
]

ATTACK_CHOICES = [
    (f"{attack_id}. {meta[2]}", attack_id)
    for attack_id, meta in ATTACK_REGISTRY.items()
]

RESULT_FILES = {
    "深度模型基础评测 eval_results.csv": RESULTS / "eval_results.csv",
    "强攻击评测 strong_attack_results.csv": RESULTS / "strong_attack_results.csv",
    "经典基准明细 classic_baseline_results.csv": RESULTS / "classic_baseline_results.csv",
    "经典基准 F1 透视表 classic_baseline_f1_pivot.csv": RESULTS / "classic_baseline_f1_pivot.csv",
    "内容类型分布 content_type_distribution.csv": RESULTS / "content_type_distribution.csv",
    "内容类型-攻击分布 content_type_attack_distribution.csv": RESULTS / "content_type_attack_distribution.csv",
    "内容类型示例 content_type_examples.csv": RESULTS / "content_type_examples.csv",
    "内容类型模型召回 content_type_model_recall.csv": RESULTS / "content_type_model_recall.csv",
    "内容类型模型逐条预测 content_type_model_predictions.csv": RESULTS / "content_type_model_predictions.csv",
}


def _missing_model_files() -> list[str]:
    required = ["baseline_bert.pth", "fusion_model.pth"]
    return [name for name in required if not (PROCESSED / name).exists()]


def load_models() -> str:
    """Lazy load BERT and Fusion models once."""
    global BERT_MODEL, FUSION_MODEL, LOAD_MESSAGE

    with MODEL_LOCK:
        if BERT_MODEL is not None and FUSION_MODEL is not None:
            return LOAD_MESSAGE

        missing = _missing_model_files()
        if missing:
            LOAD_MESSAGE = (
                "模型文件缺失：" + ", ".join(missing) +
                "。请先将训练好的 .pth 文件放入 data/processed/。"
            )
            return LOAD_MESSAGE

        try:
            bert = BertClassifier().to(DEVICE)
            bert.load_state_dict(torch.load(PROCESSED / "baseline_bert.pth", map_location=DEVICE))
            bert.eval()

            fusion = FusionClassifier(freeze_channels=False, device=DEVICE).to(DEVICE)
            checkpoint = torch.load(PROCESSED / "fusion_model.pth", map_location=DEVICE)
            state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
            fusion.load_state_dict(state_dict)
            fusion.eval()

            BERT_MODEL = bert
            FUSION_MODEL = fusion
            LOAD_MESSAGE = f"模型加载成功。当前设备：{DEVICE_NAME} ({DEVICE})"
        except Exception as exc:  # noqa: BLE001
            LOAD_MESSAGE = f"模型加载失败：{type(exc).__name__}: {exc}"

        return LOAD_MESSAGE


def _ensure_models() -> tuple[BertClassifier, FusionClassifier]:
    message = load_models()
    if BERT_MODEL is None or FUSION_MODEL is None:
        raise RuntimeError(message)
    return BERT_MODEL, FUSION_MODEL


def _prob_row(model_name: str, text: str, normal_prob: float, spam_prob: float) -> dict[str, Any]:
    label = "垃圾短信" if spam_prob >= normal_prob else "正常短信"
    return {
        "模型": model_name,
        "文本": text,
        "正常概率": round(normal_prob, 4),
        "垃圾概率": round(spam_prob, 4),
        "预测类别": label,
    }


@torch.inference_mode()
def predict_text(text: str, use_bert: bool = True, use_fusion: bool = True) -> tuple[pd.DataFrame, str]:
    text = (text or "").strip()
    if not text:
        return pd.DataFrame(), "请输入短信文本。"
    if not use_bert and not use_fusion:
        return pd.DataFrame(), "请至少选择一个模型。"

    try:
        bert, fusion = _ensure_models()
        rows: list[dict[str, Any]] = []

        if use_bert:
            probs = torch.softmax(bert([text]), dim=1)[0].detach().cpu().tolist()
            rows.append(_prob_row("BERT", text, probs[0], probs[1]))

        if use_fusion:
            probs = torch.softmax(fusion([text]), dim=1)[0].detach().cpu().tolist()
            rows.append(_prob_row("四通道融合", text, probs[0], probs[1]))

        return pd.DataFrame(rows), "预测完成。"
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame(), f"预测失败：{type(exc).__name__}: {exc}"


@torch.inference_mode()
def attack_and_predict(text: str, attack_id: str, use_bert: bool = True, use_fusion: bool = True) -> tuple[str, pd.DataFrame, str]:
    text = (text or "").strip()
    if not text:
        return "", pd.DataFrame(), "请输入原始短信文本。"

    try:
        attacked_text = apply_attack(text, attack_id)
        bert, fusion = _ensure_models()
        rows: list[dict[str, Any]] = []

        for stage, sample in [("原始文本", text), ("攻击后文本", attacked_text)]:
            if use_bert:
                probs = torch.softmax(bert([sample]), dim=1)[0].detach().cpu().tolist()
                row = _prob_row("BERT", sample, probs[0], probs[1])
                row["阶段"] = stage
                rows.append(row)
            if use_fusion:
                probs = torch.softmax(fusion([sample]), dim=1)[0].detach().cpu().tolist()
                row = _prob_row("四通道融合", sample, probs[0], probs[1])
                row["阶段"] = stage
                rows.append(row)

        columns = ["阶段", "模型", "文本", "正常概率", "垃圾概率", "预测类别"]
        return attacked_text, pd.DataFrame(rows)[columns], "攻击生成与预测完成。"
    except Exception as exc:  # noqa: BLE001
        return "", pd.DataFrame(), f"攻击或预测失败：{type(exc).__name__}: {exc}"


def load_result_table(result_name: str) -> tuple[pd.DataFrame, str]:
    path = RESULT_FILES.get(result_name)
    if path is None:
        return pd.DataFrame(), "未知结果文件。"
    if not path.exists():
        return pd.DataFrame(), f"结果文件不存在：{path.relative_to(BASE)}"
    try:
        df = pd.read_csv(path)
        return df, f"已加载：{path.relative_to(BASE)}，共 {len(df)} 行。"
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame(), f"读取失败：{type(exc).__name__}: {exc}"


def chat_detect(message: str, history: list[dict[str, str]] | None) -> tuple[str, list[dict[str, str]]]:
    history = history or []
    message = (message or "").strip()
    if not message:
        return "", history

    table, status = predict_text(message, use_bert=True, use_fusion=True)
    if table.empty:
        reply = f"检测失败：{status}"
    else:
        lines = ["检测结果："]
        for _, row in table.iterrows():
            lines.append(
                f"{row['模型']}：{row['预测类别']} "
                f"(正常={row['正常概率']:.4f}, 垃圾={row['垃圾概率']:.4f})"
            )
        reply = "\n".join(lines)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return "", history


def chat_attack(message: str, attack_id: str, history: list[dict[str, str]] | None) -> tuple[str, list[dict[str, str]]]:
    history = history or []
    message = (message or "").strip()
    if not message:
        return "", history

    attacked_text, table, status = attack_and_predict(message, attack_id, use_bert=True, use_fusion=True)
    if table.empty:
        reply = f"攻击演示失败：{status}"
    else:
        attack_name = ATTACK_REGISTRY[attack_id][2]
        lines = [f"已执行 {attack_id}. {attack_name}", f"攻击后文本：{attacked_text}", "", "攻击前后检测："]
        for _, row in table.iterrows():
            lines.append(
                f"{row['阶段']} / {row['模型']}：{row['预测类别']} "
                f"(正常={row['正常概率']:.4f}, 垃圾={row['垃圾概率']:.4f})"
            )
        reply = "\n".join(lines)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return "", history


def model_status() -> str:
    missing = _missing_model_files()
    model_file_status = "完整" if not missing else "缺失：" + ", ".join(missing)
    return (
        f"运行设备：**{DEVICE_NAME}**  \n"
        f"设备类型：**{DEVICE}**  \n"
        f"{TORCH_INFO}  \n"
        f"模型文件：**{model_file_status}**  \n"
        f"加载状态：**{LOAD_MESSAGE}**"
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="中文垃圾短信对抗检测系统") as demo:
        gr.Markdown("# 中文垃圾短信对抗检测系统")
        gr.Markdown(
            "基于 BERT 与四通道融合模型的中文垃圾短信检测、对抗攻击演示与实验结果展示。"
        )

        with gr.Tab("系统状态"):
            status_md = gr.Markdown(model_status())
            load_btn = gr.Button("加载/刷新模型状态", variant="primary")
            load_output = gr.Textbox(label="加载信息", interactive=False)
            load_btn.click(load_models, outputs=load_output).then(model_status, outputs=status_md)

        with gr.Tab("短信检测"):
            text_input = gr.Textbox(label="输入短信", lines=4, value=EXAMPLES[0])
            gr.Examples(EXAMPLES, inputs=text_input)
            with gr.Row():
                use_bert = gr.Checkbox(label="BERT", value=True)
                use_fusion = gr.Checkbox(label="四通道融合", value=True)
            predict_btn = gr.Button("开始检测", variant="primary")
            predict_table = gr.Dataframe(label="预测结果", wrap=True)
            predict_msg = gr.Textbox(label="状态", interactive=False)
            predict_btn.click(
                predict_text,
                inputs=[text_input, use_bert, use_fusion],
                outputs=[predict_table, predict_msg],
            )

        with gr.Tab("微信式演示"):
            gr.Markdown("模拟聊天检测助手：输入一条短信，系统以聊天气泡形式返回检测结果。")
            chatbot = gr.Chatbot(label="垃圾短信检测助手", height=420)
            chat_input = gr.Textbox(label="输入短信", placeholder="例如：加微信领取免费提现额度", lines=2)
            with gr.Row():
                chat_detect_btn = gr.Button("发送并检测", variant="primary")
                chat_attack_dropdown = gr.Dropdown(label="攻击方式", choices=ATTACK_CHOICES, value="F")
                chat_attack_btn = gr.Button("攻击后检测")
                chat_clear_btn = gr.Button("清空聊天")
            chat_detect_btn.click(chat_detect, inputs=[chat_input, chatbot], outputs=[chat_input, chatbot])
            chat_attack_btn.click(chat_attack, inputs=[chat_input, chat_attack_dropdown, chatbot], outputs=[chat_input, chatbot])
            chat_clear_btn.click(lambda: [], outputs=chatbot)

        with gr.Tab("对抗攻击演示"):
            attack_text = gr.Textbox(label="原始短信", lines=4, value="加微信领取免费提现额度")
            attack_dropdown = gr.Dropdown(
                label="攻击方式",
                choices=ATTACK_CHOICES,
                value="F",
            )
            with gr.Row():
                attack_use_bert = gr.Checkbox(label="BERT", value=True)
                attack_use_fusion = gr.Checkbox(label="四通道融合", value=True)
            attack_btn = gr.Button("生成攻击并检测", variant="primary")
            attacked_output = gr.Textbox(label="攻击后文本", lines=4, interactive=False)
            attack_table = gr.Dataframe(label="攻击前后预测对比", wrap=True)
            attack_msg = gr.Textbox(label="状态", interactive=False)
            attack_btn.click(
                attack_and_predict,
                inputs=[attack_text, attack_dropdown, attack_use_bert, attack_use_fusion],
                outputs=[attacked_output, attack_table, attack_msg],
            )

        with gr.Tab("实验结果"):
            result_dropdown = gr.Dropdown(
                label="选择结果文件",
                choices=list(RESULT_FILES.keys()),
                value="经典基准 F1 透视表 classic_baseline_f1_pivot.csv",
            )
            result_btn = gr.Button("加载结果", variant="primary")
            result_table = gr.Dataframe(label="结果表", wrap=True)
            result_msg = gr.Textbox(label="状态", interactive=False)
            result_btn.click(load_result_table, inputs=result_dropdown, outputs=[result_table, result_msg])

        with gr.Tab("项目说明"):
            gr.Markdown(
                """
## 功能说明

- **短信检测**：输入单条短信，展示 BERT 与四通道融合模型的正常/垃圾概率。
- **对抗攻击演示**：选择攻击方式，生成攻击文本，并比较攻击前后模型预测变化。
- **微信式演示**：用聊天气泡模拟短信检测助手，支持直接检测和攻击后检测。
- **实验结果**：直接查看 `results/` 目录中的深度模型评测、经典基准算法结果和内容类型分析。

## 设备兼容

前端自动检测运行设备：若 `torch.cuda.is_available()` 为 `True`，自动使用 GPU；否则使用 CPU。

## 注意事项

- 首次加载模型需要下载或读取 `bert-base-chinese`、ResNet18 和 `.pth` 权重，可能需要等待。
- 如 HuggingFace 下载失败，可先在 PowerShell 设置：`$env:HF_ENDPOINT="https://hf-mirror.com"`。
- 前端适合单条推理与结果展示，不建议在页面中执行完整训练流程。
"""
            )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue(default_concurrency_limit=1)
    app.launch(server_name="127.0.0.1", server_port=7860)
