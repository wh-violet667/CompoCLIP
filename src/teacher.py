"""教师特征缓存的读写与按 batch 索引取用（E3 蒸馏，不依赖 open_clip）。"""
from __future__ import annotations

from pathlib import Path

import torch


def save_teacher_cache(path: Path, image: torch.Tensor, text: torch.Tensor, meta: dict) -> None:
    """缓存与训练集逐行对齐：第 i 行对应训练 JSONL 的第 i 条样本。"""
    if image.shape[0] != text.shape[0]:
        raise ValueError(f"图像/文本样本数不一致：{image.shape[0]} vs {text.shape[0]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"image": image, "text": text, "meta": meta}, path)


def load_teacher_cache(path: Path) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} 不存在，先运行 python -m src.distill --config <config>")
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def teacher_similarity(cache: dict, index: torch.Tensor, device) -> torch.Tensor:
    """按 batch 的样本下标取教师特征，返回 (B, B) 相似度矩阵。

    只缓存特征而不是 145k x 145k 的完整相似度矩阵（后者约 84GB，无法落盘）。
    """
    teacher_image = cache["image"][index.cpu()].to(device)
    teacher_text = cache["text"][index.cpu()].to(device)
    return teacher_image @ teacher_text.t()


def check_alignment(cache: dict, expected: int) -> None:
    n = cache["image"].shape[0]
    if n != expected:
        raise ValueError(
            f"教师缓存有 {n} 条样本，训练集有 {expected} 条，无法按行对齐。"
            "请确保 src.distill 与 src.train 使用同一个 train_data 文件。"
        )
