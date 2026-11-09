"""纯 torch 的调度器工具（不依赖 open_clip，便于单测）。"""
from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine_lambda(step: int, warmup: int, total: int) -> float:
    """warmup 线性升到 1，随后余弦退到 0。step 0 起。"""
    if total <= 0:
        return 1.0
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_scheduler(optimizer: Optimizer, warmup: int, total_steps: int, name: str = "cosine"):
    if name in (None, "", "none"):
        return None
    if name != "cosine":
        raise ValueError(f"不支持的 lr_scheduler: {name}（当前只实现 cosine）")
    return LambdaLR(optimizer, lambda s: warmup_cosine_lambda(s, warmup, total_steps))
