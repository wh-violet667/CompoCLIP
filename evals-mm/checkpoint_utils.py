"""Checkpoint 格式兼容层：让评测端同时支持训练 checkpoint 与老的裸 state_dict。

`src/train.py` 现在保存的是：

    {"model": state_dict, "optimizer": ..., "scaler": ..., "epoch": ..., "config": ...}

而早期脚本/手动导出的 checkpoint 可能是裸 state_dict，也可能带 `module.` 前缀。
LoRA 训练额外导出 `*_merged.pt`，其键名与基座一致，也应能直接加载。
"""
from __future__ import annotations

from pathlib import Path

import torch


def extract_state_dict(payload) -> dict:
    """从 checkpoint payload 中取出模型 state_dict。

    支持：
    - 裸 state_dict：{param_name: tensor, ...}
    - 训练 checkpoint：{"model": {...}, "optimizer": ..., ...}
    - 其他常见包装：{"state_dict": {...}}
    - DataParallel/DDP 的 "module." 前缀
    """
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint 应为 dict，实际为 {type(payload).__name__}")

    state = None
    for key in ("model", "state_dict"):
        value = payload.get(key)
        if isinstance(value, dict):
            state = value
            break
    if state is None:
        # 裸 state_dict：所有值都是 tensor
        state = payload

    if not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError(
            "checkpoint 中没有找到模型 state_dict；"
            "期望裸 state_dict 或包含 'model' / 'state_dict' 的 dict"
        )

    if any(k.startswith("module.") for k in state):
        state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    return state


def load_checkpoint_state(path: str | Path) -> dict:
    """加载 checkpoint 并返回可直接传给 model.load_state_dict 的 dict。"""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return extract_state_dict(payload)
