"""checkpoint 格式兼容层单测：不依赖 open_clip / GPU / 数据集。"""
from __future__ import annotations

import torch

from checkpoint_utils import extract_state_dict, load_checkpoint_state


def _toy_state() -> dict[str, torch.Tensor]:
    return {"weight": torch.ones(2, 2), "logit_scale": torch.tensor(4.6052)}


def test_raw_state_dict_passthrough():
    state = _toy_state()
    assert extract_state_dict(state) is state


def test_train_checkpoint_extracts_model():
    state = _toy_state()
    payload = {"model": state, "optimizer": {}, "epoch": 3, "config": {"model": "ViT-B-32"}}
    assert extract_state_dict(payload) is state


def test_state_dict_wrapper():
    state = _toy_state()
    assert extract_state_dict({"state_dict": state}) is state


def test_strips_dataparallel_prefix():
    state = _toy_state()
    wrapped = {f"module.{k}": v for k, v in state.items()}
    out = extract_state_dict({"model": wrapped})
    assert set(out) == set(state)
    assert torch.equal(out["weight"], state["weight"])


def test_rejects_payload_without_state_dict():
    try:
        extract_state_dict({"optimizer": {"lr": 1e-5}, "epoch": 0})
    except ValueError as exc:
        assert "state_dict" in str(exc)
    else:
        raise AssertionError("没有模型权重时应报错")


def test_load_checkpoint_state_roundtrip(tmp_path):
    state = _toy_state()
    path = tmp_path / "epoch0.pt"
    torch.save({"model": state, "epoch": 0, "config": {"strategy": "full"}}, path)
    loaded = load_checkpoint_state(path)
    assert set(loaded) == set(state)
    assert torch.equal(loaded["weight"], state["weight"])
