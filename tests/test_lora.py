"""LoRA 注入、合并与导出的单测（只依赖 torch，不依赖 open_clip）。"""
import pytest
import torch
from torch import nn

from src.lora import (
    LoRALinear,
    LoRAMultiheadAttention,
    apply_lora,
    count_trainable,
    merge_lora,
    merged_state_dict,
)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.mlp = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))

    def forward(self, x):
        return self.mlp(self.proj(x))


def test_zero_init_keeps_output_identical():
    torch.manual_seed(0)
    model = Tiny()
    x = torch.randn(3, 4)
    expected = model(x)
    apply_lora(model, r=2, targets=("proj",))
    assert torch.allclose(model(x), expected, atol=1e-6)


def test_base_frozen_only_adapters_trainable():
    model = Tiny()
    apply_lora(model, r=2, targets=("proj",))
    trainable, total = count_trainable(model)
    assert 0 < trainable < total
    assert not model.proj.base.weight.requires_grad
    assert model.proj.adapters[0].A.requires_grad


def test_merge_restores_plain_output():
    torch.manual_seed(0)
    model = Tiny()
    x = torch.randn(3, 4)
    apply_lora(model, r=2, targets=("proj",))
    with torch.no_grad():                       # 模拟训练后的非零增量
        model.proj.adapters[0].B.normal_(0, 0.1)
    wrapped_out = model(x)
    assert isinstance(model.proj, LoRALinear)

    assert merge_lora(model) == 1
    assert isinstance(model.proj, nn.Linear)
    assert torch.allclose(model(x), wrapped_out, atol=1e-6)


def test_qv_row_groups_only_touch_q_and_v():
    base = nn.Linear(12, 12)                    # 融合 qkv：3 x 4 行
    lora = LoRALinear(base, r=2, row_groups=[(0, 4), (8, 12)])
    with torch.no_grad():
        lora.adapters[0].B.fill_(1.0)
        lora.adapters[1].B.fill_(1.0)
    delta = lora.delta_weight()
    assert delta[0:4].abs().sum() > 0          # q 段被改动
    assert delta[4:8].abs().sum() == 0         # k 段保持 0
    assert delta[8:12].abs().sum() > 0         # v 段被改动


def test_multihead_attention_identity_and_merge():
    torch.manual_seed(0)
    base = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=False)
    base.eval()
    x = torch.randn(5, 2, 8)
    with torch.no_grad():
        expected, _ = base(x, x, x, need_weights=False)

    wrapped = LoRAMultiheadAttention(base, r=2)
    wrapped.eval()
    with torch.no_grad():
        got, _ = wrapped(x, x, x, need_weights=False)
    assert torch.allclose(got, expected, atol=1e-6)

    with torch.no_grad():
        wrapped.adapters[0].B.normal_(0, 0.1)
        trained, _ = wrapped(x, x, x, need_weights=False)
        wrapped.merge_()
        merged, _ = base(x, x, x, need_weights=False)
    assert not torch.allclose(trained, expected, atol=1e-6)
    assert torch.allclose(trained, merged, atol=1e-6)


def test_merged_state_dict_matches_base_keys():
    torch.manual_seed(0)
    model = Tiny()
    apply_lora(model, r=2, targets=("proj",))
    with torch.no_grad():
        model.proj.adapters[0].B.normal_(0, 0.1)
    wrapped_out = model(torch.ones(1, 4))

    merged = merged_state_dict(model)
    plain = Tiny()
    assert set(merged) == set(plain.state_dict()), "合并后键名必须与基座一致"
    plain.load_state_dict(merged)
    assert torch.allclose(plain(torch.ones(1, 4)), wrapped_out, atol=1e-6)


def test_no_target_raises_with_hint():
    model = nn.Sequential(nn.Linear(2, 2))
    with pytest.raises(RuntimeError, match="targets"):
        apply_lora(model, targets=("nonexistent",))
