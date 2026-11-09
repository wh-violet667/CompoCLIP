"""手写 LoRA：适配 open_clip 的真实模块结构，并在导出时合并回基座权重。

为什么不用 peft：
- open_clip 的视觉注意力是融合的 `qkv` Linear（不是 q_proj/v_proj），文本塔是
  `nn.MultiheadAttention`（融合 `in_proj_weight`），peft 的两者都识别不了。
- 训练完直接 merge 成与基座**完全相同的键名**，评测端不需要任何 LoRA 相关代码。

适配位置：
- `nn.Linear` 且名字属于 targets（默认 qkv / out_proj / c_fc / c_proj）
- `nn.MultiheadAttention`：只对 q、v 两段行加低秩增量

注意：B 初始化为 0，因此训练起步时输出与基座逐位一致（单测覆盖）。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

DEFAULT_TARGETS = ("qkv", "out_proj", "c_fc", "c_proj")


class _Adapter(nn.Module):
    """单个低秩增量：delta = B @ A，B 初始为 0。"""

    def __init__(self, r: int, d_in: int, d_out: int):
        super().__init__()
        self.A = nn.Parameter(torch.empty(r, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def delta(self) -> Tensor:
        return self.B @ self.A


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16,
                 row_groups: list[tuple[int, int]] | None = None):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad = False
        d_out, d_in = base.weight.shape
        self.row_groups = row_groups or [(0, d_out)]
        self.scaling = alpha / r
        self.adapters = nn.ModuleList([
            _Adapter(r, d_in, end - start) for start, end in self.row_groups
        ])

    def delta_weight(self) -> Tensor:
        delta = torch.zeros_like(self.base.weight)
        for (start, end), adapter in zip(self.row_groups, self.adapters):
            delta[start:end] = adapter.delta()
        return delta * self.scaling

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.base.weight + self.delta_weight(), self.base.bias)

    @torch.no_grad()
    def merged_weight(self) -> Tensor:
        return self.base.weight + self.delta_weight()

    @torch.no_grad()
    def merge_(self) -> nn.Linear:
        """把增量写回基座权重，返回可直接替换的普通 Linear。"""
        self.base.weight.add_(self.delta_weight())
        return self.base


class LoRAMultiheadAttention(nn.Module):
    """包装 nn.MultiheadAttention，只给 q、v 对应的行加低秩增量。"""

    def __init__(self, base: nn.MultiheadAttention, r: int = 8, alpha: int = 16,
                 adapt_qv_only: bool = True):
        super().__init__()
        if base.in_proj_weight is None:
            raise ValueError("只支持使用融合 in_proj_weight 的 MultiheadAttention")
        if base.bias_k is not None or base.bias_v is not None:
            raise NotImplementedError("不支持 add_bias_kv=True 的注意力，请改用其它层做 LoRA")
        self.base = base
        for p in base.parameters():
            p.requires_grad = False
        d = base.embed_dim
        self.row_groups = [(0, d), (2 * d, 3 * d)] if adapt_qv_only else [(0, 3 * d)]
        self.scaling = alpha / r
        self.adapters = nn.ModuleList([
            _Adapter(r, d, end - start) for start, end in self.row_groups
        ])

    def delta_weight(self) -> Tensor:
        delta = torch.zeros_like(self.base.in_proj_weight)
        for (start, end), adapter in zip(self.row_groups, self.adapters):
            delta[start:end] = adapter.delta()
        return delta * self.scaling

    def forward(self, query, key=None, value=None, key_padding_mask=None,
                need_weights=True, attn_mask=None, average_attn_weights=True,
                is_causal=False):
        base = self.base
        key = query if key is None else key
        value = key if value is None else value
        if base.batch_first:
            query, key, value = (x.transpose(0, 1) for x in (query, key, value))

        out, attn = F.multi_head_attention_forward(
            query, key, value,
            embed_dim_to_check=base.embed_dim,
            num_heads=base.num_heads,
            in_proj_weight=base.in_proj_weight + self.delta_weight(),
            in_proj_bias=base.in_proj_bias,
            bias_k=None,
            bias_v=None,
            add_zero_attn=base.add_zero_attn,
            dropout_p=base.dropout,
            out_proj_weight=base.out_proj.weight,
            out_proj_bias=base.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        if base.batch_first:
            out = out.transpose(0, 1)
        return out, attn

    @torch.no_grad()
    def merge_(self) -> nn.MultiheadAttention:
        self.base.in_proj_weight.add_(self.delta_weight())
        return self.base


LORA_TYPES = (LoRALinear, LoRAMultiheadAttention)


def apply_lora(model: nn.Module, r: int = 8, alpha: int = 16,
               targets: tuple[str, ...] = DEFAULT_TARGETS,
               adapt_qv_only: bool = True) -> list[str]:
    """原地注入 LoRA，返回被包装的模块名列表。找不到任何目标时直接报错。"""
    replaced: list[str] = []
    parents = {name: mod for name, mod in model.named_modules()}

    for name, module in list(parents.items()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.MultiheadAttention):
                setattr(module, child_name,
                        LoRAMultiheadAttention(child, r, alpha, adapt_qv_only))
                replaced.append(full)
            elif isinstance(child, nn.Linear) and child_name in targets:
                row_groups = None
                d_out = child.weight.shape[0]
                if child_name == "qkv" and adapt_qv_only and d_out % 3 == 0:
                    d = d_out // 3
                    row_groups = [(0, d), (2 * d, 3 * d)]     # 只适配 q 与 v
                setattr(module, child_name, LoRALinear(child, r, alpha, row_groups))
                replaced.append(full)

    if not replaced:
        sample = ", ".join(sorted({n.split(".")[-1] for n, _ in parents.items()})[:15])
        raise RuntimeError(
            "没有找到可注入 LoRA 的模块；当前模型末级模块名示例："
            f"{sample}。请用 targets 显式指定。"
        )
    return replaced


@torch.no_grad()
def merge_lora(model: nn.Module) -> int:
    """把所有权量增量合并回基座并替换为普通模块（键名恢复成基座形式）。"""
    merged = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LORA_TYPES):
                setattr(module, child_name, child.merge_())
                merged += 1
    return merged


@torch.no_grad()
def merged_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """生成与基座模型键名一致的 state_dict（LoRA 增量已合并，适配器参数丢弃）。

    不复制模型、不占额外显存（只 clone 参数张量本身）。
    """
    raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
    lora_modules = {name: m for name, m in model.named_modules() if isinstance(m, LORA_TYPES)}
    lora_names = set(lora_modules)

    out: dict[str, Tensor] = {}
    for key, tensor in raw.items():
        for name in lora_names:
            if key.startswith(name + "."):
                rest = key[len(name) + 1:]
                if rest.startswith("base."):
                    out[name + "." + rest[len("base."):]] = tensor
                break  # 其余是适配器参数，丢弃
        else:
            out[key] = tensor

    for name, module in lora_modules.items():
        weight_key = (f"{name}.weight" if isinstance(module, LoRALinear)
                      else f"{name}.in_proj_weight")
        if weight_key not in out:
            raise KeyError(f"合并后找不到 {weight_key}，请检查 LoRA 包装层级")
        out[weight_key] = out[weight_key] + module.delta_weight().cpu()
    return out


def count_trainable(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
