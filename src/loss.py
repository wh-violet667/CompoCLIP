"""硬负例加权对比损失 + 教师蒸馏 KL 正则（创新点 2 核心，见 docs/dev-doc.md 3.3 节）。

L_i2t = -log( exp(s_ii/t) / ( sum_j exp(s_ij/t) + w_hn * sum_{k in HN(i)} exp(s_ik/t) ) )
L     = L_hardneg + lambda * KL( P_student || P_teacher )

framework 无关的纯 PyTorch 实现，输入为已 L2 归一化的特征。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class HardNegClipLoss(nn.Module):
    def __init__(self, hard_negative_weight: float = 1.0, distill_lambda: float = 0.0):
        super().__init__()
        self.w_hn = hard_negative_weight
        self.distill_lambda = distill_lambda

    def forward(
        self,
        image_features: Tensor,          # (B, d) 已归一化
        text_features: Tensor,           # (B, d) 已归一化，对角线为正样本
        logit_scale: Tensor,             # 标量（已取 exp）
        hard_neg_features: Tensor | None = None,  # (B, K, d) 每条正样本的 K 个硬负例
        teacher_sim: Tensor | None = None,        # (B, B) 教师相似度矩阵（蒸馏用）
    ) -> dict[str, Tensor]:
        scale = logit_scale.clamp(max=4.6052)  # ln(100)，防 fp16 下发散
        logits = scale * image_features @ text_features.t()  # (B, B)

        if hard_neg_features is not None:
            # (B, K)：每个图像与其硬负例文本的相似度
            hn_logits = scale * torch.einsum("bd,bkd->bk", image_features, hard_neg_features)
            # 拼接进分母，权重 w_hn 以对数偏移实现
            hn_logits = hn_logits + torch.log(
                torch.tensor(self.w_hn, device=hn_logits.device, dtype=hn_logits.dtype)
            )
            denom = torch.cat([logits, hn_logits], dim=1)
        else:
            denom = logits

        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i2t = F.cross_entropy(denom.log_softmax(dim=1), labels)
        loss_t2i = F.cross_entropy(logits.t().log_softmax(dim=1), labels)
        loss = (loss_i2t + loss_t2i) / 2

        out = {"loss": loss, "contrastive_loss": loss.detach()}

        if teacher_sim is not None and self.distill_lambda > 0:
            # 学生与教师在同一批样本上的分布对齐
            p_student = F.log_softmax(logits, dim=1)
            p_teacher = F.softmax(scale * teacher_sim, dim=1)
            kd = F.kl_div(p_student, p_teacher, reduction="batchmean")
            out["distill_loss"] = kd.detach()
            out["loss"] = loss + self.distill_lambda * kd

        return out
