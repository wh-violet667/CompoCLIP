"""硬负例加权对比损失 + 教师蒸馏 KL 正则（创新点 2 核心，见 docs/dev-doc.md 3.3 节）。

对比项（双向 InfoNCE）：
    L_i2t = -log( exp(s_ii/t) / ( sum_j exp(s_ij/t) + w_hn * sum_{k in HN(i)} exp(s_ik/t) ) )
    L_t2i = -log( exp(s_ii/t) / sum_j exp(s_ji/t) )          # 标准对称项
    L     = (L_i2t + L_t2i)/2

关于「硬负例只进 i2t」：硬负例是由正 caption 改写而来，它没有对应的真实图像，
因此只能作为「图 -> 文本」方向的负样本。若强行把它当作 t2i 的查询行，等于教模型
把错误语序的 caption 匹配到原图，与目标相反。真正的双向对称需要「硬负例图像」
（反事实图像生成），不在本项目范围内，报告里需如实说明这一点。

蒸馏项（标准 KD 写法，方向为 KL(P_teacher || P_student)）：
    p_t = softmax(teacher_sim / T_kd)
    p_s = log_softmax(student_logits / T_kd)
    L   = L_hardneg + lambda * T_kd^2 * KL(p_t || p_s)
注意 F.kl_div(log_p_s, p_t) 计算的正是 KL(p_t || p_s)。

输入特征需已 L2 归一化。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

SCALE_MAX = 100.0  # 1/tau 的上限；fp16 下 logit 过大会溢出


class HardNegClipLoss(nn.Module):
    def __init__(
        self,
        hard_negative_weight: float = 1.0,
        distill_lambda: float = 0.0,
        kd_temperature: float = 1.0,
    ):
        super().__init__()
        self.w_hn = hard_negative_weight
        self.distill_lambda = distill_lambda
        self.kd_temperature = kd_temperature

    def forward(
        self,
        image_features: Tensor,                    # (B, d) 已归一化
        text_features: Tensor,                     # (B, d) 已归一化，对角线为正样本
        logit_scale: Tensor,                       # 标量（已取 exp）
        hard_neg_features: Tensor | None = None,   # (B, K, d) 每条正样本的 K 个硬负例
        hard_neg_mask: Tensor | None = None,       # (B, K) bool，False 表示该位置无效
        teacher_sim: Tensor | None = None,         # (B, B) 教师相似度（不含学生温度缩放）
    ) -> dict[str, Tensor]:
        scale = logit_scale.clamp(max=SCALE_MAX)
        logits = scale * image_features @ text_features.t()  # (B, B)

        loss_i2t_denom = logits
        if hard_neg_features is not None:
            # (B, K)：每个图像与其硬负例文本的相似度
            hn_logits = scale * torch.einsum("bd,bkd->bk", image_features, hard_neg_features)
            if hard_neg_mask is None:
                hard_neg_mask = torch.ones_like(hn_logits, dtype=torch.bool)
            else:
                hard_neg_mask = hard_neg_mask.to(torch.bool)
            hn_logits = hn_logits + torch.log(
                torch.tensor(self.w_hn, device=hn_logits.device, dtype=hn_logits.dtype)
            )
            # 无效位置置 -inf：既不进分母，也不产生梯度
            hn_logits = hn_logits.masked_fill(~hard_neg_mask, float("-inf"))
            loss_i2t_denom = torch.cat([logits, hn_logits], dim=1)

        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i2t = F.cross_entropy(loss_i2t_denom, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        loss = (loss_i2t + loss_t2i) / 2

        out = {
            "loss": loss,
            "contrastive_loss": loss.detach(),
            "loss_i2t": loss_i2t.detach(),
            "loss_t2i": loss_t2i.detach(),
        }

        if teacher_sim is not None and self.distill_lambda > 0:
            temp = self.kd_temperature
            p_student = F.log_softmax(logits / temp, dim=1)
            p_teacher = F.softmax(teacher_sim.detach() / temp, dim=1)
            kd = F.kl_div(p_student, p_teacher, reduction="batchmean") * (temp ** 2)
            out["distill_loss"] = kd.detach()
            out["loss"] = loss + self.distill_lambda * kd

        return out
