"""硬负例 mask、蒸馏项与对称项的单测（CPU，合成张量）。"""
import torch

from src.loss import HardNegClipLoss


def _feats(b=4, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    image = torch.randn(b, d, generator=g)
    text = torch.randn(b, d, generator=g)
    neg = torch.randn(b, 2, d, generator=g)
    image = image / image.norm(dim=-1, keepdim=True)
    text = text / text.norm(dim=-1, keepdim=True)
    neg = neg / neg.norm(dim=-1, keepdim=True)
    return image, text, neg


def test_all_masked_equals_no_hard_negative():
    image, text, neg = _feats()
    crit = HardNegClipLoss()
    scale = torch.tensor(100.0)
    base = crit(image, text, scale)["loss"]
    masked = crit(image, text, scale, neg, torch.zeros(4, 2, dtype=torch.bool))["loss"]
    assert torch.allclose(base, masked, atol=1e-6)


def test_masked_positions_do_not_affect_loss():
    """空负例不再污染分母：被 mask 的位置改动任意值都不应改变 loss。"""
    image, text, neg = _feats()
    mask = torch.zeros(4, 2, dtype=torch.bool)
    mask[0, 0] = True
    perturbed = neg.clone()
    perturbed[~mask] = 7.0         # 只在无效位置塞入极端值
    crit = HardNegClipLoss()
    scale = torch.tensor(100.0)
    a = crit(image, text, scale, neg, mask)["loss"]
    b = crit(image, text, scale, perturbed, mask)["loss"]
    assert torch.allclose(a, b, atol=1e-6)


def test_partial_mask_changes_loss_but_stays_finite():
    image, text, _ = _feats()
    # 让硬负例与图像高度相似（等价于高分负样本），确保纳入分母后损失可观测地变化
    neg = text.unsqueeze(1).repeat(1, 2, 1)
    crit = HardNegClipLoss()
    scale = torch.tensor(10.0)
    mask = torch.tensor([[True, False], [True, False], [False, False], [False, False]])
    out = crit(image, text, scale, neg, mask)
    assert torch.isfinite(out["loss"])
    base = crit(image, text, scale)["loss"]
    assert not torch.allclose(out["loss"], base)


def test_k_greater_than_one_is_supported():
    image, text, _ = _feats()
    neg = text.unsqueeze(1).repeat(1, 2, 1).requires_grad_(True)
    crit = HardNegClipLoss()
    out = crit(image, text, torch.tensor(10.0), neg, torch.ones(4, 2, dtype=torch.bool))
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert neg.grad is not None and torch.isfinite(neg.grad).all()


def test_lambda_zero_ignores_teacher():
    image, text, _ = _feats()
    crit = HardNegClipLoss(hard_negative_weight=1.0, distill_lambda=0.0)
    scale = torch.tensor(10.0)
    teacher = image @ text.t()
    with_teacher = crit(image, text, scale, None, None, teacher)
    without = crit(image, text, scale)
    assert torch.allclose(with_teacher["loss"], without["loss"])
    assert "distill_loss" not in with_teacher


def test_matching_teacher_gives_zero_kd():
    image, text, _ = _feats()
    crit = HardNegClipLoss(distill_lambda=0.5, kd_temperature=1.0)
    scale = torch.tensor(1.0)                     # 学生与教师同温度
    teacher = image @ text.t()
    out = crit(image, text, scale, None, None, teacher)
    assert out["distill_loss"].item() < 1e-6


def test_teacher_does_not_inherit_student_scale():
    """教师分布不应随学生 logit_scale 漂移。"""
    image, text, _ = _feats()
    crit = HardNegClipLoss(distill_lambda=0.5)
    teacher = image @ text.t()
    low = crit(image, text, torch.tensor(1.0), None, None, teacher)["distill_loss"]
    high = crit(image, text, torch.tensor(100.0), None, None, teacher)["distill_loss"]
    assert not torch.isclose(low, high)


def test_anti_correlated_teacher_increases_loss():
    image, text, _ = _feats()
    crit = HardNegClipLoss(distill_lambda=0.5)
    scale = torch.tensor(10.0)
    base = crit(image, text, scale)
    bad_teacher = -(image @ text.t())
    out = crit(image, text, scale, None, None, bad_teacher)
    assert out["loss"].item() > base["loss"].item()
