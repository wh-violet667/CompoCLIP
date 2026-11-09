"""warmup + cosine 调度器的单测。"""
import pytest
import torch

from src.optim import build_scheduler, warmup_cosine_lambda


def test_warmup_ramps_up_then_decays():
    assert warmup_cosine_lambda(0, warmup=10, total=100) == pytest.approx(0.1)
    assert warmup_cosine_lambda(9, warmup=10, total=100) == pytest.approx(1.0)
    assert warmup_cosine_lambda(10, warmup=10, total=100) == pytest.approx(1.0)
    assert warmup_cosine_lambda(100, warmup=10, total=100) == pytest.approx(0.0)
    # 余弦段单调下降
    assert warmup_cosine_lambda(30, 10, 100) > warmup_cosine_lambda(80, 10, 100)


def test_scheduler_updates_lr():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=1.0)
    scheduler = build_scheduler(opt, warmup=2, total_steps=4)
    assert scheduler is not None
    lrs = []
    for _ in range(5):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        scheduler.step()
    assert lrs[0] < 1.0            # warmup 起点
    assert lrs[2] == pytest.approx(1.0)   # warmup 结束


def test_scheduler_disabled_or_invalid():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=1.0)
    assert build_scheduler(opt, 0, 10, "none") is None
    with pytest.raises(ValueError):
        build_scheduler(opt, 0, 10, "polynomial")
