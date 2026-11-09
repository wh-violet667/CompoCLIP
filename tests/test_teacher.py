"""教师缓存的对齐校验与按下标取用。"""
import pytest
import torch

from src.teacher import (
    check_alignment,
    load_teacher_cache,
    save_teacher_cache,
    teacher_similarity,
)


def _cache(tmp_path, n=3, d=3):
    path = tmp_path / "teacher.pt"
    image = torch.eye(n, d)
    text = torch.eye(n, d)
    save_teacher_cache(path, image, text, meta={"n": n})
    return path


def test_roundtrip_and_indexed_similarity(tmp_path):
    path = _cache(tmp_path)
    cache = load_teacher_cache(path)
    index = torch.tensor([2, 0])
    sim = teacher_similarity(cache, index, "cpu")
    # 两组特征都是单位阵，取行后内积应为置换后的单位阵
    assert torch.allclose(sim, torch.eye(2))


def test_alignment_guard(tmp_path):
    cache = load_teacher_cache(_cache(tmp_path))
    check_alignment(cache, 3)
    with pytest.raises(ValueError, match="无法按行对齐"):
        check_alignment(cache, 5)


def test_mismatched_lengths_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_teacher_cache(tmp_path / "bad.pt", torch.zeros(2, 3), torch.zeros(3, 3), meta={})


def test_missing_cache_has_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.distill"):
        load_teacher_cache(tmp_path / "nope.pt")
