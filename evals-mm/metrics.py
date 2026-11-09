"""检索与组合推理指标（移植自 Mnemo evals 的评测纪律，dev-doc 4.2 节）。"""
from __future__ import annotations

import numpy as np


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    """ranks[i] = 第 i 条 query 的正样本在结果中的名次（0 起）。"""
    return float((ranks < k).mean())


def median_rank(ranks: np.ndarray) -> float:
    return float(np.median(ranks) + 1)


def ranks_from_scores(scores: np.ndarray, gt_indices: np.ndarray) -> np.ndarray:
    """scores: (Q, N) 相似度；gt_indices[i] = query i 的正样本列号。返回 0 起名次。"""
    order = np.argsort(-scores, axis=1)
    positions = np.empty_like(order)
    np.put_along_axis(positions, order, np.arange(scores.shape[1])[None, :], axis=1)
    return positions[np.arange(scores.shape[0]), gt_indices]


def winoground_scores(text_correct: np.ndarray, image_correct: np.ndarray) -> dict[str, float]:
    """text_correct[i]: 给定图 i，两个 caption 中选对；image_correct[i]: 给定 caption 对，选对图。"""
    text_score = float(text_correct.mean())
    image_score = float(image_correct.mean())
    group_score = float((text_correct & image_correct).mean())
    return {"text": text_score, "image": image_score, "group": group_score}


def summary_row(name: str, ranks_t2i: np.ndarray, ranks_i2t: np.ndarray) -> dict:
    return {
        "benchmark": name,
        "t2i_R@1": recall_at_k(ranks_t2i, 1),
        "t2i_R@5": recall_at_k(ranks_t2i, 5),
        "t2i_R@10": recall_at_k(ranks_t2i, 10),
        "t2i_MedR": median_rank(ranks_t2i),
        "i2t_R@1": recall_at_k(ranks_i2t, 1),
        "i2t_R@5": recall_at_k(ranks_i2t, 5),
        "i2t_R@10": recall_at_k(ranks_i2t, 10),
    }
