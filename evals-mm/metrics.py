"""检索与组合推理指标（修正多 caption 协议与并列名次问题）。

协议（与 Flickr30k / COCO 文献一致）：
- t2i：每条 caption 为一次 query，正样本是它所属的那张图。
- i2t：每张图为一次 query，其 5 条 caption 全部是正样本；
  名次取「评分最高的那个正样本」的位置，即「top-k 至少命中一条 caption」。
- 并列（重复 caption 会让分数完全相同）：默认 average 名次，
  即只在严格更高分的候选后排名，再与并列的非正样本均分位置。
"""
from __future__ import annotations

import numpy as np


def rank_of_positive(scores: np.ndarray, positive_mask: np.ndarray, tie: str = "average") -> np.ndarray:
    """返回每个 query 的 0 起名次（支持一个 query 多个正样本）。

    scores: (Q, C) 相似度；positive_mask: (Q, C) bool，每行至少一个 True。
    tie="average"：并列的非正样本与正样本均分位置；tie="strict"：只数严格更高分的候选。
    """
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(positive_mask, dtype=bool)
    if scores.shape != mask.shape:
        raise ValueError(f"shape 不一致：scores {scores.shape} vs mask {mask.shape}")
    if not mask.any(axis=1).all():
        raise ValueError("每行 query 至少需要一个正样本")

    best = np.where(mask, scores, -np.inf).max(axis=1, keepdims=True)
    better = (scores > best).sum(axis=1)
    if tie == "strict":
        return better.astype(np.float64)

    at_best = scores == best
    gt_ties = (at_best & mask).sum(axis=1)
    other_ties = at_best.sum(axis=1) - gt_ties
    return better + 0.5 * other_ties


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    """ranks 为 0 起名次（可为平均名次）。"""
    return float((np.asarray(ranks) < k).mean())


def median_rank(ranks: np.ndarray) -> float:
    return float(np.median(ranks) + 1)


def summary_row(name: str, ranks_t2i: np.ndarray, ranks_i2t: np.ndarray) -> dict:
    return {
        "benchmark": name,
        "n_text_queries": int(len(ranks_t2i)),
        "n_image_queries": int(len(ranks_i2t)),
        "t2i_R@1": recall_at_k(ranks_t2i, 1),
        "t2i_R@5": recall_at_k(ranks_t2i, 5),
        "t2i_R@10": recall_at_k(ranks_t2i, 10),
        "t2i_MedR": median_rank(ranks_t2i),
        "i2t_R@1": recall_at_k(ranks_i2t, 1),
        "i2t_R@5": recall_at_k(ranks_i2t, 5),
        "i2t_R@10": recall_at_k(ranks_i2t, 10),
        "i2t_MedR": median_rank(ranks_i2t),
    }


def winoground_scores(text_correct: np.ndarray, image_correct: np.ndarray) -> dict:
    text_correct = np.asarray(text_correct, dtype=bool)
    image_correct = np.asarray(image_correct, dtype=bool)
    return {
        "wg_text": float(text_correct.mean()),
        "wg_image": float(image_correct.mean()),
        "wg_group": float((text_correct & image_correct).mean()),
        "n_pairs": int(len(text_correct)),
    }


def pairwise_accuracy(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """组合推理二选一：正描述得分高于负描述即正确。"""
    pos_scores = np.asarray(pos_scores, dtype=np.float64)
    neg_scores = np.asarray(neg_scores, dtype=np.float64)
    return float((pos_scores > neg_scores).mean())
