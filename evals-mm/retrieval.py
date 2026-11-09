"""FAISS 检索后端：FlatIP 精确内积（课程项目数据量小，不需要近似索引）。"""
from __future__ import annotations

import numpy as np


def build_index(image_feats: np.ndarray):
    import faiss
    index = faiss.IndexFlatIP(image_feats.shape[1])
    index.add(image_feats.astype("float32"))
    return index


def search(index, text_feats: np.ndarray, top_k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """返回 (scores, indices)，形状 (Q, top_k)。"""
    return index.search(text_feats.astype("float32"), top_k)


def full_scores(image_feats: np.ndarray, text_feats: np.ndarray) -> np.ndarray:
    """小数据集直接算全量相似度矩阵，供 R@k / Winoground 指标使用。"""
    return text_feats @ image_feats.T
