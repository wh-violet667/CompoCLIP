"""检索协议与指标的 CPU 单测（不依赖 GPU、不依赖数据集）。"""
import numpy as np

from metrics import pairwise_accuracy, rank_of_positive, recall_at_k, summary_row


def test_perfect_ranking():
    scores = np.eye(3)
    mask = np.eye(3, dtype=bool)
    ranks = rank_of_positive(scores, mask)
    assert np.array_equal(ranks, np.zeros(3))


def test_multi_positive_i2t_uses_best_caption():
    # 2 张图 x 2 条 caption：列 0,1 属于图 0，列 2,3 属于图 1
    scores = np.array([
        [0.90, 0.80, 0.10, 0.10],   # 图 0：正样本占据前两名 -> 名次 0
        [0.90, 0.85, 0.70, 0.60],   # 图 1：最好的正样本排第 3 -> 名次 2
    ])
    mask = np.array([
        [True, True, False, False],
        [False, False, True, True],
    ])
    ranks = rank_of_positive(scores, mask)
    assert np.allclose(ranks, [0.0, 2.0])
    assert recall_at_k(ranks, 1) == 0.5
    assert recall_at_k(ranks, 3) == 1.0


def test_duplicate_captions_do_not_inflate_rank():
    """同一图的两条 caption 分数相同（此前协议会把重复图当多个候选抬高名次）。"""
    scores = np.array([
        [0.50, 0.50, 0.90],   # 图 0：正样本并列且都低于最后一个候选 -> 名次 1
        [0.90, 0.85, 0.20],   # 图 1：唯一正样本排第 3 -> 名次 2
    ])
    mask = np.array([
        [True, True, False],
        [False, False, True],
    ])
    ranks = rank_of_positive(scores, mask)
    assert np.isclose(ranks[0], 1.0)
    assert np.isclose(ranks[1], 2.0)


def test_tie_average_half_position():
    scores = np.zeros((1, 4))
    mask = np.array([[False, True, False, False]])
    ranks = rank_of_positive(scores, mask, tie="average")
    # 3 个并列的非正样本，平均分到 1.5 名次
    assert np.isclose(ranks[0], 1.5)
    assert np.isclose(rank_of_positive(scores, mask, tie="strict")[0], 0.0)


def test_every_query_needs_positive():
    scores = np.eye(2)
    mask = np.array([[True, False], [False, False]])
    try:
        rank_of_positive(scores, mask)
    except ValueError:
        pass
    else:
        raise AssertionError("缺少正样本时应报错")


def test_summary_row_counts():
    ranks_t2i = np.array([0.0, 1.0, 5.0, 20.0])
    ranks_i2t = np.array([0.0, 0.0])
    row = summary_row("unit", ranks_t2i, ranks_i2t)
    assert row["n_text_queries"] == 4
    assert row["n_image_queries"] == 2
    assert row["t2i_R@1"] == 0.25
    assert row["t2i_R@10"] == 0.75
    assert row["i2t_R@1"] == 1.0


def test_pairwise_accuracy():
    pos = np.array([0.3, 0.2, 0.5])
    neg = np.array([0.1, 0.4, 0.5])
    assert pairwise_accuracy(pos, neg) == 1 / 3
