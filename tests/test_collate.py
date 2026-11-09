"""collate 的补齐与 mask 逻辑（用手写 fake tokenizer 隔离 open_clip）。"""
import torch

from src.data import collate


class FakeTokenizer:
    """返回形状 (..., L) 的整数张量，长度取文本词数，用于验证补齐与 mask。"""

    def tokenize(self, texts):
        if isinstance(texts, str):
            return torch.tensor([len(texts.split())], dtype=torch.long)
        return torch.tensor([[len(str(t).split())] for t in texts], dtype=torch.long)

    def __call__(self, texts):
        if texts and isinstance(texts[0], list):
            return torch.tensor(
                [[[len(str(t).split())] for t in row] for row in texts], dtype=torch.long
            )
        return self.tokenize(texts)


def _item(idx, negs):
    return idx, torch.zeros(3, 4, 4), f"caption {idx}", negs


def test_no_negatives_returns_none():
    batch = collate([_item(0, []), _item(1, [])], FakeTokenizer())
    assert batch["negative"] is None
    assert batch["neg_mask"] is None
    assert batch["index"].tolist() == [0, 1]


def test_variable_length_padding_and_mask():
    batch = collate(
        [_item(0, ["a b", "c d e"]), _item(1, []), _item(2, ["f"])],
        FakeTokenizer(),
    )
    assert batch["negative"].shape[:2] == (3, 2)
    mask = batch["neg_mask"]
    assert mask.tolist() == [[True, True], [False, False], [True, False]]


def test_index_preserved_for_teacher_lookup():
    batch = collate([_item(7, []), _item(3, [])], FakeTokenizer())
    assert batch["index"].tolist() == [7, 3]


def test_caption_tokenized_separately_from_negatives():
    batch = collate([_item(0, ["x y z"])], FakeTokenizer())
    # 正 caption "caption 0" 2 个词；负例 "x y z" 3 个词
    assert batch["text"].tolist() == [[2]]
    assert batch["negative"].tolist() == [[[3]]]
