"""训练数据集与 collate（只依赖 torch + PIL，便于单测 mask 逻辑）。"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class Flickr30kHardNeg(Dataset):
    """正样本 caption + 每条 K 个硬负例；返回原始下标供教师缓存索引。"""

    def __init__(self, split_path: str | Path, hardneg_path: str | Path | None,
                 image_root: Path, preprocess):
        self.items = [json.loads(l) for l in Path(split_path).open(encoding="utf-8") if l.strip()]
        self.image_root = image_root
        self.preprocess = preprocess
        self.negatives: dict[int, list[str]] = {}
        if hardneg_path:
            key_to_idx = {(it["image"], it["caption"]): i for i, it in enumerate(self.items)}
            for line in Path(hardneg_path).open(encoding="utf-8"):
                if not line.strip():
                    continue
                neg = json.loads(line)
                idx = key_to_idx.get((neg["image"], neg["caption"]))
                if idx is not None:
                    self.negatives.setdefault(idx, []).append(neg["negative"])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        image = self.preprocess(Image.open(self.image_root / it["image"]).convert("RGB"))
        return idx, image, it["caption"], self.negatives.get(idx, [])

    def num_with_negatives(self) -> int:
        return len(self.negatives)


def collate(batch, tokenizer):
    """把变长硬负例补齐成 (B, K)，并返回 (B, K) 的布尔 mask 供 loss 屏蔽。"""
    index = torch.tensor([b[0] for b in batch], dtype=torch.long)
    images = torch.stack([b[1] for b in batch])
    captions = [b[2] for b in batch]

    max_k = max((len(b[3]) for b in batch), default=0)
    if max_k == 0:
        return {
            "index": index,
            "image": images,
            "text": tokenizer(captions),
            "negative": None,
            "neg_mask": None,
        }

    padded, mask = [], []
    for b in batch:
        negs = b[3][:max_k]
        padded.append(negs + [""] * (max_k - len(negs)))
        mask.append([True] * len(negs) + [False] * (max_k - len(negs)))

    return {
        "index": index,
        "image": images,
        "text": tokenizer(captions),
        "negative": tokenizer(padded),          # (B, K, L)
        "neg_mask": torch.tensor(mask, dtype=torch.bool),
    }
