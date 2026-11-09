"""可插拔 encoder 协议：E0–E5 全部实验组走同一接口（dev-doc 4.1 节）。

    enc = OpenClipEncoder.from_pretrained("ViT-B-32", "laion2b_s34b_b79k")   # E0
    enc = OpenClipEncoder.from_checkpoint("ViT-B-32", "runs/e2/epoch3.pt")   # 微调组
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
from open_clip import create_model_and_transforms, get_tokenizer
from PIL import Image

from checkpoint_utils import load_checkpoint_state


class ImageTextEncoder(Protocol):
    def encode_images(self, paths: list[str]) -> np.ndarray: ...  # (N, d) L2 归一化
    def encode_texts(self, texts: list[str]) -> np.ndarray: ...   # (N, d) L2 归一化


class OpenClipEncoder:
    def __init__(self, model, preprocess, tokenizer, device: str):
        self.model = model.eval()
        self.preprocess, self.tokenizer, self.device = preprocess, tokenizer, device

    @classmethod
    def from_pretrained(cls, model_name: str, pretrained: str, device: str | None = None) -> "OpenClipEncoder":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, _, preprocess = create_model_and_transforms(model_name, pretrained=pretrained)
        return cls(model.to(device), preprocess, get_tokenizer(model_name), device)

    @classmethod
    def from_checkpoint(cls, model_name: str, ckpt: str, device: str | None = None) -> "OpenClipEncoder":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, _, preprocess = create_model_and_transforms(model_name, pretrained=None)
        # 兼容 train.py 的训练 checkpoint（含 model/optimizer/config）和裸 state_dict；
        # LoRA 请传 epoch{n}_merged.pt，键名与基座一致。
        model.load_state_dict(load_checkpoint_state(ckpt))
        return cls(model.to(device), preprocess, get_tokenizer(model_name), device)

    @torch.no_grad()
    def encode_images(self, paths: list[str], batch_size: int = 128) -> np.ndarray:
        feats = []
        for i in range(0, len(paths), batch_size):
            imgs = torch.stack([
                self.preprocess(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]
            ]).to(self.device)
            f = self.model.encode_image(imgs)
            feats.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
        return np.concatenate(feats)

    @torch.no_grad()
    def encode_texts(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        feats = []
        for i in range(0, len(texts), batch_size):
            t = self.tokenizer(texts[i:i + batch_size]).to(self.device)
            f = self.model.encode_text(t)
            feats.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
        return np.concatenate(feats)
