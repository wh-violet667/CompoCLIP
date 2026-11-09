"""微调训练入口：open_clip 基座 + HardNegClipLoss（含蒸馏）+ fp16 AMP（T4 适配）。

用法：
    python -m src.train --config configs/finetune_e2.yaml

支持三种策略（config.strategy）：
    full       全量微调
    lock_image 锁图像塔，只训文本塔（显存省约 40%）
    lora       q/v 投影加 LoRA（需 pip install peft）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from open_clip import create_model_and_transforms, get_tokenizer
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.loss import HardNegClipLoss


class Flickr30kHardNeg(Dataset):
    """正样本 caption + 每条 K 个硬负例。hardneg JSONL 需含 negative 字段。"""

    def __init__(self, split_path: str, hardneg_path: str | None, image_root: Path, preprocess):
        self.items = [json.loads(l) for l in Path(split_path).open(encoding="utf-8")]
        self.image_root = image_root
        self.preprocess = preprocess
        self.negatives: dict[int, list[str]] = {}
        if hardneg_path:
            for i, it in enumerate(self.items):
                # 简化假设：硬负例文件与正样本同序对齐（生成时保序）
                pass
            neg_items = [json.loads(l) for l in Path(hardneg_path).open(encoding="utf-8")]
            key_to_idx = {(it["image"], it["caption"]): i for i, it in enumerate(self.items)}
            for neg in neg_items:
                idx = key_to_idx.get((neg["image"], neg["caption"]))
                if idx is not None:
                    self.negatives.setdefault(idx, []).append(neg["negative"])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        image = self.preprocess(Image.open(self.image_root / it["image"]).convert("RGB"))
        negs = self.negatives.get(idx, [])
        return image, it["caption"], negs


def collate(batch, tokenizer):
    images = torch.stack([b[0] for b in batch])
    captions = [b[1] for b in batch]
    # 硬负例对齐到 K=1（不足补空串，loss 端用 mask 忽略；v1 简化：只取第一个）
    negs = [b[2][0] if b[2] else "" for b in batch]
    return images, tokenizer(captions), tokenizer(negs), [bool(b[2]) for b in batch]


def apply_strategy(model, strategy: str, lora_r: int = 8):
    if strategy == "lock_image":
        for p in model.visual.parameters():
            p.requires_grad = False
    elif strategy == "lora":
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=lora_r, lora_alpha=16, target_modules=["q_proj", "v_proj"])
        model = get_peft_model(model, cfg)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = create_model_and_transforms(
        cfg["model"], pretrained=cfg["pretrained"],
        force_custom_text=cfg.get("strategy") == "lora",  # LoRA 需要 nn.Linear 命名
    )
    model = apply_strategy(model, cfg.get("strategy", "full"), cfg.get("lora_r", 8)).to(device)
    tokenizer = get_tokenizer(cfg["model"])

    train_data = Path(cfg["train_data"])
    dataset = Flickr30kHardNeg(
        str(train_data), cfg.get("hardneg_data"), train_data.parent.parent / "images", preprocess
    )
    loader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=4,
        collate_fn=lambda b: collate(b, tokenizer), drop_last=True,
    )

    # 教师特征缓存（E3 启用蒸馏时加载）
    teacher = None
    if cfg.get("distill_lambda", 0) > 0:
        cache = torch.load(cfg["distill_teacher_cache"])
        teacher = (cache["image"], cache["text"])  # CPU 张量，训练时按 batch 索引取用

    criterion = HardNegClipLoss(cfg.get("hard_negative_weight", 1.0), cfg.get("distill_lambda", 0.0))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=cfg["wd"],
    )
    scaler = torch.amp.GradScaler(enabled=(cfg["precision"] == "amp"))
    accum = cfg.get("accum_freq", 1)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for images, texts, negs, has_neg in pbar:
            images, texts, negs = images.to(device), texts.to(device), negs.to(device)
            with torch.amp.autocast("cuda", enabled=(cfg["precision"] == "amp")):
                fi = model.encode_image(images)
                ft = model.encode_text(texts)
                fi, ft = fi / fi.norm(dim=-1, keepdim=True), ft / ft.norm(dim=-1, keepdim=True)
                fn = model.encode_text(negs)
                fn = fn / fn.norm(dim=-1, keepdim=True)
                # (B, 1, d)；无负例样本的零向量不影响分母的近似性（v1 简化，后续加 mask）
                hn = fn.unsqueeze(1) if any(has_neg) else None
                teacher_sim = None  # TODO(E3): 按全局 batch 索引取 teacher 特征切片计算
                losses = criterion(fi, ft, model.logit_scale.exp(), hn, teacher_sim)
                loss = losses["loss"] / accum

            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            pbar.set_postfix(loss=f"{losses['loss'].item():.4f}",
                             logit_scale=f"{model.logit_scale.item():.3f}")

        ckpt = out_dir / f"epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt)
        print(f"saved {ckpt}；请立刻跑 evals-mm 的 Flickr30k val + COCO 子集监控遗忘（dev-doc 3.6 节）")


if __name__ == "__main__":
    main()
