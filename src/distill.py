"""教师特征预计算（蒸馏抗遗忘的数据准备，dev-doc 3.3 节）。

只缓存特征，不缓存 145k x 145k 的相似度矩阵（约 84GB，无法落盘）；
训练时按 batch 的样本下标取特征，再算 B x B 的教师分布（见 src/teacher.py）。

缓存与 train_data 的 JSONL **逐行对齐**：第 i 行特征对应第 i 条样本。
若要把额外数据（如 COCO 子集）混进训练，请先合并成同一个 JSONL，
并让 distill 与 train 使用同一文件，否则训练启动时会因长度不一致报错。

用法：
    python -m src.distill --config configs/finetune_e3.yaml
    python -m src.distill --config configs/finetune_e3.yaml --limit 200   # 冒烟
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from open_clip import create_model_and_transforms, get_tokenizer
from PIL import Image
from tqdm import tqdm

from src.teacher import save_teacher_cache


@torch.no_grad()
def encode_all(model, preprocess, tokenizer, items, image_root: Path,
               batch_size: int, device: str):
    img_feats, txt_feats = [], []
    skipped = 0
    for i in tqdm(range(0, len(items), batch_size), desc="teacher encode"):
        chunk = items[i:i + batch_size]
        images, texts = [], []
        for it in chunk:
            path = image_root / it["image"]
            if not path.exists():
                skipped += 1
                continue
            images.append(preprocess(Image.open(path).convert("RGB")))
            texts.append(it["caption"])
        if not images:
            continue
        batch_images = torch.stack(images).to(device)
        fi = model.encode_image(batch_images)
        ft = model.encode_text(tokenizer(texts).to(device))
        img_feats.append((fi / fi.norm(dim=-1, keepdim=True)).cpu())
        txt_feats.append((ft / ft.norm(dim=-1, keepdim=True)).cpu())
    if skipped:
        raise RuntimeError(
            f"有 {skipped} 张图片找不到（{image_root}），缓存会与训练集错位。"
            "请先修好图片路径再重跑。"
        )
    return torch.cat(img_feats), torch.cat(txt_feats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256, help="推理无梯度，可以开大")
    parser.add_argument("--limit", type=int, help="只处理前 N 条样本（冒烟用，缓存不能用于正式训练）")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out_path = Path(cfg["distill_teacher_cache"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = create_model_and_transforms(cfg["model"], pretrained=cfg["pretrained"])
    model = model.to(device).eval()
    tokenizer = get_tokenizer(cfg["model"])

    train_path = Path(cfg["train_data"])
    items = [json.loads(line) for line in train_path.open(encoding="utf-8") if line.strip()]
    if args.limit:
        items = items[: args.limit]
        print(f"! 冒烟模式：只编码前 {len(items)} 条，该缓存不可用于正式训练")

    image_root = train_path.parent.parent / "images"
    img_feats, txt_feats = encode_all(model, preprocess, tokenizer, items, image_root,
                                      args.batch_size, device)

    save_teacher_cache(out_path, img_feats, txt_feats, meta={
        "model": cfg["model"],
        "pretrained": cfg["pretrained"],
        "train_data": str(train_path),
        "n_samples": len(items),
        "limit": args.limit or "",
    })
    print(f"教师特征已缓存：{out_path}")
    print(f"  图像 {tuple(img_feats.shape)}，文本 {tuple(txt_feats.shape)}，与 {train_path} 逐行对齐")


if __name__ == "__main__":
    main()
