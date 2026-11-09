"""教师 logits 预计算（蒸馏抗遗忘的数据准备，dev-doc 3.3 节）。

用未微调的 E0（原始 CLIP）在 Flickr30k train + CC3M 子集上跑一次前向，
把图像/文本特征缓存到磁盘；训练时按 batch 索引切片计算相似度分布，
不占用训练显存。

用法：
    python -m src.distill --config configs/finetune_e2.yaml
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


@torch.no_grad()
def encode_all(model, preprocess, tokenizer, items: list[dict], image_root: Path,
               batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    img_feats, txt_feats = [], []
    for i in tqdm(range(0, len(items), batch_size), desc="teacher encode"):
        chunk = items[i:i + batch_size]
        images = torch.stack([preprocess(Image.open(image_root / it["image"]).convert("RGB")) for it in chunk])
        texts = tokenizer([it["caption"] for it in chunk])
        fi = model.encode_image(images.to(device))
        ft = model.encode_text(texts.to(device))
        img_feats.append((fi / fi.norm(dim=-1, keepdim=True)).cpu())
        txt_feats.append((ft / ft.norm(dim=-1, keepdim=True)).cpu())
    return torch.cat(img_feats), torch.cat(txt_feats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256, help="推理无梯度，可以开大")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out_path = Path(cfg["distill_teacher_cache"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = create_model_and_transforms(cfg["model"], pretrained=cfg["pretrained"])
    model = model.to(device).eval()
    tokenizer = get_tokenizer(cfg["model"])

    items = [json.loads(l) for l in Path(cfg["train_data"]).open(encoding="utf-8")]
    image_root = Path(cfg["train_data"]).parent.parent / "images"

    img_feats, txt_feats = encode_all(model, preprocess, tokenizer, items, image_root,
                                      args.batch_size, device)
    torch.save({"image": img_feats, "text": txt_feats}, out_path)
    print(f"教师特征已缓存：{out_path}（图像 {tuple(img_feats.shape)}，文本 {tuple(txt_feats.shape)}）")


if __name__ == "__main__":
    main()
