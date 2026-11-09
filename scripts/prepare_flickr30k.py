"""把 Flickr30k 原始标注转换成训练/评测统一使用的 JSONL split。

背景：Kaggle 打包的 Flickr30k 只有图片 + 原始标注，没有 Karpathy split；
本项目所有脚本（train / run_eval / hardneg）都按 JSONL 逐行读取，必须先跑本脚本。

两种输入模式：
  1) --karpathy-json（推荐）：官方 dataset_flickr30k.json
     来源 http://cs.stanford.edu/people/karpathy/deepimagesent/flickr30k.zip
  2) --from-hf：HuggingFace nlphuji/flickr30k，可选把图片落盘

输出：$DATA_ROOT/flickr30k/splits/{train,val,test}.jsonl
字段：image / caption / image_id / caption_id

用法：
    python scripts/prepare_flickr30k.py --karpathy-json /path/dataset_flickr30k.json
    python scripts/prepare_flickr30k.py --karpathy-json ... --limit 200   # 冒烟
    python scripts/prepare_flickr30k.py --from-hf nlphuji/flickr30k --save-images
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", ROOT / "data"))

# Karpathy split 官方规模（图数, caption 数）；仅用于校验，不做强制
EXPECTED = {"train": (29000, 145000), "val": (1014, 5070), "test": (1000, 5000)}


def resolve_data_root(cli_value: str | None) -> Path:
    return Path(cli_value) if cli_value else DEFAULT_DATA_ROOT


def write_split(rows_by_split: dict[str, list[dict]], out_dir: Path, limit: int | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        seen_images: list[str] = []
        kept: list[dict] = []
        for row in rows:
            if row["image"] not in seen_images:
                if limit is not None and len(seen_images) >= limit:
                    continue
                seen_images.append(row["image"])
            kept.append(row)
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in kept:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{split}] 图像 {len(seen_images)}，caption {len(kept)} -> {path}")
        if limit is None and split in EXPECTED:
            exp_img, exp_cap = EXPECTED[split]
            if (len(seen_images), len(kept)) != (exp_img, exp_cap):
                print(f"  ! 与 Karpathy 官方规模 {exp_img}/{exp_cap} 不一致，R@k 将无法与文献直接比较")


def from_karpathy(json_path: Path, image_dir: Path) -> dict[str, list[dict]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if "images" not in data:
        sys.exit(f"{json_path} 不是 Karpathy/COCO 格式（缺少 images 字段）")
    rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for img in data["images"]:
        split = img.get("split")
        if split not in rows:
            continue
        filename = img.get("filename") or f"{img['imgid']}.jpg"
        for sent in img.get("sentences", []):
            caption = sent.get("raw") or " ".join(sent.get("tokens", []))
            if not caption:
                continue
            rows[split].append({
                "image": filename,
                "caption": caption,
                "image_id": str(img.get("imgid", filename)),
                "caption_id": str(sent.get("sentid", "")),
            })
    # 抽样检查图片是否真实存在，避免 split 正确但图片目录错位
    missing = [r["image"] for r in rows["train"][:20] if not (image_dir / r["image"]).exists()]
    if missing:
        print(f"  ! 前 20 张训练图有 {len(missing)} 张在 {image_dir} 找不到，例如 {missing[0]}")
        print("    请确认 --image-dir 指向解压后的 flickr30k 图片目录")
    return rows


def from_hf(dataset_name: str, image_dir: Path, save_images: bool) -> dict[str, list[dict]]:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("缺少 datasets，请先 pip install datasets")
    ds = load_dataset(dataset_name)
    rows: dict[str, list[dict]] = {}
    for split, part in ds.items():
        if split not in EXPECTED:
            continue
        rows[split] = []
        cols = set(part.column_names)
        if save_images:
            image_dir.mkdir(parents=True, exist_ok=True)
        for i, ex in enumerate(part):
            filename = ex.get("filename") or f"{ex.get('img_id', i)}.jpg"
            if save_images and "image" in cols:
                target = image_dir / filename
                if not target.exists():
                    ex["image"].convert("RGB").save(target)
            captions = ex.get("caption") or ex.get("captions") or []
            if isinstance(captions, str):
                captions = [captions]
            for j, cap in enumerate(captions):
                if not cap:
                    continue
                rows[split].append({
                    "image": filename,
                    "caption": cap,
                    "image_id": str(ex.get("img_id", filename)),
                    "caption_id": f"{ex.get('img_id', i)}_{j}",
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--karpathy-json", type=Path, help="dataset_flickr30k.json 路径")
    parser.add_argument("--from-hf", help="HF 数据集名，如 nlphuji/flickr30k")
    parser.add_argument("--image-dir", type=Path, help="图片目录，默认 $DATA_ROOT/flickr30k/images")
    parser.add_argument("--out-dir", type=Path, help="输出目录，默认 $DATA_ROOT/flickr30k/splits")
    parser.add_argument("--data-root", help="覆盖 DATA_ROOT 环境变量")
    parser.add_argument("--save-images", action="store_true", help="HF 模式下落盘图片")
    parser.add_argument("--limit", type=int, help="每个 split 只保留 N 张图（冒烟用）")
    args = parser.parse_args()

    if not args.karpathy_json and not args.from_hf:
        parser.error("必须指定 --karpathy-json 或 --from-hf")

    data_root = resolve_data_root(args.data_root)
    image_dir = args.image_dir or data_root / "flickr30k" / "images"
    out_dir = args.out_dir or data_root / "flickr30k" / "splits"

    if args.karpathy_json:
        rows = from_karpathy(args.karpathy_json, image_dir)
    else:
        rows = from_hf(args.from_hf, image_dir, args.save_images)

    write_split(rows, out_dir, args.limit)

    # 下游脚本直接写 data/hardneg/，这里提前建好，避免 FileNotFoundError
    (data_root / "hardneg").mkdir(parents=True, exist_ok=True)
    (data_root / "golden").mkdir(parents=True, exist_ok=True)
    print(f"\n完成。数据根目录：{data_root}")
    print("下一步：python scripts/prepare_pairwise_bench.py --help  # 生成 ARO/SugarCrepe 评测集")


if __name__ == "__main__":
    main()
