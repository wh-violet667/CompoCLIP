"""数据集 loader：所有基准统一返回「一个 query 对应多个正样本」的结构。

设计要点（针对审核指出的协议错误）：
- Flickr30k/COCO 是 N 张图 x 5 条 caption，不是 N*5 张独立图；
  因此按 image 分组，i2t 检索时 5 条 caption 都是正样本。
- 分组顺序稳定，便于特征缓存与结果复现。

统一结构：
    ImageGroup(image: Path, captions: list[str], image_id: str)   # 检索类基准
    PairItem(image: Path, positive: str, negative: str, subset)    # 组合推理类基准
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))


@dataclass
class ImageGroup:
    image: Path
    captions: list[str]
    image_id: str


@dataclass
class PairItem:
    image: Path
    positive: str
    negative: str
    subset: str
    fine: str = ""  # 更细的子类（如 SugarCrepe++ 的 replace_att），只用于诊断，不参与主表宏平均


def resolve_data_root(cli_value: str | Path | None) -> Path:
    if cli_value:
        return Path(cli_value)
    return Path(os.environ.get("DATA_ROOT", "data"))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 不存在。检索类基准需先运行 scripts/prepare_flickr30k.py 生成 split"
        )
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _group(rows: list[dict], image_root: Path) -> list[ImageGroup]:
    """按 image_id 分组并保持首次出现顺序；同一图的多条 caption 视为多正样本。"""
    groups: dict[str, ImageGroup] = {}
    for row in rows:
        key = str(row.get("image_id") or row["image"])
        if key not in groups:
            groups[key] = ImageGroup(image=image_root / row["image"], captions=[], image_id=key)
        groups[key].captions.append(row["caption"])

    result = list(groups.values())
    odd = [g for g in result if len(g.captions) != 5]
    if odd:
        warnings.warn(
            f"{len(odd)}/{len(result)} 张图的 caption 数不是 5（示例 {odd[0].image.name} 有 {len(odd[0].captions)} 条），"
            "R@k 与文献不可直接比较",
            stacklevel=2,
        )
    return result


def load_flickr30k(data_root: Path, split: str = "test", limit: int | None = None) -> list[ImageGroup]:
    rows = _read_jsonl(data_root / "flickr30k" / "splits" / f"{split}.jsonl")
    groups = _group(rows, data_root / "flickr30k" / "images")
    return groups[:limit] if limit else groups


def load_coco(data_root: Path, split: str = "val", limit: int | None = None) -> list[ImageGroup]:
    """COCO 2017 val：annotations/captions_val2017.json + val2017/*.jpg。"""
    ann_path = data_root / "coco2017" / "annotations" / f"captions_{split}2017.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"{ann_path} 不存在；COCO 需要 annotations 目录，不能复用 Flickr30k 的 loader")
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
    image_root = data_root / "coco2017" / f"{split}2017"

    groups: dict[int, ImageGroup] = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in groups:
            groups[img_id] = ImageGroup(
                image=image_root / id_to_file[img_id], captions=[], image_id=str(img_id)
            )
        groups[img_id].captions.append(ann["caption"].strip())

    result = list(groups.values())
    return result[:limit] if limit else result


def load_pairwise(golden_path: Path, limit: int | None = None) -> list[PairItem]:
    """ARO / SugarCrepe++ 归一化后的逐行 JSON（见 scripts/prepare_pairwise_bench.py）。"""
    rows = _read_jsonl(golden_path)
    items = [
        PairItem(image=Path(r["image"]), positive=r["positive"], negative=r["negative"],
                 subset=r.get("subset", "all"), fine=r.get("name") or r.get("fine") or r.get("subset", "all"))
        for r in rows
    ]
    return items[:limit] if limit else items


def load_winoground(limit: int | None = None, data_root: Path | None = None) -> list[dict]:
    """优先读本地 pairs.jsonl（scripts/prepare_winoground.py 产物）；否则回退 HF gated。

    本地格式：{"group_id", "image_0", "image_1", "caption_0", "caption_1", "tag", ...}
    返回结构与 HF 版一致：image_0/image_1 为 PIL.Image，可直接被 run_eval 使用。
    """
    from PIL import Image

    root = data_root or DEFAULT_DATA_ROOT
    local = root / "winoground" / "pairs.jsonl"
    if local.exists():
        rows = _read_jsonl(local)
        items = []
        for row in rows:
            items.append({
                **row,
                "image_0": Image.open(row["image_0"]).convert("RGB"),
                "image_1": Image.open(row["image_1"]).convert("RGB"),
            })
        return items[:limit] if limit else items

    from datasets import load_dataset
    ds = load_dataset("facebook/winoground", split="test")
    items = list(ds)
    return items[:limit] if limit else items
