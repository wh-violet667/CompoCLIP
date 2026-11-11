"""把 mm-eval/Winoground 的 parquet 还原成 400 对本地图片 + pairs.jsonl。

Winoground 官方 HF 是 gated（401）。D0 未拿到权限时，成员 1 用公开镜像
（mm-eval/Winoground，含原图）评测；正式结果仍需在官方 gated 数据上复核一遍。
本脚本只保留评测所需字段，不改变官方 pair 语义。

用法：
    python scripts/prepare_winoground.py --parquet "data/winoground/*.parquet" \
        --image-dir data/winoground/images --output data/winoground/pairs.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd


def save_image(media, path: Path) -> None:
    item = media[0] if len(media) else None
    if item is None:
        raise ValueError(f"空 media：{path}")
    if isinstance(item, dict) and item.get("bytes") is not None:
        path.write_bytes(item["bytes"])
    elif isinstance(item, dict) and item.get("path"):
        Path(item["path"]).replace(path)
    else:
        raise ValueError(f"无法识别的 media 格式：{type(item)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default="data/winoground/*.parquet")
    parser.add_argument("--image-dir", type=Path, default=Path("data/winoground/images"))
    parser.add_argument("--output", type=Path, default=Path("data/winoground/pairs.jsonl"))
    args = parser.parse_args()

    files = sorted(glob.glob(args.parquet))
    if not files:
        raise SystemExit(f"没有匹配到 parquet：{args.parquet}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    args.image_dir.mkdir(parents=True, exist_ok=True)
    pairs: dict[str, dict] = {}
    seen: set[tuple[str, str]] = set()

    for _, row in df.iterrows():
        gid = str(row["group_id"])
        idx = str(row["image_index"])
        pair = pairs.setdefault(gid, {
            "group_id": gid,
            "caption_0": row["caption_0"],
            "caption_1": row["caption_1"],
            "tag": row.get("tag", ""),
            "secondary_tag": row.get("secondary_tag", ""),
            "collapsed_tag": row.get("collapsed_tag", ""),
        })
        if (gid, idx) not in seen:
            out = args.image_dir / f"{gid}_{idx}.png"
            save_image(row["media"], out)
            pair[f"image_{idx}"] = str(out)
            seen.add((gid, idx))

    rows = []
    for gid in sorted(pairs, key=lambda x: int(x) if x.isdigit() else x):
        p = pairs[gid]
        if "image_0" not in p or "image_1" not in p:
            continue
        rows.append(p)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"完成：{len(rows)} 对 / {len(seen)} 张图 -> {args.output}")
    if len(rows) != 400:
        print(f"警告：Winoground 应为 400 对，实际 {len(rows)} 对；请检查 parquet 是否完整。")


if __name__ == "__main__":
    main()
