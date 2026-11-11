"""归一化 ARO 的 VG-Relation / VG-Attribution（以及可选 Flickr-Order）。

数据来源（D0 已下载）：
- data/aro/vg_relation.json     23,937 对；字段 true_caption / false_caption / image_path
- data/aro/vg_attribution.json  28,748 对；同上
- data/aro/images/              6,603 张 Visual Genome 图

ARO 官方还有 COCO-Order / Flickr-Order 两个 order 子集：
- Flickr-Order 可用 scripts/prepare_flickr30k_wds.py 的图片 + 官方扰动函数重建；
- COCO-Order 需要 COCO test2014 原图（成员 2 尚未提供），本脚本默认跳过并打印原因。

用法：
    python scripts/prepare_aro.py --output data/golden/aro.jsonl
    python scripts/prepare_aro.py --include-flickr-order
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.diagnostics.perturbations import (  # noqa: E402
    shuffle_all_but_nouns_and_adj,
    shuffle_nouns_and_adj,
    shuffle_trigrams,
    shuffle_within_trigrams,
)

ORDER_FNS = [
    ("shuffle_nouns_and_adj", shuffle_nouns_and_adj),
    ("shuffle_allbut_nouns_and_adj", shuffle_all_but_nouns_and_adj),
    ("shuffle_within_trigrams", shuffle_within_trigrams),
    ("shuffle_trigrams", shuffle_trigrams),
]


def load_json(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"缺少 {path}；先按 dev-doc D0 下载 ARO 标注")
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(rows: list[dict], out) -> None:
    for r in rows:
        out.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aro-root", type=Path, default=ROOT / "data/aro")
    parser.add_argument("--flickr-split", type=Path, default=ROOT / "data/flickr30k/splits/test.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/golden/aro.jsonl")
    parser.add_argument("--include-flickr-order", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    image_root = args.aro_root / "images"
    rows: list[dict] = []

    for fname, subset in [("vg_relation.json", "vg_relation"), ("vg_attribution.json", "vg_attribution")]:
        data = load_json(args.aro_root / fname)
        for r in data:
            img = image_root / r["image_path"]
            rows.append({
                "image": str(img),
                "positive": r["true_caption"].strip(),
                "negative": r["false_caption"].strip(),
                "subset": subset,
            })
        print(f"{subset}: {len(data)} 对（图片 {image_root}）")

    if args.include_flickr_order:
        if not args.flickr_split.exists():
            print(f"跳过 Flickr-Order：{args.flickr_split} 不存在（先跑 prepare_flickr30k_wds.py）")
        else:
            rng = random.Random(args.seed)
            n = 0
            for line in args.flickr_split.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                caption = item["caption"].strip().rstrip(".")
                if len(caption.split()) < 4:
                    continue
                for name, fn in ORDER_FNS:
                    neg = fn(caption, rng)
                    if neg is None:
                        continue
                    rows.append({
                        "image": str(ROOT / "data/flickr30k/images" / item["image"]),
                        "positive": caption,
                        "negative": neg,
                        "subset": "flickr_order",
                        "order_rule": name,
                    })
                    n += 1
            print(f"flickr_order: {n} 对（官方 4 种扰动，逐条保留）")

    if not (args.aro_root / "coco_order_placeholder").exists():
        print("跳过 COCO-Order：需要 COCO test2014 原图，当前 D0 数据未提供（成员 2 依赖）。")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        write_rows(rows, f)
    print(f"完成：{len(rows)} 对 -> {args.output}")


if __name__ == "__main__":
    main()
