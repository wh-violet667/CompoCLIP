"""把 ARO / SugarCrepe++ 的原始标注归一化成统一样本格式。

评测器只认识一种格式（逐行 JSON）：
    {"image": "...", "positive": "...", "negative": "...", "subset": "..."}

ARO 与 SugarCrepe++ 各自的字段名不同，本脚本自动识别常见字段名；
识别失败时会把首条记录的字段打印出来，按提示用 --image-key 等显式指定。

用法：
    python scripts/prepare_pairwise_bench.py --input data/aro/vg_relation.json --subset vg_relation \
        --image-root data/aro/images --output data/golden/aro.jsonl
    python scripts/prepare_pairwise_bench.py --input data/sugarcrepepp/*.json --output data/golden/sugarcrepe.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", ROOT / "data"))

IMAGE_KEYS = ["image", "img", "image_path", "filename", "image_name"]
POS_KEYS = ["positive", "caption_pos", "caption_0", "caption_true", "caption1", "pos", "correct", "caption"]
NEG_KEYS = ["negative", "caption_neg", "caption_1", "caption_false", "caption2", "neg", "incorrect"]


def pick(record: dict, candidates: list[str], override: str | None) -> tuple[str, str] | None:
    if override:
        return (override, record[override]) if override in record else None
    for key in candidates:
        if key in record and isinstance(record[key], str):
            return key, record[key]
    return None


def iter_records(path: Path, json_key: str | None) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if json_key and json_key in data:
        return data[json_key]
    for key in ("data", "annotations", "samples", "records"):
        if key in data and isinstance(data[key], list):
            return data[key]
    sys.exit(f"无法定位 {path} 中的样本列表；顶层键为 {list(data)[:10]}，可用 --json-key 指定")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="单个文件或 glob，如 data/aro/*.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subset", help="子集名；多文件输入时默认用文件名")
    parser.add_argument("--image-root", type=Path, help="图片路径前缀（原始标注只给相对路径时使用）")
    parser.add_argument("--image-key", help="显式指定图片字段名")
    parser.add_argument("--positive-key", help="显式指定正 caption 字段名")
    parser.add_argument("--negative-key", help="显式指定负 caption 字段名")
    parser.add_argument("--label-key", help="若正负由 label 决定（如 caption_0/caption_1 + label），指定该字段")
    parser.add_argument("--json-key", help="JSON 中存放样本列表的键名")
    parser.add_argument("--limit", type=int, help="只处理前 N 条（冒烟用）")
    parser.add_argument("--data-root", help="覆盖 DATA_ROOT 环境变量")
    args = parser.parse_args()

    files = sorted(glob.glob(args.input))
    if not files:
        sys.exit(f"没有匹配到文件：{args.input}")

    data_root = Path(args.data_root) if args.data_root else DEFAULT_DATA_ROOT
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with args.output.open("w", encoding="utf-8") as fout:
        for file in files:
            path = Path(file)
            records = iter_records(path, args.json_key)
            subset = args.subset or path.stem
            written = 0
            for i, rec in enumerate(records):
                if args.limit is not None and i >= args.limit:
                    break
                img = pick(rec, IMAGE_KEYS, args.image_key)
                pos = pick(rec, POS_KEYS, args.positive_key)
                neg = pick(rec, NEG_KEYS, args.negative_key)
                if not (img and pos and neg):
                    print(f"[{path.name}] 无法识别字段，首条记录键：{list(rec)[:12]}")
                    sys.exit("请用 --image-key / --positive-key / --negative-key 显式指定")
                image = img[1] if args.image_root is None else str(args.image_root / Path(img[1]).name)
                positive, negative = pos[1], neg[1]
                if args.label_key and args.label_key in rec:
                    # label 指明哪一条才是正确描述：0 -> caption_0 正确
                    if int(rec[args.label_key]) == 1:
                        positive, negative = negative, positive
                fout.write(json.dumps({
                    "image": image,
                    "positive": positive,
                    "negative": negative,
                    "subset": subset,
                }, ensure_ascii=False) + "\n")
                written += 1
            total += written
            print(f"[{path.name}] subset={subset} 写入 {written} 条")

    print(f"\n完成：{total} 条 -> {args.output}")
    print("评测：python evals-mm/run_eval.py --tag e0 --benchmark sugarcrepe（aro 同理）")


if __name__ == "__main__":
    main()
