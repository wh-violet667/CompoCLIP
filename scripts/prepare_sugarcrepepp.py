"""把 SugarCrepe++ 的 examples.jsonl 归一化成评测用 golden。

原始字段（Mayfull/sugarcrepepp_vlms）：
    id, filename, caption, caption2, negative_caption, name
其中：
- caption        = 原始 COCO 正描述（主评测用）
- caption2       = 同义改写正描述（词法鲁棒性检查，不参与主指标）
- negative_caption = 语义/词法扰动后的负描述
- name           = data_{replace,swap,add}_{att,obj,rel}

注意：通用 prepare_pairwise_bench.py 会把 caption2 当成负例（因为 NEG_KEYS 里有 caption2），
这是错的：caption2 是正例改写。因此 SugarCrepe++ 单独用本脚本，避免指标虚高。

用法：
    python scripts/prepare_sugarcrepepp.py --examples data/sugarcrepepp/examples.jsonl \
        --image-root data/sugarcrepepp/images_tmp --output data/golden/sugarcrepe.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAMILIES = ("replace", "swap", "add")


def subset_of(name: str) -> str:
    m = re.search(r"(replace|swap|add)", name)
    return m.group(1) if m else name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=Path, default=ROOT / "data/sugarcrepepp/examples.jsonl")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data/sugarcrepepp/images_tmp")
    parser.add_argument("--output", type=Path, default=ROOT / "data/golden/sugarcrepe.jsonl")
    parser.add_argument("--paraphrase-output", type=Path,
                        default=ROOT / "data/golden/sugarcrepe_paraphrase.jsonl",
                        help="正例用 caption2 的鲁棒性版本；不参与主指标")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.examples.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = [r["filename"] for r in rows if not (args.image_root / r["filename"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} 张图缺失（示例 {missing[:3]}）；先解压 images.zip 到 {args.image_root}")

    counts = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f, \
         args.paraphrase_output.open("w", encoding="utf-8") as fp:
        for r in rows:
            item = {
                "image": str(args.image_root / r["filename"]),
                "positive": r["caption"],
                "negative": r["negative_caption"],
                "subset": subset_of(r["name"]),
                "id": r.get("id"),
                "name": r.get("name"),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            fp.write(json.dumps({
                "image": item["image"],
                "positive": r.get("caption2") or r["caption"],
                "negative": r["negative_caption"],
                "subset": item["subset"],
                "id": r.get("id"),
            }, ensure_ascii=False) + "\n")
            counts[item["subset"]] += 1

    print(f"完成：{len(rows)} 条 -> {args.output}")
    print("子类分布：" + ", ".join(f"{k}={counts[k]}" for k in FAMILIES if k in counts))
    print("宏平均口径：replace / swap / add 三类等权，不按样本数加权。")
    if args.check_only:
        print("check-only：未写文件")


if __name__ == "__main__":
    main()
