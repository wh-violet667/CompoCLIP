"""硬负例 100 条人工质检表（成员 1，D6 交付物）。

自动检查：
- 词多重集是否与正例完全一致（硬约束）；
- 长度比（负例词数 / 正例词数）；
- 是否与正例重复；
- 是否只改了标点/大小写（近似无效）。

输出：
- `hardneg_qc_100.csv`：含 `human_ok` 空列，人工逐条填 1/0；
- `hardneg_qc_100.md`：按类别分组，方便直接贴进报告附录。

用法：
    python scripts/sample_hardneg_100.py --input data/hardneg/llm_generated.jsonl --n 100
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/hardneg"))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < args.n:
        print(f"警告：只有 {len(rows)} 条，少于 {args.n}；将全量质检。")
    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for i, r in enumerate(sample, 1):
        cap, neg = r["caption"].strip(), r["negative"].strip()
        wc, wn = Counter(words(cap)), Counter(words(neg))
        ratio = len(words(neg)) / max(1, len(words(cap)))
        out_rows.append({
            "id": i,
            "image": r.get("image", ""),
            "positive": cap,
            "negative": neg,
            "word_multiset_ok": int(wc == wn),
            "length_ratio": round(ratio, 3),
            "identical": int(cap.lower() == neg.lower()),
            "near_identical": int(cap.lower().replace(" ", "") == neg.lower().replace(" ", "")),
            "human_ok": "",
            "human_note": "",
        })

    csv_path = args.output_dir / "hardneg_qc_100.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    md = [f"# 硬负例质检（n={len(out_rows)}）", "",
          "人工列 `human_ok`：1=可用，0=不可用；`human_note` 写原因（语义没变/语法崩坏/指代错误等）。", ""]
    for r in out_rows:
        md += [
            f"## {r['id']}. {r['image']}",
            f"- 正例：{r['positive']}",
            f"- 负例：{r['negative']}",
            f"- 自动检查：词袋={'OK' if r['word_multiset_ok'] else 'FAIL'}；长度比={r['length_ratio']}；重复={r['identical']}",
            "- 人工：☐ 可用  ☐ 不可用；备注：",
            "",
        ]
    (args.output_dir / "hardneg_qc_100.md").write_text("\n".join(md), encoding="utf-8")

    auto_pass = sum(r["word_multiset_ok"] and not r["identical"] for r in out_rows)
    print(f"抽样 {len(out_rows)} 条 -> {csv_path}")
    print(f"自动检查通过 {auto_pass}/{len(out_rows)}（{auto_pass/len(out_rows):.1%}）；还需人工填 human_ok。")


if __name__ == "__main__":
    main()
