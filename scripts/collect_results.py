"""汇总 evals-mm/results/*.csv 为报告用 Markdown 表（统稿工具）。

用法：
    python scripts/collect_results.py
    python scripts/collect_results.py --tag e0
"""
from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def fmt(x: str | float, nd=3) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "" if x is None else str(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="e0")
    parser.add_argument("--out", type=Path, default=ROOT / "report/results_table.md")
    args = parser.parse_args()

    rows_by_key: dict[tuple[str, str, str], dict] = {}
    for f in sorted(glob.glob(str(ROOT / "evals-mm/results/*.csv"))):
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if "benchmark" not in r or not r.get("tag"):
                    continue
                if args.tag and r.get("tag") != args.tag:
                    continue
                rows_by_key[(r["benchmark"], r.get("subset") or "", r["tag"])] = r
    rows = list(rows_by_key.values())

    lines = ["# 结果汇总（自动生成）", "",
             "> 数据来源：`evals-mm/results/*.csv`；同 tag/limit 才可横向比较。", ""]
    by_bench: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_bench[r["benchmark"]].append(r)

    for bench, rs in by_bench.items():
        lines += [f"## {bench}", ""]
        if "wg_group" in rs[0]:
            lines += ["| tag | text | image | group | n |", "|---|---|---|---|---|"]
            for r in rs:
                lines.append(f"| {r['tag']} | {fmt(r['wg_text'])} | {fmt(r['wg_image'])} | "
                             f"{fmt(r['wg_group'])} | {r.get('n_pairs','')} |")
        else:
            lines += ["| tag | subset | n | accuracy | 95% CI |", "|---|---|---|---|---|"]
            for r in rs:
                ci = ""
                lo, hi = r.get("accuracy_ci_low"), r.get("accuracy_ci_high")
                if lo not in (None, "", "nan") and hi not in (None, "", "nan"):
                    ci = f"[{fmt(lo)}, {fmt(hi)}]"
                lines.append(f"| {r['tag']} | {r.get('subset','')} | {r.get('n','')} | "
                             f"{fmt(r.get('accuracy'))} | {ci} |")
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(rows)} 行 -> {args.out}")


if __name__ == "__main__":
    main()
