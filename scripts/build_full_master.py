"""Merge common ARO/Sugar + retrieval/COCO into report/results/master.csv."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "report" / "results"
TAGS = {"e0", "e1", "e2", "e2a", "e3", "e4b"}
FIELDS = ["tag", "benchmark", "subset", "metric", "value", "n", "source"]


def add(rows: dict, tag: str, bench: str, subset: str, metric: str, value, n, source: str):
    if value in ("", None):
        return
    rows[(tag, bench, subset, metric)] = {
        "tag": tag, "benchmark": bench, "subset": subset,
        "metric": metric, "value": str(value), "n": str(n), "source": source,
    }


def main() -> None:
    rows: dict[tuple, dict] = {}

    # Keep already archived E0 retrieval/COCO/Winoground rows.
    old = RESULTS / "master.csv"
    if old.exists():
        for r in csv.DictReader(old.open(encoding="utf-8")):
            if r.get("tag") == "e0" and r.get("benchmark") in {"flickr30k", "coco", "winoground"}:
                add(rows, r["tag"], r["benchmark"], r.get("subset", ""), r["metric"],
                    r["value"], r.get("n", ""), r.get("source", "archive"))

    # Common ARO/Sugar table.
    common = RESULTS / "master_common.csv"
    if common.exists():
        for r in csv.DictReader(common.open(encoding="utf-8")):
            add(rows, r["tag"], r["benchmark"], r["subset"], r["benchmark"] + "_accuracy",
                r["accuracy"], r["n"], "common 2/4 ARO + 2/3 Sugar")

    # Per-benchmark retrieval CSVs from evals-mm.
    for bench in ("flickr30k", "winoground", "coco"):
        path = ROOT / "evals-mm" / "results" / f"{bench}.csv"
        if not path.exists():
            continue
        for r in csv.DictReader(path.open(encoding="utf-8")):
            tag = r.get("tag", "")
            if tag not in TAGS or str(r.get("limit", "")).strip():
                continue
            if bench == "flickr30k":
                add(rows, tag, bench, "", "t2i_R@1", r.get("t2i_R@1"), r.get("n_text_queries", ""), "evals-mm")
                add(rows, tag, bench, "", "i2t_R@1", r.get("i2t_R@1"), r.get("n_image_queries", ""), "evals-mm")
            elif bench == "winoground":
                for metric in ("wg_text", "wg_image", "wg_group"):
                    add(rows, tag, bench, "", metric, r.get(metric), r.get("n_pairs", ""), "evals-mm")
            else:
                add(rows, tag, bench, "", "t2i_R@1", r.get("t2i_R@1"), r.get("n_text_queries", ""), "evals-mm")
                add(rows, tag, bench, "", "i2t_R@1", r.get("i2t_R@1"), r.get("n_image_queries", ""), "evals-mm")

    # COCO from A2_eval evaluation.json (E3/E4b).
    for tag in ("e3", "e4b"):
        path = ROOT / "A2_eval" / "results" / tag / "evaluation.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        coco = data.get("benchmarks", {}).get("coco")
        if coco:
            add(rows, tag, "coco", "", "t2i_R@1", coco.get("t2i_R@1"), coco.get("n_text_queries", ""), "A2_eval")
            add(rows, tag, "coco", "", "i2t_R@1", coco.get("i2t_R@1"), coco.get("n_image_queries", ""), "A2_eval")

    # Full A2 4/3 metrics (E1/E2/E2a/E3/E4b; E0 has no full archive).
    full = RESULTS / "full_4of3.csv"
    if full.exists():
        for r in csv.DictReader(full.open(encoding="utf-8")):
            add(rows, r["tag"], "aro", "macro_4", "aro4_macro", r["aro4_macro"], 82695, "A2 4/3 formal")
            add(rows, r["tag"], "sugarcrepe", "macro_3", "sugar3_macro", r["sugar3_macro"], 7511, "A2 4/3 formal")
            add(rows, r["tag"], "combined", "4of3", "composite", r["composite"], "", "A2 4/3 formal")

    out = RESULTS / "master.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows.values())
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
