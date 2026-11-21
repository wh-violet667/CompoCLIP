"""Build the full A2 4/3 table (E1/E2/E2a/E3/E4b) from evaluation.json files.

E0 does not have a full 4/3 ARO/Sugar archive, so it is left blank here and is
reported separately in the common 2/2 table.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "report" / "results"
TAGS = ("e1", "e2", "e2a", "e3", "e4b")


def main() -> None:
    rows = []
    for tag in TAGS:
        path = ROOT / "A2_eval" / "results" / tag / "evaluation.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        aro = data["benchmarks"]["aro"]["macro_accuracy"]
        sugar = data["benchmarks"]["sugarcrepe"]["macro_accuracy"]
        coco = data["benchmarks"].get("coco", {})
        rows.append({
            "tag": tag,
            "aro4_macro": aro,
            "sugar3_macro": sugar,
            "composite": (aro + sugar) / 2,
            "coco_t2i_R@1": coco.get("t2i_R@1", ""),
            "coco_i2t_R@1": coco.get("i2t_R@1", ""),
            "source": "A2 4/3 formal",
        })

    out = RESULTS / "full_4of3.csv"
    fields = ["tag", "aro4_macro", "sugar3_macro", "composite", "coco_t2i_R@1", "coco_i2t_R@1", "source"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")

    lines = [
        "# A2 4/3 全量结果",
        "",
        "> ARO = 4 子集 macro；SugarCrepe = 3 家族 macro；组合均值 = 两者平均。"
        "E0 没有完整的 4/3 ARO/Sugar 归档，只在共同 2/2 表中报告。",
        "",
        "| 模型 | ARO 4子集 macro | SugarCrepe 3家族 macro | 组合均值 | COCO t2i R@1 | COCO i2t R@1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        def pct(v):
            return f"{v * 100:.3f}%" if isinstance(v, float) else "—"
        display = {"e1": "E1", "e2": "E2", "e2a": "E2a", "e3": "E3", "e4b": "E4b"}[r["tag"]]
        lines.append(
            f"| {display} | {pct(r['aro4_macro'])} | {pct(r['sugar3_macro'])} | "
            f"{pct(r['composite'])} | {pct(r['coco_t2i_R@1'])} | {pct(r['coco_i2t_R@1'])} |"
        )
    lines.append("")
    lines.append(
        "> E2 的组合均值 62.569% 与 B 的 selection score 62.5683% 在舍入误差内一致，"
        "可用于校验本表口径。"
    )
    (RESULTS / "full_4of3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {RESULTS / 'full_4of3.md'}")


if __name__ == "__main__":
    main()
