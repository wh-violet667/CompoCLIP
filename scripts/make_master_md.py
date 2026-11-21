"""Render report/results/master.md from master.csv (available values only)."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = list(csv.DictReader((ROOT / "report/results/master.csv").open(encoding="utf-8")))
vals = {(r["tag"], r["benchmark"], r["subset"], r["metric"]): r["value"] for r in rows}
tags = ["e0", "e1", "e2", "e2a", "e3", "e4b"]


def get(tag, bench, subset, metric):
    value = vals.get((tag, bench, subset, metric), "—")
    if value == "—":
        return value
    try:
        return f"{float(value) * 100:.3f}%"
    except ValueError:
        return value


lines = [
    "# 结果总表（当前可用数据）",
    "",
    "> 完整口径见 `report/06_method.md`；本表 ARO 为共同 VG-Relation + VG-Attribution 两子集；"
    "Sugar 为 replace + swap。E3/E4b 的 ARO、检索、COCO 已由 T4 补齐；"
    "E1/E2/E2a 的 COCO 未同批次评测，用“—”表示。A2 4/3 全量表见 `full_4of3.md`。",
    "",
    "| 模型 | ARO macro | SugarCrepe macro | Flickr30k t2i R@1 | Winoground group | COCO t2i R@1 |",
    "|---|---:|---:|---:|---:|---:|",
]
for t in tags:
    lines.append(
        f"| {t} | {get(t, 'aro', 'macro', 'aro_accuracy')} | "
        f"{get(t, 'sugarcrepe', 'macro', 'sugarcrepe_accuracy')} | "
        f"{get(t, 'flickr30k', '', 't2i_R@1')} | "
        f"{get(t, 'winoground', '', 'wg_group')} | "
        f"{get(t, 'coco', '', 't2i_R@1')} |"
    )
lines += ["", "> E4a = E2。E0 的 Flickr30k/COCO/Winoground 来自 A2/A1 已核验归档；其他未完成项用“—”表示，不填猜测值。"]
(ROOT / "report/results/master.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
