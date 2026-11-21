"""Render the final report-level result summary from CSV artifacts."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "report" / "results"
OUT = ROOT / "report" / "results_table.md"
TAGS = ("e0", "e1", "e2", "e2a", "e3", "e4b")
DISPLAY = {"e0": "E0", "e1": "E1", "e2": "E2", "e2a": "E2a", "e3": "E3", "e4b": "E4b"}


def pct(value: str | float | None) -> str:
    if value in (None, "", "nan"):
        return "—"
    return f"{float(value) * 100:.3f}%"


def main() -> None:
    full = {r["tag"]: r for r in csv.DictReader((RESULTS / "full_4of3.csv").open(encoding="utf-8"))}
    master = {(r["tag"], r["benchmark"], r["subset"], r["metric"]): r["value"]
              for r in csv.DictReader((RESULTS / "master.csv").open(encoding="utf-8"))}

    lines = [
        "# 结果汇总（最终）",
        "",
        "> 自动生成，数据源：`report/results/full_4of3.csv`、`master.csv`。"
        "详细配对统计与失败案例见 `report/results/paired_full_4of3.md`、"
        "`report/results/paired_stats.md`、`report/results/failure_cases.md`。",
        "",
        "## A2 4/3 全量",
        "",
        "| 模型 | ARO 4子集 macro | SugarCrepe 3家族 macro | 组合均值 | COCO t2i R@1 | COCO i2t R@1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for tag in TAGS:
        r = full.get(tag, {})
        lines.append(
            f"| {DISPLAY[tag]} | {pct(r.get('aro4_macro'))} | {pct(r.get('sugar3_macro'))} | "
            f"{pct(r.get('composite'))} | {pct(r.get('coco_t2i_R@1'))} | {pct(r.get('coco_i2t_R@1'))} |"
        )

    lines += [
        "",
        "## 共同 2/2（E0 也可对齐）",
        "",
        "| 模型 | ARO macro | SugarCrepe macro |",
        "|---|---:|---:|",
    ]
    for tag in TAGS:
        lines.append(
            f"| {DISPLAY[tag]} | {pct(master.get((tag, 'aro', 'macro', 'aro_accuracy')))} | "
            f"{pct(master.get((tag, 'sugarcrepe', 'macro', 'sugarcrepe_accuracy')))} |"
        )

    lines += [
        "",
        "## 检索与 Winoground",
        "",
        "| 模型 | Flickr30k t2i R@1 | Flickr30k i2t R@1 | Winoground group | COCO t2i R@1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for tag in TAGS:
        lines.append(
            f"| {DISPLAY[tag]} | {pct(master.get((tag, 'flickr30k', '', 't2i_R@1')))} | "
            f"{pct(master.get((tag, 'flickr30k', '', 'i2t_R@1')))} | "
            f"{pct(master.get((tag, 'winoground', '', 'wg_group')))} | "
            f"{pct(master.get((tag, 'coco', '', 't2i_R@1')))} |"
        )

    lines += [
        "",
        "> E4a = E2。E1/E2/E2a 的 COCO 没有在同一批次评测；E2 的 COCO 缺失，"
        "因此“E3 的 COCO 遗忘比 E-best 更小”仍未验证。",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
