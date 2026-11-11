"""生成报告/PPT 用图（读取 evals-mm/results 下的 CSV/NPZ）。

用法：
    python scripts/make_figures.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "evals-mm/results"
OUT = ROOT / "report/figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 160
NAVY, ORANGE, GREY, GREEN = "#1F3A5F", "#E08A3C", "#8A94A6", "#3F7D5B"


def read_csv(name: str) -> list[dict]:
    p = RES / name
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def latest(rows: list[dict], tag="e0") -> list[dict]:
    out: dict[tuple[str, str], dict] = {}
    for r in rows:
        if r.get("tag") != tag:
            continue
        out[(r.get("benchmark", ""), r.get("subset", ""))] = r
    return list(out.values())


def fig_e0():
    wg = latest(read_csv("winoground.csv"))
    aro = [r for r in latest(read_csv("aro.csv")) if r.get("subset") in ("vg_relation", "vg_attribution", "flickr_order", "macro")]
    sc = [r for r in latest(read_csv("sugarcrepe.csv")) if r.get("subset") in ("replace", "swap", "macro")]

    panels = []
    if wg:
        r = wg[-1]
        panels.append(("Winoground", ["text", "image", "group"],
                       [float(r["wg_text"]) * 100, float(r["wg_image"]) * 100, float(r["wg_group"]) * 100],
                       [50, 50, 16.7]))
    if aro:
        labels = {"vg_relation": "VG-Rel", "vg_attribution": "VG-Att", "flickr_order": "Flickr-Order", "macro": "宏平均"}
        panels.append(("ARO（3 子集）", [labels.get(r["subset"], r["subset"]) for r in aro],
                       [float(r["accuracy"]) * 100 for r in aro], [50] * len(aro)))
    if sc:
        labels = {"replace": "replace", "swap": "swap", "macro": "宏平均"}
        panels.append(("SugarCrepe++", [labels.get(r["subset"], r["subset"]) for r in sc],
                       [float(r["accuracy"]) * 100 for r in sc], [50] * len(sc)))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, labels, vals, baselines) in zip(axes, panels):
        colors = [ORANGE if v >= b else GREY for v, b in zip(vals, baselines)]
        bars = ax.bar(labels, vals, color=colors, width=0.62)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", fontsize=9)
        for y in set(baselines):
            ax.axhline(y, color=NAVY, linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=11, color=NAVY, fontweight="bold")
        ax.set_ylabel("准确率 (%)")
        ax.tick_params(axis="x", labelsize=8.5)
    fig.suptitle("E0：冻结 CLIP ViT-B/32 的组合性基线（虚线为随机水平）", color=NAVY, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_e0.png", bbox_inches="tight")
    plt.close(fig)


def fig_0b():
    p = RES / "0b_probe.csv"
    if not p.exists():
        return
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    rows = [r for r in rows if r["probe"] != "unrelated"]
    rows.sort(key=lambda r: float(r["accuracy"]))
    labels = [r["probe"] for r in rows]
    vals = [float(r["accuracy"]) * 100 for r in rows]
    lo = [(float(r["accuracy"]) - float(r["ci_low"])) * 100 for r in rows]
    hi = [(float(r["ci_high"]) - float(r["accuracy"])) * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bars = ax.barh(labels, vals, xerr=[lo, hi], color=ORANGE, height=0.6, capsize=3)
    for b, v, e in zip(bars, vals, hi):
        ax.text(v + e + 0.8, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", fontsize=9)
    ax.axvline(50, color=NAVY, linestyle="--", linewidth=0.9)
    ax.set_xlim(40, 80)
    ax.set_xlabel("准确率 (%)，随机=50%")
    ax.set_title("0-B：保持词袋、只改词序（Flickr30k test）", color=NAVY, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_0b.png", bbox_inches="tight")
    plt.close(fig)


def fig_0c():
    p = RES / "0c_taxonomy.csv"
    if not p.exists():
        return
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    pairs = []
    for r in rows:
        if r["benchmark"] == "winoground":
            pairs.append((f"WG-{r['category']}", (1 - float(r["group_acc"])) * 100))
        elif "error_rate" in r and r["error_rate"]:
            pairs.append((f"{r['benchmark'][:5]}-{r['category']}", float(r["error_rate"]) * 100))
    if not pairs:
        return
    pairs.sort(key=lambda x: x[1])
    labels, vals = zip(*pairs)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bars = ax.barh(labels, vals, color=[GREY if v < 50 else ORANGE for v in vals], height=0.62)
    for b, v in zip(bars, vals):
        ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", fontsize=8.5)
    ax.set_xlim(0, 105)
    ax.set_xlabel("错误率 (%)")
    ax.set_title("0-C：按语言类型的错误率（WG 为 group 错误）", color=NAVY, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_0c.png", bbox_inches="tight")
    plt.close(fig)


def fig_margin():
    p = RES / "e0_sugarcrepe_persample.npz"
    if not p.exists():
        return
    d = np.load(p)
    margin = d["pos_score"] - d["neg_score"]
    correct = d["correct"].astype(bool)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.hist(margin[correct], bins=80, alpha=0.65, color=GREEN, label="正确")
    ax.hist(margin[~correct], bins=80, alpha=0.65, color=ORANGE, label="错误")
    ax.axvline(0, color=NAVY, linewidth=1)
    ax.axvline(-0.01, color=NAVY, linewidth=0.8, linestyle="--")
    ax.set_xlim(-0.08, 0.12)
    ax.set_xlabel("margin = sim(正例) − sim(负例)")
    ax.set_ylabel("样本数")
    ax.set_title("SugarCrepe++：错误集中在 0 附近（近失手）", color=NAVY, fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_margin.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_e0()
    fig_0b()
    fig_0c()
    fig_margin()
    print(f"图 -> {OUT}")
