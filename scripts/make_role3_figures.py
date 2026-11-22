"""Generate role-3 result figures for report chapter 6."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "report" / "results"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 180

NAVY, ORANGE, GREY, GREEN, RED = "#1F3A5F", "#E08A3C", "#8A94A6", "#3F7D5B", "#B64B4B"
TAGS = ["e0", "e1", "e2", "e2a", "e3", "e4b"]
LABELS = {"e0": "E0", "e1": "E1", "e2": "E2", "e2a": "E2a", "e3": "E3", "e4b": "E4b"}


def read_full():
    with (RES / "full_4of3.csv").open(encoding="utf-8") as f:
        return {r["tag"]: {k: (float(v) if v not in ("", None) else None)
                           for k, v in r.items() if k not in ("tag", "source")}
                for r in csv.DictReader(f)}


def read_paired():
    data = json.loads((RES / "paired_full_4of3.json").read_text(encoding="utf-8"))
    out = {}
    for c in data:
        out[(c["benchmark"], c["tag_a"], c["tag_b"])] = c
    return out


def fig_main(full):
    models = [LABELS[t] for t in TAGS]
    aro = [full[t]["aro4_macro"] * 100 for t in TAGS]
    sugar = [full[t]["sugar3_macro"] * 100 for t in TAGS]
    comp = [full[t]["composite"] * 100 for t in TAGS]
    x = np.arange(len(TAGS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    b1 = ax.bar(x - w, aro, w, label="ARO 4子集", color=NAVY)
    b2 = ax.bar(x, sugar, w, label="Sugar 3家族", color=ORANGE)
    b3 = ax.bar(x + w, comp, w, label="组合均值", color=GREEN)
    for bars in (b1, b2, b3):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                    f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, models)
    ax.set_ylim(0, 95)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("A2 4/3：E0–E4b 组合推理主结果")
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.00))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = FIG / "fig_role3_4of3.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_delta(full, paired):
    aro = paired[("aro", "e0", "e3")]["subsets"]
    aro2 = paired[("aro", "e2", "e3")]["subsets"]
    sugar = paired[("sugarcrepe", "e0", "e3")]["subsets"]
    sugar2 = paired[("sugarcrepe", "e2", "e3")]["subsets"]
    aro_names = ["vg_relation", "vg_attribution", "coco_order", "flickr_order"]
    aro_labels = ["VG-Rel", "VG-Attr", "COCO-Order", "Flickr-Order"]
    sugar_names = ["replace", "swap", "add"]
    sugar_labels = ["replace", "swap", "add"]

    def vals(d, names):
        # d.diff is tag_a - tag_b; negate to show E3 - baseline
        return [-d[n]["diff"] * 100 for n in names]

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2))
    ax = axes[0, 0]
    x = np.arange(len(aro_names))
    w = 0.36
    ax.bar(x - w / 2, vals(aro, aro_names), w, label="E3 − E0", color=NAVY)
    ax.bar(x + w / 2, vals(aro2, aro_names), w, label="E3 − E2", color=ORANGE)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, aro_labels, rotation=15, ha="right")
    ax.set_ylabel("Δ accuracy (pp)")
    ax.set_title("ARO 子集")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[0, 1]
    x = np.arange(len(sugar_names))
    ax.bar(x - w / 2, vals(sugar, sugar_names), w, label="E3 − E0", color=NAVY)
    ax.bar(x + w / 2, vals(sugar2, sugar_names), w, label="E3 − E2", color=ORANGE)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, sugar_labels)
    ax.set_ylabel("Δ accuracy (pp)")
    ax.set_title("SugarCrepe 子类")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 0]
    components = ["ARO 4子集", "Sugar 3家族", "组合均值", "COCO t2i"]
    e0 = full["e0"]
    e3 = full["e3"]
    e2 = full["e2"]
    deltas_e0 = [
        (e3["aro4_macro"] - e0["aro4_macro"]) * 100,
        (e3["sugar3_macro"] - e0["sugar3_macro"]) * 100,
        (e3["composite"] - e0["composite"]) * 100,
        (e3["coco_t2i_R@1"] - e0["coco_t2i_R@1"]) * 100,
    ]
    deltas_e2 = [
        (e3["aro4_macro"] - e2["aro4_macro"]) * 100,
        (e3["sugar3_macro"] - e2["sugar3_macro"]) * 100,
        (e3["composite"] - e2["composite"]) * 100,
        None,
    ]
    x = np.arange(len(components))
    ax.bar(x - w / 2, deltas_e0, w, label="E3 − E0", color=NAVY)
    ax.bar(x + w / 2, [v if v is not None else 0 for v in deltas_e2], w,
           label="E3 − E2", color=ORANGE)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, components, rotation=12, ha="right")
    ax.set_ylabel("Δ (pp)")
    ax.set_title("E3 相对基线（E2 COCO 未测）")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 1]
    labels = ["E0", "E3", "E4b"]
    values = [full[t]["coco_t2i_R@1"] * 100 for t in ("e0", "e3", "e4b")]
    ax.bar(labels, values, color=[GREY, NAVY, GREEN], width=0.5)
    for i, v in enumerate(values):
        ax.text(i, v + 0.6, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylim(30, 45)
    ax.set_ylabel("COCO t2i R@1 (%)")
    ax.set_title("COCO 检索")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("E3/E4b 的收益与代价", fontsize=12)
    fig.tight_layout()
    out = FIG / "fig_role3_delta.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    full = read_full()
    paired = read_paired()
    fig_main(full)
    fig_delta(full, paired)


if __name__ == "__main__":
    main()
