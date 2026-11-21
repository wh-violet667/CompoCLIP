"""Build 30 failure cases from E3 per-sample predictions (ARO + SugarCrepe)."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "results" / "failure_cases.md"


def load(path: Path):
    d = np.load(path, allow_pickle=True)
    return d


def pick(path: Path, n: int):
    d = load(path)
    correct = d["correct"].astype(bool)
    pos = d["pos_score"].astype(float)
    neg = d["neg_score"].astype(float)
    margin = pos - neg
    wrong = np.where(~correct)[0]
    # smallest margin first = closest misses
    order = wrong[np.argsort(margin[wrong])][:n]
    rows = []
    for i in order:
        rows.append({
            "image": Path(str(d["image"][i])).name,
            "subset": str(d["subset"][i]),
            "positive": str(d["positive"][i]),
            "negative": str(d["negative"][i]),
            "pos_score": float(pos[i]),
            "neg_score": float(neg[i]),
            "margin": float(margin[i]),
        })
    return rows


def main() -> None:
    aro = pick(ROOT / "evals-mm" / "results" / "e3_aro_persample.npz", 15)
    sugar = pick(ROOT / "evals-mm" / "results" / "e3_sugarcrepe_persample.npz", 15)
    lines = ["# E3 失败案例（30 条）", "",
             "> 选择规则：E3 预测错误中 margin 最小的样本；ARO 15 条 + SugarCrepe 15 条。",
             "> 图片名只保留 basename；完整路径见对应的 golden JSONL。", ""]
    for title, rows in (("ARO", aro), ("SugarCrepe", sugar)):
        lines += [f"## {title}", "",
                  "| # | image | subset | positive | negative | pos | neg | margin |",
                  "|---:|---|---|---|---|---:|---:|---:|"]
        for i, r in enumerate(rows, 1):
            lines.append(
                f"| {i} | {r['image']} | {r['subset']} | {r['positive'][:80]} | "
                f"{r['negative'][:80]} | {r['pos_score']:.4f} | {r['neg_score']:.4f} | {r['margin']:+.4f} |"
            )
        lines.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
