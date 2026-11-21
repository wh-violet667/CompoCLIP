"""Build 30 failure cases from the formal A2 E3 per-sample results.

15 ARO cases (stratified across the four subsets) and 15 SugarCrepe cases
(stratified across replace/swap/add), selected by the smallest positive-vs-best-
negative margin among wrong predictions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "results" / "failure_cases.md"


def load_rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def pick(bench: str, per_group: int):
    data = np.load(ROOT / "A2_eval" / "results" / "e3" / f"{bench}_persample.npz", allow_pickle=True)
    rows = load_rows(ROOT / "A2_eval" / "data" / f"{bench}.jsonl")
    assert len(rows) == len(data["ids"])
    pos = data["positive_score"].astype(float)
    neg = data["negative_scores"].astype(float)
    max_neg = data["max_negative_score"].astype(float)
    correct = data["correct"].astype(bool)
    subsets = data["subsets"]
    margin = pos - max_neg

    picked = []
    for subset in dict.fromkeys(subsets.tolist()):
        idx = np.where((subsets == subset) & (~correct))[0]
        idx = idx[np.argsort(margin[idx])][:per_group]
        for i in idx:
            scores = neg[i]
            valid = np.where(np.isfinite(scores))[0]
            j = int(valid[np.argmax(scores[valid])]) if valid.size else 0
            row = rows[int(i)]
            negatives = row.get("negatives", [])
            picked.append({
                "image": Path(str(row.get("image", ""))).name,
                "subset": str(subset),
                "fine": str(row.get("fine_subset", "")),
                "positive": str(row.get("positive", "")),
                "negative": str(negatives[j]) if j < len(negatives) else "",
                "pos": float(pos[i]),
                "neg": float(scores[j]),
                "margin": float(margin[i]),
            })
    return picked


def main() -> None:
    aro = pick("aro", 4)[:15]
    sugar = pick("sugarcrepe", 5)[:15]
    lines = [
        "# E3 失败案例（formal A2 4/3，30 条）",
        "",
        "> 数据：`A2_eval/results/e3/{aro,sugarcrepe}_persample.npz`，T4 GPU 正式运行。",
        "> 选择规则：预测错误中 positive − best negative margin 最小的样本；"
        "ARO 按 4 个子集各取 4 条（截断到 15），SugarCrepe 按 replace/swap/add 各取 5 条。",
        "> 图片只保留 basename；完整路径见 `A2_eval/data/*.jsonl`。",
        "",
    ]
    for title, rows in (("ARO", aro), ("SugarCrepe", sugar)):
        lines += [
            f"## {title}",
            "",
            "| # | image | subset | positive | negative | pos | neg | margin |",
            "|---:|---|---|---|---|---:|---:|---:|",
        ]
        for i, r in enumerate(rows, 1):
            subset = r["subset"] + (f" / {r['fine']}" if r["fine"] else "")
            lines.append(
                f"| {i} | {r['image']} | {subset} | {r['positive'][:80]} | "
                f"{r['negative'][:80]} | {r['pos']:.4f} | {r['neg']:.4f} | {r['margin']:+.4f} |"
            )
        lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT} ({len(aro)} ARO + {len(sugar)} SugarCrepe)")


if __name__ == "__main__":
    main()
