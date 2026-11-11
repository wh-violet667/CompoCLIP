"""0-C 失败模式诊断：把 E0 的逐样本错误按语言学类型分桶。

输入：evals-mm/run_eval.py 产出的 results/{tag}_{bench}_persample.npz
输出：
    0c_taxonomy.csv    每个 benchmark × 类型 的 n / 错误率 / 平均 margin / 近失手比例
    0c_summary.json    汇总 + Winoground 的 text/image/both 分解
    0c_examples.md    每类 4 条典型错误（供报告第 4 章直接引用）

分桶规则（显式、可审计）：
    order     子集名含 order/swap，或正负句词袋完全相同（Counter 相等）
    attribute 子集名含 att/attr，或两侧形容词集合不同
    relation  子集名含 rel
    object    子集名含 obj
    add       子集名含 add
    other     其余

用法：
    python scripts/diagnose_failures.py --inputs evals-mm/results/e0_*_persample.npz
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evals-mm"))


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def bag(text: str) -> Counter:
    return Counter(words(text))


def classify(pos: str, neg: str, subset: str, fine: str = "") -> str:
    s = f"{subset} {fine}".lower()
    if "order" in s:
        return "order"
    if "attr" in s or "att" in s:
        return "attribute"
    if "rel" in s:
        return "relation"
    if "obj" in s:
        return "object"
    if "add" in s:
        return "add"
    if "swap" in s:
        return "order"
    if bag(pos) == bag(neg) and len(words(pos)) >= 3:
        return "order"
    return "other"


def summarize_pairwise(npz_path: Path) -> tuple[list[dict], dict, list[dict]]:
    d = np.load(npz_path)
    pos = d["positive"].astype(str)
    neg = d["negative"].astype(str)
    subset = d["subset"].astype(str)
    fine = d["fine"].astype(str) if "fine" in d.files else subset
    correct = d["correct"].astype(bool)
    margin = d["pos_score"] - d["neg_score"]

    cats = np.array([classify(p, n, s, f) for p, n, s, f in zip(pos, neg, subset, fine)])
    rows, examples = [], []
    for cat in sorted(set(cats)):
        sel = cats == cat
        m = margin[sel]
        rows.append({
            "benchmark": npz_path.stem.split("_")[1] if "_" in npz_path.stem else npz_path.stem,
            "category": cat,
            "n": int(sel.sum()),
            "errors": int((~correct[sel]).sum()),
            "error_rate": float((~correct[sel]).mean()),
            "mean_margin": float(m.mean()),
            "near_miss_share": float(((m <= 0) & (m > -0.01)).sum() / max(1, (~correct[sel]).sum())),
            "confident_error_share": float(((m <= -0.05).sum()) / max(1, (~correct[sel]).sum())),
        })
        wrong = np.where(sel & (~correct))[0]
        wrong = wrong[np.argsort(margin[wrong])][:4]
        for i in wrong:
            examples.append({
                "benchmark": rows[-1]["benchmark"], "category": cat,
                "image": str(d["image"][i]), "positive": pos[i], "negative": neg[i],
                "margin": float(margin[i]),
            })
    info = {
        "n": int(len(pos)),
        "accuracy": float(correct.mean()),
        "categories": {r["category"]: r for r in rows},
    }
    return rows, info, examples


def summarize_winoground(npz_path: Path) -> tuple[list[dict], dict, list[dict]]:
    d = np.load(npz_path)
    tc = d["text_correct"].astype(bool)
    ic = d["image_correct"].astype(bool)
    gc = tc & ic
    tags = d["collapsed_tag"].astype(str)
    rows = []
    for tag in sorted(set(tags)):
        sel = tags == tag
        rows.append({
            "benchmark": "winoground", "category": tag, "n": int(sel.sum()),
            "text_acc": float(tc[sel].mean()), "image_acc": float(ic[sel].mean()),
            "group_acc": float(gc[sel].mean()),
            "text_only_fail": int((~tc[sel] & ic[sel]).sum()),
            "image_only_fail": int((tc[sel] & ~ic[sel]).sum()),
            "both_fail": int((~tc[sel] & ~ic[sel]).sum()),
        })
    info = {
        "n": int(len(tc)), "text_acc": float(tc.mean()), "image_acc": float(ic.mean()),
        "group_acc": float(gc.mean()),
        "text_only_fail": int((~tc & ic).sum()),
        "image_only_fail": int((tc & ~ic).sum()),
        "both_fail": int((~tc & ~ic).sum()),
    }
    # 典型 both-fail 样本
    wrong = np.where(~gc)[0][:8]
    examples = [{
        "benchmark": "winoground", "category": "both_fail" if (~tc & ~ic)[i] else "partial",
        "image": str(d["group_id"][i]),
        "positive": d["caption_0"][i], "negative": d["caption_1"][i],
        "margin": float("nan"),
    } for i in wrong]
    return rows, info, examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default=str(ROOT / "evals-mm/results/e0_*_persample.npz"))
    parser.add_argument("--out-dir", type=Path, default=ROOT / "evals-mm/results")
    args = parser.parse_args()

    files = sorted(glob.glob(args.inputs))
    if not files:
        raise SystemExit(f"没有找到逐样本文件：{args.inputs}")

    all_rows, summary, examples = [], {}, []
    for f in files:
        path = Path(f)
        with np.load(path) as d:
            is_wg = "text_correct" in d.files
        if is_wg:
            rows, info, ex = summarize_winoground(path)
        else:
            rows, info, ex = summarize_pairwise(path)
        key = "winoground" if is_wg else path.stem.replace("e0_", "").replace("_persample", "")
        summary[key] = info
        all_rows.extend(rows)
        examples.extend(ex)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "0c_taxonomy.csv").open("w", newline="", encoding="utf-8") as f:
        fields = sorted({k for r in all_rows for k in r})
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    (args.out_dir / "0c_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 0-C 典型失败样本（自动抽取）", ""]
    for e in examples:
        margin = "n/a" if e["margin"] != e["margin"] else f"{e['margin']:+.4f}"
        lines += [
            f"## {e['benchmark']} / {e['category']} / margin {margin}",
            f"- image: `{e['image']}`",
            f"- positive: {e['positive']}",
            f"- negative: {e['negative']}",
            "",
        ]
    (args.out_dir / "0c_examples.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"完成：{len(files)} 个 benchmark -> {args.out_dir / '0c_taxonomy.csv'}")
    for k, v in summary.items():
        if "accuracy" in v:
            print(f"  {k}: n={v['n']} acc={v['accuracy']:.3f}")
        else:
            print(f"  {k}: n={v['n']} text={v['text_acc']:.3f} image={v['image_acc']:.3f} group={v['group_acc']:.3f}")


if __name__ == "__main__":
    main()
