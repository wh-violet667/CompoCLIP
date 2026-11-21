"""Full A2 4/3 paired statistics for E0/E2/E3/E4b.

Complements assemble_role3_results.py, which only tests the common 2/2 subset.
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "report" / "results"
ARO_SUBSETS = ("vg_relation", "vg_attribution", "coco_order", "flickr_order")
SUGAR_FAMILIES = ("replace", "swap", "add")
COMPARISONS = (("e0", "e3"), ("e2", "e3"), ("e2", "e4b"))


def load(tag: str, bench: str):
    path = ROOT / "A2_eval" / "results" / tag / f"{bench}_persample.npz"
    data = np.load(path, allow_pickle=True)
    return np.asarray(data["subsets"]), np.asarray(data["correct"], dtype=bool)


def paired(bench: str, tag_a: str, tag_b: str, wanted, n_boot: int = 1000, seed: int = 0):
    sa, ca = load(tag_a, bench)
    sb, cb = load(tag_b, bench)
    if not np.array_equal(sa, sb) or len(ca) != len(cb):
        raise ValueError(f"{tag_a}/{tag_b} {bench}: subset order mismatch")
    per_subset = {}
    diffs = []
    for name in wanted:
        sel = sa == name
        a, b = ca[sel].astype(float), cb[sel].astype(float)
        diff = float(a.mean() - b.mean())
        diffs.append(diff)
        rng = np.random.default_rng(seed + len(name))
        idx = rng.integers(0, len(a), size=(n_boot, len(a)))
        boot = a[idx].mean(axis=1) - b[idx].mean(axis=1)
        b01 = int(np.sum((~a.astype(bool)) & b.astype(bool)))
        b10 = int(np.sum(a.astype(bool) & (~b.astype(bool))))
        if b01 + b10 == 0:
            p = 1.0
        else:
            k = min(b01, b10)
            p = min(1.0, 2 * sum(comb(b01 + b10, i) for i in range(k + 1)) / (2 ** (b01 + b10)))
        per_subset[name] = {
            "n": int(sel.sum()), "acc_a": float(a.mean()), "acc_b": float(b.mean()),
            "diff": diff, "ci_low": float(np.quantile(boot, 0.025)),
            "ci_high": float(np.quantile(boot, 0.975)),
            "mcnemar_b01": b01, "mcnemar_b10": b10, "mcnemar_p": p,
        }
    rng = np.random.default_rng(seed + 999)
    boot_macro = []
    for _ in range(n_boot):
        vals = []
        for name in wanted:
            sel = sa == name
            a, b = ca[sel].astype(float), cb[sel].astype(float)
            idx = rng.integers(0, len(a), size=len(a))
            vals.append(a[idx].mean() - b[idx].mean())
        boot_macro.append(float(np.mean(vals)))
    return {
        "benchmark": bench, "tag_a": tag_a, "tag_b": tag_b,
        "n": int(len(ca)), "diff_macro": float(np.mean(diffs)),
        "ci_low": float(np.quantile(boot_macro, 0.025)),
        "ci_high": float(np.quantile(boot_macro, 0.975)),
        "subsets": per_subset,
    }


def main() -> None:
    out = []
    for bench, wanted in (("aro", ARO_SUBSETS), ("sugarcrepe", SUGAR_FAMILIES)):
        for a, b in COMPARISONS:
            out.append(paired(bench, a, b, wanted))
    (RESULTS / "paired_full_4of3.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Full A2 4/3 paired statistics", "",
             "> diff = tag_a − tag_b；负值表示 tag_b 更高。", ""]
    for c in out:
        lines += [f"## {c['benchmark']}: {c['tag_a']} vs {c['tag_b']}", ""]
        lines.append(f"- macro diff: {c['diff_macro']*100:+.3f} pp, "
                     f"95% CI [{c['ci_low']*100:+.3f}, {c['ci_high']*100:+.3f}] pp, n={c['n']}")
        for name, s in c["subsets"].items():
            lines.append(f"- {name}: {s['acc_a']*100:.3f}% vs {s['acc_b']*100:.3f}%, "
                         f"diff {s['diff']*100:+.3f} pp, "
                         f"95% CI [{s['ci_low']*100:+.3f}, {s['ci_high']*100:+.3f}], "
                         f"McNemar p={s['mcnemar_p']:.3g}")
        lines.append("")
    (RESULTS / "paired_full_4of3.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {RESULTS / 'paired_full_4of3.json'} and .md")


if __name__ == "__main__":
    main()
