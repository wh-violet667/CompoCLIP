"""Assemble role-3 common-subset metrics and paired statistics.

The A2 handoff uses 4 ARO subsets + 3 SugarCrepe families for E-best selection.
For the CPU-side common evaluation we use the archived A1/E0 scope:
ARO = vg_relation + vg_attribution (common across A1/A2); SugarCrepe = replace + swap.
E1/E2/E2a per-sample files are filtered from the A2 4/3 records; E0 comes from
the A1 per-sample files; E3/E4b come from the local CPU evaluation.

Outputs:
    report/results/aro_common.csv
    report/results/sugarcrepe_common.csv
    report/results/master_common.csv
    report/results/paired_stats.json
    report/results/paired_stats.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "report" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

ARO_SUBSETS = ("vg_relation", "vg_attribution")
SUGAR_FAMILIES = ("replace", "swap")
TAGS = ("e0", "e1", "e2", "e2a", "e3", "e4b")


def bootstrap_ci(correct: np.ndarray, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    correct = np.asarray(correct, dtype=np.float64).reshape(-1)
    if correct.size == 0 or n <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = correct[rng.integers(0, correct.size, size=(n, correct.size))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def load_correct(tag: str, bench: str) -> tuple[np.ndarray, np.ndarray]:
    candidates = []
    if tag in ("e1", "e2", "e2a", "e3"):
        candidates.append(ROOT / "A2_eval" / "results" / tag / f"{bench}_persample.npz")
    candidates.append(ROOT / "evals-mm" / "results" / f"{tag}_{bench}_persample.npz")
    path = next((p for p in candidates if p.exists()), candidates[0])
    if not path.exists():
        raise FileNotFoundError(f"missing per-sample file: {path}")
    data = np.load(path, allow_pickle=True)
    subset = data["subsets"] if "subsets" in data.files else data["subset"]
    correct = np.asarray(data["correct"], dtype=bool)
    return np.asarray(subset), correct


def filter_common(tag: str, bench: str):
    subset, correct = load_correct(tag, bench)
    wanted = ARO_SUBSETS if bench == "aro" else SUGAR_FAMILIES
    mask = np.isin(subset, wanted)
    return subset[mask], correct[mask]


def metrics_rows(tag: str, bench: str):
    subset, correct = filter_common(tag, bench)
    rows = []
    accs = []
    for name in (ARO_SUBSETS if bench == "aro" else SUGAR_FAMILIES):
        sel = subset == name
        acc = float(correct[sel].mean())
        lo, hi = bootstrap_ci(correct[sel], seed=len(tag) + len(name))
        accs.append(acc)
        rows.append({"tag": tag, "benchmark": bench, "subset": name, "n": int(sel.sum()),
                     "accuracy": acc, "ci_low": lo, "ci_high": hi})
    rows.append({"tag": tag, "benchmark": bench, "subset": "macro", "n": int(len(correct)),
                 "accuracy": float(np.mean(accs)), "ci_low": float("nan"), "ci_high": float("nan")})
    return rows


def paired_stats(bench: str, tag_a: str, tag_b: str, n_boot: int = 1000, seed: int = 0):
    sa, ca = filter_common(tag_a, bench)
    sb, cb = filter_common(tag_b, bench)
    wanted = ARO_SUBSETS if bench == "aro" else SUGAR_FAMILIES
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
        # two-sided exact McNemar on discordant pairs
        b01 = int(np.sum((~a.astype(bool)) & b.astype(bool)))
        b10 = int(np.sum(a.astype(bool) & (~b.astype(bool))))
        if b01 + b10 == 0:
            p = 1.0
        else:
            from math import comb
            k = min(b01, b10)
            p = min(1.0, 2 * sum(comb(b01 + b10, i) for i in range(k + 1)) / (2 ** (b01 + b10)))
        per_subset[name] = {
            "n": int(sel.sum()), "acc_a": float(a.mean()), "acc_b": float(b.mean()),
            "diff": diff, "ci_low": float(np.quantile(boot, 0.025)),
            "ci_high": float(np.quantile(boot, 0.975)),
            "mcnemar_b01": b01, "mcnemar_b10": b10, "mcnemar_p": p,
        }
    # macro CI: resample each subset independently and average differences
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
    for bench, out_name in (("aro", "aro_common.csv"), ("sugarcrepe", "sugarcrepe_common.csv")):
        rows = []
        for tag in TAGS:
            try:
                rows.extend(metrics_rows(tag, bench))
            except FileNotFoundError as exc:
                print(f"skip {tag}/{bench}: {exc}")
        path = RESULTS / out_name
        import csv
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["tag", "benchmark", "subset", "n", "accuracy", "ci_low", "ci_high"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path} ({len(rows)} rows)")

    # Long master for the common compositional table
    import csv
    master = []
    for bench in ("aro", "sugarcrepe"):
        for tag in TAGS:
            try:
                master.extend(metrics_rows(tag, bench))
            except FileNotFoundError as exc:
                print(f"skip master {tag}/{bench}: {exc}")
    with (RESULTS / "master_common.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["tag", "benchmark", "subset", "n", "accuracy", "ci_low", "ci_high"])
        writer.writeheader()
        writer.writerows(master)
    print(f"wrote {RESULTS / 'master_common.csv'} ({len(master)} rows)")

    comparisons = []
    skipped = []
    for bench in ("aro", "sugarcrepe"):
        for a, b in (("e0", "e3"), ("e2", "e3"), ("e2", "e4b")):
            try:
                comparisons.append(paired_stats(bench, a, b))
            except FileNotFoundError as exc:
                skipped.append(f"{bench}: {a} vs {b}: {exc}")
                print(f"skip paired {bench} {a} vs {b}: {exc}")
    (RESULTS / "paired_stats.json").write_text(json.dumps(comparisons, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Paired statistics (common 2/4 ARO + 2/3 SugarCrepe)", ""]
    for c in comparisons:
        lines.append(f"## {c['benchmark']}: {c['tag_a']} vs {c['tag_b']}")
        lines.append("")
        lines.append(f"- macro diff: {c['diff_macro']*100:+.3f} pp, 95% CI [{c['ci_low']*100:+.3f}, {c['ci_high']*100:+.3f}] pp, n={c['n']}")
        for name, s in c["subsets"].items():
            lines.append(f"- {name}: {s['acc_a']*100:.3f}% vs {s['acc_b']*100:.3f}%, diff {s['diff']*100:+.3f} pp, "
                         f"95% CI [{s['ci_low']*100:+.3f}, {s['ci_high']*100:+.3f}], McNemar p={s['mcnemar_p']:.3g}")
        lines.append("")
    (RESULTS / "paired_stats.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {RESULTS / 'paired_stats.md'}")


if __name__ == "__main__":
    main()
