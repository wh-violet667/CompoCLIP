"""把 run_eval.py 生成的 per-benchmark CSV 汇总成 long-format master.csv。

run_eval 按 benchmark 追加 `flickr30k.csv` / `coco.csv` / ...；本脚本负责：
- 保留 tag / benchmark / subset / weights / seed / limit 等来源信息；
- 把不同 benchmark 的不同指标列展开为 metric/value；
- 标记 `--limit` 产生的冒烟结果（is_smoke）；
- 按 `(tag, benchmark, subset, metric, limit, seed, weights)` 去重，保留最后一次；
- 加 git commit，便于报告/答辩追溯。

用法：
    python evals-mm/collect_results.py
    python evals-mm/collect_results.py --results-dir evals-mm/results --output evals-mm/results/master.csv
"""
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS = ("flickr30k", "coco", "winoground", "aro", "sugarcrepe")

META_COLUMNS = {
    "tag", "benchmark", "subset", "n", "n_text_queries", "n_image_queries", "n_pairs",
    "model", "weights", "checkpoint", "seed", "limit", "bootstrap", "git_commit",
    "config_hash", "ts", "source_file", "is_smoke",
}

OUTPUT_COLUMNS = [
    "tag", "benchmark", "subset", "metric", "value",
    "n", "n_text_queries", "n_image_queries", "n_pairs",
    "model", "weights", "checkpoint", "seed", "limit", "bootstrap",
    "is_smoke", "git_commit", "config_hash", "source_file", "ts",
]


def current_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return ""


def is_smoke(limit: str | int | None) -> bool:
    return str(limit).strip().lower() not in ("", "full", "none", "0")


def read_per_bench(results_dir: Path):
    for bench in BENCHMARKS:
        path = results_dir / f"{bench}.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["source_file"] = path.name
                yield bench, row


def collect(results_dir: Path, output: Path, git_commit: str | None = None) -> Path:
    """读取 per-bench CSV，写出 long-format master.csv，返回输出路径。"""
    git_commit = current_git_commit() if git_commit is None else git_commit
    records: dict[tuple, dict] = {}

    for bench, row in read_per_bench(results_dir):
        tag = str(row.get("tag", "")).strip()
        subset = str(row.get("subset", "")).strip()
        limit = str(row.get("limit", "")).strip()
        seed = str(row.get("seed", "")).strip()
        weights = str(row.get("weights", "")).strip()
        checkpoint = str(row.get("checkpoint") or weights)
        source_file = row.get("source_file", "")
        smoke = is_smoke(limit)

        base = {
            "tag": tag,
            "benchmark": bench,
            "subset": subset,
            "n": row.get("n", ""),
            "n_text_queries": row.get("n_text_queries", ""),
            "n_image_queries": row.get("n_image_queries", ""),
            "n_pairs": row.get("n_pairs", ""),
            "model": row.get("model", ""),
            "weights": weights,
            "checkpoint": checkpoint,
            "seed": seed,
            "limit": limit,
            "bootstrap": row.get("bootstrap", ""),
            "is_smoke": str(smoke).lower(),
            "git_commit": row.get("git_commit") or git_commit,
            "config_hash": row.get("config_hash", ""),
            "source_file": source_file,
            "ts": row.get("ts", ""),
        }

        for key, value in row.items():
            if key in META_COLUMNS or value in ("", None):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            metric_key = (tag, bench, subset, key, limit, seed, weights)
            records[metric_key] = {
                **base,
                "metric": key,
                "value": number,
            }

    output.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        records.values(),
        key=lambda r: (r["tag"], r["benchmark"], r["subset"], r["metric"], r["limit"]),
    )
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"写入 {len(rows)} 行 -> {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "master.csv")
    parser.add_argument("--git-commit", help="覆盖自动读取的 git commit（测试/离线用）")
    args = parser.parse_args()
    collect(args.results_dir, args.output, args.git_commit)


if __name__ == "__main__":
    main()
