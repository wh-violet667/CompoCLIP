"""collect_results.py 的 CPU 单测：合成 per-bench CSV，验证 long-format 汇总。"""
from __future__ import annotations

import csv

from collect_results import collect


def _write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_collect_long_format_and_dedup(tmp_path):
    results = tmp_path / "results"
    results.mkdir()

    flickr_fields = [
        "tag", "benchmark", "n_text_queries", "n_image_queries",
        "t2i_R@1", "i2t_R@1", "limit", "seed", "weights", "git_commit", "ts",
    ]
    _write_csv(results / "flickr30k.csv", [
        {
            "tag": "e0", "benchmark": "flickr30k", "n_text_queries": "100",
            "n_image_queries": "20", "t2i_R@1": "0.50", "i2t_R@1": "0.40",
            "limit": "", "seed": "0", "weights": "base", "git_commit": "abc", "ts": "t1",
        },
        {
            "tag": "e0", "benchmark": "flickr30k", "n_text_queries": "100",
            "n_image_queries": "20", "t2i_R@1": "0.60", "i2t_R@1": "0.45",
            "limit": "", "seed": "0", "weights": "base", "git_commit": "abc", "ts": "t2",
        },
        {
            "tag": "e0", "benchmark": "flickr30k", "n_text_queries": "100",
            "n_image_queries": "20", "t2i_R@1": "0.99", "i2t_R@1": "0.99",
            "limit": "200", "seed": "0", "weights": "base", "git_commit": "abc", "ts": "t3",
        },
    ], flickr_fields)

    aro_fields = [
        "tag", "benchmark", "subset", "n", "accuracy", "accuracy_ci_low",
        "accuracy_ci_high", "limit", "seed", "weights", "git_commit", "ts",
    ]
    _write_csv(results / "aro.csv", [
        {
            "tag": "e0", "benchmark": "aro", "subset": "vg_relation", "n": "50",
            "accuracy": "0.70", "accuracy_ci_low": "0.55", "accuracy_ci_high": "0.82",
            "limit": "", "seed": "0", "weights": "base", "git_commit": "abc", "ts": "t4",
        },
    ], aro_fields)

    out = results / "master.csv"
    collect(results, out, git_commit="deadbeef")

    with out.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # 去重后 t2i_R@1 保留最后一次 0.60，而不是 0.50
    t2i = [r for r in rows if r["benchmark"] == "flickr30k" and r["metric"] == "t2i_R@1" and r["limit"] == ""]
    assert len(t2i) == 1
    assert float(t2i[0]["value"]) == 0.60
    assert t2i[0]["git_commit"] == "abc"  # 行内已有 commit 时优先保留
    assert t2i[0]["source_file"] == "flickr30k.csv"

    smoke = [r for r in rows if r["benchmark"] == "flickr30k" and r["limit"] == "200"]
    assert smoke and all(r["is_smoke"] == "true" for r in smoke)

    aro = [r for r in rows if r["benchmark"] == "aro" and r["metric"] == "accuracy"]
    assert len(aro) == 1
    assert aro[0]["subset"] == "vg_relation"
    assert aro[0]["git_commit"] == "abc"

    # 缺失行内 commit 时才用外部 git_commit
    assert all(r["git_commit"] in ("abc", "deadbeef") for r in rows)
