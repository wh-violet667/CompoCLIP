"""评测入口：encoder x 基准 -> 指标行 + 未命中样本。

支持的基准（审核修复后不再混用 loader）：
    flickr30k   Karpathy test 1k 图 x 5 caption，双向检索
    coco        COCO 2017 val 5k 图 x 5 caption，双向检索
    winoground  Meta Winoground 400 对，三项指标
    aro         ARO 归一化样本（data/golden/aro.jsonl），二选一准确率
    sugarcrepe  SugarCrepe++ 归一化样本（data/golden/sugarcrepe.jsonl）

用法：
    python evals-mm/run_eval.py --tag e0 --benchmark flickr30k
    python evals-mm/run_eval.py --tag e0 --benchmark coco --limit 200
    python evals-mm/run_eval.py --tag e2 --checkpoint runs/e2/epoch3.pt --benchmark winoground
    python evals-mm/run_eval.py --tag e2 --checkpoint runs/e2/epoch3.pt --benchmark aro

评测纪律：同批次数字才可横向比较；未命中样本必落盘；结果 CSV 记录 checkpoint 与 seed。
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import loaders  # noqa: E402
from encoders import OpenClipEncoder  # noqa: E402
from metrics import (  # noqa: E402
    bootstrap_ci,
    pairwise_accuracy,
    rank_of_positive,
    summary_row,
    winoground_scores,
)

RESULTS_DIR = Path(__file__).parent / "results"
CACHE_DIR = Path(__file__).parent / ".cache"
ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def current_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return ""


def encoder_id(args) -> str:
    """缓存键：模型 + 权重标识 + checkpoint 文件指纹，防止换了权重还复用旧特征。"""
    if args.checkpoint:
        p = Path(args.checkpoint)
        stamp = f"{p.name}_{p.stat().st_size}_{int(p.stat().st_mtime)}" if p.exists() else p.name
        return f"{args.model}__ckpt_{stamp}"
    return f"{args.model}__{args.pretrained}"


def cached_arrays(npz_path: Path, compute, use_cache: bool) -> dict[str, np.ndarray]:
    if use_cache and npz_path.exists():
        with np.load(npz_path) as data:
            return {k: data[k] for k in data.files}
    arrays = compute()
    if use_cache:
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz_path, **arrays)
    return arrays


def eval_retrieval(enc, groups, tag, bench, cache_path: Path, use_cache: bool,
                   bootstrap: int = 1000, seed: int = 0):
    images = [str(g.image) for g in groups]
    captions = [c for g in groups for c in g.captions]
    cap_to_img = np.repeat(np.arange(len(groups)), [len(g.captions) for g in groups])

    arrays = cached_arrays(
        cache_path,
        lambda: {"image": enc.encode_images(images), "text": enc.encode_texts(captions)},
        use_cache,
    )
    img_f, txt_f = arrays["image"], arrays["text"]
    scores = txt_f @ img_f.T  # (N_caption, N_image)

    # t2i：每条 caption 一次 query，正样本只有它所属的那张图
    pos_t2i = np.zeros(scores.shape, dtype=bool)
    pos_t2i[np.arange(len(captions)), cap_to_img] = True
    ranks_t2i = rank_of_positive(scores, pos_t2i)

    # i2t：每张图一次 query，它的 5 条 caption 全部是正样本
    pos_i2t = np.zeros((len(groups), len(captions)), dtype=bool)
    pos_i2t[cap_to_img, np.arange(len(captions))] = True
    ranks_i2t = rank_of_positive(scores.T, pos_i2t)

    row = summary_row(bench, ranks_t2i, ranks_i2t)
    t2i_r1 = (ranks_t2i < 1).astype(np.float64)
    i2t_r1 = (ranks_i2t < 1).astype(np.float64)
    row["t2i_R@1_ci_low"], row["t2i_R@1_ci_high"] = bootstrap_ci(
        t2i_r1, n_bootstrap=bootstrap, seed=seed
    )
    row["i2t_R@1_ci_low"], row["i2t_R@1_ci_high"] = bootstrap_ci(
        i2t_r1, n_bootstrap=bootstrap, seed=seed
    )

    misses = [
        {
            "caption": captions[i],
            "gt_image": images[int(cap_to_img[i])],
            "rank": float(ranks_t2i[i]),
        }
        for i in np.where(ranks_t2i >= 10)[0][:200]
    ]
    return [row], misses


def eval_pairwise(enc, items, tag, bench, cache_path: Path, use_cache: bool,
                  bootstrap: int = 1000, seed: int = 0):
    images = [str(it.image) for it in items]

    arrays = cached_arrays(
        cache_path,
        lambda: {
            "image": enc.encode_images(images),
            "positive": enc.encode_texts([it.positive for it in items]),
            "negative": enc.encode_texts([it.negative for it in items]),
        },
        use_cache,
    )
    pos_scores = (arrays["image"] * arrays["positive"]).sum(axis=1)
    neg_scores = (arrays["image"] * arrays["negative"]).sum(axis=1)

    subsets = np.array([it.subset for it in items])
    correct = (pos_scores > neg_scores).astype(np.float64)
    rows = []
    for subset in sorted(set(subsets)):
        sel = subsets == subset
        lo, hi = bootstrap_ci(correct[sel], n_bootstrap=bootstrap, seed=seed)
        rows.append({
            "benchmark": bench,
            "subset": subset,
            "n": int(sel.sum()),
            "accuracy": pairwise_accuracy(pos_scores[sel], neg_scores[sel]),
            "accuracy_ci_low": lo,
            "accuracy_ci_high": hi,
        })
    lo, hi = bootstrap_ci(correct, n_bootstrap=bootstrap, seed=seed)
    rows.append({
        "benchmark": bench,
        "subset": "all",
        "n": int(len(items)),
        "accuracy": pairwise_accuracy(pos_scores, neg_scores),
        "accuracy_ci_low": lo,
        "accuracy_ci_high": hi,
    })
    misses = [
        {"image": images[i], "positive": items[i].positive, "negative": items[i].negative,
         "subset": items[i].subset}
        for i in np.where(pos_scores <= neg_scores)[0][:200]
    ]
    return rows, misses


def eval_winoground(enc, items, tag, bootstrap: int = 1000, seed: int = 0):
    text_correct, image_correct = [], []
    for ex in items:
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for k, im in enumerate([ex["image_0"], ex["image_1"]]):
                p = Path(td) / f"{k}.png"
                im.save(p)
                paths.append(str(p))
            img_f = enc.encode_images(paths)
        txt_f = enc.encode_texts([ex["caption_0"], ex["caption_1"]])
        s = txt_f @ img_f.T  # s[a, b] = caption_a 与 image_b
        text_correct.append(s[0, 0] > s[1, 0] and s[1, 1] > s[0, 1])
        image_correct.append(s[0, 0] > s[0, 1] and s[1, 1] > s[1, 0])
    row = winoground_scores(text_correct, image_correct)
    text_arr = np.asarray(text_correct, dtype=np.float64)
    image_arr = np.asarray(image_correct, dtype=np.float64)
    group_arr = (np.asarray(text_correct, dtype=bool) & np.asarray(image_correct, dtype=bool)).astype(np.float64)
    row["wg_text_ci_low"], row["wg_text_ci_high"] = bootstrap_ci(
        text_arr, n_bootstrap=bootstrap, seed=seed
    )
    row["wg_image_ci_low"], row["wg_image_ci_high"] = bootstrap_ci(
        image_arr, n_bootstrap=bootstrap, seed=seed
    )
    row["wg_group_ci_low"], row["wg_group_ci_high"] = bootstrap_ci(
        group_arr, n_bootstrap=bootstrap, seed=seed
    )
    row["benchmark"] = "winoground"
    return [row], []


def write_rows(bench: str, rows: list[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{bench}.csv"
    extra = [k for r in rows for k in r if k not in ("benchmark",)]
    fieldnames: list[str] = []
    for key in ["tag", "benchmark"] + extra:
        if key not in fieldnames:
            fieldnames.append(key)

    existing_header: list[str] = []
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            existing_header = next(csv.reader(f), [])
        if existing_header and set(existing_header) != set(fieldnames):
            backup = csv_path.with_name(csv_path.stem + f".bak.{int(time.time())}" + csv_path.suffix)
            csv_path.replace(backup)
            print(f"结果表 schema 变化，旧文件备份到 {backup.name}；新文件使用新表头")
            existing_header = []
        elif existing_header:
            # 保持旧表头顺序，避免追加时列错位
            fieldnames = existing_header

    write_header = not existing_header
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="实验组名，写进结果 CSV")
    parser.add_argument("--benchmark", required=True,
                        choices=["flickr30k", "coco", "winoground", "aro", "sugarcrepe"])
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--checkpoint", help="微调组 checkpoint；不给则为 E0 基线")
    parser.add_argument("--data-root", help="覆盖 DATA_ROOT 环境变量，默认 data/")
    parser.add_argument("--split", help="Flickr30k 用 test/val；COCO 默认 val")
    parser.add_argument("--limit", type=int, help="只评测前 N 个 query（冒烟用，结果不可比）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="bootstrap 次数；0 表示关闭 CI（冒烟时可设 0）")
    parser.add_argument("--no-cache", action="store_true", help="禁用特征缓存")
    args = parser.parse_args()

    set_seed(args.seed)
    data_root = loaders.resolve_data_root(args.data_root)
    enc = (OpenClipEncoder.from_checkpoint(args.model, args.checkpoint)
           if args.checkpoint else OpenClipEncoder.from_pretrained(args.model, args.pretrained))

    enc_id = encoder_id(args)
    limit_tag = f"limit{args.limit}" if args.limit else "full"
    use_cache = not args.no_cache

    if args.benchmark in ("flickr30k", "coco"):
        split = args.split or ("test" if args.benchmark == "flickr30k" else "val")
        groups = (loaders.load_flickr30k(data_root, split=split, limit=args.limit)
                  if args.benchmark == "flickr30k"
                  else loaders.load_coco(data_root, split=split, limit=args.limit))
        cache = CACHE_DIR / enc_id / f"{args.benchmark}_{split}_{limit_tag}.npz"
        rows, misses = eval_retrieval(
            enc, groups, args.tag, args.benchmark, cache, use_cache,
            bootstrap=args.bootstrap, seed=args.seed,
        )
    elif args.benchmark == "winoground":
        rows, misses = eval_winoground(
            enc, loaders.load_winoground(limit=args.limit), args.tag,
            bootstrap=args.bootstrap, seed=args.seed,
        )
    else:
        golden = data_root / "golden" / f"{args.benchmark}.jsonl"
        cache = CACHE_DIR / enc_id / f"{args.benchmark}_{limit_tag}.npz"
        rows, misses = eval_pairwise(
            enc, loaders.load_pairwise(golden, limit=args.limit), args.tag, args.benchmark,
            cache, use_cache, bootstrap=args.bootstrap, seed=args.seed,
        )

    meta = {
        "tag": args.tag,
        "model": args.model,
        "weights": args.checkpoint or args.pretrained,
        "seed": args.seed,
        "limit": args.limit or "",
        "bootstrap": args.bootstrap,
        "git_commit": current_git_commit(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for r in rows:
        r.update(meta)

    csv_path = write_rows(args.benchmark, rows)
    if misses:
        miss_path = RESULTS_DIR / f"{args.tag}_{args.benchmark}_misses.json"
        miss_path.write_text(json.dumps(misses, ensure_ascii=False, indent=2), encoding="utf-8")

    for r in rows:
        print({k: v for k, v in r.items() if k not in ("ts",)})
    print(f"\n结果追加到 {csv_path}；未命中 {len(misses)} 条")
    print("注意：同批次、同 limit 的数字才可横向比较。")


if __name__ == "__main__":
    main()
