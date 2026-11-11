"""0-A（减配）：组合信息在不在 CLIP 表征里？——Winoground 版线性探针

为什么用 Winoground：Flickr 的扰动负例常常不自然，线性探针可以靠“句子流畅度”作弊。
Winoground 的两句 caption 都是人写的、词袋相同、语法都通，唯一区别是图文绑定，
因此“能否线性解码正确绑定”是更干净的机制证据。

构造：
    每个 pair 两张图 g0/g1、两句 c0/c1；
    正例特征 = [g0,g1,t0,t1, g0*t0, g0*t1, g1*t0, g1*t1, |g0-g1|, |t0-t1|]，标签 1；
    反例 = 把 (g0,g1) 与 (t0,t1) 的对应关系对调，标签 0；
    5 折 GroupKFold（按 group_id），逻辑回归。
对比同一批 pair 的零样本 group 准确率（CLIP 原始读数）。

同时给出 Flickr 版的补充结果（--benchmark flickr），并明确标注“可能受流畅度捷径影响”。

用法：
    python scripts/run_0a_probe.py --benchmark winoground
    python scripts/run_0a_probe.py --benchmark flickr --limit 1200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evals-mm"))
from encoders import OpenClipEncoder  # noqa: E402

sys.path.insert(0, str(ROOT))
from src.diagnostics.perturbations import PROBES  # noqa: E402


def fit_probe(X, y, groups):
    cv = GroupKFold(n_splits=5)
    accs = []
    for tr, te in cv.split(X, y, groups):
        clf = LogisticRegression(max_iter=800, C=1.0, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        accs.append(float(clf.score(X[te], y[te])))
    return float(np.mean(accs)), float(np.std(accs))


def run_winoground(enc, pairs_path: Path, seed: int):
    rows = [json.loads(l) for l in pairs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = rows[:400]
    g0 = enc.encode_images([r["image_0"] for r in rows])
    g1 = enc.encode_images([r["image_1"] for r in rows])
    t0 = enc.encode_texts([r["caption_0"] for r in rows])
    t1 = enc.encode_texts([r["caption_1"] for r in rows])

    X, y, groups = [], [], []
    zs_text, zs_image, zs_group = [], [], []
    for i, r in enumerate(rows):
        a0, a1, b0, b1 = g0[i], g1[i], t0[i], t1[i]
        feat = np.concatenate([a0, a1, b0, b1, a0 * b0, a0 * b1, a1 * b0, a1 * b1,
                               np.abs(a0 - a1), np.abs(b0 - b1)])
        X.extend([feat, feat])
        y.extend([1, 0])
        groups.extend([i, i])
        s00, s01 = a0 @ b0, a0 @ b1
        s10, s11 = a1 @ b0, a1 @ b1
        tc = (s00 > s01) and (s11 > s10)   # caption0->image0, caption1->image1
        ic = (s00 > s10) and (s11 > s01)
        zs_text.append(tc)
        zs_image.append(ic)
        zs_group.append(tc and ic)

    acc, std = fit_probe(np.asarray(X, dtype=np.float32), np.asarray(y), np.asarray(groups))
    return {
        "benchmark": "winoground",
        "n_pairs": len(rows),
        "zero_shot_text": float(np.mean(zs_text)),
        "zero_shot_image": float(np.mean(zs_image)),
        "zero_shot_group": float(np.mean(zs_group)),
        "linear_probe_group": acc,
        "linear_probe_std": std,
        "gain_over_group": acc - float(np.mean(zs_group)),
    }


def run_flickr(enc, split: Path, image_root: Path, limit: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = [json.loads(line) for line in split.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng.shuffle(rows)
    rows = rows[:limit]

    samples = []
    for r in rows:
        cap = r["caption"].strip()
        negs = {}
        for name, fn in PROBES.items():
            neg = fn(cap) if name not in ("noun_swap", "adj_swap", "word_shuffle", "trigram_shuffle") else fn(cap, None)
            if neg:
                negs[name] = neg
        if negs:
            samples.append({"image": str(image_root / r["image"]), "caption": cap, "negs": negs})

    images = list(dict.fromkeys(s["image"] for s in samples))
    texts = list(dict.fromkeys([s["caption"] for s in samples] + [n for s in samples for n in s["negs"].values()]))
    ii = {p: i for i, p in enumerate(images)}
    ti = {t: i for i, t in enumerate(texts)}
    gf = enc.encode_images(images)
    tf = enc.encode_texts(texts)

    out = {}
    for name in PROBES:
        X, y, groups, zs = [], [], [], []
        for k, s in enumerate(samples):
            if name not in s["negs"]:
                continue
            g = gf[ii[s["image"]]]
            p = tf[ti[s["caption"]]]
            n = tf[ti[s["negs"][name]]]
            feat = np.concatenate([g, p, n, g * p, g * n, np.abs(p - n)])
            X.extend([feat, feat])
            y.extend([1, 0])
            groups.extend([k, k])
            zs.append(float(p @ g > n @ g))
        if len(zs) < 30:
            continue
        acc, std = fit_probe(np.asarray(X, dtype=np.float32), np.asarray(y), np.asarray(groups))
        out[name] = {
            "n_pairs": len(zs), "zero_shot_acc": float(np.mean(zs)),
            "linear_probe_acc": acc, "linear_probe_std": std, "gain": acc - float(np.mean(zs)),
        }
    return {"benchmark": "flickr", "n_captions": len(samples), "probes": out,
            "note": "扰动负例可能不自然，线性探针可能利用流畅度捷径；仅作补充。"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["winoground", "flickr"], default="winoground")
    parser.add_argument("--pairs", type=Path, default=ROOT / "data/winoground/pairs.jsonl")
    parser.add_argument("--split", type=Path, default=ROOT / "data/flickr30k/splits/test.jsonl")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data/flickr30k/images")
    parser.add_argument("--limit", type=int, default=1200)
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "evals-mm/results")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    enc = OpenClipEncoder.from_pretrained(args.model, args.pretrained)

    if args.benchmark == "winoground":
        out = run_winoground(enc, args.pairs, args.seed)
        print(f"Winoground：零样本 text={out['zero_shot_text']:.3f} image={out['zero_shot_image']:.3f} "
              f"group={out['zero_shot_group']:.3f}；线性探针 group={out['linear_probe_group']:.3f}"
              f"（gain {out['gain_over_group']:+.3f}）")
        name = "0a_winoground.json"
    else:
        out = run_flickr(enc, args.split, args.image_root, args.limit, args.seed)
        print(f"{'probe':<20}{'n':>7}{'zero-shot':>11}{'linear':>9}{'gain':>8}")
        for k, v in sorted(out["probes"].items(), key=lambda kv: kv[1]["gain"], reverse=True):
            print(f"{k:<20}{v['n_pairs']:>7}{v['zero_shot_acc']:>11.3f}{v['linear_probe_acc']:>9.3f}{v['gain']:>+8.3f}")
        name = "0a_flickr.json"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / name).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果 -> {args.out_dir / name}")


if __name__ == "__main__":
    main()
