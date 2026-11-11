"""0-B 词序/句法探针：CLIP 对“同一组词、不同词序”的敏感度。

对每条 Flickr30k caption 生成 5 类保持词袋的负例：
    subj_obj_swap     交换主语/宾语（句法角色）
    noun_swap         交换两个名词
    adj_swap          交换两个形容词（属性错配）
    word_shuffle      完全打乱词序（词袋基线）
    trigram_shuffle   三元组内部打乱（局部词序）
另加 unrelated 对照：用另一张图的 caption 当负例，校准“任务本身有多难”。

指标：image 对原句的相似度 > 对负例的相似度 的准确率（随机 50%）。
若 word_shuffle 仍显著高于 50%，说明模型主要靠词袋而非句法。

用法：
    python scripts/run_0b_probe.py --limit 5000
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evals-mm"))
from encoders import OpenClipEncoder  # noqa: E402
from metrics import bootstrap_ci  # noqa: E402

sys.path.insert(0, str(ROOT))
from src.diagnostics.perturbations import PROBES  # noqa: E402


def dedupe(values: list[str]) -> tuple[list[str], np.ndarray]:
    index: dict[str, int] = {}
    unique: list[str] = []
    inv = np.empty(len(values), dtype=np.int64)
    for i, v in enumerate(values):
        if v not in index:
            index[v] = len(unique)
            unique.append(v)
        inv[i] = index[v]
    return unique, inv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=ROOT / "data/flickr30k/splits/test.jsonl")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data/flickr30k/images")
    parser.add_argument("--limit", type=int, default=5000, help="最多取多少条 caption")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "evals-mm/results")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    rows = [json.loads(line) for line in args.split.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng.shuffle(rows)
    rows = rows[:args.limit]

    samples = []
    for r in rows:
        caption = r["caption"].strip()
        negs = {}
        for name, fn in PROBES.items():
            neg = fn(caption, rng) if name in ("noun_swap", "adj_swap", "word_shuffle", "trigram_shuffle") else fn(caption)
            if neg:
                negs[name] = neg
        if negs:
            samples.append({"image": str(args.image_root / r["image"]), "caption": caption, "negs": negs})

    # unrelated 对照：从另一张图随机取一条 caption
    captions = [r["caption"] for r in rows]
    for s in samples:
        other = rng.choice(captions)
        for _ in range(5):
            if other.strip().lower() != s["caption"].lower():
                break
            other = rng.choice(captions)
        s["negs"]["unrelated"] = other

    enc = OpenClipEncoder.from_pretrained(args.model, args.pretrained)

    images = [s["image"] for s in samples]
    uniq_images, img_idx = dedupe(images)
    print(f"图片 {len(images)} -> 去重 {len(uniq_images)}；caption {len(samples)}")
    img_f = enc.encode_images(uniq_images)[img_idx]

    # 所有 probe 文本一次性去重编码
    all_texts: list[str] = []
    for s in samples:
        all_texts.append(s["caption"])
        all_texts.extend(s["negs"].values())
    uniq_texts, text_idx = dedupe(all_texts)
    print(f"文本 {len(all_texts)} -> 去重 {len(uniq_texts)}")
    txt_f = enc.encode_texts(uniq_texts)

    offset = 0
    pos_scores = np.empty(len(samples))
    per_sample = []
    for i, s in enumerate(samples):
        pos_i = text_idx[offset]
        pos_scores[i] = float(img_f[i] @ txt_f[pos_i])
        offset += 1
        rec = {"image": images[i], "caption": s["caption"], "pos_score": float(pos_scores[i])}
        for name, neg in s["negs"].items():
            score = float(img_f[i] @ txt_f[text_idx[offset]])
            offset += 1
            rec[name] = {"negative": neg, "score": score, "margin": float(pos_scores[i] - score)}
        per_sample.append(rec)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in list(PROBES) + ["unrelated"]:
        margins = np.array([r[name]["margin"] for r in per_sample if name in r])
        if len(margins) == 0:
            continue
        correct = (margins > 0).astype(np.float64)
        lo, hi = bootstrap_ci(correct, n_bootstrap=args.bootstrap, seed=args.seed)
        summary.append({
            "probe": name,
            "n": int(len(correct)),
            "accuracy": float(correct.mean()),
            "ci_low": lo,
            "ci_high": hi,
            "mean_margin": float(margins.mean()),
            "median_margin": float(np.median(margins)),
        })

    summary.sort(key=lambda x: x["accuracy"], reverse=True)
    with (args.out_dir / "0b_probe.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    (args.out_dir / "0b_summary.json").write_text(json.dumps({
        "model": args.model, "pretrained": args.pretrained, "seed": args.seed,
        "n_samples": len(samples), "probes": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.out_dir / "0b_persample.jsonl").open("w", encoding="utf-8") as f:
        for r in per_sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'probe':<20}{'n':>7}{'acc':>9}{'95% CI':>18}{'mean margin':>14}")
    for r in summary:
        ci = f"[{r['ci_low']:.3f},{r['ci_high']:.3f}]"
        print(f"{r['probe']:<20}{r['n']:>7}{r['accuracy']:>9.3f}{ci:>18}{r['mean_margin']:>14.4f}")
    print(f"\n结果 -> {args.out_dir / '0b_probe.csv'}")


if __name__ == "__main__":
    main()
