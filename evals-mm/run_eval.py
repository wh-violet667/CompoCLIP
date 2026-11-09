"""评测入口：一组 encoder x 一个基准 -> 指标行 + 未命中样本。

用法：
    # E0 基线（Flickr30k test 双向检索）
    python evals-mm/run_eval.py --tag e0 --benchmark flickr30k

    # 微调组（Winoground）
    python evals-mm/run_eval.py --tag e2 --checkpoint runs/e2/epoch3.pt --benchmark winoground

评测纪律（dev-doc 4.3 节）：同批次数字才可横向比较；未命中样本必落盘。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))  # evals-mm 目录内互相 import
from encoders import OpenClipEncoder  # noqa: E402
from metrics import ranks_from_scores, summary_row, winoground_scores  # noqa: E402
from retrieval import full_scores  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"


def load_flickr30k(data_root: Path):
    """Karpathy test split：每图取一条 caption 做 t2i；i2t 按图检索。"""
    items = [json.loads(l) for l in (data_root / "flickr30k/splits/test.json").open(encoding="utf-8")]
    images = [str(data_root / "flickr30k/images" / it["image"]) for it in items]
    captions = [it["caption"] for it in items]
    return images, captions


def load_winoground():
    from datasets import load_dataset
    return load_dataset("facebook/winoground", split="test")


def eval_retrieval(enc, images, captions, tag, bench):
    img_f = enc.encode_images(images)
    txt_f = enc.encode_texts(captions)
    scores = full_scores(img_f, txt_f)  # (Q_text, N_img)
    gt = np.arange(len(images))
    ranks_t2i = ranks_from_scores(scores, gt)
    ranks_i2t = ranks_from_scores(scores.T, gt)
    row = summary_row(bench, ranks_t2i, ranks_i2t)
    row["tag"] = tag
    misses = [{"caption": captions[i], "gt_image": images[i], "rank": int(ranks_t2i[i])}
              for i in np.where(ranks_t2i >= 10)[0]]
    return row, misses


def eval_winoground(enc, ds, tag):
    text_correct, image_correct = [], []
    for ex in ds:
        # winoground 图片为 PIL 对象，临时存盘复用 encode_images
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for k, im in enumerate([ex["image_0"], ex["image_1"]]):
                p = Path(td) / f"{k}.png"
                im.save(p)
                paths.append(str(p))
            img_f = enc.encode_images(paths)
        txt_f = enc.encode_texts([ex["caption_0"], ex["caption_1"]])
        s = txt_f @ img_f.T  # (2, 2)
        # text score：给定图，两个 caption 中选对（两张图都要对）
        text_correct.append(s[0, 0] > s[1, 0] and s[1, 1] > s[0, 1])
        # image score：给定 caption，两张图中选对（两个 caption 都要对）
        image_correct.append(s[0, 0] > s[0, 1] and s[1, 1] > s[1, 0])
    row = winoground_scores(np.array(text_correct), np.array(image_correct))
    row["tag"], row["benchmark"] = tag, "winoground"
    return row, []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="实验组名，写进结果 CSV")
    parser.add_argument("--benchmark", choices=["flickr30k", "coco", "winoground"], required=True)
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--checkpoint", help="微调组 checkpoint；不给则为 E0 基线")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()

    enc = (OpenClipEncoder.from_checkpoint(args.model, args.checkpoint)
           if args.checkpoint else OpenClipEncoder.from_pretrained(args.model, args.pretrained))

    if args.benchmark == "winoground":
        row, misses = eval_winoground(enc, load_winoground(), args.tag)
    else:
        row, misses = eval_retrieval(enc, *load_flickr30k(args.data_root), args.tag, args.benchmark)

    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / f"{args.benchmark}.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            w.writeheader()
        w.writerow(row)
    if misses:
        (RESULTS_DIR / f"{args.tag}_{args.benchmark}_misses.json").write_text(
            json.dumps(misses, ensure_ascii=False, indent=2), encoding="utf-8")
    print(row)
    print(f"结果追加到 {csv_path}；未命中 {len(misses)} 条已落盘。同批次数字才可横向比较。")


if __name__ == "__main__":
    main()
