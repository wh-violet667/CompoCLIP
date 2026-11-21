"""Gradio 最小 demo：E0 与微调模型的盲区对照。

用法：
    python demo/app.py --finetuned runs/e2/epoch3.pt
    python demo/app.py --finetuned A2_models/e2/e2_model.pt --limit 500

特性：
- 图像特征缓存：同一模型/图片集合第二次启动不再重复编码；
- 预置组合 query；
- 没有 --finetuned 时只展示 E0，便于先调布局。
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "evals-mm"))
from encoders import OpenClipEncoder  # noqa: E402
from retrieval import build_index, search  # noqa: E402

TOP_K = 8
PRESETS = [
    "the dog is to the left of the cat",
    "the cat is to the left of the dog",
    "a man in a red shirt is riding a bike",
    "a man in a red bike is riding a shirt",
    "a red car next to a blue house",
    "a blue car next to a red house",
    "the taller person hugs the shorter person",
    "the shorter person hugs the taller person",
]


def image_paths(image_dir: Path, limit: int) -> list[str]:
    paths = sorted(
        str(p) for p in image_dir.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    return paths[:limit]


def cache_key(model_id: str, paths: list[str]) -> str:
    h = hashlib.sha1()
    h.update(model_id.encode("utf-8"))
    for p in paths:
        st = Path(p).stat()
        h.update(f"{p}:{st.st_size}:{int(st.st_mtime)}".encode("utf-8"))
    return h.hexdigest()[:16]


class Gallery:
    def __init__(self, enc: OpenClipEncoder, paths: list[str], cache_dir: Path, name: str):
        self.paths = paths
        self.enc = enc
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = cache_dir / f"{name}_{cache_key(name, paths)}.npz"
        if cache.exists():
            with np.load(cache) as data:
                feats = data["image"]
            print(f"demo: 复用特征缓存 {cache}")
        else:
            print(f"demo: 编码 {len(paths)} 张图 -> {cache}")
            feats = enc.encode_images(paths)
            np.savez(cache, image=feats)
        self.index = build_index(feats)

    def query(self, text: str) -> list[tuple[str, str]]:
        q = self.enc.encode_texts([text])
        k = min(TOP_K, len(self.paths))
        scores, idx = search(self.index, q, k)
        return [(self.paths[i], f"{s:.3f}") for s, i in zip(scores[0], idx[0])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="laion2b_s34b_b79k", help="E0 预训练标识")
    parser.add_argument("--finetuned", help="微调 checkpoint；不填则只展示 E0")
    parser.add_argument("--image-dir", type=Path, default=Path("data/flickr30k/images"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--cache-dir", type=Path, default=Path("demo/.cache"))
    args = parser.parse_args()

    paths = image_paths(args.image_dir, args.limit)
    if not paths:
        raise SystemExit(f"没有在 {args.image_dir} 找到图片")
    print(f"demo: {len(paths)} 张图")

    g_base = Gallery(OpenClipEncoder.from_pretrained("ViT-B-32", args.base),
                     paths, args.cache_dir, "e0")
    g_ft = None
    if args.finetuned:
        g_ft = Gallery(OpenClipEncoder.from_checkpoint("ViT-B-32", args.finetuned),
                       paths, args.cache_dir, "ft")

    def compare(query: str):
        base = g_base.query(query)
        ft = g_ft.query(query) if g_ft else []
        return base, ft

    outputs = [
        gr.Gallery(label="E0 原始 CLIP top-8", columns=4),
        gr.Gallery(label="微调后 top-8" if g_ft else "未提供微调模型", columns=4),
    ]
    demo = gr.Interface(
        fn=compare,
        inputs=gr.Textbox(label="query", placeholder=PRESETS[0]),
        outputs=outputs,
        examples=[[q] for q in PRESETS],
        title="CompoCLIP 盲区对照",
        description="左：未微调基线；右：硬负例微调模型。组合 query 下观察主宾颠倒图的排名变化。",
    )
    demo.launch()


if __name__ == "__main__":
    main()
