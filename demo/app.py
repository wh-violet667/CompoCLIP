"""Gradio 最小 demo：微调前后「盲区对照」（dev-doc 第 5 节，1-2 天工作量）。

    python demo/app.py --base laion2b_s34b_b79k --finetuned runs/e2/epoch3.pt

并排展示 E0 与微调模型对同一 query 的 top-8，组合 query 下差异最直观。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "evals-mm"))
from encoders import OpenClipEncoder  # noqa: E402
from retrieval import build_index, search  # noqa: E402

TOP_K = 8


class Gallery:
    def __init__(self, enc, image_paths: list[str]):
        self.paths = image_paths
        self.index = build_index(enc.encode_images(image_paths))
        self.enc = enc

    def query(self, text: str) -> list[tuple[str, str]]:
        q = self.enc.encode_texts([text])
        scores, idx = search(self.index, q, TOP_K)
        return [(self.paths[i], f"{s:.3f}") for s, i in zip(scores[0], idx[0])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="laion2b_s34b_b79k", help="E0 预训练标识")
    parser.add_argument("--finetuned", required=True, help="微调 checkpoint")
    parser.add_argument("--image-dir", type=Path, default=Path("data/flickr30k/images"))
    args = parser.parse_args()

    paths = sorted(str(p) for p in args.image_dir.glob("*.jpg"))[:5000]
    g_base = Gallery(OpenClipEncoder.from_pretrained("ViT-B-32", args.base), paths)
    g_ft = Gallery(OpenClipEncoder.from_checkpoint("ViT-B-32", args.finetuned), paths)

    def compare(query: str):
        return g_base.query(query), g_ft.query(query)

    demo = gr.Interface(
        fn=compare,
        inputs=gr.Textbox(label="query", placeholder="the dog is to the left of the cat"),
        outputs=[
            gr.Gallery(label="E0 原始 CLIP top-8", columns=4),
            gr.Gallery(label="微调后 top-8", columns=4),
        ],
        title="CompoCLIP 盲区对照",
        description="左：未微调基线；右：硬负例微调模型。组合 query 下观察主宾颠倒图的排名变化。",
    )
    demo.launch()


if __name__ == "__main__":
    main()
