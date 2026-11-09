"""对抗过滤分桶（创新点 1 的关键步骤）：用当前 CLIP 给负例打分，按相似度分桶。

只保留模型盲区——CLIP 认为相似度高但语义不同的样本。
桶划分（dev-doc 3.2 节）：>0.85 高相似（训练用）/ 0.6-0.85 中 / <0.6 低（丢弃）。

输入：负例 JSONL（含 image / caption / negative）
输出：adversarial_filtered.jsonl（仅高相似桶）+ 分桶统计打印
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from open_clip import create_model_and_transforms, get_tokenizer
from tqdm import tqdm

HIGH, MID = 0.85, 0.60


@torch.no_grad()
def score_pairs(model, tokenizer, texts_a: list[str], texts_b: list[str], device: str) -> torch.Tensor:
    """正负 caption 文本-文本相似度（图文对版可扩展为图像-文本，此处用文本代理更快）。"""
    sims = []
    for i in range(0, len(texts_a), 512):
        fa = model.encode_text(tokenizer(texts_a[i:i + 512]).to(device))
        fb = model.encode_text(tokenizer(texts_b[i:i + 512]).to(device))
        fa, fb = fa / fa.norm(dim=-1, keepdim=True), fb / fb.norm(dim=-1, keepdim=True)
        sims.append((fa * fb).sum(dim=-1).cpu())
    return torch.cat(sims)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(device).eval()
    tokenizer = get_tokenizer(args.model)

    items = [json.loads(l) for l in args.input.open(encoding="utf-8")]
    sims = score_pairs(model, tokenizer, [it["caption"] for it in items], [it["negative"] for it in items], device)

    buckets = {"high": 0, "mid": 0, "low": 0}
    with args.output.open("w", encoding="utf-8") as fout:
        for item, s in zip(items, sims):
            s = float(s)
            if s > HIGH:
                buckets["high"] += 1
                item["neg_sim"] = round(s, 4)
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            elif s > MID:
                buckets["mid"] += 1
            else:
                buckets["low"] += 1

    total = len(items)
    print(f"分桶统计：高相似(>{HIGH}) {buckets['high']} ({buckets['high']/total:.0%})，"
          f"中 {buckets['mid']}，低(<{MID}) {buckets['low']}（丢弃）")
    print(f"高相似桶已写入 {args.output}（E2a 训练集；约为 E2 的 {buckets['high']/total:.0%} 规模）")


if __name__ == "__main__":
    main()
