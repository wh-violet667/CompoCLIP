"""LLM 法硬负例生成（SCRAMBLe 轻量版，创新点 1 之来源二）。

用 few-shot prompt 让 LLM 生成"使用完全相同的词但表达不同/相反含义"的句子。
兼容 OpenAI 协议（OpenAI / DashScope 兼容模式均可，改 base-url 即可）。

输入/输出格式同 rule_based.py，rule 字段固定 "llm"。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

PROMPT_TEMPLATE = """Given an image caption, rewrite it using EXACTLY the same words but in a different order, so that it describes a DIFFERENT or OPPOSITE scene. Keep every original word; do not add or remove words.

Caption: "a dog sits to the left of a cat"
Rewrite: "a cat sits to the left of a dog"

Caption: "a man in a red shirt is riding a bike"
Rewrite: "a man in a red bike is riding a shirt"

Caption: "{caption}"
Rewrite:"""


def generate_negative(client: OpenAI, model: str, caption: str) -> str | None:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(caption=caption)}],
        temperature=0.7,
        max_tokens=64,
    )
    text = resp.choices[0].message.content.strip().strip('"')
    # 粗过滤：改写为空或与原文相同则丢弃；完整词集合校验由对抗过滤阶段兜底
    if not text or text.lower() == caption.lower():
        return None
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen-turbo", help="低成本模型即可，约 1.5 万条")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url)  # API key 走 OPENAI_API_KEY 环境变量

    kept, dropped = 0, 0
    with args.input.open(encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            item = json.loads(line)
            try:
                negative = generate_negative(client, args.model, item["caption"])
            except Exception as e:  # 限流/解析失败，跳过并计数
                print(f"[{i}] 生成失败：{e}")
                negative = None
            if negative is None:
                dropped += 1
                continue
            fout.write(json.dumps({
                "image": item["image"],
                "caption": item["caption"],
                "negative": negative,
                "rule": "llm",
            }, ensure_ascii=False) + "\n")
            kept += 1
            if kept % 500 == 0:
                print(f"已生成 {kept} 条")
    print(f"完成：保留 {kept} 条，丢弃 {dropped} 条。下一步跑 adversarial_filter.py 分桶。")


if __name__ == "__main__":
    main()
