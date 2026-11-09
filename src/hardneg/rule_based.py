"""规则法硬负例生成（NegCLIP 路线，创新点 1 之来源一）。

策略：spaCy 依存解析找主语/宾语名词并交换；找不到则退化为交换两个普通名词。
输入：captions JSONL，每行 {"image": "xxx.jpg", "caption": "..."}
输出：JSONL，每行 {"image", "caption", "negative", "rule"}
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import spacy

nlp = spacy.load("en_core_web_sm")

SUBJ_DEPS = {"nsubj", "nsubjpass"}
OBJ_DEPS = {"dobj", "obj", "pobj"}


def swap_tokens(doc, i: int, j: int) -> str:
    tokens = [t.text for t in doc]
    tokens[i], tokens[j] = tokens[j], tokens[i]
    return " ".join(tokens)


def make_negative(caption: str) -> tuple[str, str] | None:
    doc = nlp(caption)
    subj = [t.i for t in doc if t.dep_ in SUBJ_DEPS]
    obj = [t.i for t in doc if t.dep_ in OBJ_DEPS]
    if subj and obj:
        return swap_tokens(doc, subj[0], obj[0]), "swap_subj_obj"
    nouns = [t.i for t in doc if t.pos_ in {"NOUN", "PROPN"}]
    if len(nouns) >= 2:
        i, j = random.sample(nouns, 2)
        return swap_tokens(doc, i, j), "swap_two_nouns"
    return None  # 过短或解析失败，由调用方决定丢弃


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="captions JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-caption", type=int, default=1, help="每条正样本生成几个负例")
    args = parser.parse_args()

    kept, dropped = 0, 0
    with args.input.open(encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line)
            for _ in range(args.per_caption):
                result = make_negative(item["caption"])
                if result is None:
                    dropped += 1
                    continue
                negative, rule = result
                fout.write(json.dumps({
                    "image": item["image"],
                    "caption": item["caption"],
                    "negative": negative,
                    "rule": rule,
                }, ensure_ascii=False) + "\n")
                kept += 1
    print(f"完成：保留 {kept} 条，丢弃 {dropped} 条。请抽样 100 条人工质检（dev-doc 3.2 节）。")


if __name__ == "__main__":
    main()
