"""LLM 法硬负例生成（SCRAMBLe 轻量版，创新点 1 之来源二）。

用 few-shot prompt 让 LLM 生成“使用完全相同的词但表达不同/相反含义”的句子。
兼容 OpenAI 协议（OpenAI / DashScope 兼容模式均可，改 base-url 即可）。

产线特性（成员 1 在 D2 补齐）：
- `--workers` 并发、`--qps` 限流、指数退避重试；
- `--resume` 按 (image, caption) 去重续跑，输出 append；
- 词多重集校验（默认开启）：负例必须与正例词袋完全一致，否则重试/丢弃；
- `--dry-run` 用确定性打乱代替 API，跑通端到端与单元测试；
- 产出 `quality_report.json`，记录保留率、词袋通过率、长度比、重复率。

输入/输出格式同 rule_based.py，rule 字段固定 "llm"。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROMPT_TEMPLATE = """Given an image caption, rewrite it using EXACTLY the same words but in a different order, so that it describes a DIFFERENT or OPPOSITE scene. Keep every original word; do not add or remove words. Keep the same punctuation.

Caption: "a dog sits to the left of a cat"
Rewrite: "a cat sits to the left of a dog"

Caption: "a man in a red shirt is riding a bike"
Rewrite: "a man in a red bike is riding a shirt"

Caption: "{caption}"
Rewrite:"""


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def word_multiset_ok(original: str, negative: str) -> bool:
    return Counter(words(original)) == Counter(words(negative))


def key_of(item: dict) -> str:
    return hashlib.sha1(f"{item.get('image','')}\x00{item['caption']}".encode("utf-8")).hexdigest()[:16]


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            done.add(key_of(json.loads(line)))
        except Exception:
            continue
    return done


class RateLimiter:
    def __init__(self, qps: float):
        self.interval = 1.0 / qps if qps > 0 else 0.0
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self) -> None:
        if not self.interval:
            return
        with self.lock:
            now = time.monotonic()
            sleep = max(0.0, self.next_at - now)
            self.next_at = max(now, self.next_at) + self.interval
        if sleep:
            time.sleep(sleep)


def mock_generate(caption: str, seed: int) -> str:
    rng = random.Random(seed)
    ws = caption.split()
    for _ in range(20):
        rng.shuffle(ws)
        if " ".join(ws).lower() != caption.lower():
            break
    return " ".join(ws)


def call_llm(client, model: str, caption: str, temperature: float, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(caption=caption)}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip().strip('"').strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen-turbo", help="低成本模型即可，约 1.5 万条")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--qps", type=float, default=2.0, help="全局请求速率上限")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="跳过输出文件里已有的 (image, caption)")
    parser.add_argument("--no-word-check", action="store_true", help="关闭词多重集校验（不建议）")
    parser.add_argument("--dry-run", action="store_true", help="不调用 API，用打乱词序模拟，供测试")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.output) if args.resume else set()
    items = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    items = [it for it in items if not (args.resume and key_of(it) in done)]
    if args.limit is not None:
        items = items[:args.limit]
    if not items:
        print("没有待生成样本（可能已全部完成）。")
        return

    client = None
    if not args.dry_run:
        from openai import OpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("缺少 OPENAI_API_KEY；无 key 时请加 --dry-run 或先配置环境变量。")
        client = OpenAI(base_url=args.base_url)

    limiter = RateLimiter(args.qps)
    stats = Counter()
    length_ratios: list[float] = []
    lock = threading.Lock()
    mode = "a" if args.resume else "w"
    out = args.output.open(mode, encoding="utf-8")

    def work(idx_item: tuple[int, dict]) -> tuple[int, dict, str | None, str]:
        idx, item = idx_item
        caption = item["caption"].strip()
        last_err = ""
        for attempt in range(args.retries + 1):
            try:
                limiter.wait()
                if args.dry_run:
                    neg = mock_generate(caption, args.seed + idx + attempt)
                else:
                    neg = call_llm(client, args.model, caption, args.temperature, args.max_tokens)
                if not neg:
                    last_err = "empty"
                    continue
                if neg.lower() == caption.lower():
                    last_err = "same"
                    continue
                if not args.no_word_check and not word_multiset_ok(caption, neg):
                    last_err = "word_multiset"
                    continue
                return idx, item, neg, ""
            except Exception as e:  # 限流/网络/解析失败
                last_err = f"exception:{type(e).__name__}"
                time.sleep(min(2 ** attempt, 8))
        return idx, item, None, last_err

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(work, pair) for pair in enumerate(items)]
            for fut in as_completed(futures):
                idx, item, neg, err = fut.result()
                with lock:
                    if neg is None:
                        stats[f"dropped_{err or 'unknown'}"] += 1
                        continue
                    out.write(json.dumps({
                        "image": item.get("image"),
                        "caption": item["caption"],
                        "negative": neg,
                        "rule": "llm",
                    }, ensure_ascii=False) + "\n")
                    stats["kept"] += 1
                    length_ratios.append(len(words(neg)) / max(1, len(words(item["caption"]))))
                    if stats["kept"] % 200 == 0:
                        out.flush()
                        print(f"已生成 {stats['kept']} 条")
    finally:
        out.close()

    total = sum(stats.values()) or 1
    report = {
        "input_pending": len(items),
        "kept": stats["kept"],
        "dropped": total - stats["kept"],
        "keep_rate": stats["kept"] / total,
        "drop_reasons": {k: v for k, v in stats.items() if k.startswith("dropped_")},
        "mean_length_ratio": float(sum(length_ratios) / len(length_ratios)) if length_ratios else None,
        "word_check": not args.no_word_check,
        "dry_run": args.dry_run,
        "model": args.model,
    }
    report_path = args.report or args.output.with_name("quality_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：保留 {stats['kept']} 条，丢弃 {total - stats['kept']} 条；报告 -> {report_path}")
    print("下一步：python scripts/sample_hardneg_100.py --input <output> 做人工质检。")


if __name__ == "__main__":
    main()
