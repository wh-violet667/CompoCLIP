"""把 clip-benchmark/wds_flickr30k 的 test tar 还原成 E0/0-B 可用的 JSONL。

背景：正式数据由成员 2 用 Kaggle + Karpathy JSON 生成；在 D0 依赖未就绪时，
成员 1 用公开的 webdataset 镜像（与原图同尺寸、5 caption/图）做组合探针与 E0 冒烟。
产物与 prepare_flickr30k.py 完全同 schema，可直接被 evals-mm/loaders.py 读取。

用法：
    python scripts/prepare_flickr30k_wds.py --tars "data/flickr30k/test_tars/*.tar"
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tars", default=str(ROOT / "data/flickr30k/test_tars/*.tar"))
    parser.add_argument("--image-dir", type=Path, default=ROOT / "data/flickr30k/images")
    parser.add_argument("--output", type=Path, default=ROOT / "data/flickr30k/splits/test.jsonl")
    args = parser.parse_args()

    args.image_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(args.tars))
    if not files:
        raise SystemExit(f"没有匹配到 tar：{args.tars}")

    images = 0
    rows: list[dict] = []
    for path in files:
        with tarfile.open(path) as tf:
            members = {m.name: m for m in tf.getmembers()}
            keys = sorted({Path(n).stem for n in members if n.endswith(".jpg")})
            for key in keys:
                img_member = members[f"{key}.jpg"]
                txt_member = members.get(f"{key}.txt")
                if txt_member is None:
                    continue
                img_bytes = tf.extractfile(img_member).read()
                (args.image_dir / f"{key}.jpg").write_bytes(img_bytes)
                captions = [
                    line.strip()
                    for line in io.BytesIO(tf.extractfile(txt_member).read()).read().decode("utf-8").splitlines()
                    if line.strip()
                ]
                for cid, caption in enumerate(captions):
                    rows.append({
                        "image": f"{key}.jpg",
                        "caption": caption,
                        "image_id": key,
                        "caption_id": cid,
                    })
                images += 1

    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"完成：{images} 张图 / {len(rows)} 条 caption -> {args.output}")
    print("提示：这是公开镜像，不是 Kaggle 原始包；只用于组合探针与 D0 冒烟，正式 E0 以成员 2 的 Karpathy split 为准。")


if __name__ == "__main__":
    main()
