"""一键拉取全部数据集并摆成 dev-doc 2.2 节的目录结构。

用法：
    python scripts/download_data.py --all
    python scripts/download_data.py --dataset flickr30k
    python scripts/download_data.py --list

前置条件：
    - Kaggle 数据集需要 ~/.kaggle/kaggle.json
      （Kaggle 账号 -> Settings -> API -> Create New Token）
    - Winoground 需要先在 HF 页面同意条款，并执行 huggingface-cli login
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

KAGGLE_DATASETS = {
    "flickr8k": ("adityajn105/flickr8k", "flickr8k"),
    "flickr30k": ("nunenuh/flickr30k", "flickr30k"),
    "coco2017": ("awsaf49/coco-2017-dataset", "coco2017"),
}


def check_kaggle() -> None:
    if not (Path.home() / ".kaggle" / "kaggle.json").exists():
        sys.exit("缺少 ~/.kaggle/kaggle.json，请先从 Kaggle 账号设置页创建 API token")
    if shutil.which("kaggle") is None:
        sys.exit("缺少 kaggle CLI，请先 pip install kaggle")


def download_kaggle(slug: str, target: str) -> None:
    check_kaggle()
    dest = DATA_ROOT / target
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[kaggle] {slug} -> {dest}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"],
        check=True,
    )


def download_winoground() -> None:
    """Winoground 不落盘图片，仅验证可加载并打印样本数。"""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("缺少 datasets，请先 pip install datasets")
    print("[hf] facebook/winoground（需已同意条款并 huggingface-cli login）")
    ds = load_dataset("facebook/winoground", split="test")
    print(f"[hf] winoground 加载成功：{len(ds)} 对样本，评测时由 evals-mm 直接调用")


def download_aro() -> None:
    dest = DATA_ROOT / "aro"
    if dest.exists():
        print(f"[aro] 已存在：{dest}")
        return
    print("[aro] clone ARO 生成脚本仓库")
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/ARO-benchmark/ARO.git", str(dest)],
        check=True,
    )
    print("[aro] 请按该仓库 README 运行生成脚本产出组合样本")


def download_sugarcrepe() -> None:
    print("[sugarcrepe++] 请从 CLIC 项目页面获取：https://clic-compositional-clip.github.io")
    print("               下载后放入 data/sugarcrepepp/（许可以其页面为准，本脚本不自动拉取）")


DOWNLOADERS = {
    **{k: (lambda s=s, t=t: download_kaggle(s, t)) for k, (s, t) in KAGGLE_DATASETS.items()},
    "winoground": download_winoground,
    "aro": download_aro,
    "sugarcrepepp": download_sugarcrepe,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", choices=list(DOWNLOADERS), help="只下载指定数据集")
    parser.add_argument("--all", action="store_true", help="下载全部（sugarcrepepp 仍需手动）")
    parser.add_argument("--list", action="store_true", help="列出支持的数据集")
    args = parser.parse_args()

    if args.list:
        for name in DOWNLOADERS:
            print(f"  {name}")
        return

    DATA_ROOT.mkdir(exist_ok=True)
    if args.all:
        for name, fn in DOWNLOADERS.items():
            print(f"\n=== {name} ===")
            fn()
    elif args.dataset:
        DOWNLOADERS[args.dataset]()
    else:
        parser.print_help()

    print("\n完成。目录结构应匹配 docs/dev-doc.md 2.2 节；Flickr30k 请确认使用 Karpathy split。")


if __name__ == "__main__":
    main()
