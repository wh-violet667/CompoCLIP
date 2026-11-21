"""Role-3 unified evaluation for E0 / E3 / E4b on the frozen A2 benchmark data.

Uses the frozen A2 benchmark records and scoring functions. Runs on CPU for
local verification or on CUDA for the formal GPU evaluation. E1/E2/E2a
ARO/SugarCrepe results are reused from the E-best selection archive; this script
fills the missing models.

Usage:
    python scripts/evaluate_role3_models.py --tags e0 e3 e4b --benchmarks sugarcrepe
    python scripts/evaluate_role3_models.py --tags e3 e4b --benchmarks aro sugarcrepe coco --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "A2_eval_tools"))

import open_clip  # noqa: E402
import evaluate_models as em  # noqa: E402


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_e0(device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", precision="fp32", device=device
    )
    model.float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, preprocess, open_clip.get_tokenizer("ViT-B-32")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, default=ROOT / "A2_eval")
    parser.add_argument("--toolkit", type=Path, default=ROOT / "A2" / "E_best" / "workspace")
    parser.add_argument("--tags", nargs="+", default=["e3", "e4b"], choices=["e0", "e1", "e2", "e2a", "e3", "e4b"])
    parser.add_argument("--benchmarks", nargs="+", default=["aro", "sugarcrepe", "coco"],
                        choices=["aro", "sugarcrepe", "coco"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-batch", type=int, default=16)
    parser.add_argument("--text-batch", type=int, default=64)
    args = parser.parse_args()

    work = args.work.resolve()
    protocol = read_json(work / "selection_protocol.json")
    manifest = read_json(work / "benchmark_manifest.json")
    expected_data = protocol["expected_data"]
    rows = {}
    for name in args.benchmarks:
        if name == "coco":
            rows[name] = em.read_coco(work, manifest["benchmarks"]["coco"], expected_data["benchmarks"]["coco"])
        else:
            rows[name] = em.read_benchmark(work, name, manifest["benchmarks"][name], expected_data["benchmarks"][name])
    print(f"loaded benchmarks: { {k: len(v) for k, v in rows.items()} }", flush=True)

    eval_args = SimpleNamespace(device=args.device, image_batch=args.image_batch, text_batch=args.text_batch)
    for tag in args.tags:
        output_dir = work / "results" / tag
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {tag} ===", flush=True)
        started = time.time()
        if tag == "e0":
            model, preprocess, tokenizer = load_e0(args.device)
            checkpoint_sha = "e0_pretrained_laion2b_s34b_b79k"
        else:
            checkpoint = ROOT / "A2_models" / tag / f"{tag}_model.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            model, preprocess, tokenizer = em.load_model(checkpoint, args.device)
            checkpoint_sha = em.sha256_file(checkpoint)
        evaluation = {
            "tag": tag, "benchmarks": {},
            "checkpoint_sha256": checkpoint_sha,
            "runtime": em.runtime_versions(),
            "precision": "fp32",
        }
        for name in args.benchmarks:
            expected = expected_data["benchmarks"][name]
            identity = {
                "tag": tag, "benchmark": name, "checkpoint_sha256": checkpoint_sha,
                "dataset_sha256": expected["dataset_sha256"],
                "image_batch": args.image_batch, "text_batch": args.text_batch,
            }
            cache_path = work / "cache" / tag / f"{name}_features.npz"
            print(f"{tag}/{name}: start", flush=True)
            if name == "coco":
                result = em.evaluate_coco(model, preprocess, tokenizer, rows[name], expected,
                                          output_dir, cache_path, identity, eval_args)
            else:
                result = em.evaluate_benchmark(model, preprocess, tokenizer, rows[name], name, expected,
                                               output_dir, cache_path, identity, eval_args)
            evaluation["benchmarks"][name] = result
            (output_dir / "evaluation.partial.json").write_text(
                json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{tag}/{name}: done", flush=True)
        evaluation["elapsed_seconds"] = round(time.time() - started, 1)
        (output_dir / "evaluation.json").write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"=== {tag} done in {evaluation['elapsed_seconds']}s -> {output_dir} ===", flush=True)
        del model, preprocess, tokenizer


if __name__ == "__main__":
    main()
