"""Make an A2_eval directory portable (Windows absolute paths -> Colab/Linux paths).

Run this on Colab after uploading A2_eval, before evaluate_role3_models.py:

    python scripts/make_a2_eval_portable.py \
      --root /content/compoclip_data/A2_eval \
      --data-root /content/CompoCLIP/data

The script rewrites image paths in data/*.jsonl and refreshes the dataset hashes
in benchmark_manifest.json and selection_protocol.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rewrite_rows(path: Path, new_root: Path, data_root: Path) -> int:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    changed = 0
    for row in rows:
        image = str(row.get("image", "")).replace("\\", "/")
        name = Path(image).name
        subset = str(row.get("subset", ""))
        if subset == "flickr_order":
            new = data_root / "flickr30k" / "images" / name
        elif subset in ("vg_relation", "vg_attribution"):
            new = new_root / "images" / "vg" / name
        elif subset == "coco_order":
            new = new_root / "images" / "coco2014" / name
        elif path.name == "coco.jsonl":
            new = new_root / "images" / "coco2017" / name
        else:
            # SugarCrepe and any remaining COCO-2017 rows
            new = new_root / "images" / "coco2017" / name
        row["image"] = str(new)
        changed += 1
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="A2_eval directory on the VM")
    parser.add_argument("--data-root", type=Path, required=True, help="repo data directory with flickr30k")
    args = parser.parse_args()

    root = args.root.resolve()
    data_root = args.data_root.resolve()
    files = ["aro.jsonl", "sugarcrepe.jsonl", "coco.jsonl", "coco_order.jsonl", "flickr_order.jsonl"]
    hashes = {}
    for name in files:
        path = root / "data" / name
        if not path.exists():
            print(f"skip missing {path}")
            continue
        n = rewrite_rows(path, root, data_root)
        hashes[name.replace(".jsonl", "")] = sha256(path)
        print(f"rewrote {path} ({n} rows)")

    # Refresh dataset hashes in manifest/protocol.
    manifest_path = root / "benchmark_manifest.json"
    protocol_path = root / "selection_protocol.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    for bench, key in (("aro", "aro"), ("sugarcrepe", "sugarcrepe"), ("coco", "coco")):
        if key not in hashes:
            continue
        if bench in manifest.get("benchmarks", {}):
            manifest["benchmarks"][bench]["dataset_sha256"] = hashes[key]
        expected = protocol.get("expected_data", {}).get("benchmarks", {})
        if bench in expected:
            expected[bench]["dataset_sha256"] = hashes[key]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    print("updated benchmark_manifest.json and selection_protocol.json")


if __name__ == "__main__":
    main()
