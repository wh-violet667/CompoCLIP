"""Build Drive-ready tar packages for the Colab notebook (local helper)."""
from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "colab_packages"
OUT.mkdir(exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def add_tree(tar: tarfile.TarFile, path: Path, arcname: str, skip: set[str] | None = None):
    skip = skip or set()
    if not path.exists():
        print(f"skip missing {path}")
        return
    for p in path.rglob("*"):
        if any(part in skip for part in p.relative_to(path).parts):
            continue
        rel = Path(arcname) / p.relative_to(path)
        tar.add(p, arcname=str(rel), recursive=False)


def main() -> None:
    packages = [
        ("A2_eval.tar", ROOT / "A2_eval", "A2_eval", {"downloads", "models"}),
        ("A2_models.tar", ROOT / "A2_models", "A2_models", set()),
    ]
    for name, path, arc, skip in packages:
        target = OUT / name
        print(f"building {target} ...", flush=True)
        with tarfile.open(target, "w") as tar:
            add_tree(tar, path, arc, skip)
        print(f"{target} {target.stat().st_size / 1e9:.2f} GB", flush=True)

    target = OUT / "data_role3.tar"
    print(f"building {target} ...", flush=True)
    with tarfile.open(target, "w") as tar:
        for sub in ("golden", "flickr30k", "winoground"):
            add_tree(tar, ROOT / "data" / sub, f"data/{sub}")
    print(f"{target} {target.stat().st_size / 1e9:.2f} GB", flush=True)

    # E0 is not rerun on the GPU; its archived per-sample files are required by
    # assemble_role3_results.py for the common table and E0-vs-E3 paired stats.
    target = OUT / "role3_archive.tar"
    print(f"building {target} ...", flush=True)
    archive_files = [
        "evals-mm/results/e0_aro_persample.npz",
        "evals-mm/results/e0_sugarcrepe_persample.npz",
        "evals-mm/results/e0_winoground_persample.npz",
        "evals-mm/results/flickr30k.csv",
        "evals-mm/results/sugarcrepe.csv",
    ]
    with tarfile.open(target, "w") as tar:
        for name in archive_files:
            path = ROOT / name
            if path.exists():
                tar.add(path, arcname=name)
            else:
                print(f"skip missing {path}")
    print(f"{target} {target.stat().st_size / 1e9:.2f} GB", flush=True)

    with (OUT / "SHA256SUMS.txt").open("w", encoding="utf-8") as f:
        for p in sorted(OUT.glob("*.tar")):
            f.write(f"{sha256(p)}  {p.name}\n")
    print("done")


if __name__ == "__main__":
    main()
