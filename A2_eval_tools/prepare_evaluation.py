"""Prepare model identities, benchmark records, and images for E-best selection."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import getpass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import urllib.request
from zipfile import ZipFile

from select_ebest import read_json, require, sha256_file


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_lines(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_lines(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def download(url, target):
    target = Path(target)
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "CompoCLIP-Ebest"})
            with urllib.request.urlopen(request, timeout=75) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            require(temporary.stat().st_size > 0, "Empty download")
            os.replace(temporary, target)
            return target
        except Exception:
            if attempt == 3:
                raise RuntimeError(f"Download failed: {target.name}. Rerun to retry missing files.") from None
            time.sleep(2 ** attempt)


def safe_extract(archive, destination, prefix=None):
    destination = Path(destination).resolve()
    with ZipFile(archive) as package:
        for item in package.infolist():
            if item.is_dir() or (prefix and not item.filename.startswith(prefix)):
                continue
            target = (destination / item.filename).resolve()
            require(target.is_relative_to(destination), "Archive entry escapes output directory")
            if target.is_file() and target.stat().st_size == item.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".part")
            with package.open(item) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
            os.replace(temporary, target)


def find_file(roots, filename):
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        direct = root / filename
        if direct.is_file():
            return direct
        for path in root.rglob(filename):
            if path.is_file():
                return path
    return None


def environment(work):
    import importlib
    packages = {"open_clip": "open_clip_torch==4.2.2", "PIL": "Pillow",
                "numpy": "numpy", "gdown": "gdown", "nltk": "nltk==3.9.2",
                "spacy": "spacy==3.8.15"}
    missing = []
    for module, package in packages.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
    if missing:
        print("Installing missing packages: " + ", ".join(p.split("==")[0] for p in missing), flush=True)
        with (work / "dependencies.log").open("a", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-m", "pip", "install", *missing], stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    import torch
    # CPU-mode local evaluation: CUDA check disabled in this copy.
    import spacy
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        with (work / "dependencies.log").open("a", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-m", "pip", "install",
                            "https://github.com/explosion/spacy-models/releases/download/"
                            "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    import nltk
    for name in ("punkt", "punkt_tab"):
        if not nltk.download(name, quiet=True):
            raise RuntimeError(f"NLTK resource unavailable: {name}")
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "torchvision", "open_clip_torch", "Pillow", "numpy", "spacy", "en_core_web_sm", "nltk")}
    result = {"gpu": "cpu-local", "versions": versions}
    write_json(work / "environment.json", result)
    print(json.dumps(result), flush=True)


def prepare_models(toolkit, work, roots):
    catalog = read_json(toolkit / "inputs/models.json")
    models = {}
    for tag, entry in catalog.items():
        model_root = work / "models" / tag
        model_root.mkdir(parents=True, exist_ok=True)
        expected = dict(entry["identity"])
        for key, item in entry["files"].items():
            expected[key + "_path"] = str(Path("models") / tag / Path(item["member"]).name)
        complete = all((work / expected[key + "_path"]).is_file()
                       and sha256_file(work / expected[key + "_path"]) == item["sha256"]
                       for key, item in entry["files"].items())
        if not complete:
            archive = find_file(roots, entry["archive"])
            if archive is None:
                import gdown
                url = getpass.getpass(f"{tag.upper()} handoff ZIP Drive link (hidden): ").strip()
                require(bool(url), f"Upload {entry['archive']} to /content and rerun.")
                archive = work / "downloads" / entry["archive"]
                archive.parent.mkdir(parents=True, exist_ok=True)
                temporary = archive.with_name(archive.name + ".part")
                try:
                    result = gdown.download(url=url, output=str(temporary), fuzzy=True, quiet=True)
                    require(result and temporary.is_file(), "Download did not complete")
                    os.replace(temporary, archive)
                except Exception:
                    raise RuntimeError(f"{tag} download failed. Upload the handoff ZIP and rerun.") from None
            print(f"Checking {tag.upper()} handoff...", flush=True)
            require(sha256_file(archive) == entry["archive_sha256"], f"Wrong or incomplete {tag} handoff ZIP")
            with ZipFile(archive) as package:
                for key, item in entry["files"].items():
                    target = work / expected[key + "_path"]
                    temporary = target.with_name(target.name + ".part")
                    with package.open(item["member"]) as source, temporary.open("wb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                    require(sha256_file(temporary) == item["sha256"], f"Invalid {tag}/{key}")
                    os.replace(temporary, target)
        models[tag] = expected
        print(f"{tag.upper()} model and provenance ready.", flush=True)
    return models


def valid_image(path):
    from PIL import Image
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.load()
    except (OSError, ValueError, SyntaxError):
        path.replace(path.with_name(path.name + ".invalid"))
        return False
    return True


def extract_images(archive, image_root, needed, prefix=None):
    with ZipFile(archive) as package:
        entries = {Path(item.filename).name: item for item in package.infolist()
                   if not item.is_dir() and (prefix is None or item.filename.startswith(prefix))}
        for name in needed:
            target = image_root / name
            if target.is_file():
                continue
            require(name in entries, f"Missing ARO image in archive: {name}")
            temporary = target.with_name(target.name + ".part")
            with package.open(entries[name]) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
            require(valid_image(temporary), f"Invalid ARO image: {name}")
            os.replace(temporary, target)


def vg_images(toolkit, work, roots, needed):
    image_root = work / "images" / "vg"
    image_root.mkdir(parents=True, exist_ok=True)
    missing = [name for name in sorted(needed) if not valid_image(image_root / name)]
    if not missing:
        return image_root
    a1 = find_file(roots, "pkg_eval_images.zip")
    if a1:
        extract_images(a1, image_root, missing, "data/aro/images/")
    else:
        import gdown
        archive = work / "downloads" / "vgr_vga_images.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.is_file():
            print("Downloading ARO region images...", flush=True)
            temporary = archive.with_name(archive.name + ".part")
            try:
                result = gdown.download(id="1qaPlrwhGNMrR3a11iopZUT_GPP_LrgP9", output=str(temporary), quiet=True)
                require(result and temporary.is_file(), "ARO download failed")
                os.replace(temporary, archive)
            except Exception:
                raise RuntimeError("ARO image download failed. Upload pkg_eval_images.zip and rerun.") from None
        extract_images(archive, image_root, missing)
    return image_root


def download_images(entries, work):
    pending = [item for item in entries if not valid_image(work / "images" / item["image"])]
    print(f"COCO images: {len(entries) - len(pending)}/{len(entries)} already available.", flush=True)
    def fetch(item):
        target = download(item["url"], work / "images" / item["image"])
        require(valid_image(target), f"Invalid image download: {target.name}")
    failures = []
    start = last = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = {pool.submit(fetch, item): item for item in pending}
        for count, future in enumerate(as_completed(jobs), 1):
            try:
                future.result()
            except Exception:
                failures.append(jobs[future]["image"])
            now = time.monotonic()
            if now - last >= 20 or count == len(pending):
                print(f"Images downloaded: {count}/{len(pending)}; failed: {len(failures)}; "
                      f"elapsed: {(now-start)/60:.1f} min.", flush=True)
                last = now
    write_json(work / "image_download_failures.json", failures)
    require(not failures, f"{len(failures)} image downloads failed. Rerun preparation to retry them.")


def order_rows(toolkit, work, annotation_file, subset, image_root):
    cached = work / "data" / (subset + ".jsonl")
    source_hash = sha256_file(toolkit / "inputs" / annotation_file)
    settings = {"annotation_sha256": source_hash, "seed": 42, "max_words": 30,
                "generator_sha256": sha256_file(toolkit / "vendor" / "aro_text.py"),
                "spacy": importlib.metadata.version("spacy"),
                "model": importlib.metadata.version("en_core_web_sm"),
                "nltk": importlib.metadata.version("nltk"), "image_root": str(image_root)}
    cache_meta = cached.with_suffix(".meta.json")
    if cached.is_file() and cache_meta.is_file():
        previous = read_json(cache_meta)
        require(previous["settings"] == settings and previous["sha256"] == sha256_file(cached),
                f"{subset} preparation changed. Use a new WORK directory to freeze a new scope.")
        return read_lines(cached)
    import numpy as np
    sys.path.insert(0, str(toolkit / "vendor"))
    from aro_text import TextShuffler, pre_caption
    random.seed(42)
    np.random.seed(42)
    shuffler = TextShuffler()
    functions = [shuffler.shuffle_nouns_and_adj, shuffler.shuffle_allbut_nouns_and_adj,
                 shuffler.shuffle_within_trigrams, shuffler.shuffle_trigrams]
    annotations = read_json(toolkit / "inputs" / annotation_file)
    rows = []
    last = time.monotonic()
    for image_index, entry in enumerate(annotations):
        image = str((image_root / Path(entry["image"]).name).resolve())
        for caption_index, caption in enumerate(entry["caption"]):
            rows.append({"id": f"{subset}:{image_index}:{caption_index}", "image": image,
                         "subset": subset, "positive": pre_caption(caption, 30),
                         "negatives": [pre_caption(fn(caption), 30) for fn in functions],
                         "tie_policy": "positive_first"})
        if time.monotonic() - last > 25:
            print(f"Preparing {subset}: {image_index+1}/{len(annotations)} images.", flush=True)
            last = time.monotonic()
    write_lines(cached, rows)
    write_json(cache_meta, {"settings": settings, "sha256": sha256_file(cached)})
    return rows


def image_inventory(work, rows):
    from PIL import Image
    paths = sorted({row["image"] for row in rows})
    records = []
    for filename in paths:
        path = Path(filename)
        require(path.is_file(), f"Missing image: {path.name}")
        with Image.open(path) as image:
            image.load()
        records.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(work / "image_inventory.json", {"count": len(records), "files": records})
    return sha256_file(work / "image_inventory.json")


def prepare(toolkit, work, data_root, include_coco=False):
    work.mkdir(parents=True, exist_ok=True)
    environment(work)
    roots = [Path("/content/compoclip_enterprise/role2_mainline_runs"), Path("/content"), work / "downloads"]
    models = prepare_models(toolkit, work, roots)
    source = read_json(toolkit / "inputs/source_manifest.json")
    for entry in source["files"]:
        require(sha256_file(toolkit / entry["path"]) == entry["sha256"], "Frozen source input was changed")
    flickr_root = data_root / "flickr30k/images"
    flickr = read_json(toolkit / "inputs/flickr30k_test.json")
    require(all((flickr_root / Path(row["image"]).name).is_file() for row in flickr),
            "Flickr30k images are missing. Set DATA_ROOT to the existing E2a data directory.")
    vg = read_lines(toolkit / "inputs/vg.jsonl")
    sugar = read_lines(toolkit / "inputs/sugarcrepe.jsonl")
    vg_images(toolkit, work, roots, {Path(row["image"]).name for row in vg})
    downloads = read_json(toolkit / "inputs/image_downloads.json")
    coco_groups = []
    if include_coco:
        annotation = data_root / "coco2017/annotations/captions_val2017.json"
        if not annotation.is_file():
            archive = download("https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip",
                               work / "downloads/annotations_trainval2017.zip")
            safe_extract(archive, data_root / "coco2017", "annotations/captions_val2017.json")
        data = read_json(annotation)
        grouped = {item["id"]: {"image_id": str(item["id"]),
                  "image": str((work / "images/coco2017" / item["file_name"]).resolve()), "captions": []}
                   for item in data["images"]}
        for item in data["annotations"]:
            grouped[item["image_id"]]["captions"].append(item["caption"].strip())
        coco_groups = list(grouped.values())
        require(len(coco_groups) == 5000 and all(row["captions"] for row in coco_groups),
                "COCO must have all 5000 validation images and their captions")
        have = {entry["image"] for entry in downloads}
        for row in coco_groups:
            key = "coco2017/" + Path(row["image"]).name
            if key not in have:
                downloads.append({"image": key, "url": "https://s3.amazonaws.com/images.cocodataset.org/val2017/" + Path(key).name})
    for item in downloads:
        target = work / "images" / item["image"]
        if target.is_file():
            continue
        name = Path(item["image"]).name
        old = data_root / "coco2017/val2017" / name
        if old.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(old, target)
    download_images(downloads, work)
    for row in vg + sugar:
        row["image"] = str((work / "images" / row["image"]).resolve())
    aro = vg + order_rows(toolkit, work, "coco_karpathy_test.json", "coco_order", work / "images/coco2014")
    aro += order_rows(toolkit, work, "flickr30k_test.json", "flickr_order", flickr_root)
    print("Checking image files and freezing benchmark identities...", flush=True)
    inventory_hash = image_inventory(work, aro + sugar + coco_groups)
    benchmarks = {}
    for name, rows in (("aro", aro), ("sugarcrepe", sugar)):
        counts = dict(Counter(row["subset"] for row in rows))
        require(counts == source["expected_counts"][name], f"Incomplete {name} scope: {counts}")
        write_lines(work / "data" / (name + ".jsonl"), rows)
        benchmarks[name] = {"path": "data/" + name + ".jsonl", "subsets": counts,
                            "dataset_sha256": sha256_file(work / "data" / (name + ".jsonl"))}
    if coco_groups:
        write_lines(work / "data/coco.jsonl", coco_groups)
        benchmarks["coco"] = {"path": "data/coco.jsonl", "dataset_sha256": sha256_file(work / "data/coco.jsonl"),
                              "subset": "val2017_5k", "n_images": len(coco_groups),
                              "n_text_queries": sum(len(row["captions"]) for row in coco_groups)}
    scope = {"benchmarks": benchmarks, "image_inventory_sha256": inventory_hash,
             "source_manifest_sha256": sha256_file(toolkit / "inputs/source_manifest.json")}
    fingerprint = hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()
    manifest = {"status": "ready", "data_fingerprint": fingerprint, "benchmarks": benchmarks,
                "image_inventory_path": "image_inventory.json", "image_inventory_sha256": inventory_hash}
    protocol = read_json(toolkit / "inputs/protocol_template.json")
    protocol.update(expected_models=models, expected_data={
        "data_fingerprint": fingerprint, "image_inventory_sha256": inventory_hash,
        "benchmarks": benchmarks})
    write_json(work / "benchmark_manifest.json", manifest)
    write_json(work / "selection_protocol.json", protocol)
    print("Preparation complete. ARO and SugarCrepe are frozen for all three models.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolkit", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--include-coco", action="store_true")
    args = parser.parse_args()
    prepare(args.toolkit.resolve(), args.work.resolve(), args.data_root.resolve(), args.include_coco)
