"""Evaluate frozen E-best candidates using complete ARO and SugarCrepe data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
from zipfile import BadZipFile

from select_ebest import (ARO, SUGAR, TAGS, is_sha256, read_json, require, sha256_file,
                          validate_evaluation, validate_protocol)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def work_path(work, value):
    path = Path(value)
    return path if path.is_absolute() else work / path


def stable_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def image_key(row):
    crop = row.get("crop")
    if crop is not None:
        require(isinstance(crop, list) and len(crop) == 4
                and all(type(v) in (int, float) and math.isfinite(v) for v in crop)
                and crop[2] > 0 and crop[3] > 0, "Invalid image crop")
        crop = tuple(float(v) for v in crop)
    return (str(Path(row["image"]).resolve()), crop)


def read_benchmark(work, name, record, expected):
    path = work_path(work, record["path"])
    require(path.is_file(), f"Missing prepared benchmark: {name}")
    digest = sha256_file(path)
    require(digest == record["dataset_sha256"] == expected["dataset_sha256"],
            f"Prepared dataset hash mismatch: {name}")
    rows = []
    ids = set()
    counts = {}
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            require(isinstance(row, dict), f"{name}:{line_number}: expected object")
            require(isinstance(row.get("id"), (str, int)) and not isinstance(row["id"], bool),
                    f"{name}:{line_number}: missing sample id")
            identifier = str(row["id"])
            require(identifier not in ids, f"{name}: duplicate sample id {identifier}")
            ids.add(identifier)
            row["id"] = identifier
            require(isinstance(row.get("image"), str) and Path(row["image"]).is_absolute()
                    and Path(row["image"]).is_file(), f"{name}:{identifier}: missing absolute image path")
            image_key(row)
            require(isinstance(row.get("positive"), str) and row["positive"].strip(),
                    f"{name}:{identifier}: missing positive text")
            require(isinstance(row.get("negatives"), list) and row["negatives"]
                    and all(isinstance(v, str) and v.strip() for v in row["negatives"]),
                    f"{name}:{identifier}: missing negative texts")
            subset = row.get("subset")
            require(subset in expected["subsets"], f"{name}:{identifier}: unexpected subset")
            if name == "aro" and subset in ("coco_order", "flickr_order"):
                require(len(row["negatives"]) == 4,
                        f"{name}:{identifier}: order samples require four negative captions")
                require(row.get("tie_policy") == "positive_first",
                        f"{name}:{identifier}: order ties must follow the official positive-first ordering")
            else:
                require(len(row["negatives"]) == 1,
                        f"{name}:{identifier}: pairwise samples require one negative caption")
                require(row.get("tie_policy", "negative_first") == "negative_first",
                        f"{name}:{identifier}: pairwise ties must fail")
            counts[subset] = counts.get(subset, 0) + 1
            rows.append(row)
    require(counts == expected["subsets"], f"{name}: frozen subset counts differ")
    return rows


def deduplicated_inputs(rows):
    images, image_indices = [], {}
    texts, text_indices = [], {}
    row_images, row_positives, row_negatives = [], [], []
    for row in rows:
        key = image_key(row)
        if key not in image_indices:
            image_indices[key] = len(images)
            images.append(key)
        row_images.append(image_indices[key])
        indices = []
        for text in [row["positive"], *row["negatives"]]:
            if text not in text_indices:
                text_indices[text] = len(texts)
                texts.append(text)
            indices.append(text_indices[text])
        row_positives.append(indices[0])
        row_negatives.append(indices[1:])
    return images, texts, row_images, row_positives, row_negatives


def score_rows(image_features, text_features, image_indices, positive_indices, negative_indices,
               positive_first=None):
    import numpy as np
    positives = np.empty(len(image_indices), dtype=np.float32)
    maximum_negatives = max(map(len, negative_indices))
    negatives = np.full((len(negative_indices), maximum_negatives), np.nan, dtype=np.float32)
    for i, indices in enumerate(negative_indices):
        candidate_indices = [positive_indices[i], *indices]
        scores = text_features[candidate_indices] @ image_features[image_indices[i]]
        positives[i] = scores[0]
        negatives[i, :len(indices)] = scores[1:]
    negative_max = np.nanmax(negatives, axis=1)
    correct = positives > negative_max
    if positive_first is not None:
        positive_first = np.asarray(positive_first, dtype=bool)
        correct = correct | (positive_first & (positives == negative_max))
    return positives, negatives, negative_max, correct


class Progress:
    def __init__(self, label):
        self.label = label
        self.start = time.monotonic()
        self.last = self.start

    def update(self, done, total):
        now = time.monotonic()
        if done == total or now - self.last >= 30:
            elapsed = now - self.start
            remaining = elapsed * (total - done) / done if done else 0
            print(f"{self.label}: {done}/{total}; elapsed {elapsed / 60:.1f} min; "
                  f"ETA {remaining / 60:.1f} min", flush=True)
            self.last = now


def verify_image_inventory(work, manifest, expected_data, benchmarks):
    inventory_name = manifest.get("image_inventory_path")
    require(isinstance(inventory_name, str) and inventory_name,
            "Missing prepared image inventory path")
    inventory_hash = expected_data.get("image_inventory_sha256")
    require(is_sha256(inventory_hash)
            and inventory_hash == manifest.get("image_inventory_sha256"),
            "Image inventory differs from the frozen protocol")
    inventory_path = work_path(work, inventory_name)
    require(inventory_path.is_file() and sha256_file(inventory_path) == inventory_hash,
            "Prepared image inventory hash mismatch")
    inventory = read_json(inventory_path)
    require(isinstance(inventory, dict) and isinstance(inventory.get("files"), list),
            "Invalid image inventory")
    files = inventory["files"]
    require(type(inventory.get("count")) is int and inventory["count"] == len(files) > 0,
            "Image inventory count mismatch")
    indexed = {}
    for entry in files:
        require(isinstance(entry, dict) and isinstance(entry.get("path"), str)
                and Path(entry["path"]).is_absolute(), "Invalid image inventory path")
        filename = Path(entry["path"]).resolve()
        require(filename not in indexed, "Image inventory contains duplicate file paths")
        require(type(entry.get("bytes")) is int and entry["bytes"] > 0
                and is_sha256(entry.get("sha256")), "Invalid image inventory identity")
        indexed[filename] = entry
    referenced = {Path(row["image"]).resolve() for rows in benchmarks.values() for row in rows}
    require(referenced <= set(indexed), "Benchmark images are missing from the frozen inventory")
    print(f"Verifying {len(files)} frozen image files before evaluation or cache reuse", flush=True)
    progress = Progress("Image integrity")
    for index, (filename, entry) in enumerate(indexed.items(), 1):
        require(filename.is_file() and filename.stat().st_size == entry["bytes"],
                f"Image size differs from the frozen inventory: {filename.name}")
        require(sha256_file(filename) == entry["sha256"],
                f"Image hash differs from the frozen inventory: {filename.name}")
        progress.update(index, len(files))
    return inventory_hash


def valid_features(values, expected):
    import numpy as np
    if values.ndim != 2 or values.shape != (expected, 512):
        return False
    if values.dtype != np.float32 or not np.isfinite(values).all():
        return False
    return bool(np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=2e-4, rtol=2e-4))


def load_cached_features(cache_path, identity, image_count, text_count):
    import numpy as np
    metadata_path = cache_path.with_suffix(".json")
    if not (cache_path.is_file() and metadata_path.is_file()):
        return None, None
    try:
        metadata = read_json(metadata_path)
        if (metadata.get("identity") != identity
                or metadata.get("npz_sha256") != sha256_file(cache_path)):
            return None, None
        with np.load(cache_path, allow_pickle=False) as archive:
            image_features, text_features = archive["image"], archive["text"]
        if valid_features(image_features, image_count) and valid_features(text_features, text_count):
            return image_features, text_features
    except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, BadZipFile):
        pass
    return None, None


def encode_images(model, preprocess, images, device, batch_size, label):
    import numpy as np
    import torch
    from PIL import Image
    features = []
    progress = Progress(label + " images")
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = []
            for filename, crop in images[start:start + batch_size]:
                with Image.open(filename) as original:
                    image = original.convert("RGB")
                    if crop is not None:
                        x, y, width, height = crop
                        image = image.crop((x, y, x + width, y + height))
                    batch.append(preprocess(image))
            encoded = model.encode_image(torch.stack(batch).to(device))
            encoded = torch.nn.functional.normalize(encoded.float(), dim=-1)
            features.append(encoded.cpu().numpy())
            progress.update(min(start + batch_size, len(images)), len(images))
    values = np.concatenate(features).astype(np.float32, copy=False)
    require(valid_features(values, len(images)), "Invalid encoded image features")
    return values


def encode_texts(model, tokenizer, texts, device, batch_size, label):
    import numpy as np
    import torch
    features = []
    progress = Progress(label + " texts")
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            tokens = tokenizer(texts[start:start + batch_size]).to(device)
            encoded = model.encode_text(tokens)
            encoded = torch.nn.functional.normalize(encoded.float(), dim=-1)
            features.append(encoded.cpu().numpy())
            progress.update(min(start + batch_size, len(texts)), len(texts))
    values = np.concatenate(features).astype(np.float32, copy=False)
    require(valid_features(values, len(texts)), "Invalid encoded text features")
    return values


def feature_cache(model, preprocess, tokenizer, rows, cache_path, identity, device,
                  image_batch, text_batch):
    import numpy as np
    images, texts, image_indices, positive_indices, negative_indices = deduplicated_inputs(rows)
    metadata_path = cache_path.with_suffix(".json")
    fingerprint = {**identity, "image_count": len(images), "text_count": len(texts),
                   "input_order_sha256": stable_hash({"images": images, "texts": texts})}
    image_features, text_features = load_cached_features(cache_path, fingerprint, len(images), len(texts))
    if image_features is None:
        print(f"Encoding {identity['tag']}/{identity['benchmark']}: "
              f"{len(images)} unique image/crop pairs; {len(texts)} unique texts", flush=True)
        image_features = encode_images(model, preprocess, images, device, image_batch,
                                       identity["benchmark"])
        text_features = encode_texts(model, tokenizer, texts, device, text_batch,
                                     identity["benchmark"])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(cache_path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, image=image_features, text=text_features)
        os.replace(temporary, cache_path)
        write_json(metadata_path, {"identity": fingerprint, "npz_sha256": sha256_file(cache_path)})
    else:
        print(f"Reusing verified features for {identity['tag']}/{identity['benchmark']}", flush=True)
    return score_rows(image_features, text_features, image_indices, positive_indices, negative_indices,
                      [row.get("tie_policy") == "positive_first" for row in rows])


def evaluate_benchmark(model, preprocess, tokenizer, rows, bench, expected, output_dir,
                       cache_path, identity, args):
    import numpy as np
    positives, negatives, negative_max, correct = feature_cache(
        model, preprocess, tokenizer, rows, cache_path, identity, args.device,
        args.image_batch, args.text_batch)
    subsets = np.asarray([r["subset"] for r in rows])
    metrics = {}
    for subset in expected["subsets"]:
        selected = subsets == subset
        metrics[subset] = {"accuracy": float(correct[selected].mean()), "n": int(selected.sum())}
    sample_path = output_dir / (bench + "_persample.npz")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = sample_path.with_name(sample_path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, ids=np.asarray([r["id"] for r in rows]), subsets=subsets,
                            positive_score=positives, negative_scores=negatives,
                            negative_count=np.asarray([len(r["negatives"]) for r in rows]),
                            max_negative_score=negative_max, correct=correct,
                            positive_first_ties=np.asarray([r.get("tie_policy") == "positive_first" for r in rows]),
                            checkpoint_sha256=np.asarray(identity["checkpoint_sha256"]),
                            dataset_sha256=np.asarray(identity["dataset_sha256"]))
    os.replace(temporary, sample_path)
    return {"dataset_sha256": expected["dataset_sha256"], "subsets": metrics,
            "macro_accuracy": sum(v["accuracy"] for v in metrics.values()) / len(metrics),
            "persample_path": sample_path.name, "persample_sha256": sha256_file(sample_path),
            "scoring": "positive versus all negatives; ARO order ties follow official positive-first argmax, pairwise ties fail"}


def read_coco(work, record, expected):
    path = work_path(work, record["path"])
    require(sha256_file(path) == record["dataset_sha256"] == expected["dataset_sha256"],
            "COCO prepared dataset hash mismatch")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    require(len(rows) == expected["n_images"] == 5000, "COCO requires all 5000 validation images")
    identifiers, paths = set(), set()
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get("image"), str), "Invalid COCO record")
        require(Path(row["image"]).is_absolute() and Path(row["image"]).is_file(), "Missing COCO image")
        require(isinstance(row.get("image_id"), (str, int)) and not isinstance(row["image_id"], bool),
                "Missing COCO image id")
        identifier, image = str(row["image_id"]), str(Path(row["image"]).resolve())
        require(identifier not in identifiers and image not in paths, "COCO images must be unique")
        identifiers.add(identifier)
        paths.add(image)
        require(isinstance(row.get("captions"), list) and row["captions"]
                and all(isinstance(caption, str) and caption.strip() for caption in row["captions"]),
                "Missing COCO captions")
    require(sum(len(row["captions"]) for row in rows) == expected["n_text_queries"],
            "COCO caption count differs from frozen scope")
    return rows


def retrieval_top_one(image_features, text_features, caption_images, chunk_size=256):
    import numpy as np
    caption_images = np.asarray(caption_images, dtype=np.int64)
    text_predictions, text_scores = [], []
    for start in range(0, len(text_features), chunk_size):
        scores = text_features[start:start + chunk_size] @ image_features.T
        text_predictions.append(np.argmax(scores, axis=1))
        text_scores.append(np.max(scores, axis=1))
    image_predictions, image_scores = [], []
    for start in range(0, len(image_features), chunk_size):
        scores = image_features[start:start + chunk_size] @ text_features.T
        image_predictions.append(np.argmax(scores, axis=1))
        image_scores.append(np.max(scores, axis=1))
    text_predictions, image_predictions = np.concatenate(text_predictions), np.concatenate(image_predictions)
    return {"text_correct": text_predictions == caption_images,
            "image_correct": caption_images[image_predictions] == np.arange(len(image_features)),
            "text_top_image": text_predictions, "image_top_caption": image_predictions,
            "text_top_score": np.concatenate(text_scores), "image_top_score": np.concatenate(image_scores)}


def evaluate_coco(model, preprocess, tokenizer, rows, expected, output_dir, cache_path, identity, args):
    import numpy as np
    images = [image_key(row) for row in rows]
    unique_texts, text_indices = [], {}
    caption_indices, caption_images = [], []
    for image_index, row in enumerate(rows):
        for caption in row["captions"]:
            if caption not in text_indices:
                text_indices[caption] = len(unique_texts)
                unique_texts.append(caption)
            caption_indices.append(text_indices[caption])
            caption_images.append(image_index)
    metadata_path = cache_path.with_suffix(".json")
    fingerprint = {**identity, "input_order_sha256": stable_hash({"images": images, "texts": unique_texts})}
    image_features, text_features = load_cached_features(
        cache_path, fingerprint, len(images), len(unique_texts))
    if image_features is None:
        image_features = encode_images(model, preprocess, images, args.device, args.image_batch, "coco")
        text_features = encode_texts(model, tokenizer, unique_texts, args.device, args.text_batch, "coco")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(cache_path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, image=image_features, text=text_features)
        os.replace(temporary, cache_path)
        write_json(metadata_path, {"identity": fingerprint, "npz_sha256": sha256_file(cache_path)})
    else:
        print(f"Reusing verified features for {identity['tag']}/coco", flush=True)
    print(f"Scoring {identity['tag']}/coco retrieval in bounded chunks", flush=True)
    scores = retrieval_top_one(image_features, text_features[caption_indices], caption_images)
    sample_path = output_dir / "coco_persample.npz"
    temporary = sample_path.with_name(sample_path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, image_ids=np.asarray([str(row["image_id"]) for row in rows]),
                            caption_image_index=np.asarray(caption_images), **scores,
                            checkpoint_sha256=np.asarray(identity["checkpoint_sha256"]),
                            dataset_sha256=np.asarray(identity["dataset_sha256"]))
    os.replace(temporary, sample_path)
    return {"dataset_sha256": expected["dataset_sha256"], "subset": "val2017_5k",
            "n_images": len(rows), "n_text_queries": len(caption_images),
            "t2i_R@1": float(scores["text_correct"].mean()),
            "i2t_R@1": float(scores["image_correct"].mean()),
            "persample_path": sample_path.name, "persample_sha256": sha256_file(sample_path),
            "tie_policy": "first index in the frozen image/caption order"}


def load_model(checkpoint_path, device):
    import open_clip
    import torch
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(isinstance(state, dict) and state and all(torch.is_tensor(v) for v in state.values()),
            "Expected a bare model state_dict")
    require(all(bool(torch.isfinite(v).all()) for v in state.values()), "Checkpoint contains nonfinite tensors")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None,
                                                                 precision="fp32", device="cpu")
    model.load_state_dict(state, strict=True)
    del state
    model.float().eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, preprocess, open_clip.get_tokenizer("ViT-B-32")


def runtime_versions():
    import importlib.metadata
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-local"
    return {"torch": torch.__version__, "open_clip_torch": importlib.metadata.version("open_clip_torch"),
            "torchvision": importlib.metadata.version("torchvision"),
            "numpy": importlib.metadata.version("numpy"), "Pillow": importlib.metadata.version("Pillow"),
            "cuda": torch.version.cuda or "none", "gpu": gpu}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--toolkit", type=Path, required=True)
    parser.add_argument("--tags", nargs="+", choices=TAGS, default=list(TAGS))
    parser.add_argument("--image-batch", type=int, default=32)
    parser.add_argument("--text-batch", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    require(args.image_batch > 0 and args.text_batch > 0, "Batch sizes must be positive")
    # CPU-mode local evaluation: CUDA check disabled in this copy.
    work = args.work.resolve()
    require(args.toolkit.is_dir(), "Missing toolkit directory")
    protocol_path = work / "selection_protocol.json"
    protocol = read_json(protocol_path)
    validate_protocol(protocol)
    protocol_hash = sha256_file(protocol_path)
    manifest = read_json(work / "benchmark_manifest.json")
    require(manifest.get("status") == "ready", "Benchmark preparation has not passed the full-scope contract")
    expected_data = protocol.get("expected_data", {})
    require(manifest.get("data_fingerprint") == expected_data.get("data_fingerprint")
            and expected_data.get("data_fingerprint"), "Prepared data fingerprint differs from frozen protocol")
    rows = {}
    for name, subsets in (("aro", ARO), ("sugarcrepe", SUGAR)):
        require(set(expected_data["benchmarks"][name]["subsets"]) == set(subsets),
                f"Incomplete formal subset contract: {name}")
        rows[name] = read_benchmark(work, name, manifest["benchmarks"][name],
                                    expected_data["benchmarks"][name])
    if "coco" in expected_data["benchmarks"]:
        require("coco" in manifest["benchmarks"], "Frozen COCO scope has not been prepared")
        rows["coco"] = read_coco(work, manifest["benchmarks"]["coco"], expected_data["benchmarks"]["coco"])
    inventory_hash = verify_image_inventory(work, manifest, expected_data, rows)
    import torch
    # CPU-mode local evaluation: CUDA check disabled in this copy.
    torch.manual_seed(0)
    versions = runtime_versions()
    evaluator_hash = sha256_file(__file__)
    for tag in dict.fromkeys(args.tags):
        frozen = protocol["expected_models"][tag]
        checkpoint = work_path(work, frozen["checkpoint_path"])
        require(checkpoint.is_file() and sha256_file(checkpoint) == frozen["checkpoint_sha256"],
                f"{tag}: checkpoint hash mismatch")
        for name, path_key in (("training_config", "training_config_path"), ("negative", "negative_path")):
            if path_key in frozen:
                require(sha256_file(work_path(work, frozen[path_key])) == frozen[name + "_sha256"],
                        f"{tag}: {name} hash mismatch")
        output_dir = work / "results" / tag
        evaluation_path = output_dir / "evaluation.json"
        if evaluation_path.is_file():
            try:
                old = read_json(evaluation_path)
                validate_evaluation(old, tag, protocol, protocol_hash)
                require(old["evaluator_sha256"] == evaluator_hash, "Evaluator source changed")
                require(old.get("image_inventory_sha256") == inventory_hash, "Image inventory changed")
                require(old.get("runtime") == versions, "Evaluation runtime changed")
                require(all(sha256_file(output_dir / b["persample_path"]) == b["persample_sha256"]
                            for b in old["benchmarks"].values()), "Missing or changed per-sample evidence")
            except (OSError, ValueError, KeyError, TypeError):
                print(f"Existing {tag} evaluation needs recomputation with the current frozen inputs", flush=True)
            else:
                print(f"Verified {tag} evaluation already exists; skipping model inference", flush=True)
                continue
        print(f"Loading {tag} in FP32 on {versions['gpu']}", flush=True)
        model, preprocess, tokenizer = load_model(checkpoint, args.device)
        evaluation = {"schema_version": 1, "tag": tag, "created_utc": datetime.now(timezone.utc).isoformat(),
                      "protocol_sha256": protocol_hash, "model": protocol["model"],
                      "pretrained": protocol["pretrained"], "strategy": "full", "distill_lambda": 0,
                      "seed": 42, "formal": True, "smoke": False, "limit": None,
                      "checkpoint_sha256": frozen["checkpoint_sha256"],
                      "negative_sha256": frozen["negative_sha256"],
                      "training_config_sha256": frozen["training_config_sha256"],
                      "data_fingerprint": manifest["data_fingerprint"], "evaluator_sha256": evaluator_hash,
                      "image_inventory_sha256": inventory_hash,
                      "runtime": versions, "precision": "fp32", "benchmarks": {}}
        try:
            for name in rows:
                expected = expected_data["benchmarks"][name]
                identity = {"tag": tag, "benchmark": name, "checkpoint_sha256": frozen["checkpoint_sha256"],
                            "dataset_sha256": expected["dataset_sha256"], "evaluator_sha256": evaluator_hash,
                            "data_fingerprint": manifest["data_fingerprint"], "runtime": versions,
                            "image_inventory_sha256": inventory_hash,
                            "precision": "fp32", "image_batch": args.image_batch, "text_batch": args.text_batch}
                cache_path = work / "cache" / tag / (name + "_features.npz")
                if name == "coco":
                    evaluation["benchmarks"][name] = evaluate_coco(
                        model, preprocess, tokenizer, rows[name], expected, output_dir, cache_path, identity, args)
                else:
                    evaluation["benchmarks"][name] = evaluate_benchmark(
                        model, preprocess, tokenizer, rows[name], name, expected, output_dir, cache_path, identity, args)
                write_json(output_dir / "partial_metrics.json", evaluation)
            validate_evaluation(evaluation, tag, protocol, protocol_hash)
            write_json(evaluation_path, evaluation)
            print(f"Completed {tag}: full ARO and SugarCrepe evidence saved", flush=True)
        finally:
            del model, preprocess, tokenizer
            gc.collect()


if __name__ == "__main__":
    main()
