"""Select E-best from complete, comparable formal evaluations; never use training monitors."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


TAGS = ("e1", "e2", "e2a")
ARO = ("vg_relation", "vg_attribution", "coco_order", "flickr_order")
SUGAR = ("replace", "swap", "add")
HASH_FIELDS = ("checkpoint_sha256", "negative_sha256", "training_config_sha256")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique_object)


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def probability(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 1, f"{label}: expected finite fraction in [0, 1]")
    return float(value)


def positive_int(value, label):
    require(type(value) is int and value > 0, f"{label}: expected positive integer")
    return value


def validate_protocol(protocol):
    require(isinstance(protocol, dict), "Protocol must be a JSON object")
    require(protocol.get("schema_version") == 1, "Unsupported protocol schema")
    require(bool(protocol.get("protocol_id")), "Missing frozen protocol_id")
    require(protocol.get("aro_subsets") == list(ARO), "Protocol must include all four ARO subsets")
    require(protocol.get("sugarcrepe_families") == list(SUGAR), "Protocol must include replace/swap/add")
    require(protocol.get("tie_metric") == "coco.t2i_R@1", "Freeze COCO t2i R@1 tie rule before evaluation")
    tolerance = protocol.get("tie_tolerance", 1e-12)
    require(isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool)
            and math.isfinite(tolerance) and 0 <= tolerance <= 1e-12, "Tie tolerance exceeds numerical-only tolerance")
    require(protocol.get("model") == "ViT-B-32", "Unexpected model")
    require(protocol.get("pretrained") == "laion2b_s34b_b79k", "Unexpected pretrained weights")
    expected = protocol.get("expected_models", {})
    require(isinstance(expected, dict), "Protocol expected_models must be an object")
    require(set(expected) == set(TAGS), "Protocol must freeze e1/e2/e2a model identities")
    for tag in TAGS:
        require(isinstance(expected[tag], dict), f"{tag}: expected model identity object")
        for field in HASH_FIELDS:
            require(is_sha256(expected[tag].get(field)), f"{tag}: missing frozen {field}")
    require(len({expected[tag]["checkpoint_sha256"] for tag in TAGS}) == 3,
            "Expected checkpoints must be three distinct trained models")
    expected_data = protocol.get("expected_data", {})
    require(isinstance(expected_data, dict) and is_sha256(expected_data.get("data_fingerprint")),
            "Protocol must freeze expected_data.data_fingerprint before formal selection")
    require(is_sha256(expected_data.get("image_inventory_sha256")),
            "Protocol must freeze expected_data.image_inventory_sha256 before formal selection")
    benchmarks = expected_data.get("benchmarks", {})
    require(isinstance(benchmarks, dict), "Protocol expected_data.benchmarks must be an object")
    for name, subsets in (("aro", ARO), ("sugarcrepe", SUGAR)):
        benchmark = benchmarks.get(name, {})
        require(isinstance(benchmark, dict) and is_sha256(benchmark.get("dataset_sha256")),
                f"Protocol must freeze {name} dataset fingerprint")
        counts = benchmark.get("subsets", {})
        require(isinstance(counts, dict) and set(counts) == set(subsets),
                f"Protocol must freeze {name} subset counts")
        for subset, count in counts.items():
            positive_int(count, f"protocol/{name}/{subset}/n")
    if "coco" in benchmarks:
        coco = benchmarks["coco"]
        require(isinstance(coco, dict) and is_sha256(coco.get("dataset_sha256")),
                "Protocol must freeze COCO dataset fingerprint")
        require(coco.get("subset") == "val2017_5k" and coco.get("n_images") == 5000,
                "Protocol COCO scope must be full val2017 5k")
        positive_int(coco.get("n_text_queries"), "protocol/coco/n_text_queries")


def validate_evaluation(record, tag, protocol, protocol_sha256):
    require(isinstance(record, dict), f"{tag}: evaluation must be a JSON object")
    require(record.get("schema_version") == 1, f"{tag}: unsupported evaluation schema")
    require(record.get("tag") == tag, f"{tag}: wrong model tag")
    require(record.get("protocol_sha256") == protocol_sha256, f"{tag}: protocol fingerprint mismatch")
    require(record.get("formal") is True and record.get("smoke") is False
            and "limit" in record and record["limit"] is None, f"{tag}: formal, full-scope evidence required")
    for key in ("model", "pretrained"):
        require(record.get(key) == protocol[key], f"{tag}: {key} mismatch")
    require(record.get("strategy") == "full" and type(record.get("distill_lambda")) in (int, float)
            and record["distill_lambda"] == 0,
            f"{tag}: selection candidates must be full-tuning lambda=0 runs")
    require(type(record.get("seed")) is int and record["seed"] == 42, f"{tag}: unexpected training seed")
    for field in HASH_FIELDS:
        require(record.get(field) == protocol["expected_models"][tag][field], f"{tag}: {field} mismatch")
    for field in ("data_fingerprint", "evaluator_sha256"):
        require(is_sha256(record.get(field)), f"{tag}: missing {field}")
    require(record["data_fingerprint"] == protocol["expected_data"]["data_fingerprint"],
            f"{tag}: data fingerprint differs from frozen scope")
    require(record.get("image_inventory_sha256") == protocol["expected_data"]["image_inventory_sha256"],
            f"{tag}: image inventory differs from frozen scope")
    benchmarks = record.get("benchmarks", {})
    require(isinstance(benchmarks, dict), f"{tag}: benchmarks must be an object")
    macros, scopes = {}, {}
    for name, expected_subsets in (("aro", ARO), ("sugarcrepe", SUGAR)):
        bench = benchmarks.get(name, {})
        require(isinstance(bench, dict), f"{tag}/{name}: benchmark must be an object")
        require(is_sha256(bench.get("dataset_sha256")), f"{tag}/{name}: missing dataset fingerprint")
        subsets = bench.get("subsets", {})
        require(isinstance(subsets, dict), f"{tag}/{name}: subsets must be an object")
        require(set(subsets) == set(expected_subsets), f"{tag}/{name}: missing or unexpected subsets")
        values, sizes = [], {}
        for subset in expected_subsets:
            row = subsets[subset]
            require(isinstance(row, dict), f"{tag}/{name}/{subset}: expected result object")
            values.append(probability(row.get("accuracy"), f"{tag}/{name}/{subset}"))
            sizes[subset] = positive_int(row.get("n"), f"{tag}/{name}/{subset}/n")
        macros[name] = math.fsum(values) / len(values)
        scopes[name] = {"dataset_sha256": bench["dataset_sha256"], "subsets": sizes}
        expected_scope = protocol["expected_data"]["benchmarks"][name]
        require(scopes[name] == {key: expected_scope[key] for key in ("dataset_sha256", "subsets")},
                f"{tag}/{name}: data or counts differ from frozen scope")
    coco = benchmarks.get("coco")
    coco_t2i = None
    if coco is not None:
        require(isinstance(coco, dict), f"{tag}/coco: benchmark must be an object")
        require(is_sha256(coco.get("dataset_sha256")), f"{tag}/coco: missing dataset fingerprint")
        require(coco.get("subset") == "val2017_5k" and coco.get("n_images") == 5000,
                f"{tag}/coco: full val2017 5k required")
        positive_int(coco.get("n_text_queries"), f"{tag}/coco/n_text_queries")
        coco_t2i = probability(coco.get("t2i_R@1"), f"{tag}/coco/t2i_R@1")
        probability(coco.get("i2t_R@1"), f"{tag}/coco/i2t_R@1")
        scopes["coco"] = {key: coco[key] for key in ("dataset_sha256", "subset", "n_images", "n_text_queries")}
        expected_coco = protocol["expected_data"]["benchmarks"].get("coco")
        require(isinstance(expected_coco, dict) and scopes["coco"] == {
            key: expected_coco.get(key) for key in ("dataset_sha256", "subset", "n_images", "n_text_queries")},
            f"{tag}/coco: data or counts differ from frozen scope")
    return {
        "tag": tag,
        "aro_macro": macros["aro"],
        "sugarcrepe_macro": macros["sugarcrepe"],
        "selection_score": (macros["aro"] + macros["sugarcrepe"]) / 2,
        "coco_t2i_R@1": coco_t2i,
        "scopes": scopes,
        "data_fingerprint": record["data_fingerprint"],
        "evaluator_sha256": record["evaluator_sha256"],
    }


def select(results_dir, protocol_path):
    """Return a decision and comparison rows, including explicit blocked reasons."""
    protocol_path, results_dir = Path(protocol_path), Path(results_dir)
    decision = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "blocked",
        "winner": None,
        "blocked_by": [],
        "selection_owner": "role2",
        "evaluation_owner": "role3",
        "score_formula": "(mean(ARO 4 subsets) + mean(SugarCrepe 3 families)) / 2",
        "tie_rule": "COCO t2i R@1, then e1 if tied; otherwise explicit protocol decision required",
        "evidence": {},
    }
    comparison = []
    try:
        protocol = read_json(protocol_path)
        validate_protocol(protocol)
        protocol_hash = sha256_file(protocol_path)
        decision["protocol_id"] = protocol["protocol_id"]
        decision["protocol_sha256"] = protocol_hash
    except (OSError, ValueError, TypeError, KeyError) as error:
        decision["blocked_by"].append(f"protocol: {error}")
        return decision, comparison
    for tag in TAGS:
        file_path = results_dir / tag / "evaluation.json"
        row = {"tag": tag, "evidence_status": "invalid", "error": ""}
        try:
            record = read_json(file_path)
            valid = validate_evaluation(record, tag, protocol, protocol_hash)
            row.update(valid)
            row["evidence_status"] = "valid"
            decision["evidence"][tag] = {"path": str(file_path), "sha256": sha256_file(file_path),
                                         **protocol["expected_models"][tag]}
        except (OSError, ValueError, TypeError, KeyError) as error:
            row["error"] = str(error)
            decision["blocked_by"].append(f"{tag}: {error}")
        comparison.append(row)
    if decision["blocked_by"]:
        return decision, comparison
    for key in ("data_fingerprint", "evaluator_sha256"):
        if len({row[key] for row in comparison}) != 1:
            decision["blocked_by"].append(f"Candidates use different {key}")
    for name in ("aro", "sugarcrepe"):
        if any(row["scopes"][name] != comparison[0]["scopes"][name] for row in comparison[1:]):
            decision["blocked_by"].append(f"Candidates use different {name} data or subset counts")
    if decision["blocked_by"]:
        return decision, comparison
    tolerance = protocol.get("tie_tolerance", 1e-12)
    maximum = max(row["selection_score"] for row in comparison)
    contenders = [row for row in comparison if math.isclose(row["selection_score"], maximum,
                                                           abs_tol=tolerance, rel_tol=0)]
    decision["primary_ties"] = [row["tag"] for row in contenders]
    reason = "highest_composition_macro"
    if len(contenders) > 1:
        if any(row["coco_t2i_R@1"] is None for row in contenders):
            decision["blocked_by"].append("Primary tie requires formal COCO t2i R@1 for every tied candidate")
            return decision, comparison
        if any(row["scopes"]["coco"] != contenders[0]["scopes"]["coco"] for row in contenders[1:]):
            decision["blocked_by"].append("Tied candidates use different COCO data or query counts")
            return decision, comparison
        maximum = max(row["coco_t2i_R@1"] for row in contenders)
        contenders = [row for row in contenders if math.isclose(row["coco_t2i_R@1"], maximum,
                                                               abs_tol=tolerance, rel_tol=0)]
        reason = "composition_tie_coco_t2i_R@1"
    if len(contenders) > 1:
        if any(row["tag"] == "e1" for row in contenders):
            contenders = [row for row in contenders if row["tag"] == "e1"]
            reason = "composition_and_coco_tie_prefer_rule_e1"
        else:
            decision["blocked_by"].append("E2 and E2a remain tied; no predeclared final tie-break applies")
            return decision, comparison
    decision.update({"status": "selected", "winner": contenders[0]["tag"], "reason": reason,
                     "winner_identity": protocol["expected_models"][contenders[0]["tag"]]})
    return decision, comparison


def write_outputs(output_dir, decision, comparison):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ebest_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    columns = ["tag", "evidence_status", "aro_macro", "sugarcrepe_macro", "selection_score", "coco_t2i_R@1", "error"]
    with (output_dir / "ebest_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparison)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    decision, comparison = select(args.results_dir, args.protocol)
    write_outputs(args.output_dir, decision, comparison)
    print(json.dumps({key: decision[key] for key in ("status", "winner", "blocked_by")}, ensure_ascii=False))
    raise SystemExit(0 if decision["status"] == "selected" else 2)


if __name__ == "__main__":
    main()
