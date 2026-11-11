import json
import subprocess
import sys
from collections import Counter
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def words(text: str) -> Counter:
    return Counter(re.findall(r"[A-Za-z0-9']+", text.lower()))


def test_llm_dry_run_pipeline(tmp_path):
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    rows = [
        {"image": f"img{i}.jpg", "caption": caption}
        for i, caption in enumerate([
            "a dog sits to the left of a cat",
            "a man in a red shirt is riding a bike",
            "two children are playing with a ball",
        ])
    ]
    inp.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    subprocess.run([
        sys.executable, str(ROOT / "src/hardneg/llm_gen.py"),
        "--input", str(inp), "--output", str(out),
        "--limit", "3", "--workers", "2", "--qps", "0", "--dry-run",
    ], check=True, cwd=ROOT)

    got = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(got) == 3
    for row in got:
        assert words(row["caption"]) == words(row["negative"])
    report = json.loads((tmp_path / "quality_report.json").read_text(encoding="utf-8"))
    assert report["kept"] == 3
    assert report["word_check"] is True
