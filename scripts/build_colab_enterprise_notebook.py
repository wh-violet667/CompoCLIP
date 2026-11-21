"""Generate notebooks/colab_enterprise_role3_gpu.ipynb (Colab Enterprise + GCS).

Run from the repository root:

    python scripts/build_colab_enterprise_notebook.py

The generated notebook does not use google.colab.drive or /content. It assumes
the three tarballs are in a GCS prefix and the runtime service account can read
them (and write the results prefix).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "colab_enterprise_role3_gpu.ipynb"


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip("\n"))


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip("\n"))


cells = [
    md(
        """
        # CompoCLIP role-3 evaluation on Colab Enterprise

        This notebook is for a **Colab Enterprise runtime** in project
        `cs5489-gp-121`. It does **not** use `google.colab.drive` or `/content`.

        You can get the three packages onto the runtime in either of two ways:

        **A. GCS (recommended).** Upload them to one GCS prefix, for example
        `gs://YOUR_BUCKET/compoclip/`, and set `BUCKET_URI` in cell 1:

        ```text
        A2_eval.tar      2.67 GB
        A2_models.tar    3.03 GB
        data_role3.tar   0.55 GB
        ```

        **B. Notebook file browser (no bucket).** Upload the same three tar
        files into the runtime from the notebook's file panel, then leave
        `BUCKET_URI` empty in cell 1. The notebook searches the current folder,
        `/home/jupyter`, `/content` and `/tmp/compoclip/downloads`.
        This avoids GCS, but a 6.25 GB browser upload is slow and can fail, and
        the runtime disk is deleted when the runtime is deleted.

        For GCS, give the runtime service account read access to the bucket and
        write access to the results prefix. `roles/storage.objectAdmin` on a
        private course bucket is the simplest choice.

        Create/select a Colab Enterprise runtime with a **T4 or L4 GPU** and at
        least **50 GB disk**. The default 100 GB boot disk is enough.

        The repository is cloned from GitHub, so the code (including
        `A2_eval_tools/`) comes from the pushed `main` branch, not from this
        notebook upload.
        """
    ),
    md("## 1. Configuration"),
    code(
        """
        import os
        from pathlib import Path

        # GCS mode: set this to gs://YOUR_BUCKET/compoclip
        # Local-upload mode: leave it empty and upload the tars in the file panel.
        BUCKET_URI = os.environ.get("COMPOCLIP_BUCKET", "").strip()
        REPO_URL = os.environ.get(
            "COMPOCLIP_REPO", "https://github.com/wh-violet667/CompoCLIP.git"
        )

        WORK = Path("/tmp/compoclip")
        REPO = WORK / "CompoCLIP"
        DATA_ROOT = WORK / "data_root"
        DOWNLOADS = WORK / "downloads"
        for path in (WORK, DATA_ROOT, DOWNLOADS):
            path.mkdir(parents=True, exist_ok=True)

        print("bucket :", BUCKET_URI or "(empty: expect local uploads)")
        print("repo   :", REPO_URL)
        print("work   :", WORK)
        """
    ),
    md("## 2. Check GPU and disk"),
    code(
        """
        !nvidia-smi
        """
    ),
    code(
        """
        import shutil
        import torch

        free_gb = shutil.disk_usage("/tmp").free / 1e9
        print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
        print("free disk under /tmp: %.1f GB" % free_gb)
        assert torch.cuda.is_available(), "Select a GPU runtime (T4/L4), not a CPU runtime."
        print("GPU:", torch.cuda.get_device_name(0))
        assert free_gb > 20, "Need at least ~20 GB free under /tmp for data + results."
        """
    ),
    md("## 3. Clone the repository"),
    code(
        """
        import subprocess

        if (REPO / ".git").exists():
            subprocess.check_call(["git", "-C", str(REPO), "pull", "--ff-only"])
        else:
            subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, str(REPO)])
        print(subprocess.check_output(["git", "-C", str(REPO), "log", "-1", "--oneline"], text=True))
        """
    ),
    md(
        """
        ## 4. Install evaluation dependencies

        Colab Enterprise GPU images usually already provide CUDA-enabled
        `torch`/`torchvision`. We install `open_clip_torch` with `--no-deps` so
        pip cannot replace the matching torch wheel.
        """
    ),
    code(
        """
        import importlib.util
        import subprocess
        import sys

        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torchvision") is None:
            raise RuntimeError(
                "This runtime has no torch/torchvision. Choose a Colab Enterprise GPU "
                "runtime image that includes PyTorch, or install the CUDA wheel that "
                "matches the runtime driver before continuing."
            )

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "open_clip_torch==2.24.0"]
        )
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q",
             "timm", "ftfy", "regex", "huggingface_hub", "safetensors",
             "tqdm", "pandas", "pyyaml", "google-cloud-storage"]
        )
        import open_clip, timm
        print("open_clip:", open_clip.__file__)
        print("timm:", timm.__version__)
        """
    ),
    md(
        """
        ## 5. Locate or download the three packages

        If `BUCKET_URI` is set, they are downloaded from GCS. If it is empty,
        the notebook looks for files already uploaded through the notebook file
        browser.
        """
    ),
    code(
        """
        from pathlib import Path
        from urllib.parse import urlparse

        PACKAGE_NAMES = ("A2_eval.tar", "A2_models.tar", "data_role3.tar")

        def find_local_packages():
            roots = [DOWNLOADS, Path.cwd(), Path("/home/jupyter"), Path("/content"), Path("/tmp")]
            checked = []
            for root in roots:
                if root in checked or not root.is_dir():
                    continue
                checked.append(root)
                if all((root / name).is_file() for name in PACKAGE_NAMES):
                    return root
                for child in root.iterdir():
                    if child.is_dir() and all((child / name).is_file() for name in PACKAGE_NAMES):
                        return child
            return None

        PACKAGE_DIR = find_local_packages()
        if PACKAGE_DIR is not None:
            print("using local uploads:", PACKAGE_DIR)
            for name in PACKAGE_NAMES:
                print(f"  {name}: {(PACKAGE_DIR / name).stat().st_size / 1e9:.2f} GB")
        else:
            if not BUCKET_URI.startswith("gs://"):
                raise FileNotFoundError(
                    "No local tar files found. Either upload A2_eval.tar, "
                    "A2_models.tar and data_role3.tar in the notebook file "
                    "browser, or set COMPOCLIP_BUCKET=gs://YOUR_BUCKET/compoclip."
                )
            from google.cloud import storage

            parsed = urlparse(BUCKET_URI)
            bucket_name = parsed.netloc
            prefix = parsed.path.strip("/")
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            for name in PACKAGE_NAMES:
                key = f"{prefix}/{name}" if prefix else name
                target = DOWNLOADS / name
                blob = bucket.get_blob(key)
                if blob is None:
                    raise FileNotFoundError(f"gs://{bucket_name}/{key} does not exist")
                if target.exists() and target.stat().st_size == blob.size:
                    print("cached", target, f"({target.stat().st_size / 1e9:.2f} GB)")
                    continue
                print(f"downloading gs://{bucket_name}/{key} -> {target}", flush=True)
                blob.download_to_filename(str(target))
                print(f"  done {target.stat().st_size / 1e9:.2f} GB", flush=True)
            PACKAGE_DIR = DOWNLOADS
        """
    ),
    md("## 6. Extract and link the data"),
    code(
        """
        import shutil
        import tarfile

        for name in ("A2_eval.tar", "A2_models.tar", "data_role3.tar"):
            print("extracting", name, flush=True)
            with tarfile.open(PACKAGE_DIR / name) as tar:
                try:
                    tar.extractall(DATA_ROOT, filter="data")
                except TypeError:  # Python < 3.12
                    tar.extractall(DATA_ROOT)
            print("  ok", flush=True)

        def replace_with_symlink(source: Path, target: Path) -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.exists():
                shutil.rmtree(target)
            target.symlink_to(source, target_is_directory=True)
            print("linked", target, "->", source)

        replace_with_symlink(DATA_ROOT / "A2_eval", REPO / "A2_eval")
        replace_with_symlink(DATA_ROOT / "A2_models", REPO / "A2_models")
        for subset in ("golden", "flickr30k", "winoground"):
            replace_with_symlink(DATA_ROOT / "data" / subset, REPO / "data" / subset)

        assert (REPO / "A2_eval_tools" / "evaluate_models.py").is_file(), (
            "A2_eval_tools missing from the repo clone; push it or set COMPOCLIP_REPO."
        )
        print("data layout ready")
        """
    ),
    md(
        """
        ## 7. Rewrite the Windows absolute paths

        `A2_eval/data/*.jsonl` and `data/winoground/pairs.jsonl` were prepared on
        Windows. Both must be rewritten before evaluation, otherwise every image
        path fails to open on Linux.
        """
    ),
    code(
        """
        import subprocess
        import sys

        subprocess.check_call(
            [
                sys.executable,
                "scripts/make_a2_eval_portable.py",
                "--root", str(REPO / "A2_eval"),
                "--data-root", str(REPO / "data"),
            ],
            cwd=REPO,
        )
        """
    ),
    code(
        """
        import json
        from pathlib import Path

        pairs = Path(REPO / "data" / "winoground" / "pairs.jsonl")
        rows = [json.loads(line) for line in pairs.read_text(encoding="utf-8").splitlines() if line.strip()]
        image_root = Path(REPO / "data" / "winoground" / "images")
        for row in rows:
            for key in ("image_0", "image_1"):
                row[key] = str(image_root / Path(str(row[key]).replace("\\\\", "/")).name)
        pairs.write_text(
            "\\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\\n",
            encoding="utf-8",
        )
        print("winoground paths fixed:", len(rows))
        """
    ),
    md(
        """
        ## 8. E3/E4b: A2 4/3 protocol (ARO + SugarCrepe + COCO)

        This is the long GPU job (roughly tens of minutes on a T4). Results go to
        `A2_eval/results/<tag>/evaluation.json` and per-sample `.npz` files.
        """
    ),
    code(
        """
        import subprocess
        import sys

        subprocess.check_call(
            [
                sys.executable,
                "scripts/evaluate_role3_models.py",
                "--tags", "e3", "e4b",
                "--benchmarks", "aro", "sugarcrepe", "coco",
                "--device", "cuda",
                "--image-batch", "64",
                "--text-batch", "256",
            ],
            cwd=REPO,
        )
        """
    ),
    md(
        """
        ## 9. Flickr30k + Winoground retrieval (E1/E2/E2a/E3/E4b)

        E0 is already archived and E4a = E2, so neither is rerun. Each run writes
        a row to `evals-mm/results/<benchmark>.csv`.
        """
    ),
    code(
        """
        import subprocess
        import sys

        tags = ["e1", "e2", "e2a", "e3", "e4b"]
        for tag in tags:
            checkpoint = REPO / "A2_models" / tag / f"{tag}_model.pt"
            assert checkpoint.is_file(), checkpoint
            for benchmark in ("flickr30k", "winoground"):
                cmd = [
                    sys.executable,
                    "evals-mm/run_eval.py",
                    "--tag", tag,
                    "--checkpoint", str(checkpoint),
                    "--benchmark", benchmark,
                ]
                print("\\n$", " ".join(cmd), flush=True)
                subprocess.check_call(cmd, cwd=REPO)
        """
    ),
    md("## 10. Rebuild the result tables"),
    code(
        """
        import subprocess
        import sys

        for script in (
            "scripts/assemble_role3_results.py",
            "scripts/build_full_master.py",
            "scripts/make_master_md.py",
        ):
            print("\\n$", script, flush=True)
            subprocess.check_call([sys.executable, script], cwd=REPO)
        """
    ),
    md(
        """
        ## 11. Upload the results (GCS mode) or download them manually (local mode)

        `/tmp` disappears when the runtime is deleted. In GCS mode this cell
        copies the artifacts to the bucket; in local-upload mode it creates
        `results_role3.tar.gz` in the notebook working directory, which you
        download from the notebook file browser before deleting the runtime.
        """
    ),
    code(
        """
        from pathlib import Path
        from urllib.parse import urlparse

        sources = [
            (REPO / "report" / "results", "report_results"),
            (REPO / "evals-mm" / "results", "evals_mm_results"),
            (REPO / "A2_eval" / "results" / "e3", "a2_eval_e3"),
            (REPO / "A2_eval" / "results" / "e4b", "a2_eval_e4b"),
        ]

        if not BUCKET_URI:
            import tarfile

            archive = Path.cwd() / "results_role3.tar.gz"
            print("BUCKET_URI is empty: results stay on this runtime.")
            with tarfile.open(archive, "w:gz") as tar:
                for source, label in sources:
                    if source.exists():
                        tar.add(source, arcname=label)
            print(f"created {archive} ({archive.stat().st_size / 1e6:.1f} MB)")
            print("Download this single file from the notebook file browser before deleting the runtime.")
        else:
            from google.cloud import storage

            parsed = urlparse(BUCKET_URI)
            prefix = parsed.path.strip("/")
            out_prefix = f"{prefix}/results_role3" if prefix else "results_role3"
            client = storage.Client()
            bucket = client.bucket(parsed.netloc)

            uploaded = 0
            for source, label in sources:
                if not source.exists():
                    print("skip missing", source)
                    continue
                for path in source.rglob("*"):
                    if not path.is_file() or path.suffix in {".pyc"}:
                        continue
                    relative = path.relative_to(source).as_posix()
                    blob = bucket.blob(f"{out_prefix}/{label}/{relative}")
                    blob.upload_from_filename(str(path))
                    uploaded += 1
            print(f"uploaded {uploaded} files to gs://{parsed.netloc}/{out_prefix}")
        """
    ),
    md(
        """
        ## Notes and failure recovery

        - **Runtime has no internet:** upload a repository zip to the bucket (or
          the notebook file browser) and change the clone cell to download and
          extract it.
        - **Local-upload mode:** the three tars live on the runtime disk. If the
          runtime is deleted or restarted, upload them again; results must be
          downloaded manually, and a 6 GB browser upload may fail. GCS mode is
          more reliable for repeated runs.
        - **Permission denied on GCS:** grant the runtime service account
          `roles/storage.objectAdmin` on the bucket, or at least
          `objectViewer` + `objectCreator` on the results prefix.
        - **A2 4/3 job dies halfway:** re-run cell 8. Feature caches under
          `A2_eval/cache/` make a re-run resume from the last completed
          benchmark. Do not change `--image-batch`/`--text-batch` between runs,
          or the cache identity changes.
        - **Retrieval job dies:** re-run cell 9 with a shorter `tags` list.
          Existing CSV rows are kept and rewritten deterministically.
        - **Do not fill missing report numbers by hand.** If a job fails, leave
          the cell marked incomplete and bring the error back.
        """
    ),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "name": "python3",
    "display_name": "Python 3",
    "language": "python",
}
nb.metadata["language_info"] = {"name": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
