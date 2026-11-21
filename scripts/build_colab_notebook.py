"""Generate notebooks/colab_role3_gpu.ipynb for the Colab + VS Code workflow."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "colab_role3_gpu.ipynb"

cells = [
    nbf.v4.new_markdown_cell(
        "# CompoCLIP role-3 GPU evaluation (Colab + VS Code)\n\n"
        "Upload `A2_eval.tar`, `A2_models.tar`, `data_role3.tar`, "
        "`role3_archive.tar` to `MyDrive/compoclip/` first. Then select the Colab "
        "T4 kernel and run all cells.\n\n"
        "See `docs/colab-vscode.md` for the full explanation."
    ),
    nbf.v4.new_code_cell(
        "!nvidia-smi\n"
        "import torch\n"
        "print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
    ),
    nbf.v4.new_code_cell(
        "from google.colab import drive\n"
        "drive.mount('/content/drive')"
    ),
    nbf.v4.new_code_cell(
        "%cd /content\n"
        "!git clone https://github.com/wh-violet667/CompoCLIP.git\n"
        "%cd CompoCLIP"
    ),
    nbf.v4.new_code_cell(
        "!mkdir -p /content/compoclip_data\n"
        "!tar -xf /content/drive/MyDrive/compoclip/A2_eval.tar -C /content/compoclip_data\n"
        "!tar -xf /content/drive/MyDrive/compoclip/A2_models.tar -C /content/compoclip_data\n"
        "!tar -xf /content/drive/MyDrive/compoclip/data_role3.tar -C /content/compoclip_data\n"
        "!tar -xf /content/drive/MyDrive/compoclip/role3_archive.tar -C /content/CompoCLIP"
    ),
    nbf.v4.new_code_cell(
        "!ln -sfn /content/compoclip_data/A2_eval /content/CompoCLIP/A2_eval\n"
        "!ln -sfn /content/compoclip_data/A2_models /content/CompoCLIP/A2_models\n"
        "!ln -sfn /content/compoclip_data/data/golden /content/CompoCLIP/data/golden\n"
        "!ln -sfn /content/compoclip_data/data/flickr30k /content/CompoCLIP/data/flickr30k\n"
        "!ln -sfn /content/compoclip_data/data/winoground /content/CompoCLIP/data/winoground"
    ),
    nbf.v4.new_code_cell(
        "!pip -q install \"open_clip_torch==2.24.0\" timm ftfy regex huggingface_hub safetensors spacy pandas tqdm pyyaml datasets faiss-cpu\n"
        "!python -m spacy download en_core_web_sm"
    ),
    nbf.v4.new_code_cell(
        "!python scripts/make_a2_eval_portable.py --root /content/CompoCLIP/A2_eval --data-root /content/CompoCLIP/data"
    ),
    nbf.v4.new_code_cell(
        "import json\n"
        "from pathlib import Path\n"
        "p = Path('/content/CompoCLIP/data/winoground/pairs.jsonl')\n"
        "rows = [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]\n"
        "img_root = Path('/content/CompoCLIP/data/winoground/images')\n"
        "for r in rows:\n"
        "    r['image_0'] = str(img_root / Path(str(r['image_0']).replace('\\\\', '/')).name)\n"
        "    r['image_1'] = str(img_root / Path(str(r['image_1']).replace('\\\\', '/')).name)\n"
        "p.write_text('\\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\\n', encoding='utf-8')\n"
        "print('winoground paths fixed', len(rows))"
    ),
    nbf.v4.new_code_cell(
        "!python scripts/evaluate_role3_models.py \\\n"
        "  --tags e3 e4b \\\n"
        "  --benchmarks aro sugarcrepe coco \\\n"
        "  --device cuda --image-batch 64 --text-batch 256"
    ),
    nbf.v4.new_code_cell(
        "!for tag in e1 e2 e2a e3 e4b; do \\\n"
        "  python evals-mm/run_eval.py --tag \"$tag\" \\\n"
        "    --checkpoint \"A2_models/$tag/${tag}_model.pt\" --benchmark flickr30k; \\\n"
        "  python evals-mm/run_eval.py --tag \"$tag\" \\\n"
        "    --checkpoint \"A2_models/$tag/${tag}_model.pt\" --benchmark winoground; \\\n"
        "done"
    ),
    nbf.v4.new_code_cell(
        "!python scripts/assemble_role3_results.py\n"
        "!python scripts/build_full_4of3.py\n"
        "!python scripts/build_full_master.py\n"
        "!python scripts/paired_full_4of3.py\n"
        "!python scripts/make_failure_cases_a2.py\n"
        "!python scripts/make_master_md.py"
    ),
    nbf.v4.new_code_cell(
        "!mkdir -p /content/drive/MyDrive/compoclip/results_role3\n"
        "!cp -r report/results/* /content/drive/MyDrive/compoclip/results_role3/\n"
        "!cp -r A2_eval/results/e3 /content/drive/MyDrive/compoclip/results_role3/\n"
        "!cp -r A2_eval/results/e4b /content/drive/MyDrive/compoclip/results_role3/\n"
        "!cp -r evals-mm/results/*.csv /content/drive/MyDrive/compoclip/results_role3/"
    ),
    nbf.v4.new_markdown_cell(
        "## Notes\n\n"
        "- `/content` resets when the runtime disconnects; re-run the extraction cells.\n"
        "- Copy results to Drive after each long job.\n"
        "- `assemble_role3_results.py` updates the common 2/4 ARO + 2/3 Sugar table; "
        "`build_full_4of3.py` writes the full A2 4/3 table. "
        "The Flickr/Winoground/COCO rows are in `evals-mm/results/*.csv` and "
        "`A2_eval/results/*/evaluation.json`.\n"
        "- Do not use scripted keep-alive; it violates Colab terms."
    ),
]

nb = nbf.v4.new_notebook(cells=cells)
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT}")
