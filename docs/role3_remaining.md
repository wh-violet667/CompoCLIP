# Role 3 remaining GPU evaluation

This file records the formal evaluation still required after the CPU-side local run. Do **not** fill the missing numbers by hand; run the commands on a GPU (Colab / Kaggle / local CUDA) and regenerate the tables.

For the Colab + VS Code route, follow [`docs/colab-vscode.md`](colab-vscode.md); for GCP Compute Engine, see [`docs/gcp-gpu.md`](gcp-gpu.md).

## Why some cells are missing

The local machine has no CUDA. The CPU-side ARO runs for E3/E4b exceeded the wall-clock budget (52k VG queries × two text towers, plus 71k for the 3-subset set). The available CPU results are still valid for the common 2/4 ARO + 2/3 SugarCrepe subset; they are marked in the report.

## 1. A2 4/3 frozen protocol (E3/E4b)

`A2_eval` already contains the frozen A2 benchmark data (4 ARO subsets, 3 SugarCrepe families, COCO val2017), including images.

```bash
# On a CUDA machine:
python scripts/evaluate_role3_models.py \
  --tags e3 e4b \
  --benchmarks aro sugarcrepe coco \
  --device cuda --image-batch 32 --text-batch 128
```

This writes `A2_eval/results/{e3,e4b}/evaluation.json` and per-sample files.

To re-evaluate E1/E2/E2a on the same frozen protocol:

```bash
python A2_eval_tools/evaluate_models.py \
  --work A2_eval --toolkit A2/E_best/workspace \
  --tags e1 e2 e2a --device cuda
```

> The copied evaluator has only the CPU guard removed; on a GPU it uses the original A2 scores. E1/E2/E2a results already exist in `A2_eval/results/` from the A2 selection run.

## 2. Flickr30k + Winoground (all fine-tuned models)

The local Flickr30k split is `data/flickr30k/splits/test.jsonl`; Winoground is `data/winoground/pairs.jsonl` + `data/winoground/images/`.

```bash
for tag in e1 e2 e2a e3 e4b; do
  python evals-mm/run_eval.py --tag "$tag" \
    --checkpoint "A2_models/$tag/${tag}_model.pt" --benchmark flickr30k
  python evals-mm/run_eval.py --tag "$tag" \
    --checkpoint "A2_models/$tag/${tag}_model.pt" --benchmark winoground
done
```

E0 is already archived: Flickr30k t2i R@1 66.76%, i2t 84.20%; Winoground group 7.75%. E4a = E2, so do not run a separate E4a model.

## 3. COCO val2017

If using the A2 prepared COCO:

```bash
python scripts/evaluate_role3_models.py \
  --tags e0 e3 e4b --benchmarks coco \
  --device cuda --image-batch 32 --text-batch 128
```

If using the official `evals-mm` COCO loader, run:

```bash
python evals-mm/run_eval.py --tag e3 \
  --checkpoint A2_models/e3/e3_model.pt --benchmark coco
python evals-mm/run_eval.py --tag e4b \
  --checkpoint A2_models/e4b/e4b_model.pt --benchmark coco
```

## 4. Regenerate tables

```bash
# Common CPU subset (already available): 2/4 ARO + 2/3 SugarCrepe
python scripts/assemble_role3_results.py

# If the A2 4/3 results are complete, merge them into the report table
python evals-mm/collect_results.py
python scripts/collect_results.py --tag e0
```

Then update:

- `report/06_method.md` E3/E4b ARO and retrieval/COCO cells;
- `report/07_conclusion.md` success criteria;
- `report/results/master.csv` / `master_common.csv`;
- `report/README.md` status table.

## 5. Acceptance checks

- ARO 4 subsets: `vg_relation=23937`, `vg_attribution=28748`, `coco_order=25010`, `flickr_order=5000`.
- SugarCrepe 3 families: `replace=3846`, `swap=911`, `add=2754`.
- COCO val2017: 5,000 images / 25,014 captions.
- Every number must have a matching `evaluation.json` or `evals-mm/results/*.csv` row.
- Do not compare the 2/4 ARO macro with the 4/4 selection score without a footnote.
