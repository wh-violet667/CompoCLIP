# Role 3 GPU evaluation: completed

Status: **E0–E4b formal A2 4/3 + E1–E4b Flickr30k/Winoground completed on a
Tesla T4** (Colab Enterprise, project `cs5489-gp-121`, bucket `demo667`).

This file now records what was run, where the artifacts are, and the remaining
gaps. It replaces the earlier "missing numbers" checklist.

## 1. Run record

- Date: 2026-11-21 (E3/E4b/retrieval), 2026-11-22 (E0 baseline)
- Runtime: Colab Enterprise, Tesla T4 16 GB
- Environment: torch 2.11.0+cu128, torchvision 0.26.0+cu128, open_clip 2.24.0,
  numpy 2.0.2
- Notebook: `notebooks/colab_enterprise_role3_gpu.ipynb`
- Code commit: `09a93e3` (E3/E4b/retrieval), `bab6f53` (E0 cell)
- Results prefix: `gs://demo667/results_role3/`
- Elapsed: E0 941.4 s; E3 937.1 s; E4b 939.1 s (ARO + SugarCrepe + COCO)
- Local download: `results_role3/` (merged into `report/results/`,
  `evals-mm/results/`, `A2_eval/results/{e0,e3,e4b}/`)

## 2. Formal results

### 2.1 A2 4/3 (E0–E4b)

| Model | ARO 4-subset macro | SugarCrepe 3-family macro | Composite | COCO t2i R@1 |
|---|---:|---:|---:|---:|
| E0 | 45.486% | 77.539% | 61.512% | 38.794% |
| E1 | 43.974% | 80.046% | 62.010% | — |
| E2 | 44.823% | 80.314% | 62.568% | — |
| E2a | 44.289% | 80.160% | 62.224% | — |
| E3 | 48.195% | 79.404% | 63.799% | 41.029% |
| E4b | 46.235% | 79.385% | 62.810% | 38.958% |

E2's composite matches B's selection score 62.5683% within rounding.
Full table: `report/results/full_4of3.csv` / `.md`.

### 2.2 Retrieval and Winoground

| Model | Flickr30k t2i R@1 | Flickr30k i2t R@1 | Winoground group |
|---|---:|---:|---:|
| E1 | 73.98% | 87.10% | 8.25% |
| E2 | 72.98% | 87.50% | 8.00% |
| E2a | 73.68% | 87.50% | 8.25% |
| E3 | 73.26% | 85.70% | 7.50% |
| E4b | 71.44% | 88.50% | 7.75% |

E0: Flickr30k 66.76% / 84.20%, Winoground group 7.75%, COCO 38.794% / 56.580%.
E0's ARO/Sugar/COCO now come from the A2 4/3 baseline, not the older A1 archive.

## 3. Commands used

```bash
# E0 A2 4/3 baseline (strict paired baseline)
python scripts/evaluate_role3_models.py \
  --tags e0 --benchmarks aro sugarcrepe coco \
  --device cuda --image-batch 64 --text-batch 256

# E3/E4b A2 4/3
python scripts/evaluate_role3_models.py \
  --tags e3 e4b --benchmarks aro sugarcrepe coco \
  --device cuda --image-batch 64 --text-batch 256

# E1/E2/E2a/E3/E4b retrieval
for tag in e1 e2 e2a e3 e4b; do
  python evals-mm/run_eval.py --tag "$tag" \
    --checkpoint "A2_models/$tag/${tag}_model.pt" --benchmark flickr30k
  python evals-mm/run_eval.py --tag "$tag" \
    --checkpoint "A2_models/$tag/${tag}_model.pt" --benchmark winoground
done

# Tables
python scripts/assemble_role3_results.py
python scripts/build_full_master.py
python scripts/build_full_4of3.py
python scripts/make_master_md.py
```

## 4. Remaining gaps

1. **E2 has no COCO 4/3 run.** The pre-registered sub-criterion
   "E3's COCO forgetting is smaller than E-best" therefore remains unverified.
   E1/E2/E2a COCO was also not run in the same batch; COCO compares E0/E3/E4b
   only.
2. E4b Sugar differs by 0.439pp between A2 4/3 (76.358%) and the older
   `evals-mm` 2-family run (76.797%); the report uses A2 4/3 consistently.
3. Flickr30k/Winoground still use public mirrors; official-data recheck is
   pending.
4. Single training seed; no matched-size control for E2a.

## 5. Acceptance checks

- ARO 4 subsets: `vg_relation=23937`, `vg_attribution=28748`,
  `coco_order=25010`, `flickr_order=5000`.
- SugarCrepe 3 families: `replace=3846`, `swap=911`, `add=2754`.
- COCO: 5,000 images / 25,014 captions.
- Every number has a matching `evaluation.json` or `evals-mm/results/*.csv` row.
- Do not compare the common 2/2 ARO macro with the 4/4 selection score without
  a footnote.
