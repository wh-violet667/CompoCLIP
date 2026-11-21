# Results artifacts (role 3)

All numbers here come from the formal T4 runs or the A2 handoff. No cell is filled
by hand; missing measurements stay `—`.

## Main tables

| File | Content |
|---|---|
| `full_4of3.csv` / `.md` | A2 4/3: ARO 4 subsets + SugarCrepe 3 families + COCO; E0–E4b |
| `master.csv` | Long-form master table (common 2/2 + full 4/3 + retrieval/COCO) |
| `master_common.csv` | Common 2/2 ARO + 2/3 SugarCrepe |
| `master.md` / `results_table.md` | Human-readable summaries |
| `aro_common.csv` / `sugarcrepe_common.csv` | Common-subset per-subset accuracy + CI |

## Statistics and failures

| File | Content |
|---|---|
| `paired_stats.json` / `.md` | Common 2/2 paired bootstrap + McNemar |
| `paired_full_4of3.json` / `.md` | Full 4/3 paired bootstrap + McNemar |
| `failure_cases.md` | E3 ARO 15 + SugarCrepe 15 closest-margin errors |

## Provenance

- E0 A2 4/3 baseline: Colab Enterprise T4, 941.4 s, `A2_eval/results/e0/`.
- E3/E4b A2 4/3: T4, 937.1 s / 939.1 s, `A2_eval/results/{e3,e4b}/`.
- E1/E2/E2a A2 4/3: A2 handoff `A2_eval/results/{e1,e2,e2a}/`.
- Retrieval: `evals-mm/results/{flickr30k,winoground}.csv`.
- Raw GCS prefix: `gs://demo667/results_role3/`.

## Regenerate

```bash
python scripts/assemble_role3_results.py
python scripts/build_full_4of3.py
python scripts/build_full_master.py
python scripts/paired_full_4of3.py
python scripts/make_failure_cases_a2.py
python scripts/make_master_md.py
python scripts/make_results_table.py
```

## Known gaps

- E2 has no COCO run, so "E3's COCO forgetting is smaller than E-best" is
  unverified; E1/E2/E2a COCO is also missing.
- E4b Sugar differs by 0.439pp between A2 4/3 and the older `evals-mm` 2-family
  run; this directory uses A2 4/3.
- E0 uses the new A2 4/3 baseline; the older A1 values (53.987/74.314) are kept
  only as a historical reference.
