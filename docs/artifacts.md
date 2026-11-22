# Large artifacts and release policy

## Short answer

**No large file is required in GitHub Releases.** Code, report, PPT, result
tables and demo source are already in the `main` branch. The benchmark data,
model weights and Colab tarballs are too large for Releases and stay in GCS.

## What is in git

- `report/CompoCLIP_Report.docx` / `.pdf` (27-page chaptered report)
- `report/CompoCLIP_Report_总报告.docx` / `.pdf` (15-page overall report)
- `report/CompoCLIP_成员3汇报.pptx` (9 slides)
- `report/results/*.csv` / `.md` / `.json`
- `report/figures/*.png`
- `demo/app.py` / `demo/README.md`
- all scripts and notebooks

These are enough for a normal course submission or a code review.

## Large files (not in git)

Stored in the private GCS bucket used for the Colab Enterprise run:

```text
gs://demo667/A2_eval.tar         2.67 GB
gs://demo667/A2_models.tar       3.03 GB
gs://demo667/data_role3.tar      0.55 GB
gs://demo667/role3_archive.tar   0.25 GB
gs://demo667/results_role3/      raw E0/E3/E4b + retrieval results
```

SHA256:

```text
26359d08d853caba1a06895e4e494fe0ecdab6b2f46f3a25c472952011838203  A2_eval.tar
e2796869ec829cac5ff523beb45b0d3bad0b6b3c8f0e213092cbd355a8c6ab3c  A2_models.tar
158cfbdb77059d80be7fa9ccc5a43fe757714655d80c8e0ce0c38e0fe8e05a99  data_role3.tar
785437ab66b911c60a90a524b0484ae30a67d34eb70075006dd8677568b1aaf2  role3_archive.tar
```

The bucket is private. To share with a teammate or grader, either:

1. grant the recipient `roles/storage.objectViewer` on the bucket/prefix, or
2. generate time-limited signed URLs for the four tarballs, or
3. for a course submission, use the LMS/Drive link instead of GitHub Releases.

Do **not** make the bucket public: `A2_models.tar` contains trained weights and
`A2_eval/images` contains benchmark images.

## Why not GitHub Releases

- GitHub release assets have a **2 GB per-file limit**; `A2_eval.tar` (2.67 GB)
  and `A2_models.tar` (3.03 GB) cannot be uploaded as single assets.
- Splitting 6.5 GB into 2 GB parts is fragile and unnecessary: GCS already
  stores the exact bytes with SHA256.
- Data/model binaries should not be versioned with code; Releases are better
  reserved for a small, stable result bundle.

## Optional small release bundle

If a non-git reviewer needs one download, a **result-only** bundle is enough and
well under 2 GB:

```text
CompoCLIP_role3_results.zip
├── CompoCLIP_Report_总报告.pdf
├── CompoCLIP_成员3汇报.pptx
├── report/results/full_4of3.csv / .md
├── report/results/master.csv
├── report/results/paired_full_4of3.json / .md
├── report/results/failure_cases.md
└── README_artifacts.md
```

Create it with:

```bash
mkdir -p release_bundle
cp report/CompoCLIP_Report_总报告.pdf release_bundle/
cp report/CompoCLIP_成员3汇报.pptx release_bundle/
mkdir -p release_bundle/results
cp report/results/full_4of3.* report/results/master.csv \
   report/results/paired_full_4of3.* report/results/failure_cases.md \
   release_bundle/results/
cp docs/artifacts.md release_bundle/README_artifacts.md
cd release_bundle && zip -r ../CompoCLIP_role3_results.zip . && cd ..
sha256sum CompoCLIP_role3_results.zip
```

This bundle contains no images, no model weights and no benchmark data; it is
safe to attach to a Release or upload to an LMS.

## E2 checkpoint (optional demo asset)

If someone needs to run `demo/app.py` without the full `A2_models.tar`, the E2
checkpoint alone is about 605 MB:

```text
A2_models/e2/e2_model.pt
```

It is under the 2 GB limit and can be a single Release asset if desired. The
full five-model package is still better delivered via GCS.
