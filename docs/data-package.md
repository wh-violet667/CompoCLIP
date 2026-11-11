# 成员 2/3 数据与结果包

> 仓库只放代码、文档、小结果；原始图片和 >100MB 的逐样本文件走网盘/Release。
> 本文件是上传清单与验收口径，路径相对于仓库根目录。

## 1. 仓库里已经有的（push 后队友直接 `git pull`）

| 内容 | 路径 | 谁用 |
|---|---|---|
| 数据准备脚本 | `scripts/prepare_*.py` | 成员 2 |
| E0 评测 harness | `evals-mm/` | 成员 3 |
| 0-A/0-B/0-C 脚本 | `scripts/run_0a_probe.py`、`run_0b_probe.py`、`diagnose_failures.py` | 成员 3 |
| 出图/汇总/报告/PPT | `scripts/make_figures.py`、`collect_results.py`、`build_report.py`、`build_ppt.py` | 全员 |
| 结果小表 | `report/results/{winoground,aro,sugarcrepe}.csv`、`0a_winoground.json`、`0b_*`、`0c_*` | 成员 3 |
| E0 misses | `report/results/misses/e0_*_misses.json` | 成员 3 |
| 逐样本小文件 | `report/results/persample/e0_winoground_persample.npz`、`e0_sugarcrepe_persample.npz`、`0b_persample.jsonl` | 成员 3 |
| LLM 100 条质检 | `report/results/hardneg/hardneg_qc_100.csv/.md`、`quality_report.json`、`llm_dryrun.jsonl` | 成员 2 |
| 报告 1–4 章 + 图 | `report/01_*.md`…`04_*.md`、`report/figures/` | 成员 2/3 |
| 交接与决策 | `docs/handoff.md`、`decision-log.md`、`experiment-log.md`、`sources.md` | 全员 |

## 2. 必须走网盘/Release 的大文件

### 2.1 `pkg_golden.zip`（约 35 MB）

包含：

- `data/golden/aro.jsonl`（71,679 对，3 子集）
- `data/golden/sugarcrepe.jsonl`（4,757 对）
- `data/winoground/pairs.jsonl`（400 对）
- `data/hardneg/captions_source.jsonl`（500 条 LLM few-shot 源）
- `data/aro/vg_relation.json`、`vg_attribution.json`
- `data/sugarcrepepp/examples.jsonl`

解压到仓库根目录，覆盖同名 `data/` 路径。成员 2 用它核对 golden 字段；成员 3 用它复跑 E0。

### 2.2 `pkg_eval_images.zip`（约 1.6 GB）

包含：

- `data/aro/images/`（6,603 张，VG-Relation + VG-Attribution）
- `data/sugarcrepepp/images_tmp/`（1,560 张）
- `data/winoground/images/`（800 张）
- `data/flickr30k/images/`（1,000 张 test 图）

解压后目录名保持不变。`images_tmp` 如与 `prepare_sugarcrepepp.py` 的默认路径不一致，先看脚本顶部常量。

### 2.3 `pkg_persample.zip`（约 260 MB）

包含：

- `e0_aro_persample.npz`（242 MB，仓库放不下）
- `e0_sugarcrepe_persample.npz`
- `e0_winoground_persample.npz`
- `0b_persample.jsonl`
- `e0_aro_misses.json`、`e0_sugarcrepe_misses.json`

放置到 `evals-mm/results/`，0-C 与成员 3 的复查直接读这里。

### 2.4 训练数据（成员 2 自备）

- Flickr30k 全量 31.8k 图 + Karpathy split：Kaggle `nunenuh/flickr30k`
- COCO 2017 val2017 + train20k 子集：Kaggle `awsaf49/coco-2017-dataset`
- ARO COCO-Order 所需 COCO test2014：官方 `test2014.zip`（约 6 GB）
- SugarCrepe++ 的 `add` 子类：CLIC 项目页手工下载

这些不进网盘包，成员 2 用官方源下载后按 `docs/sources.md` 补 SHA256。

## 3. 校验和（SHA256）

| 文件 | 字节 | SHA256 |
|---|---:|---|
| `report/results/winoground.csv` | 340 | `c4ff7c7a951511cc6e471b64fc46d468536cf5d848ed6f5368ad6c500c30a105` |
| `report/results/aro.csv` | 811 | `575220a7321c5b62ff370dbd84c06bc74e41fd920740543aa6e150998904de4e` |
| `report/results/sugarcrepe.csv` | 1,226 | `a56d11fb5a0b0d5a287f0b392bbc59ea2b7bd9abb0dcad6c19724c11cfc03f9b` |
| `report/results/0c_taxonomy.csv` | 1,093 | `9e1bdd8f2b0de688cd229c605e20cbab7d3fb8347f22c144e684dbc26915712c` |
| `report/results/persample/e0_winoground_persample.npz` | 623,790 | `7e3b516541e582738e1187e354e2fbec627e9a140c982d4ce71c4a185e8ceca2` |
| `report/results/persample/e0_sugarcrepe_persample.npz` | 10,086,830 | `8a1f86bd2e1594db6932d2544af956f4cd9ce9ca34c5eee4cb7418bff632abd6` |
| `report/results/persample/0b_persample.jsonl` | 5,599,642 | `d25fbb9dfab99640d9e143c07be8a24c20131cfb8cf70c84abd4ffa3b3bdfdf7` |
| `evals-mm/results/e0_aro_persample.npz` | 242,276,770 | `22fbfac036a400c9739838da5135706bb57f3a9d83615baace4f5c1632c86994` |
| `report/results/misses/e0_aro_misses.json` | 56,492 | `11db70989ac19bbd56f9db1b02e8aea111a1d64e98e5d84106ec10ec1c1de4d3` |
| `report/results/misses/e0_sugarcrepe_misses.json` | 67,979 | `ffc841f3a7a9ec87c6de37fcdb65e7f722442f4618b562b9c0b4ee75935d60ab` |
| `data/golden/aro.jsonl` | 17,663,301 | `d1a27893dd5d3987749f6192ada3f9e21fd8edb2a20d4c862dff6501298824a1` |
| `data/golden/sugarcrepe.jsonl` | 1,465,028 | `2ad53eb37314aba8cdd74bd69fa7d629dbac9ca1f615cb9dbe585db110c2d1c9` |
| `data/winoground/pairs.jsonl` | 128,680 | `b542f69f1b60723d4392084cec939d5d963cca00c5bd0d4940c2cfe910707f3f` |
| `data/hardneg/captions_source.jsonl` | 64,225 | `be5545b68fa193da048656ebfa34387d150a0dc5a6cb32879826a14593923875` |

## 4. 队友验收命令

```bash
# 1) 仓库代码
git pull
python -m pytest -q

# 2) 数据包解压后核对
python - <<'PY'
from pathlib import Path
checks = {
    "data/golden/aro.jsonl": 71679,
    "data/golden/sugarcrepe.jsonl": 4757,
    "data/winoground/pairs.jsonl": 400,
}
for p, n in checks.items():
    lines = sum(1 for _ in open(p, encoding="utf-8"))
    print(p, lines, "OK" if lines == n else "MISMATCH")
PY

# 3) 复跑 E0（成员 3）
python evals-mm/run_eval.py --tag e0_check --benchmark winoground --limit 50
python scripts/collect_results.py --tag e0_check
```

## 5. 不要上传的东西

- `data/**` 原图与 zip、`evals-mm/results/e0_aro_persample.npz`（242 MB）
- `runs/**`、`*.pt` 权重、`caches/**`
- `__pycache__`、`.venv`、`wandb`
- 任何 `caption2` 同义改写被误当负例的数据
