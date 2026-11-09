# 评测 runbook（成员 3）

本目录负责 E0–E4b 的统一评测：多 caption 检索、Winoground、ARO/SugarCrepe++ 二选一、特征缓存、bootstrap CI 和结果汇总。

## 1. 运行环境

- **CPU 可以跑**：`pytest -q tests`、结果汇总、报告表格。
- **GPU 建议用于**：全量 Flickr30k/COCO/ARO/SugarCrepe++ 评测和 demo。
- **VS Code + Colab**：安装官方 `Google Colab` 扩展，打开本地 `.ipynb` 后选择 `Colab` kernel 和 T4 GPU；它运行的是 Notebook，不是本地终端/SSH。
- 安装：

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### VS Code + Colab 快速连接

1. VS Code 扩展市场搜索并安装 **Google Colab**（会自动带 Jupyter 依赖）。
2. 打开仓库里的 `.ipynb`，点右上角 **Select Kernel → Colab**。
3. 登录 Google 账号，选择 **T4 GPU** runtime。
4. 在第一个 cell 运行：

```python
!nvidia-smi
!git clone https://github.com/wh-violet667/CompoCLIP.git
%cd CompoCLIP
!pip install -q -r requirements.txt
from google.colab import drive
drive.mount('/content/drive')
!ln -sfn /content/drive/MyDrive/compoclip/data data
!ln -sfn /content/drive/MyDrive/compoclip/runs runs
```

> 免费 Colab 的 T4 是临时 runtime：断连后要重连；checkpoint 必须放 Drive。它不能替代本地 SSH/终端，也不要用自动点击脚本保活。

## 2. 数据前提

| 基准 | 需要的文件 |
|---|---|
| Flickr30k | `$DATA_ROOT/flickr30k/splits/{train,val,test}.jsonl` + `flickr30k/images/` |
| COCO | `$DATA_ROOT/coco2017/annotations/captions_val2017.json` + `coco2017/val2017/` |
| Winoground | HF `facebook/winoground`，需登录并同意条款 |
| ARO | `$DATA_ROOT/golden/aro.jsonl` + 图片根路径 |
| SugarCrepe++ | `$DATA_ROOT/golden/sugarcrepe.jsonl` + 图片根路径 |

生成命令：

```bash
python scripts/prepare_flickr30k.py --karpathy-json /path/dataset_flickr30k.json
python scripts/prepare_pairwise_bench.py \
  --input "data/aro/*.json" --image-root data/aro/images --output data/golden/aro.jsonl
python scripts/prepare_pairwise_bench.py \
  --input "data/sugarcrepepp/*.json" --output data/golden/sugarcrepe.jsonl
```

> ARO/SugarCrepe++ golden 由成员 1 负责；不要另写一套字段格式。

## 3. Checkpoint 格式

`src/train.py` 保存：

```python
{"model": state_dict, "optimizer": ..., "scaler": ..., "epoch": ..., "config": ...}
```

`evals-mm/encoders.py` 通过 `checkpoint_utils.load_checkpoint_state()` 读取，兼容：

- `train.py` 的训练 checkpoint（自动取 `payload["model"]`）；
- 裸 `state_dict`；
- `{"state_dict": ...}`；
- DataParallel 的 `module.` 前缀。

LoRA **只评 `epoch{n}_merged.pt`**，不要评原始 `epoch{n}.pt`。

## 4. E0 分工

- **成员 2**：E0 `flickr30k` + `coco`（数据 owner，检索/遗忘基线）。
- **成员 1**：E0 `winoground` + `aro` + `sugarcrepe`（组合基线，Winoground 与 0-C 共用一次）。
- **成员 3**：维护 harness、汇总 `master.csv`、抽查数字；**不重复跑**其他人已完成的 E0。

每个 benchmark 只跑一次，统一 `--tag e0`、同一 git commit、同一 `DATA_ROOT`；正式结果不加 `--limit`。

## 5. 常用命令

```bash
# E0：不加 --checkpoint
python evals-mm/run_eval.py --tag e0 --benchmark flickr30k
python evals-mm/run_eval.py --tag e0 --benchmark flickr30k --split val
python evals-mm/run_eval.py --tag e0 --benchmark coco --limit 200
python evals-mm/run_eval.py --tag e0 --benchmark winoground
python evals-mm/run_eval.py --tag e0 --benchmark aro
python evals-mm/run_eval.py --tag e0 --benchmark sugarcrepe

# 微调组
python evals-mm/run_eval.py --tag e2 --checkpoint runs/e2/epoch3.pt --benchmark flickr30k
python evals-mm/run_eval.py --tag e2 --checkpoint runs/e2/epoch3.pt --benchmark winoground

# LoRA：只认 merged
python evals-mm/run_eval.py --tag e4c --checkpoint runs/e4c/epoch3_merged.pt --benchmark flickr30k

# 冒烟：结果不可与全量比较
python evals-mm/run_eval.py --tag smoke --benchmark flickr30k --limit 200 --bootstrap 0
```

### 参数说明

| 参数 | 说明 |
|---|---|
| `--limit N` | 只评测前 N 个 query；结果标记为 smoke |
| `--split val` | Flickr30k 用 val split；默认 test |
| `--bootstrap N` | bootstrap 次数，默认 1000；0 关闭 CI |
| `--seed` | 评测采样/CI 随机种子 |
| `--no-cache` | 禁用特征缓存，强制重新编码 |
| `--data-root` | 覆盖 `DATA_ROOT` |

## 6. 输出文件

```text
evals-mm/
├── .cache/<encoder_id>/<bench>_<split>_<limit>.npz   # 特征缓存
└── results/
    ├── flickr30k.csv
    ├── coco.csv
    ├── winoground.csv
    ├── aro.csv
    ├── sugarcrepe.csv
    ├── master.csv
    └── {tag}_{bench}_misses.json
```

- `run_eval` 按 benchmark 追加 CSV；列包含 `tag/benchmark/model/weights/seed/limit/bootstrap/git_commit/ts`。
- 检索行包含 `t2i_R@1/5/10`、`i2t_R@1/5/10`、MedR，以及 R@1 的 bootstrap CI。
- Winoground 行包含 `wg_text/wg_image/wg_group` 及三项 CI。
- ARO/SugarCrepe 行按 `subset` 分行，包含 `accuracy` 和 CI，并额外写一行 `subset=all`。
- 结果表 schema 变化时，`run_eval.py` 会把旧 CSV 备份成 `*.bak.<timestamp>.csv`，避免列错位。

## 7. 汇总 `master.csv`

`run_eval` 只写 per-bench CSV；报告前统一汇总：

```bash
python evals-mm/collect_results.py
# 或
python evals-mm/collect_results.py \
  --results-dir evals-mm/results \
  --output evals-mm/results/master.csv
```

`master.csv` 是 **long format**，一行一个指标：

```text
tag, benchmark, subset, metric, value,
n, n_text_queries, n_image_queries, n_pairs,
model, weights, checkpoint, seed, limit, bootstrap,
is_smoke, git_commit, config_hash, source_file, ts
```

规则：

- `--limit` 结果 `is_smoke=true`，不能和全量结果混排；
- 同一 `(tag, benchmark, subset, metric, limit, seed, weights)` 保留最后一次；
- `git_commit` 优先用行内记录，缺失时自动取当前仓库；
- 报告只引用 `is_smoke=false` 的数字。

## 8. 常见错误

| 现象 | 原因 / 处理 |
|---|---|
| `ModuleNotFoundError: open_clip` | `pip install -r requirements.txt` |
| `FileNotFoundError: .../splits/test.jsonl` | 先跑 `scripts/prepare_flickr30k.py` |
| `FileNotFoundError: .../golden/aro.jsonl` | 等成员 1 的 golden，或确认 `DATA_ROOT` |
| Winoground 加载失败 | 先在 HF 页面同意条款并 `huggingface-cli login` |
| checkpoint 加载报 missing/unexpected keys | 确认是 `train.py` 新格式；LoRA 用 `*_merged.pt` |
| CSV 追加报列不匹配 | 新 schema 会自动备份旧文件；不要手工拼 CSV |
| CUDA OOM | 降低 `encode_images` 的 batch size；或先 `--limit` 冒烟 |

## 9. 评测纪律

- 同 batch、同 split、同 `limit` 的数字才可横向比较。
- `--limit` 只用于冒烟；不能进报告主表。
- Winoground 只作辅助证据；主证据用 ARO + SugarCrepe++。
- 每个 benchmark 至少抽 30 条 misses 做归因。
- 结果 CSV 必须带 `git_commit`、`seed`、`weights`；没有来源的数字不进报告。
- 不引用未修复硬负例 pipeline 的 E2a 数字作为主结果。
