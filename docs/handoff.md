# 成员 2/3 交接清单（成员 1 → 队友）

> 仓库内代码/文档/小结果已提交（commit `bb57deb`）；原始图片与 ARO 逐样本 npz 走网盘。上传/解压清单与 SHA256 见 `docs/data-package.md`。

## 0. 先拉仓库，再解压数据包

```bash
git pull
# 把 pkg_golden.zip / pkg_eval_images.zip / pkg_persample.zip 解压到仓库根目录
```

- `pkg_golden.zip` → `data/golden/`、`data/winoground/pairs.jsonl`、`data/aro/*.json`、`data/hardneg/captions_source.jsonl`
- `pkg_eval_images.zip` → `data/{aro,sugarcrepepp,winoground,flickr30k}/images*`
- `pkg_persample.zip` → `evals-mm/results/`

## 1. 直接可用的产物

| 产物 | 路径 | 说明 |
|---|---|---|
| 0-B 探针 | `scripts/run_0b_probe.py`、`evals-mm/results/0b_*` | 5 类保持词袋变换 + unrelated 对照 |
| 0-A 线性探针 | `scripts/run_0a_probe.py`、`results/0a_winoground.json` | Winoground 上零样本 vs 线性探针 |
| 0-C 失败分类 | `scripts/diagnose_failures.py`、`results/0c_*` | 读 E0 逐样本 npz，自动分桶 |
| E0 基线 | `results/winoground.csv`、`aro.csv`、`sugarcrepe.csv` | 统一 tag=e0、bootstrap 1000 |
| 数据准备 | `scripts/prepare_*.py` | 镜像 → 本地 golden，可直接换 Kaggle 路径 |

## 2. 需要你们补的三件事

### 2.1 官方数据复核（最高优先级）

当前 Flickr30k/Winoground 用公开镜像，ARO 缺 COCO-Order，SugarCrepe++ 缺 add 子类。拿到 Kaggle/Winoground 权限后：

```bash
# 1) 用官方 Karpathy split 覆盖 data/flickr30k/splits/{train,val,test}.jsonl
# 2) 重新跑 E0
python evals-mm/run_eval.py --tag e0_official --benchmark flickr30k
python evals-mm/run_eval.py --tag e0_official --benchmark winoground
# 3) 对比 e0 vs e0_official，差异 >1pt 写进报告脚注
```

Winoground 官方 loader 已保留：删除 `data/winoground/pairs.jsonl` 后会回退 HF `datasets`（需 `HF_TOKEN`）。

### 2.2 LLM 硬负例 15k（E2 前置）

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...        # DashScope 兼容模式填对应 base_url
python src/hardneg/llm_gen.py \
  --input data/hardneg/captions_source.jsonl \
  --output data/hardneg/llm_generated.jsonl \
  --model qwen-turbo --workers 4 --qps 2 --resume
python scripts/sample_hardneg_100.py --input data/hardneg/llm_generated.jsonl
```

- 已内置词多重集校验、续跑、限流、`quality_report.json`。
- 若保留率 <70%，先看 `drop_reasons`：`word_multiset` 高说明 prompt 约束不够，需在 prompt 里强调“逐词相同、只换顺序”。
- `data/hardneg/captions_source.jsonl` 目前只有 500 条示例，正式跑请用完整 train split。

### 2.3 把 E1–E4 结果接入报告

评测统一用：

```bash
python evals-mm/run_eval.py --tag e1 --benchmark winoground
python evals-mm/run_eval.py --tag e1 --benchmark aro
python evals-mm/run_eval.py --tag e1 --benchmark sugarcrepe
```

结果会自动追加到 `evals-mm/results/*.csv`，并生成 `e1_*_persample.npz`（0-C 可直接复用）。之后：

```bash
python scripts/collect_results.py     # 生成 report/results_table.md
python scripts/make_figures.py        # 生成 report/figures/*.png
python scripts/build_report.py        # 合成全文 DOCX/PDF
```

## 3. 报告写作接口

- 你们只需新增 `report/05_data_training.md`、`06_method.md`、`07_conclusion.md`，格式：
  - `# 5 ...` 一级标题；
  - `## 5.1 ...` 二级；
  - 表格用 Markdown `| ... |`；
  - 图片放 `report/figures/`，用 `![图 5-1 ...](figures/xxx.png)`。
- 不要改第 1–4 章的数字；若必须改，先改 `evals-mm/results/*.csv` 再重新生成。
- 引用编号见 `docs/references.md`；新增文献追加编号。

## 4. 验收口径（建议写进组会）

1. E0/E1/E2/E3 全部使用同一 `encode_images/encode_texts` 接口、同一 golden、同一 bootstrap=1000。
2. 主表报 **Winoground group + ARO 宏平均 + SugarCrepe++ macro**；不要只报 replace。
3. 所有提升都要给 95% CI；CI 跨 0 的“提升”不能写成结论。
4. 每个实验报告近失手比例（margin∈(-0.01,0]），区分“学会”与“边界挪动”。
5. 最终 PDF 的每个数字都能在 `evals-mm/results/*.csv` 找到对应行。
