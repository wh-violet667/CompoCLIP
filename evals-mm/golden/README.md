# golden 评测集规范

本目录只放 **格式说明**，不放数据。真实评测数据由脚本生成到 `$DATA_ROOT/golden/`，不提交仓库（含图片路径与第三方标注，受许可约束）。

## 两类格式

### 1. 检索类（Flickr30k / COCO）

由 `scripts/prepare_flickr30k.py` 生成 `$DATA_ROOT/flickr30k/splits/{train,val,test}.jsonl`：

```json
{"image": "1000092795.jpg", "caption": "A man in a blue shirt...", "image_id": "0", "caption_id": "0"}
```

- 每张图 5 行（5 条 caption），`image_id` 相同
- `image` 是相对 `$DATA_ROOT/flickr30k/images/` 的文件名
- **不要**把每个 caption 行当成独立图片评测，`evals-mm/loaders.py` 会按 `image_id` 分组

COCO 走官方标注，无需转换：

```text
$DATA_ROOT/coco2017/annotations/captions_val2017.json
$DATA_ROOT/coco2017/val2017/*.jpg
```

### 2. 组合推理二选一（ARO / SugarCrepe++ / 自定义）

由 `scripts/prepare_pairwise_bench.py` 归一化生成，统一为逐行 JSON：

```json
{"image": "path/to/img.jpg", "positive": "正确描述", "negative": "同词集但语义不同", "subset": "vg_relation"}
```

要求：
- `positive` 与 `negative` 来自同一张图，二选一必须唯一正确
- `subset` 用于分类型报准确率（ARO 的 4 个子集必须区分）
- 生成后抽 100 条人工核对，确认正负没有标反

## 生成命令

```bash
# Flickr30k split（必须，训练与评测都依赖）
python scripts/prepare_flickr30k.py --karpathy-json /path/dataset_flickr30k.json

# ARO：按子集分别转换后合并
python scripts/prepare_pairwise_bench.py \
  --input "data/aro/*.json" --image-root data/aro/images \
  --output data/golden/aro.jsonl

# SugarCrepe++
python scripts/prepare_pairwise_bench.py \
  --input "data/sugarcrepepp/*.json" \
  --output data/golden/sugarcrepe.jsonl
```

## 评测

```bash
python evals-mm/run_eval.py --tag e0 --benchmark flickr30k
python evals-mm/run_eval.py --tag e0 --benchmark coco --limit 200
python evals-mm/run_eval.py --tag e0 --benchmark aro
python evals-mm/run_eval.py --tag e0 --benchmark sugarcrepe
```

## 纪律

- `--limit` 只用于冒烟，带 `limit` 的结果与全量结果不可横向比较
- 每次评测自动记录 `weights / seed / ts`，CSV 里必须能看到来源
- 未命中样本自动落盘到 `evals-mm/results/{tag}_{benchmark}_misses.json`，抽样 30 条做归因
