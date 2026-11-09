# CompoCLIP

> Compositionally-Aware Image Retrieval: Diagnosing and Fixing Contrastive Vision-Language Models

机器学习课程三人小组项目：诊断 CLIP 的组合推理盲区，通过硬负例微调与教师蒸馏抗遗忘修复，在 Flickr30k / Winoground / ARO / SugarCrepe++ / COCO 上量化“组合增益 vs 通用能力代价”。

**当前状态**：训练、损失、教师蒸馏、五基准评测和 LoRA 已经可运行，并有 CPU 单测；**硬负例 pipeline 仍需在 D1–D2 补齐**（LLM 生成限流/词集校验、对抗过滤改为 image–text 打分）。在这些项完成前，E2a 只能作为 pipeline 冒烟，不作为主结果。

## 简化后的范围

保留 6 个主线实验组；E4c LoRA 作为可选附录；砍掉 E5、0-D 全量、完整 0-A 和“五基准 × 所有组”的全量矩阵。

| 组 | 负例 | 损失 | 策略 | 目的 |
|---|---|---|---|---|
| **E0** | — | — | 不微调 | 基线 |
| **E1** | 规则法 | 硬负例 InfoNCE | 全量 | 负例来源① |
| **E2** | LLM 法 | 同上 | 全量 | 负例来源② |
| **E2a** | LLM + 对抗过滤 | 同上 | 全量 | 过滤 pipeline 消融 |
| **E3** | E-best | + 教师蒸馏 λ=0.5 | 全量 | 抗遗忘 |
| **E4b** | E-best | λ=0 | 锁图像塔 | 策略矩阵对比 |
| *E4c（可选）* | *E-best* | *λ=0* | *LoRA* | *时间富余才做* |

E4a 直接复用 E-best（全量、λ=0），不新增训练。E2r（随机子集对照）为可选；不跑时不能把 E2 vs E2a 解释成“质量比数量重要”。

## 快速开始

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt        # pytest / matplotlib / seaborn / umap / jupyter
python -m spacy download en_core_web_sm

# 可选：自定义数据目录（默认 ./data）
# export DATA_ROOT=/content/drive/MyDrive/compoclip/data

# 1. 拉取原始数据（需先配置 ~/.kaggle/kaggle.json）
python scripts/download_data.py --list
python scripts/download_data.py --dataset flickr30k
# COCO 可选；只做 Flickr30k 可跳过。COCO 需要 annotations/captions_val2017.json + val2017/
python scripts/download_data.py --dataset coco2017

# 2. 生成 Flickr30k 的 Karpathy split（训练与评测都依赖）
python scripts/prepare_flickr30k.py --karpathy-json /path/dataset_flickr30k.json
# 冒烟：python scripts/prepare_flickr30k.py --karpathy-json ... --limit 200

# 3. 生成 ARO / SugarCrepe++ 归一化评测集（可选，组合推理主证据）
python scripts/prepare_pairwise_bench.py --input "data/aro/*.json" \
    --image-root data/aro/images --output data/golden/aro.jsonl
python scripts/prepare_pairwise_bench.py --input "data/sugarcrepepp/*.json" \
    --output data/golden/sugarcrepe.jsonl

# 4. E0 基线
python evals-mm/run_eval.py --tag e0 --benchmark flickr30k
python evals-mm/run_eval.py --tag e0 --benchmark coco --limit 200
python evals-mm/run_eval.py --tag e0 --benchmark winoground
# golden 就绪后：--benchmark aro / sugarcrepe

# 5. 硬负例（三路）
python -c "from pathlib import Path; Path('data/hardneg').mkdir(parents=True, exist_ok=True)"
python -m src.hardneg.rule_based \
  --input data/flickr30k/splits/train.jsonl \
  --output data/hardneg/rule_based.jsonl

# 当前 llm_gen 还没有 --limit：先截取 15k 行；D2 再补并发/断点/词集校验
python -c "from pathlib import Path; p=Path('data/flickr30k/splits/train.jsonl'); q=Path('data/flickr30k/splits/train_15k.jsonl'); q.write_text(''.join(p.open(encoding='utf-8').readlines()[:15000]), encoding='utf-8')"
python -m src.hardneg.llm_gen \
  --input data/flickr30k/splits/train_15k.jsonl \
  --output data/hardneg/llm_generated.jsonl

# 注意：当前 adversarial_filter 仍是 text-text 代理；D2 改为 image-negative 后再把 E2a 当主结果
python -m src.hardneg.adversarial_filter \
  --input data/hardneg/llm_generated.jsonl \
  --output data/hardneg/adversarial_filtered.jsonl

# 6. 微调（先 E1/E2/E2a，再按 E-best 跑 E3/E4b）
python -m src.train --config configs/finetune_e1.yaml
python -m src.train --config configs/finetune_e2.yaml
python -m src.train --config configs/finetune_e2a.yaml

# E3 蒸馏前先缓存教师特征（与 train_data 逐行对齐）
python -m src.distill --config configs/finetune_e3.yaml
python -m src.train --config configs/finetune_e3.yaml
python -m src.train --config configs/finetune_e4b.yaml

# 可选：LoRA 附录
python -m src.train --config configs/finetune_e4c.yaml

# 7. 评测
python evals-mm/run_eval.py --tag e2 --checkpoint runs/e2/epoch3.pt --benchmark flickr30k
python evals-mm/run_eval.py --tag e2 --checkpoint runs/e2/epoch3.pt --benchmark winoground

# 8. 单测（CPU，无需数据集）
python -m pytest -q tests

# 9. 盲区对照 demo
python demo/app.py --finetuned runs/e2/epoch3.pt
```

## 仓库结构

```text
CompoCLIP/
├── configs/                     # E1/E2/E2a/E3/E4b/E4c
├── docs/
│   └── dev-doc.md               # 简化版完整开发文档
├── evals-mm/
│   ├── loaders.py               # 五个基准的统一 loader（多 caption 分组）
│   ├── metrics.py               # tie-aware 名次 / R@k / Winoground / 二选一准确率
│   ├── run_eval.py              # 评测入口（含特征缓存）
│   └── golden/                  # 评测集格式规范
├── notebooks/                   # Phase 0 分析实验
├── scripts/
│   ├── download_data.py
│   ├── prepare_flickr30k.py     # Karpathy split -> JSONL
│   └── prepare_pairwise_bench.py
├── src/
│   ├── loss.py                  # 硬负例 InfoNCE + 蒸馏 KL
│   ├── data.py                  # 数据集与 collate（K 个负例 + mask）
│   ├── teacher.py               # 教师特征按下标取用
│   ├── optim.py                 # warmup + cosine
│   ├── lora.py                  # 手写 LoRA（q/v）+ merge 导出
│   ├── hardneg/                 # rule / llm / adversarial filter
│   ├── distill.py               # 教师特征预计算
│   └── train.py                 # full / lock_image / lora + resume
├── tests/                       # CPU 单测
├── demo/                        # Gradio 盲区对照
├── data/                        # git 不跟踪
└── requirements.txt
```

## 当前实现状态

| 模块 | 状态 |
|---|---|
| 数据准备 | `prepare_flickr30k.py` / `prepare_pairwise_bench.py` 可用 |
| 训练 | full / lock_image / lora；warmup+cosine、grad clip、resume、val R@1、metrics.jsonl |
| 损失 | hard-neg mask、K>1、KD 温度、λ=0 等价基线 |
| 教师蒸馏 | 特征缓存 + 按样本下标取 B×B 教师相似度；与 train JSONL 逐行对齐 |
| 评测 | 五基准 loader；多 caption 协议、tie-aware 名次、特征缓存；单测覆盖 |
| 硬负例 | 规则/LLM 基本可用；**对抗过滤仍是 text-text，LLM 缺 limit/并发/词集校验（D2）** |
| Demo | Gradio 骨架；缺缓存特征与预置 query（D5） |

完整状态、D1–D2 待办、分工和 Go/No-Go 门槛见 [`docs/dev-doc.md`](docs/dev-doc.md)。

## 团队分工

| 成员 | 主责 |
|---|---|
| 成员 1 | 诊断（0-B/0-C）+ Winoground/ARO/SugarCrepe 适配器 + 报告 1–4 章 + 统稿 |
| 成员 2 | 数据转换 + 硬负例 + E1/E2/E2a/E3 训练 + 报告 5 章 + 模型卡 |
| 成员 3 | 评测 harness + E0 + COCO/Flickr 检索 + E4b + demo + 报告 6–7 章 |

## 算力

单卡 T4 16GB（fp16，无 bf16）；主线训练 + 分层评测约 **18–24 GPU 小时**；3 个 Colab T4 会话并行。详细预算见 `docs/dev-doc.md` §1.3。

## 文档

- [简化版开发文档](docs/dev-doc.md)：数据、实验矩阵、方法、评测、分工、排期、风险、实现状态。
- 本版已按上一轮审核收敛：保留主线、砍掉 E5/0-D/完整 0-A/全量矩阵，E4c LoRA 降为可选。
