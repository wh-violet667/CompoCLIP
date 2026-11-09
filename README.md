# CompoCLIP

> Compositionally-Aware Image Retrieval: Diagnosing and Fixing Contrastive Vision-Language Models

机器学习课程三人小组项目：诊断 CLIP/SigLIP 的组合推理盲区，通过硬负例微调与教师蒸馏抗遗忘修复，并在 Winoground/ARO/SugarCrepe++ 等基准上量化改进。

## 核心创新点

1. **硬负例来源系统对比**：规则生成 vs LLM 生成 vs 对抗过滤分桶，回答"负例质量 vs 数量"
2. **教师蒸馏抗遗忘**：以原始 CLIP 相似度分布做 KL 正则，显式控制组合增益与遗忘代价的权衡
3. **微调策略帕累托矩阵**：全量 / 锁图像塔 / LoRA 三维对比
4. （可选）**SigLIP 对照**：softmax vs sigmoid 对比损失对硬负例的响应差异

## 快速开始

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt        # 测试与 notebooks
python -m spacy download en_core_web_sm

# 可选：自定义数据目录（默认 ./data）
# export DATA_ROOT=/content/drive/MyDrive/compoclip/data

# 1. 拉取原始数据（需先配置 ~/.kaggle/kaggle.json）
python scripts/download_data.py --list
python scripts/download_data.py --all

# 2. 生成 Flickr30k 的 Karpathy split（训练与评测都依赖，必须先跑）
python scripts/prepare_flickr30k.py --karpathy-json /path/dataset_flickr30k.json
# 冒烟验证：python scripts/prepare_flickr30k.py ... --limit 200

# 3. 生成 ARO / SugarCrepe++ 评测集（可选，组合推理主证据）
python scripts/prepare_pairwise_bench.py --input "data/aro/*.json" \
    --image-root data/aro/images --output data/golden/aro.jsonl

# 4. E0 基线评测（结果追加到 evals-mm/results/*.csv）
python evals-mm/run_eval.py --tag e0 --benchmark flickr30k
python evals-mm/run_eval.py --tag e0 --benchmark coco --limit 200
python evals-mm/run_eval.py --tag e0 --benchmark winoground

# 5. 生成硬负例 -> 微调 -> 评测
python -m src.hardneg.rule_based --input data/flickr30k/splits/train.jsonl \
    --output data/hardneg/rule_based.jsonl
python -m src.train --config configs/finetune_e2.yaml
python evals-mm/run_eval.py --tag e2 --checkpoint runs/e2/epoch3.pt --benchmark flickr30k

# 6. 盲区对照 demo
python demo/app.py --finetuned runs/e2/epoch3.pt

# 7. 单测（CPU，无需数据集）
python -m pytest -q tests
```

## 仓库结构

```text
CompoCLIP/
├── notebooks/      # Phase 0 分析实验（温度消融 / 表示空间探查 / 词序探针）
├── configs/        # 每组实验一个 yaml，与实验记录一一对应
├── caches/         # 教师 logits 缓存（git 不跟踪）
├── scripts/
│   ├── download_data.py          # 拉取原始数据集
│   ├── prepare_flickr30k.py      # 生成 Karpathy split（必须先跑）
│   └── prepare_pairwise_bench.py # ARO/SugarCrepe++ 归一化
├── src/
│   ├── loss.py             # 硬负例加权 InfoNCE + 蒸馏 KL
│   ├── hardneg/            # rule_based / llm_gen / adversarial_filter
│   ├── distill.py          # 教师 logits 预计算
│   └── train.py            # 微调入口（full / lock_image / lora 三策略）
├── evals-mm/
│   ├── loaders.py          # 五个基准的统一 loader（多 caption 分组）
│   ├── metrics.py          # tie-aware 名次 / R@k / Winoground / 二选一准确率
│   ├── retrieval.py        # FAISS 检索后端
│   ├── run_eval.py         # 评测入口
│   └── golden/README.md    # 评测集格式规范
├── demo/           # Gradio 盲区对照
├── tests/          # CPU 单测（检索协议 / loader 分组）
├── data/           # git 不跟踪：split、hardneg、golden 均在此生成
└── docs/
    ├── dev-doc.md  # 完整开发文档
    └── report/     # 课程报告
```

## 算力基线

单卡 T4 16GB（fp16，无 bf16），全程训练预算约 15–20 GPU 小时，详见 [docs/dev-doc.md](docs/dev-doc.md)。
