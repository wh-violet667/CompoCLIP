# CompoCLIP

> Compositionally-Aware Image Retrieval: Diagnosing and Fixing Contrastive Vision-Language Models

机器学习课程三人小组项目：诊断 CLIP/SigLIP 的组合推理盲区，通过硬负例微调与教师蒸馏抗遗忘修复，并在 Winoground/ARO/SugarCrepe++ 等基准上量化改进。

## 核心创新点

1. **硬负例来源系统对比**：规则生成 vs LLM 生成 vs 对抗过滤分桶，回答"负例质量 vs 数量"
2. **教师蒸馏抗遗忘**：以原始 CLIP 相似度分布做 KL 正则，显式控制组合增益与遗忘代价的权衡
3. **微调策略帕累托矩阵**：全量 / 锁图像塔 / LoRA 三维对比
4. （可选）**SigLIP 对照**：softmax vs sigmoid 对比损失对硬负例的响应差异

## 仓库结构

```text
CompoCLIP/
├── notebooks/      # Phase 0 分析实验（温度消融 / 表示空间探查 / 词序探针）
├── configs/        # 每组实验一个 yaml，与实验记录一一对应
├── caches/         # 教师 logits 缓存（git 不跟踪）
├── src/
│   ├── loss.py             # 硬负例加权 InfoNCE + 蒸馏 KL
│   ├── hardneg/            # 规则法 / LLM 法 / 对抗过滤
│   ├── distill.py          # 教师 logits 预计算
│   └── train.py            # open_clip 微调入口
├── evals-mm/       # 五基准评测器（Flickr30k/COCO/Winoground/ARO/SugarCrepe++）
├── demo/           # Gradio 最小 demo（盲区对照）
├── data/           # git 不跟踪
└── docs/
    ├── dev-doc.md  # 完整开发文档（数据集 / 训练配置 / 评测方案 / 分工里程碑 / 报告大纲）
    └── report/     # 课程报告
```

## 算力基线

单卡 T4 16GB（fp16，无 bf16），全程训练预算约 15–20 GPU 小时，详见 [docs/dev-doc.md](docs/dev-doc.md)。
