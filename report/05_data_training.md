# 5 数据与训练

> 本章由成员 3 根据 A2 交接材料整理；数据、负例和训练事实以 `A2/README.md`、`A2/报告/Chapter_5_Methods.md`、`A2/训练分析/training_analysis.md` 为准。正式数字以 `report/results/master.csv` 和 `evals-mm/results/*.csv` 为准。

## 5.1 数据与评测划分

本项目使用 Flickr30k 的冻结划分：29,000 张训练图 / 145,000 条训练 caption，1,014 张验证图 / 5,070 条验证 caption，1,000 张测试图 / 5,000 条测试 caption。训练只使用 train split；val 用于 300 张图的训练监控；test 只进入评测和 ARO 的 Flickr-Order 子集，不进入梯度训练。所有微调实验都从同一个原始 `ViT-B-32 / laion2b_s34b_b79k` 初始化，不从彼此的权重继续训练。

| 数据 | 规模 | 用途 |
|---|---:|---|
| Flickr30k train | 29,000 图 / 145,000 caption | 正样本与负例源 |
| Flickr30k val | 1,014 图 / 5,070 caption | 300 图训练监控 |
| Flickr30k test | 1,000 图 / 5,000 caption | 检索评测、ARO Flickr-Order |
| ARO | 82,695 查询（VG-Rel / VG-Att / COCO-Order / Flickr-Order） | 组合推理评测 |
| 原版 SugarCrepe | 7,511 查询（replace / swap / add） | 组合推理评测 |
| COCO val2017 | 5,000 图 / 25,014 caption | 通用检索遗忘评测 |

> 数据来源与许可的完整说明见 `A2/报告/Data_Statement.md` 与 `A2/报告/Sources_and_Licenses.md`。Flickr30k / Winoground 的部分评测图来自公开镜像，正式提交前仍需完成官方数据复核；这部分限制不能省略。

## 5.2 硬负例构造

五组训练共享 145,000 条正样本，区别在于附加负例和训练目标。

| 组 | 负例来源 | 负例记录 | 覆盖训练行 | 覆盖图片 | 覆盖比例 | 人工状态 |
|---|---|---:|---:|---:|---:|---|
| E1 | 规则法（颜色绑定 / 主宾交换） | 1,158 | 1,031 | 915 | 0.7110% | 224 keep / 934 pending |
| E2 | LLM 定向生成 + 早期人工保留 | 1,739 | 1,739 | 1,739 | 1.1993% | 291 keep / 1,448 pending |
| E2a | E2 中按原 CLIP 图像–负例相似度筛出的高相似子集 | 522 | 522 | 522 | 0.3600% | 73 keep / 449 pending |
| E3 | 复用 E2 负例 | 1,739 | 1,739 | 1,739 | 1.1993% | 同 E2 |
| E4b | 复用 E2 负例 | 1,739 | 1,739 | 1,739 | 1.1993% | 同 E2 |

E1 的规则集合以颜色绑定交换为主，另有少量主宾交换；固定 100 条抽检中 93 条保留、7 条拒绝，但抽检不等于逐条批准。E2 的最终集合包含 195 条较早人工保留记录和 1,544 条定向生成记录；定向批次有 1,671 条输入，部分候选因词袋、重复或语义检查被排除。E2a 用原始 CLIP 对 E2 的“图像–负 caption”余弦相似度排序，取第 70 百分位以上的 522 条；它不是用 E2 模型筛的，且保留了原始人工状态。

**必须保留的口径限制**：

- 最终 E2 是 `reduced_targeted` 集合，不是最初计划的 15,000 条 LLM 负例；不能写成“15k 负例已全部人工批准”。
- E1 vs E2 同时改变负例来源和覆盖量；E2 vs E2a 同时改变筛选和保留数量。没有 matched-size 随机子集对照，因此不能把差异单独归因于“质量”或“数量”。
- 负例来自训练 split，不来自 val/test；Flickr test 只作为评测与 ARO Flickr-Order 的输入，不进入训练。

## 5.3 训练配置

五组训练使用同一套基础配置：

| 项 | 值 |
|---|---|
| 基座 | `ViT-B-32` / `laion2b_s34b_b79k` |
| epochs | 4 |
| batch / accumulation | 64 / 4（每次更新访问 256 行） |
| optimizer | AdamW，lr 1e-5，wd 0.1 |
| scheduler | cosine，warmup 200 次更新 |
| precision | AMP；T4 |
| seed | 42 |
| 更新次数 | 2,268 次（E1/E2/E2a 记录为 scheduled attempts；E3/E4b 记录 applied updates） |
| 监控 | 300 张验证图 / 1,500 条 caption 的 t2i R@1，仅用于监控，不进入正式主表 |

E1/E2/E2a 为全量微调，不使用蒸馏。E3 同样全量微调，但加入教师蒸馏：教师是原始 CLIP，教师缓存为 Flickr30k train 的 145,000 行、512 维 FP32 特征，KL 温度 2、λ=0.5。E4b 冻结图像塔（87,849,216 个参数），只训练文本塔和 logit scale，不使用蒸馏。E4a 直接复用 E-best=E2 的权重，没有单独训练。

> 训练日志、GPU 显存、吞吐和每 epoch 监控见 `A2/训练分析/training_analysis.md`。这些监控数字不能替代 formal benchmark。

## 5.4 E-best 选择

E-best 只在 E1/E2/E2a 之间选择，使用 A2 冻结的 4 个 ARO 子集和 3 个原版 SugarCrepe 家族；每个子集/家族等权，两个 benchmark 再等权平均。

| 模型 | ARO macro | SugarCrepe macro | Selection score |
|---|---:|---:|---:|
| E1 | 43.9742% | 80.0462% | 62.0102% |
| **E2** | **44.8225%** | **80.3141%** | **62.5683%** |
| E2a | 44.2889% | 80.1597% | 62.2243% |

E2 以 0.3440pp 领先 E2a、0.5582pp 领先 E1，没有触发 COCO tie-break。E2 同时作为 E4a；E3 和 E4b 复用 E2 的负例，但各自从原始 CLIP 重新初始化。上述分数是**选模指标**，不是独立泛化检验，也不建立统计显著性；ARO 和 SugarCrepe 同时用于选择与报告，必须披露这一重叠。

## 5.5 本章小结

本项目最终完成的是 `reduced_targeted` 范围下的五组训练：E1 规则负例、E2 LLM 负例、E2a 过滤子集、E3 蒸馏、E4b 锁图像塔；E4a 复用 E2。训练完成只说明模型可以评测，不说明组合增益或抗遗忘成立。E3/E4b 与 E0/E-best 的正式比较、配对统计和失败分析放在第 6 章。
