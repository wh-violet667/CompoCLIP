# CompoCLIP · 开发文档 v3.1

> 更新：2026-11-10　｜　上一版：v3.0
>
> **本版改了什么**
> 1. 新增 **D0（外部依赖日）**：数据下载、账号申请、许可确认不进 D1–D7 的关键路径。
> 2. §5 分工改为 RACI，新增「交接物」列与「横切责任」，修掉三处矛盾：backup 全压在成员 3、E4b 归属与「只有成员 2 改 config」冲突、`golden/aro.jsonl` 有三个主人。
> 3. 主指标口径写死为 **macro 平均**（ARO 四个子集不加权；SugarCrepe++ 按 replace/swap/add 不加权）。
> 4. 明确 **负例覆盖率必须对齐**，新增可选对照 E1s。
> 5. 交付物增加「逐样本正确性」与 `docs/decision-log.md`，使配对显著性检验与范围变更可追溯。

---

## 0. 项目定位

### 0.1 一句话

诊断 CLIP 的组合推理盲区，用硬负例微调修复，并用教师蒸馏控制通用能力遗忘；在 Flickr30k / Winoground / ARO / SugarCrepe++ / COCO 上量化「组合增益 vs 通用能力代价」。

### 0.2 研究问题

- **RQ1**：CLIP 的组合失败主要发生在哪一层——词序、主宾、属性，还是关系？
- **RQ2**：硬负例微调能把组合指标提升多少，代价是 COCO / Flickr 检索掉多少？
- **RQ3**：负例来源（规则 / LLM / 过滤后）的差异，是否大于负例数量带来的差异？
- **RQ4**：教师蒸馏能否在不损失组合增益的前提下减小遗忘？锁图像塔、LoRA 的代价曲线如何？

### 0.3 本版保留

6 个主线实验组（E0 / E1 / E2 / E2a / E3 / E4b）；E4c 作为可选附录。主线是「诊断 → 负例来源对比 → 抗遗忘 → 策略矩阵」。

### 0.4 本版砍掉 / 降级

| 砍掉 | 原因 | 替代 |
|---|---|---|
| E5 更大模型（ViT-B-16 / SigLIP） | 算力与时间不够 | 报告「未来工作」一节 |
| 0-D 全量消融 | 与主线不构成差异 | 只保留 smoke 级趋势 |
| 完整 0-A 网格 | 同上 | 精简版（2–3 组 lr / 数据量） |
| 五基准 × 所有组 的全量矩阵 | 评测与写作负担过大 | 分层评测（§4.1） |
| `w_hn` / λ 全网格 | 时间不够 | 主表 λ=0.5；附录 λ=1.0（可选） |
| 额外随机种子 | 时间不够 | 单 seed + 明确写局限性 |
| Mnemo 实际接入 | 不占本期开发资源 | 报告第 8 节展望 |

### 0.5 交付物与负责人

| 交付物 | 内容 | 负责人 | 验收 |
|---|---|---|---|
| 数据就绪包 | 五基准可用 + 来源 / 许可 / 校验和 | 成员 2 | 成员 1 |
| 诊断实验包 | 0-B、0-C、误差分类表 | 成员 1 | 成员 3 |
| 硬负例数据集 | 规则 / LLM / 过滤三份 + 质检表 + 覆盖率 | 成员 2（规则、过滤）、成员 1（LLM） | 成员 1 / 成员 3 |
| 主训练组 | E1/E2/E2a/E3/E4b checkpoints + 日志 + config | 成员 2 | 成员 3 |
| E0 检索基线 | Flickr30k / COCO，`tag=e0` | 成员 2 | 成员 3 |
| E0 组合基线 | Winoground / ARO / SugarCrepe++ | 成员 1 | 成员 3 |
| 评测框架与结果库 | 五基准适配 + 特征缓存 + `master.csv` + 逐样本正确性 | 成员 3 | 成员 1 |
| Demo | Gradio 检索界面（E0 vs E-best） | 成员 3 | 成员 1 |
| 课程报告 | 见 §8；成员 1 统稿 | 全员 | 成员 1 |
| 答辩 PPT | 盲区对照素材 + 主结果 | 成员 1 lead | 全员排练 |

### 0.6 成功标准（预注册，看结果前写死）

- **主指标**：ARO 四个子集的 **macro** 平均 + SugarCrepe++ 三个子类的 **macro** 平均。不加权，避免 VG 子集（各约 3 万）压过 order 子集（各约 5 千）。
- **主结论**：E3 相对 E0 提升 ≥ **2.0pp**，同时 COCO R@1 下降 ≤ **1.0pp**。
- **蒸馏的增量结论**：E3 vs E-best（λ=0）在组合指标上下降 ≤ **0.5pp**，且 COCO 遗忘更小；否则不能说「蒸馏有效」。
- **辅助指标**：Winoground group 提升 ≥ 2pp，报 95% bootstrap CI；**不单独作为成功依据**（400 对样本的 CI 约 ±4pp，2pp 落在噪声内）。
- **E-best 选择规则**：在 λ=0 的 E1/E2/E2a 中，按 ARO + SugarCrepe++ 的 macro 平均选最高；平手看 COCO R@1；再平手选规则法（更简单、更可复现）。报告必须同时列出三个来源。
- **负结果也算交付**：若 E3 不同时满足增益与遗忘约束，按「组合增益—遗忘代价」如实报告，不强行宣称成功。

---

## 1. 实验矩阵

### 1.1 实验组

| 组 | 基座 | 负例 | 损失 | 策略 | 目的 | 新增训练 |
|---|---|---|---|---|---|---|
| **E0** | CLIP ViT-B-32 | — | — | 不微调 | 对照 | 否 |
| **E1** | 同上 | 规则法（全量） | 硬负例加权 InfoNCE | 全量 | 来源① | 是 |
| **E1s**（可选） | 同上 | 规则法（15k 子集） | 同上 | 全量 | 与 E2 对齐覆盖率 | 是 |
| **E2** | 同上 | LLM 法（15k） | 同上 | 全量 | 来源② | 是 |
| **E2a** | 同上 | LLM + 对抗过滤 | 同上 | 全量 | 过滤 pipeline 消融 | 是 |
| **E2r**（建议做） | 同上 | E2 的随机子集，与 E2a 同规模 | 同上 | 全量 | 质量 vs 数量 | 是 |
| **E3** | 同上 | E-best 来源 | + 教师蒸馏 λ=0.5 | 全量 | 抗遗忘 | 是 |
| **E4a** | 同上 | E-best 来源 | λ=0 | 全量 | 策略矩阵基准 | 否，复用 E-best |
| **E4b** | 同上 | E-best 来源 | λ=0 | 锁图像塔 | 策略矩阵对比 | 是 |
| *E4c（可选）* | *同上* | *E-best 来源* | *λ=0* | *LoRA* | *附录 / future work* | *有余量才做* |

> **E2a 的解释边界**：E2a 的数据量通常小于 E2。不跑 E2r 时，E2 vs E2a 只能说明「过滤 pipeline 的整体效果」，不能声称「质量比数量重要」。

### 1.2 执行顺序

```text
D0–D1：数据、golden、E0 五基准冒烟
  → E1 / E2 / E2a（并行）
  → 按 §0.6 规则选 E-best
  → E3（λ=0.5）与 E4b（锁图像塔）并行
  → 可选：E1s、E2r、λ=1.0 附录、E4c LoRA
```

- E4a 复用 E-best checkpoint，不新增训练。
- E3 与 E4b 都用 E-best 的负例配置；E4b 固定 λ=0，避免「策略差异」与「蒸馏差异」混淆。
- 任何实验开跑前先做 100 step 吞吐实测；单组预计超过 3 小时就降 epoch / batch 或缩数据。

### 1.3 算力预算

| 项目 | 估算 GPU 时 |
|---|---|
| Phase 0（0-A 精简 + 0-B/0-C 推理） | 4–6 |
| E1 / E2 / E2a | 5–6 |
| E1s / E2r（可选） | 各 1.5–2 |
| E3 | 2 |
| E4b | 1.2 |
| E4c LoRA（可选附录） | 1.5–2 |
| 评测（分层 + 特征缓存） | 5–8 |
| **合计** | **18–24（主线）/ 22–28（含 E1s、E2r、E4c）** |

3 个 Colab 账号足以覆盖；本地机器只做评测和 demo。**真正的约束是人的时间与 D0 的外部依赖，不是 GPU 时。**

### 1.4 负例覆盖率（必须对齐，否则结论不成立）

| 组 | 负例覆盖行数 / 总行数 | 说明 |
|---|---|---|
| E1 | 145k / 145k = 100% | 规则法可全量跑 |
| E2 | 15k / 145k ≈ 10% | LLM 限流，只跑前 15k 行（≈3k 张图） |
| E2a | 过滤后 / 145k，可能 3–8% | 再过滤一次 |
| E1s | 15k / 145k ≈ 10% | 可选对照，用于隔离「覆盖率」 |

要求：

- 每次训练把 `num_with_negatives` 写进 `metrics.jsonl` 第一行。
- 报告里 E1 vs E2 必须写明覆盖率差异；不跑 E1s 就不能把差异全归因于「负例来源」。
- 时间允许时优先补 **E2r**（随机负例、与 E2a 同规模），它比 E4c 便宜，且更能支撑核心 claim。

---

## 2. 数据

### 2.1 数据集与用途

| 数据集 | 规模 | 用途 | 获取方式 | 注意 |
|---|---|---|---|---|
| Flickr30k | 31.8k 图 / 158k caption | 主训练 + 检索评测 | Kaggle `nunenuh/flickr30k` + Karpathy split JSON | 必须用 Karpathy 29k/1k/1k |
| Flickr8k | 8k 图 / 40k caption | 0-A 小消融 | Kaggle `adityajn105/flickr8k` | 只做短程趋势 |
| COCO 2017 val | 5k 图 / 25k caption | 遗忘度量 | Kaggle `awsaf49/coco-2017-dataset` | 可只下 val |
| COCO 2017 train 20k 子集 | 20k 图 | 教师蒸馏 / 通用正则 | 同上 | 替代 CC3M |
| Winoground | 400 对 | 组合推理诊断 | HF `facebook/winoground` | **gated，需提前申请**；仅评测 |
| ARO | 4 个子集 | 组合推理主证据 | ARO 官方来源；VG 两个子集需 Visual Genome 图片（约 1GB） | D0 必须验证下载路径可用 |
| SugarCrepe++ | ~7.5k 对 | 组合推理补充 | CLIC 项目页面 | **手工下载 + 许可**，D0 完成 |

### 2.2 目录与 JSONL schema

```text
data/
├── flickr30k/
│   ├── images/
│   └── splits/{train,val,test}.jsonl
├── coco2017/
│   ├── val2017/
│   └── annotations/
├── hardneg/
│   ├── rule_based.jsonl
│   ├── rule_based_15k.jsonl      # E1s，可选
│   ├── llm_generated.jsonl
│   └── adversarial_filtered.jsonl
├── aro/
├── sugarcrepepp/
└── golden/{aro,sugarcrepe}.jsonl
```

每行一条 caption：

```json
{"image":"1000092795.jpg","caption":"A man is riding a bike.","image_id":"1000092795","caption_id":0}
```

- `image` 相对 `images/` 目录；`image_id` 唯一；`caption_id` 为 0–4。
- 所有路径走 `DATA_ROOT` 环境变量。
- 训练与评测都只读 JSONL，不再混用 Karpathy 原始 JSON。

### 2.3 数据准备

```bash
# 0. 账号与许可（D0，见 §6.1）
#    Kaggle token、HF gated 申请、SugarCrepe++ 许可、LLM API key

python scripts/download_data.py --dataset flickr30k
python scripts/download_data.py --dataset coco2017
python scripts/prepare_flickr30k.py --karpathy-json /path/dataset_flickr30k.json

python scripts/prepare_pairwise_bench.py --input "data/aro/*.json" \
  --image-root data/aro/images --output data/golden/aro.jsonl
python scripts/prepare_pairwise_bench.py --input "data/sugarcrepepp/*.json" \
  --image-root data/sugarcrepepp/images --output data/golden/sugarcrepe.jsonl
```

`prepare_flickr30k.py` 校验 Karpathy 规模（29k/1k/1k 图、5 caption/图）；`prepare_pairwise_bench.py` 负责 ARO / SugarCrepe++ 的字段归一化。

> Windows PowerShell 下 `mkdir -p` 不可用，用 `New-Item -ItemType Directory -Force data\hardneg`。

**D0 出口（硬性）**：五个基准各抽 50 条，图片能打开、正负 caption 人工核对无误、`subset` 字段正确。这一步不过，D1 不允许开始。

### 2.4 硬负例生成

三路，全部只从 **train split** 生成：

1. **规则法**：spaCy 依存解析，交换主宾；失败退化为交换两个名词。全量 145k 行。
2. **LLM 法**：few-shot 让模型用**相同词集**重排，表达不同或相反含义。先跑 15k 行，必须支持限流重试、断点续跑、词多重集校验。
3. **对抗过滤**：用当前 CLIP 给每个（图像, 负例）打分，只保留模型「看不出差别」的样本。阈值用**分位数校准**，不写死 0.85 / 0.6。

质检（D1–D2，成员 1 抽检 100 条）必须回答三件事：

- 正负是否真的语义不同（不是同义改写）？
- 负例是否恰好等于同图另一条 GT caption（假负例）？
- 负例是否因标点、大小写、语序异常而「太容易」？

### 2.5 防泄漏

- Winoground 只用于评测，绝不进训练。
- ARO / SugarCrepe++ 只用于评测。
- 硬负例只能从 train split 生成，不能从 val/test 生成。
- 报告写数据使用声明与许可说明；LAION2B 可能见过 Flickr30k / COCO / Winoground，必须写污染风险。

---

## 3. 方法

### 3.1 硬负例 InfoNCE

图像→文本方向：

```text
L_i2t = -log( exp(s_ii / τ) / ( Σ_j exp(s_ij / τ) + w_hn · Σ_{k∈HN(i)} exp(s_ik / τ) ) )
```

- `w_hn` 固定 1.0；时间富余补 2.0 一组进附录。
- 缺失硬负例必须 mask，不能把空串编码后拼进分母。
- K 暂定 1；若数据里有多条负例，collate 后按 mask 支持 K>1。
- 文本→图像方向仍用批内负例；报告里说明这个不对称性（硬负例由正 caption 改写而来，没有对应的真实图像，不能当 t2i 的查询）。

### 3.2 教师蒸馏抗遗忘

- **教师特征缓存**：原始 CLIP 在 Flickr30k train 上做一次前向，缓存图像 / 文本特征，不缓存完整相似度矩阵。若要把 COCO train 20k 混进来，必须先合并成同一个 JSONL 再缓存（缓存与训练按行号严格对齐）。
- **必须同时缓存教师的 `logit_scale`**：教师相似度要用「教师自己的温度」再除 `kd_temperature`，否则教师分布接近均匀，KL 项会把学生往均匀分布拉，得到的不是蒸馏而是一次隐式的熵正则。
- **损失**：`L = L_hardneg + λ · KL(P_teacher || P_student)`，λ=0.5 为主。
- **λ=0**：直接复用 E-best；λ=1.0 仅作附录可选。
- **目的**：在组合增益与 COCO 遗忘之间给出可解释的 trade-off，而不是假设蒸馏一定有用。

### 3.3 锁图像塔

- 冻结视觉塔，只训练文本塔；图像特征可预计算复用。
- 预期显存省约 40%，训练更快；组合增益可能低于全量，但遗忘更小。
- E4a（全量）复用 E-best；E4b（锁塔）单独训练；两者同数据、同 λ=0。

### 3.4 超参数起点

| 项 | 值 | 状态 |
|---|---|---|
| 基座 | `ViT-B-32` + `laion2b_s34b_b79k` | 已有 |
| batch / accum | 128 / 2（等效 256） | 已有 |
| epochs | 4 | 已有 |
| lr | 默认 1e-5；pilot {1e-6, 5e-6, 1e-5} | D2 pilot 定 |
| wd | 0.1 | 已实现 |
| precision | AMP fp16 | 已实现 |
| grad clip | 1.0 | 已实现 |
| warmup / cosine | 200 step / cosine | 已实现 |
| logit scale clamp | ln(100) | 已实现 |
| hard_negative_weight | 1.0 | 已实现 |
| distill λ | 0.5 | 已接线 |
| seed | 42 | D1 补进训练脚本 |

> **注意**：`configs/` 的字段必须与训练脚本对齐；新增超参先改代码再改 config，避免「配置看起来生效、实际没生效」。LoRA 的 `lora_r / lora_alpha` 只在 `strategy: lora` 时生效。

---

## 4. 评测

### 4.1 分层评测

| 层级 | 基准 | 参与组 | 目的 |
|---|---|---|---|
| **Tier 1** | Flickr30k test、Winoground | E0/E1/E2/E2a/E3/E4b | 所有组都有主结果 |
| **Tier 2** | COCO val 5k、ARO、SugarCrepe++ | E0/E2/E3/E4b；E1/E2a 视时间 | 遗忘与组合推理主证据 |

- Tier 1 必须全跑；Tier 2 先保 E0/E2/E3/E4b。
- 特征缓存按 `encoder × benchmark` 存，避免重复编码。

### 4.2 正确的检索协议

**Flickr30k / COCO**：

- 测试集是 **N 张唯一图片 × 5 条 caption**，不是 5N 张独立图片。
- t2i：每条 caption 检索图片；正确图片命中即算对。
- i2t：每张图片检索 caption；5 条正确 caption 中任意一条进 top-k 即算对。
- 重复图片先去重；并列分数用确定性规则，不能依赖 `argsort` 的任意 tie 顺序。

**Winoground**：官方 Text / Image / Group 三项，400 对，报 bootstrap CI。

**ARO**：4 个子集分别报准确率，再报 **macro** 平均；pairwise 打分（正 caption vs 负 caption）。

**SugarCrepe++**：pairwise 准确率，按 replace / swap / add 分开报，再报 **macro** 平均。

### 4.3 结果产物

```text
evals-mm/
├── golden/                     # 格式规范（真实数据在 $DATA_ROOT/golden）
├── .cache/                     # encoder × benchmark 特征缓存
└── results/
    ├── {bench}.csv             # 按 benchmark 追加
    ├── master.csv              # 唯一结果库（long format）
    ├── {tag}_{bench}_persample.npz   # 逐样本正确性，用于配对检验
    └── {tag}_{bench}_misses.json     # 未命中样本
```

`master.csv` 至少包含：

```text
tag, benchmark, subset, metric, value, n, config_hash, checkpoint, git_commit, seed, timestamp
```

### 4.4 统计与报告纪律

- 同批次、同 `limit` 的数字才可横向比较。
- 主证据用 ARO + SugarCrepe++ 的 macro 平均；Winoground 只作辅助。
- 必须落盘逐样本正确性，对「E0 vs E3」「E-best vs E3」做**配对** bootstrap 或 McNemar，而不是只看两个 CI 是否重叠。
- 报告所有来源，不只报 E-best；选择规则看结果前写死。
- 单 seed 是局限，必须写明；有条件给 E-best 与 E3 各补一个 seed。
- COCO 下降 >2pp 要如实归因，不能只挑有利数字。
- 失败样本抽 30 条做归因（词序 / 属性 / 关系 / 假负例）。

---

## 5. 团队分工 v3.1

> 原则：一人一条主实验线 + 一块共享基建 + 一块报告章节 + 一块横切责任。
> **主责唯一、验收唯一、交接物写死。** 不设「人人有责」的条目。

### 5.1 RACI 主表

R = 主责（对结果负责），A = 验收（签字的人），C = 参与 / 支持。

| 工作流 | R | A | C | 交接物 | 截止 |
|---|---|---|---|---|---|
| 数据获取（Kaggle / HF / ARO / SugarCrepe++ / 许可） | 成员 2 | 成员 1 | 成员 3 | `docs/sources.md`：来源、许可、校验和、目录树 | D0 |
| Karpathy split 与 JSONL schema | 成员 2 | 成员 3 | 成员 1 | `splits/*.jsonl` + 规模校验输出 | D1 |
| ARO / SugarCrepe++ golden | 成员 2 | 成员 1 | 成员 3 | `data/golden/*.jsonl` + 20 条人工核对记录 | D1 |
| 规则法负例 | 成员 2 | 成员 1 | 成员 3 | `hardneg/rule_based.jsonl` + 质检表 | D1 |
| **LLM 负例（15k）** | **成员 1** | 成员 2 | 成员 3 | `hardneg/llm_generated.jsonl` + 词集校验日志 + 覆盖率 | D2 |
| 对抗过滤 | 成员 2 | 成员 1 | 成员 3 | `hardneg/adversarial_filtered.jsonl` + 分桶统计 + 覆盖率 | D2 |
| 训练 E1/E2/E2a/E3/**E4b** | 成员 2 | 成员 3 | 成员 1 | `runs/{tag}/epoch*.pt` + `metrics.jsonl` + config + git commit | D3–D4 |
| E0 Flickr30k / COCO | 成员 2 | 成员 3 | 成员 1 | `tag=e0` 结果行 + 特征缓存 | D1 |
| E0 Winoground / ARO / SugarCrepe++ | 成员 1 | 成员 3 | 成员 2 | `tag=e0` 结果行 + misses | D2 |
| 评测 harness + 特征缓存 + `master.csv` | 成员 3 | 成员 1 | 全员 | harness + `master.csv` + 逐样本 npz | D1 起持续 |
| 全组评测（E1/E2/E2a/E3/E4b） | 成员 3 | 成员 1 | 成员 2 | 结果行 + misses + 出图脚本 | D3–D5 |
| Demo | 成员 3 | 成员 1 | 成员 2 | Gradio app + 预置 query + 缓存特征 | D5 |
| 报告 1–4 章 | 成员 1 | 成员 2 | 成员 3 | Markdown / LaTeX 分章文件 | D5 |
| 报告 5 章 + 模型卡 | 成员 2 | 成员 1 | 成员 3 | 章节 + 模型卡 + 数据声明 | D6 |
| 报告 6–7 章 | 成员 3 | 成员 1 | 成员 2 | 章节 + 主表 + 图 | D6 |
| 统稿与终稿 | 成员 1 | 全员 | — | 完整 PDF | D6 晚 |
| 答辩 PPT | 成员 1 | 全员 | 成员 2/3 供图 | PPT + 排练记录 | D7 |

### 5.2 横切责任（v3.1 新增，之前无人负责）

| 责任 | 内容 | 负责人 |
|---|---|---|
| 决策与范围控制 | Go/No-Go 拍板、砍实验、冻结数字；范围变更写入 `docs/decision-log.md` | 成员 1 |
| 外部账号与预算 | 3 个 Colab 账号分配、Kaggle token、HF gated 申请、LLM API key 与额度、Drive 目录 | 成员 2 |
| 环境与复现 | 依赖版本锁定、`DATA_ROOT` 约定、clean clone 可跑通 | 成员 2 |
| 版本与留痕 | git tag（`e0` / `e1` …）、config 冻结、结果 → commit 可追溯 | 成员 3 |
| 图表与引用 | 统一配色 / 字号 / 表格模板；`docs/references.md`（ARO、SugarCrepe++、NegCLIP、LAION 的引用与许可） | 成员 1（图）、成员 2（许可清单） |

### 5.3 备份矩阵（v3.1 重排）

| 关键路径 | 主责 | 备份 | 说明 |
|---|---|---|---|
| 训练 | 成员 2 | 成员 1 | 成员 1 本来就要读 loss 曲线和评测结果 |
| 数据与负例 | 成员 2 | 成员 3 | 成员 3 熟悉 schema 与评测契约 |
| 评测 harness | 成员 3 | 成员 2 | 成员 2 是数据侧最熟悉格式的人 |
| 报告整合 | 成员 1 | 成员 2 | 成员 2 写第 5 章，最了解方法 |
| Demo | 成员 3 | 成员 1 | D5 前必须有一版静态截图兜底 |

**规则**：备份只在主责缺席超过 4 小时时启用；启用后主责降为 reviewer，并在 `decision-log.md` 记一行。**不要在文档里声称「每条路径都有备份」——真正能接管的只有一个人。**

### 5.4 交接契约

1. **checkpoint**：`runs/{tag}/epoch{n}.pt`，同目录必须有 `config.yaml`、`metrics.jsonl`；`metrics.jsonl` 第一行写 `num_with_negatives`。
2. **结果行**：必须带 `tag / weights / seed / limit / git_commit`，否则不入 `master.csv`。
3. **golden**：必须带 `subset`，且附 20 条人工核对记录（正负是否标反）。
4. **图表**：原始数据来自 `master.csv`，不接受手工抄数；出图脚本进 `notebooks/` 或 `scripts/figures/`。
5. 任何一次「换了数据 / 换了 checkpoint / 改了阈值」都要在 `decision-log.md` 记时间、原因、影响范围。

### 5.5 协作规则

- **只有成员 2 修改训练超参与 config**。E4b 由成员 2 训练、成员 3 评测；其他人托管 Colab 会话时不许改 lr / batch / epochs。
- **E0 每个 benchmark 只跑一次**：Flickr30k / COCO 由成员 2 跑，Winoground / ARO / SugarCrepe++ 由成员 1 跑；成员 3 负责汇总与抽查，不重复跑「验证」。
- 所有 E0 运行统一 `--tag e0`、同一 git commit、同一 `DATA_ROOT`、同一 seed；正式结果不许带 `--limit`。
- 每天 22:00 站会，只回答三个问题：昨天交付什么、今天交付什么、被谁阻塞。
- **被阻塞超过 4 小时必须立刻升级给成员 1**，不要等到站会。
- D4 晚训练冻结；D5 晚数字冻结；之后只修 bug，不新增实验。

### 5.6 工时预算（8 天，约 5 h/天/人）

| 成员 | 主要工时分布 | 合计 |
|---|---|---|
| 成员 1 | 诊断 6h、三个组合基准 4h、LLM 负例 8h、报告 1–4 章 14h、统稿 6h、PPT 6h | ≈ 44h |
| 成员 2 | 数据 8h、规则法与过滤 10h、训练托管 6h、第 5 章 + 模型卡 12h、许可与声明 3h | ≈ 39h |
| 成员 3 | harness 10h、全组评测 12h、demo 8h、第 6–7 章 10h、汇总与出图 4h | ≈ 44h |

> 三个人的总量接近，但**成员 2 全在关键路径上**（D0–D4 无缓冲）。这就是把 LLM 负例挪给成员 1 的原因：那条线不占 GPU，且成员 1 在 D1 本来就被 harness 阻塞。

---

## 6. 排期与 Go/No-Go

### 6.1 D0：外部依赖日（不进 D1–D7）

| 事项 | 负责人 | 出口 |
|---|---|---|
| Kaggle token、HF 登录、Winoground gated 申请（可能等 1 天） | 成员 2 | 三个都能拉到数据 |
| ARO / SugarCrepe++ 下载路径验证（含 VG 图片与许可） | 成员 2 | 五基准各 50 条能打开 |
| LLM API key 与额度（15k 次调用预算） | 成员 2 | 能跑通 20 条 smoke |
| Colab × 3 + Drive 目录结构 | 成员 3 | `DATA_ROOT` 约定写入文档 |
| 依赖版本锁定、clean clone 跑通 | 成员 2 | `pip install -r` 一次成功 |
| 建 `decision-log.md` / `references.md` / `sources.md` | 成员 1 | 三个空文件 + 模板 |

**D0 不过，D1 不开工。** 这一步拖到 D1 做，就是把 5 个外部依赖塞进关键路径。

### 6.2 D1–D7

| 天 | 成员 1 | 成员 2 | 成员 3 | 当天出口（唯一） |
|---|---|---|---|---|
| **D1** | 0-B + 0-C 词序探针 + 误差分类 | split + golden + 规则负例 + E0 检索冒烟 | harness + golden 契约 + 特征缓存 | split 与 golden 可用；E0 检索冒烟 |
| **D2** | LLM 负例（15k）+ E0 组合三基准 | 对抗过滤 + lr pilot + teacher cache + E3 smoke | E0 汇总 + 复核 | **E0 五基准进 master**；E3 smoke 通过 |
| **D3** | 0-A 精简 + 报告 3 章 | E1 / E2 / E2a 训练 | 随 checkpoint 评测 + 第 6 章提纲 | 3 个 checkpoint + Flickr30k/Winoground |
| **D4** | 探针复跑 + 报告 4 章 | E3 + E4b 训练 | 全组评测 + demo 骨架 | 主结果落表 |
| **D5** | 1–4 章定稿 + 图表 | 第 5 章初稿 + 模型卡 | 全矩阵收口 + demo + 未命中分析 | **数字冻结** |
| **D6** | 统稿 + 交叉校对 + 终稿 | 第 5 章定稿 + 数据声明与许可 | 第 6–7 章定稿 + 图 | **完整初稿** |
| **D7** | PPT + 排练 | buffer | buffer | 提交 / 答辩 |

> D7 是唯一 buffer，所以**报告必须 D6 晚定稿**，D7 只做 PPT 与排练。

### 6.3 Go/No-Go 门槛

| 时点 | 必须满足 | 不满足的预案 | 决策人 |
|---|---|---|---|
| **D0 22:00** | 五基准各 50 条可 open；依赖一次装好；三个账号可用 | 全员停下做数据，D1 顺延；砍 SugarCrepe++，主指标降级为 ARO | 成员 1 |
| **D1 22:00** | split 可用；E0 Flickr30k R@1 在合理区间 | 全员支援数据；Phase 0 后移；先保 E0 | 成员 1 |
| **D2 22:00** | E0 五基准进 master；LLM 负例就绪或明确回退规则法 | 缺的 E0 由成员 3 当晚补跑；回退规则法；E3 只跑规则法版 | 成员 1 |
| **D3 22:00** | E1/E2/E2a checkpoint + Flickr30k/Winoground 数字 | 砍 E2a 或 E4b，保 E1/E2 | 成员 1 |
| **D4 22:00** | E3 + E4b 结果落表 | 训练冻结，报告只写 E1/E2 + 负结果分析 | 成员 1 |
| **D5 22:00** | 全部数字 + demo 可用 | demo 降级为静态 top-k 截图，报告用表格 | 成员 1 |

### 6.4 风险与预案

| 风险 | 概率 | 影响 | 预案 | 负责人 |
|---|---|---|---|---|
| HF gated 审批慢 / 拿不到 Winoground | 高 | 少一个辅助基准 | D0 就申请；备选官方仓库下载 | 成员 2 |
| ARO / SugarCrepe++ 数据或许可拿不到 | 中 | **主指标失效** | 主指标降级为 ARO 单基准 + Winoground，报告写清 | 成员 2 / 成员 1 |
| LLM 负例限流、质量差 | 中 | E2 无法开训 | 词集校验 + 断点续跑；回退规则法；E2 降为附录 | 成员 1 |
| Colab 断连 / 配额被收回 | 高 | 训练中断 | 每 epoch 存盘、Drive 持久化、转 Kaggle | 成员 3 |
| OOM | 中 | 训练起不来 | batch 96 + accum 3；锁塔组优先 | 成员 2 |
| fp16 不稳 | 中 | loss 发散 | lr 降到 1e-6、grad clip、保底 fp32 | 成员 2 |
| E3 没有抗遗忘效果 | 中 | 主结论落空 | 如实报 trade-off，不强行宣称成功 | 成员 1 |
| 单 seed 结论弱 | 高 | 审稿被问 | 至少给 E-best 与 E3 补第二个 seed | 成员 3 |
| 关键人缺席 | 中 | 路径停摆 | §5.3 备份矩阵 + 交接物齐全 | 成员 1 |
| 报告写不完 | 高 | 交付失败 | D5 数字冻结，D6 只写作，D7 只做 PPT | 成员 1 |

---

## 7. Demo

- Gradio 单页：query 输入 + E0 / E-best 切换 + top-8 结果网格。
- **盲区对照**：组合 query 下并排展示微调前后 top-5。
- 使用 500–1000 张精选图 + **预计算特征**；不要在启动时编码 5000 张图。
- 预置 5–10 条组合 query（主宾颠倒、属性交换、关系词），否则用户不知道输入什么。
- 不做用户系统、数据库、Mnemo 对接；应用价值在报告第 8 节用一段话 + 一张架构图交代。
- D4 就要有一版可截图的骨架，D5 定稿；PPT 素材不能等 demo 完成才做。

---

## 8. 报告大纲与分工

```text
1. 引言与问题定义                        成员 1
2. 相关工作                              成员 1
3. 对比预训练机理分析（0-A / 0-B / 0-D smoke）  成员 1
4. 失败模式诊断（0-C + 误差分类）          成员 1
5. 方法（硬负例 + 蒸馏 + 锁塔）            成员 2
6. 实验（主结果 + 消融 + 未命中分析）       成员 3
7. Demo                                  成员 3
8. 局限与展望                            全员素材，成员 1 统稿
参考文献 + 附录                          成员 2（清单）/ 全员
```

**写作纪律**：

- 不写「首次」；改成「在统一协议下的系统复现与对比」。
- 不把 KD 防遗忘说成全新方法；贡献是 CLIP 组合微调场景下的量化 trade-off。
- 明确写：单 seed、Winoground 统计功效、LAION 数据污染、LLM 非确定性、E2a 数据量混淆、**E1 与 E2 的覆盖率差异**。
- 主表必须同时列出 E1 / E2 / E2a，不只报 E-best；组合指标一律报 macro 平均并附 95% CI。
- 第 6 章由成员 3 主笔，但成员 2 必须供一节「训练侧分析」（loss 曲线、负例覆盖率、耗时与显存），否则容易写成数字罗列。

---

## 9. 当前实现状态与 D1–D2 待办

> 这一节是「诚实清单」，以仓库实际代码为准。在对应项修好之前，不要引用相关数字。

| 模块 | 现状 | 剩余缺口 | 负责人 | 截止 |
|---|---|---|---|---|
| 数据获取 | 下载脚本可用 | 五基准实际落地、许可与校验和、D0 出口验证 | 成员 2 | D0 |
| 数据转换 | Karpathy split、pairwise 归一化脚本可用 | ARO / SugarCrepe++ 的字段与正负方向人工核对 | 成员 2 / 成员 1 | D1 |
| Phase 0 | `notebooks/` 为空 | 0-B / 0-C 探针；0-A 精简版 | 成员 1 | D1–D2 |
| 训练入口 | 全量 / 锁塔 / LoRA；warmup+cosine、grad clip、resume、val R@1 | 训练 seed、100-step 吞吐 pilot、E4b 由成员 2 接管 | 成员 2 | D1 |
| 损失 | 硬负例 mask、K>1、KD 温度、λ=0 等价基线 | 无 | 成员 3 | 已完成 |
| 教师缓存 | 特征缓存、按下标取 B×B、长度对齐检查 | **教师 `logit_scale` 必须一起缓存**，否则 KD 退化为熵正则 | 成员 3 | D2 |
| 硬负例 | 规则 / LLM / 过滤三脚本可跑 | 输出目录、seed、LLM `--limit`/并发/断点/词集校验；过滤改为图–文打分与分位阈值；假负例过滤 | 成员 2 / 成员 1 | D1–D2 |
| 评测 | 五基准 loader、多 caption 协议、tie-aware 名次、特征缓存、CI、`master.csv` | macro 平均、逐样本 npz、golden 就位 | 成员 3 | D1–D5 |
| 配置 | E1/E2/E2a/E3/E4b/E4c 已对齐 | E1s / E2r 的子集生成 | 成员 2 | D2 |
| Demo | Gradio 骨架 | 预计算特征、精选图、预置 query | 成员 3 | D5 |
| 测试 | `tests/` 覆盖 loader / metrics / collate / loss / scheduler / teacher / lora | 每次改代码跑 `pytest -q tests`；补一条用真实 tokenizer 的 collate 用例 | 成员 3 | 持续 |
| 文档 | 本版 v3.1 | 按实际结果回填数字与状态；decision-log 每日更新 | 成员 1 | 持续 |

**D1–D2 最小闭环**：

1. D0 出口全绿（五基准 50 条 open、依赖可装、账号可用）。
2. split / golden 产出并通过 20 条人工核对（正负方向、`subset`、图片路径）。
3. E0 五基准至少各跑一次 `--limit` 冒烟，数字进 `master.csv`（标 `is_smoke`）。
4. 规则法全量 + LLM 15k 负例产出，附覆盖率与词集校验日志。
5. 训练加 seed，跑 100-step 吞吐 pilot；`pytest -q tests` 全绿。
6. E3 teacher cache（含 `logit_scale`）先跑 Flickr30k 版 smoke。

---

## 10. Definition of Done

- [ ] clean clone 后按 README 能完成：D0 数据 → E0 → E1/E2/E2a → E3/E4b → 评测 → demo。
- [ ] 所有 config 与代码字段一致；没有「写了但没生效」的超参。
- [ ] `results/master.csv` 含 Tier 1 全部组 + Tier 2 的 E0/E2/E3/E4b，且每行可追溯到 commit 与 checkpoint。
- [ ] 逐样本正确性已落盘，主结论附配对检验。
- [ ] 未命中样本、golden、特征缓存可复现。
- [ ] 报告 + PPT + 模型卡 + 数据声明 + `references.md` 齐全。
- [ ] 明确写局限性；不把负结果包装成正结果。
- [ ] E5 不进主线；E4c 标注「可选 / 附录」，不承诺结果。
- [ ] `decision-log.md` 记录了每一次范围变更与砍掉的实验。

---

## 11. 未来工作（报告一小节）

- E5：`ViT-B-16-SigLIP` 或冻结 / LoRA 版 SO400M，补 sigmoid 版 hard-neg loss。
- LoRA：扩展到更多模块、更多层与更大模型。
- CC3M / 更大通用数据；多 seed；更大模型。
- 中文组合推理基准。
- Mnemo 接入：把 E3 / E-best 封装为 `ImageTextEncoder` 服务，作为图像记忆检索的编码内核。
