# CompoCLIP · 简化版开发文档 v3.0

> **题目**：Compositionally-Aware Image Retrieval: Diagnosing and Fixing Contrastive Vision-Language Models
> **团队**：3 人 | **周期**：1 周（D1–D7） | **算力**：单卡 T4 16GB（fp16，无 bf16）+ 3 个 Colab 免费 T4 会话
> **本版原则**：先保证主线可交付，再谈扩展。相比 v2.4，本版砍掉 E5、0-D 全量训练、完整 0-A 网格、CC3M 依赖和“五基准 × 所有实验组”的全量矩阵；E4c LoRA 从主线降为可选附录（代码已保留），把任务压到 3 人 1 周能完成的规模。

---

## 0. 项目定位

### 0.1 一句话

诊断 CLIP 的组合推理盲区，用硬负例微调修复，用教师蒸馏控制遗忘，在 Flickr30k / Winoground / ARO / SugarCrepe++ / COCO 上量化“组合增益 vs 通用能力代价”。

### 0.2 研究问题

| 编号 | 问题 | 对应实验 |
|---|---|---|
| RQ1 | 规则负例 vs LLM 负例，哪种更有效？ | E1 vs E2 |
| RQ2 | 对抗过滤（只保留高图文相似负例）能否用更少数据换来更好组合增益？ | E2 vs E2a（可选 E2r） |
| RQ3 | 教师蒸馏能否在保留组合增益的同时降低 COCO 遗忘？ | E3 vs E-best |
| RQ4 | 全量微调 vs 锁图像塔，组合增益 / 遗忘 / 成本如何权衡？ | E4a vs E4b |

### 0.3 本版保留

- **主实验**：E0、E1、E2、E2a、E3、E4b。
- **Phase 0**：保留 0-B（embedding 空间探查）和 0-C（词序探针 + Winoground 零样本）；0-A 只做 2–3 组小消融；0-D 只做 2 小时 smoke 或直接引用文献。
- **评测分层**：所有组跑 Flickr30k + Winoground；E0/E2/E3/E4b 跑 COCO + ARO + SugarCrepe++，E1/E2a 视时间补。
- **硬负例三路**：规则 / LLM / 对抗过滤。
- **教师蒸馏**：λ=0.5 为主；λ=0 复用 E-best；λ=1.0 仅作附录可选。
- **策略矩阵**：全量（E4a = E-best）vs 锁图像塔（E4b）；E4c LoRA 为可选附录。
- **Demo**：Gradio 单页，E0 vs E-best，缓存特征 + 500–1000 张精选图。
- **报告**：8–12 页，章节与实验一一对应。

### 0.4 本版砍掉 / 降级

| 项目 | 处理 | 原因 / 去向 |
|---|---|---|
| E5 SigLIP | 砍掉 | SO400M 在 T4 全量微调不可行；sigmoid 版 hard-neg loss 未实现；报告 future work |
| E4c LoRA | **降级为可选附录** | 代码与单测已在 `src/lora.py`，但主线时间紧；只有 D4 前有余量才跑，结果进附录 |
| 0-D 从头训练 | 不服务主线，训练/对照成本高 | 2h smoke 或引用文献 |
| 0-A 完整网格 | 多组短训练工期不够 | 只做 τ={0.07, 0.5}、batch={64, 256} 两组小消融 |
| CC3M 子集 | 下载链接不稳定，增加数据准备风险 | 用 COCO train2017 20k 子集做蒸馏/混数据 |
| 五基准 × 所有组 | 评测与写作负担过大 | 分层评测（见 §4.1） |
| `w_hn` / λ 全网格 | 时间不够 | 主表 λ=0.5；附录 λ=1.0（可选） |
| 额外随机种子 | 时间不够 | 单 seed + 明确写局限性 |
| Mnemo 实际接入 | 不占本期开发资源 | 报告第 8 节展望 |

### 0.5 交付物与负责人

| 交付物 | 内容 | 负责人 |
|---|---|---|
| 诊断实验包 | 0-B、0-C、误差分类表 | 成员 1 |
| 硬负例数据集 | 规则 / LLM / 对抗过滤三份 + 质检报告 | 成员 2 |
| 主训练组 | E1/E2/E2a/E3 checkpoints + 训练日志 | 成员 2 |
| 评测框架与结果库 | E0 基线 + 五基准适配 + `results/master.csv` | 成员 3 |
| 策略矩阵 | E4b 锁图像塔 + 与 E4a 的对比 | 成员 3 |
| Demo | Gradio 检索界面（E0 vs E-best） | 成员 3 |
| 课程报告 | 见 §8；成员 1 统稿 | 全员 |
| 答辩 PPT | 盲区对照素材 + 主结果 | 成员 1 lead |

### 0.6 成功标准（预注册，看结果前写死）

- **主指标**：ARO 4 个子集平均 + SugarCrepe++ 平均。
- **主结论**：E3 相对 E0 提升 ≥ **2.0pp**，同时 COCO R@1 下降 ≤ **1.0pp**。
- **蒸馏的增量结论**：E3 vs E-best（λ=0）在组合指标上下降 ≤ **0.5pp**，且 COCO 遗忘更小；否则不能说“蒸馏有效”。
- **辅助指标**：Winoground group 提升 ≥ 2pp，报告 95% bootstrap CI；不单独作为成功依据。
- **E-best 选择规则**：在 λ=0 的 E1/E2/E2a 中，按 ARO + SugarCrepe++ 平均选最高；平手看 COCO R@1；再平手选规则法（更简单、更可复现）。报告中必须同时列出三个来源，不只报 best。
- **负结果也算交付**：如果 E3 没有同时满足增益与遗忘约束，就按“组合增益—遗忘代价”曲线如实报告，不强行宣称成功。

---

## 1. 简化后的实验矩阵

### 1.1 实验组

| 组 | 基座 | 负例 | 损失 | 策略 | 目的 | 新增训练 |
|---|---|---|---|---|---|---|
| **E0** | CLIP ViT-B-32 | — | — | 不微调 | 对照 | 否 |
| **E1** | 同上 | 规则法 | 硬负例加权 InfoNCE | 全量 | 负例来源对比① | 是 |
| **E2** | 同上 | LLM 法 | 同上 | 全量 | 负例来源对比② | 是 |
| **E2a** | 同上 | LLM + 对抗过滤 | 同上 | 全量 | 过滤 pipeline 消融 | 是 |
| **E2r**（可选） | 同上 | E2 的随机子集，样本量与 E2a 对齐 | 同上 | 全量 | 质量 vs 数量的对照 | 是 |
| **E3** | 同上 | E-best 来源 | + 教师蒸馏 λ=0.5 | 全量 | 抗遗忘 | 是 |
| **E4a** | 同上 | E-best 来源 | 硬负例 InfoNCE，λ=0 | 全量 | 策略矩阵基准 | 否，复用 E-best |
| **E4b** | 同上 | E-best 来源 | 同上 | 锁图像塔 | 策略矩阵对比 | 是 |
| **E4c**（可选） | 同上 | E-best 来源 | 同上 | LoRA | 附录 / future work | 有余量才做 |

> **E2a 的解释边界**：E2a 的数据量通常小于 E2。如果不跑 E2r，E2 vs E2a 只能说明“过滤 pipeline 的整体效果”，不能直接声称“质量比数量重要”。报告里必须写清这一点。

### 1.2 执行顺序

```text
E1 / E2 / E2a
   → 按 0.6 的规则选 E-best
   → E3（λ=0.5）与 E4b（锁图像塔）并行
   → 可选：E2r、λ=1.0 附录、E4c LoRA
```

- E4a 直接复用 E-best checkpoint，不新增训练。
- E3 与 E4b 都使用 E-best 的负例配置；E4b 固定 λ=0，避免把“策略差异”和“蒸馏差异”混在一起。
- 任何实验启动前先跑 100 step 实测吞吐；单组预计超过 3 小时就降 epoch / batch 或缩数据。

### 1.3 算力预算

| 项目 | 估算 GPU 时 |
|---|---|
| Phase 0（0-A 小消融 + 0-B/0-C 推理） | 4–6 |
| E1 / E2 / E2a | 5–6 |
| E3 | 2 |
| E4b | 1.2 |
| E2r（可选） | 1.5–2 |
| E4c LoRA（可选附录） | 1.5–2 |
| 评测（分层 + 特征缓存） | 5–8 |
| **合计** | **18–24（主线）/ 20–26（含 E4c）** |

3 个 Colab 账号足以覆盖；本地 T4 只做评测和 demo。**真正的约束是人的时间，不是 GPU 时。**

---

## 2. 数据

### 2.1 数据集与用途

| 数据集 | 规模 | 用途 | 获取方式 | 注意 |
|---|---|---|---|---|
| Flickr30k | 31.8k 图 / 158k caption | 主训练 + 检索评测 | Kaggle `nunenuh/flickr30k` + Karpathy split JSON | 必须用 Karpathy 29k/1k/1k |
| Flickr8k | 8k 图 / 40k caption | 0-A 小消融 | Kaggle `adityajn105/flickr8k` | 只做短程趋势 |
| COCO 2017 val | 5k 图 / 25k caption | 遗忘度量 | Kaggle `awsaf49/coco-2017-dataset` | 只下 val 即可先跑 |
| COCO 2017 train 20k 子集 | 20k 图 | 教师蒸馏 / 通用数据正则 | 同上，取 train2017 子集 | 替代 CC3M |
| Winoground | 400 对 | 组合推理诊断 | HF `facebook/winoground` | **仅评测，严禁进训练** |
| ARO | 4 个子集 | 组合推理主证据 | ARO 官方 repo（VG-R/VG-A 为 GQA 子集，约 1GB） | COCO/Flickr Order 复用现有图 |
| SugarCrepe++ | ~7.5k 对 | 组合推理补充 | CLIC 项目页面 / 官方仓库 | 许可以页面为准 |

### 2.2 目录与 JSONL schema

```text
data/
├── flickr30k/
│   ├── images/
│   └── splits/{train,val,test}.jsonl
├── flickr8k/
├── coco2017/
│   ├── val2017/
│   └── train2017_subset/
├── hardneg/
│   ├── rule_based.jsonl
│   ├── llm_generated.jsonl
│   └── adversarial_filtered.jsonl
├── aro/
└── sugarcrepepp/
```

每行一条 caption：

```json
{"image":"1000092795.jpg","caption":"A man is riding a bike.","image_id":"1000092795","caption_id":0,"split":"train"}
```

- `image` 相对 `images/` 目录；`image_id` 唯一；`caption_id` 为 0–4。
- 所有路径走 `DATA_ROOT` 环境变量；脚本不再硬编码 `data/`。
- 训练/评测都只读 JSONL；不再混用 Karpathy 原始 JSON。

### 2.3 数据准备（已实现）

```bash
# 1. 下载原始图片
python scripts/download_data.py --dataset flickr30k
python scripts/download_data.py --dataset coco2017     # 只需 val 也可手动只放 val2017

# 2. Karpathy split -> JSONL（支持 --limit 冒烟）
python scripts/prepare_flickr30k.py --karpathy-json /path/dataset_flickr30k.json
# 或从 HF 加载：--from-hf nlphuji/flickr30k --save-images

# 3. ARO / SugarCrepe++ -> 统一 pairwise JSONL
python scripts/prepare_pairwise_bench.py \
  --input "data/aro/*.json" --image-root data/aro/images --output data/golden/aro.jsonl

# 4. 硬负例输出目录（脚本当前不会自动创建）
mkdir -p data/hardneg
```

`prepare_flickr30k.py` 会校验 Karpathy 规模（29k/1k/1k 图、5 caption/图）并打印警告；`prepare_pairwise_bench.py` 负责 ARO/SugarCrepe++ 的字段归一化。若使用自定义目录，统一设置 `DATA_ROOT`。

> Windows PowerShell 下 `mkdir -p` 不可用，改用 `New-Item -ItemType Directory -Force data\hardneg`，或直接用 README 里的 Python 一行命令。

### 2.4 硬负例生成

| 来源 | 做法 | 规模 | 质检 |
|---|---|---|---|
| 规则法 | spaCy 依存解析，交换主宾/名词；固定 seed | 每条正样本 1 条 | 100 条人工抽检 |
| LLM 法 | few-shot prompt“同词不同序”；低成本模型 | 约 15k 条（3k 图 × 5 caption 或 15k 图 × 1） | 词多重集校验 + 100 条人工 |
| 对抗过滤 | 用 CLIP 给 **图像–负 caption** 打分，取 top 30% 分位 | 约为 E2 的 30% | 保存分桶直方图 |

> **当前实现状态**：`rule_based.py` / `llm_gen.py` / `adversarial_filter.py` 仍是基础版。
> - 三个脚本都还没有自动创建输出目录；跑之前先 `mkdir -p data/hardneg`。
> - `rule_based.py` 没有固定 seed；`llm_gen.py` 没有 `--limit` / 并发 / 断点 / 词集校验。
> - `adversarial_filter.py` 仍用 text-text 相似度，且固定阈值 0.85/0.6；**在改成 image–negative 打分和分位阈值之前，E2a 只能作 pipeline 冒烟，不能当主结果。**
> 这些是 D1–D2 的硬任务，owner 为成员 2，成员 3 提供过滤指标与评测口径。

**硬性要求**：

- LLM 负例必须校验“词多重集相同”；只检查“非空/不相等”不够。
- 对抗过滤不能再用 text-text 相似度；必须用 image–negative text 相似度。
- 阈值用分位数校准，不用固定 0.85/0.6；先在小样本上看分布。
- LLM 生成要支持 `--limit`、`--workers`、`--resume`，并缓存输出；不要默认处理全部 145k caption。
- 如果 D2 晚上 LLM 负例未就绪，D3 先跑规则法 E1，E2 后补；不要让 E1 等 LLM。

### 2.5 防泄漏

- Winoground 只用于评测，绝不进训练。
- ARO / SugarCrepe++ 只用于评测。
- 硬负例只能从 train split 生成，不能从 val/test 生成。
- 报告里写数据使用声明和许可说明。

---

## 3. 方法

### 3.1 硬负例 InfoNCE

图像→文本方向：

```text
L_i2t = -log( exp(s_ii/τ) / ( Σ_j exp(s_ij/τ) + w_hn · Σ_k∈HN(i) exp(s_ik/τ) ) )
```

- `w_hn` 固定 1.0；时间富余时补 2.0 一组进附录。
- 缺失硬负例必须 mask，不能把空字符串编码后拼进分母。
- K 暂定 1；如果数据里有多条负例，collate 后按 mask 支持 K>1。
- 文本→图像方向仍用批内负例；报告里说明这个不对称性。

### 3.2 教师蒸馏抗遗忘

- **教师特征缓存**：原始 CLIP 在 Flickr30k train 上做一次前向，缓存图像/文本特征，不缓存完整相似度矩阵；若要主张“抗 COCO 遗忘”，再把 COCO train 20k 子集合并进同一个 JSONL 后重跑（见下方注意）。
- **训练时**：按 batch 索引取出教师特征，计算 `teacher_sim = teacher_img @ teacher_txt.T`，再算 KL。
- **损失**：`L = L_hardneg + λ · KL(P_teacher || P_student)`，λ=0.5 为主。
- **温度**：KD 单独用 `kd_temperature`（建议 2.0），不要直接复用对比损失的 `logit_scale=100`。
- **λ=0**：直接复用 E-best；**λ=1.0** 仅作附录可选。
- **目的**：在组合增益和 COCO 遗忘之间给出可解释的 trade-off，而不是假设蒸馏一定有用。

> `src/distill.py` 与 `src/train.py` 按 JSONL 行号严格对齐，长度不一致会直接报错（`src/teacher.py`）。如果要把 COCO train 20k 混进教师缓存，必须先把 Flickr30k train 与 COCO 子集合并成同一个 JSONL，再让 distill 和 train 使用同一个文件；当前没有自动合并脚本，D2 补。

### 3.3 锁图像塔

- 冻结 `model.visual`，只训练文本塔；图像特征可预计算并复用。
- 预期显存省约 40%，训练更快；组合增益可能低于全量，但遗忘更小。
- E4a（全量）复用 E-best；E4b（锁塔）单独训练；两者同数据、同 λ=0。

### 3.4 超参数起点

| 项 | 值 | 状态 |
|---|---|---|
| 基座 | `ViT-B-32` + `laion2b_s34b_b79k` | 已有 |
| batch / accum | 128 / 2（等效 256） | 已有 |
| epochs | 4 | 已有 |
| lr | config 默认 1e-5；pilot {1e-6, 5e-6, 1e-5} | 待 pilot |
| wd | 0.1 | 已实现 |
| precision | AMP fp16 | 已实现 |
| grad clip | 1.0 | 已实现 |
| warmup / cosine | 200 step / cosine | 已实现 |
| logit scale clamp | ln(100) | loss/trainer 内已有 |
| hard_negative_weight | 1.0 | 已实现 |
| distill λ | 0.5 | 已接线，含 KD 温度 |
| seed | 42 | 训练脚本 D1 补；评测已有 `--seed` |

> **注意**：`configs/` 里的字段现在与 `src/train.py` 对齐；如果新增超参，先改代码再改 config，避免“配置看起来生效、实际没生效”。LoRA 的 `lora_r/lora_alpha` 只在 `strategy: lora` 时生效。

---

## 4. 评测

### 4.1 分层评测

| 层级 | 基准 | 参与组 | 目的 |
|---|---|---|---|
| **Tier 1** | Flickr30k test、Winoground | E0/E1/E2/E2a/E3/E4b | 所有组都有主结果 |
| **Tier 2** | COCO val 5k、ARO、SugarCrepe++ | E0/E2/E3/E4b；E1/E2a 视时间 | 遗忘与组合推理主证据 |

- Tier 1 必须全跑；Tier 2 先保 E0/E2/E3/E4b。
- 当前 `run_eval.py` 按 benchmark 追加 `results/{bench}.csv`；D5 再把它们汇总成唯一的 `results/master.csv`。
- 特征缓存按 `encoder × benchmark` 存，避免重复编码（`evals-mm/.cache`）。

### 4.2 正确的检索协议

**Flickr30k / COCO**：

- 测试集是 **N 张唯一图片 × 5 条 caption**，不是 5N 张独立图片。
- t2i：每条 caption 检索图片；正确图片命中即算对。
- i2t：每张图片检索 caption；5 条正确 caption 中任意一条进 top-k 即算对。
- MedR 按唯一图片统计。
- 重复图片先去重；并列分数用确定性规则，不能依赖 `argsort` 的任意 tie 顺序。
- 以上协议已在 `evals-mm/loaders.py` 与 `evals-mm/metrics.py` 实现：按 `image_id` 分组、多正样本 mask、`tie="average"` 名次；`tests/test_metrics.py` 覆盖。

**Winoground**：官方 Text / Image / Group 三项；400 对，报告 bootstrap CI。

**ARO**：4 个子集分别报准确率，再报平均；pairwise 打分（正 caption vs 负 caption）。

**SugarCrepe++**：pairwise 准确率；按子类（replace/swap/add）分开报。

### 4.3 结果产物

```text
evals-mm/
├── golden/                  # 固定评测样本与 id
├── .cache/                  # encoder × benchmark 特征缓存（git 不跟踪）
└── results/
    ├── flickr30k.csv        # run_eval 当前按 benchmark 追加
    ├── coco.csv
    ├── winoground.csv
    ├── aro.csv
    ├── sugarcrepe.csv
    ├── master.csv           # D5 汇总（唯一结果库）
    └── {tag}_{bench}_misses.json
```

`master.csv` 至少包含：

```text
tag, benchmark, subset, metric, value, n, config_hash, checkpoint, git_commit, seed, timestamp
```

### 4.4 统计与报告纪律

- 同批次数字才可横向比较。
- Winoground 只作辅助证据；主证据用 ARO + SugarCrepe++。
- 报告所有来源，不只报 E-best；E-best 选择规则在看结果前写死。
- 单 seed 是局限，必须在报告中写明；有条件再补第二个 seed。
- COCO 下降 >2pp 要如实归因，不能只挑有利数字。
- LAION2B 可能见过 Flickr30k/COCO/Winoground，报告里要写数据污染风险。

---

## 5. 团队分工（唯一版本）

> 原则：每人一条主实验线 + 一块共享基础设施 + 一块报告章节；关键路径必须有 backup。

| 工作流 | 主责（对结果负责） | 支持 / 评审 | 交付与验收 |
|---|---|---|---|
| 数据转换、Karpathy split、COCO 数据 | 成员 2 | 成员 3（schema/评测）、成员 1（split 语义） | `prepare_*.py` + 校验报告；成员 3 验收 |
| 规则/LLM/对抗过滤负例 + 质检 | 成员 2 | 成员 1（QC）、成员 3（过滤指标） | 三份 JSONL + 质检表；成员 1 抽检 |
| E1/E2/E2a/E3 训练 | 成员 2 | 成员 3（E3 代码/缓存）、成员 1（评测） | checkpoints + 日志 + config；成员 3 评测 |
| E0 + 评测 harness + 特征缓存 + 结果库 | 成员 3 | 成员 1（benchmark 语义）、成员 2（数据） | `results/master.csv` + misses；成员 1 核对 |
| Winoground/ARO/SugarCrepe 适配器 | 成员 1 | 成员 3（harness）、成员 2（数据） | 五基准数字；成员 3 验收 |
| COCO/Flickr 检索协议 | 成员 2（数据）+ 成员 3（评测） | 成员 1（review） | COCO R@k + 遗忘指标；成员 1 核对 |
| E4b 锁图像塔 | 成员 3 | 成员 2（训练 config） | checkpoint + 结果；成员 2 验收 |
| E4c LoRA（可选附录） | 成员 3 | 成员 2（backup） | 只有 D4 前有余量才做；否则明确写“未做” |
| Demo | 成员 3 | 成员 1（示例 query）、成员 2（模型卡） | Gradio app；成员 1 验收 |
| 报告 | 成员 1：1–4 章 + 统稿；成员 2：5 章 + 模型卡/数据声明；成员 3：6–7 章 | 全员第 8 章 | 完整 PDF；成员 1 统稿 |
| 答辩 PPT | 成员 1 lead | 全员提供图表 | 排练通过 |

**协作规则**：

- **只有成员 2 修改训练超参和 config**；其他人可以托管 Colab 会话，但不改 lr/batch/epochs。
- 每个关键路径都有 backup：评测（成员 1）、数据转换（成员 3）、训练（成员 3）、报告整合（成员 3）。
- 每天 22:00 站会，站会只回答三个问题：昨天交付什么、今天交付什么、被谁阻塞。
- D4 晚训练冻结；D5 晚数字冻结；之后只修 bug，不新增实验。

---

## 6. 排期与 Go/No-Go

### 6.1 D1–D7

| 天 | 成员 1（诊断 + 报告 1–4） | 成员 2（数据 + 训练） | 成员 3（评测 + 策略 + demo） | 当天出口 |
|---|---|---|---|---|
| **D1** | 0-B + 0-C 词序探针 + 误差分类 | 数据转换 + resume + 0-D smoke | 评测 harness + golden + E0 Flickr30k | split 可用、E0 出数 |
| **D2** | Winoground/ARO/SugarCrepe 适配器 + 0-C Winoground 基线 | 规则/LLM 负例 + 过滤校准 + lr pilot | E0 全基准 + teacher cache + E3 smoke | E0 冻结；E4c 可选，不影响主线 |
| **D3** | 0-A 精简版 + 报告 3 章初稿 | E1/E2/E2a 训练 | 随 checkpoint 评测 + 报告 6 章提纲 | 3 个 checkpoint + Flickr30k/Winoground |
| **D4** | 探针复跑 + 报告 4 章初稿 | E3 + E4b 训练 | E2/E2a/E3/E4b 评测 + demo 骨架 | 主结果落表 |
| **D5** | 1–4 章定稿 + 图表 | 第 5 章初稿 + 模型卡 | 全矩阵收口 + demo + 未命中分析 | 数字冻结 |
| **D6** | 报告整合 + 交叉校对 | 第 5 章定稿 + 数据声明 | 第 6–7 章定稿 + demo 定稿 | 完整初稿 |
| **D7** | 答辩 PPT + 排练 | 同左 + buffer | 同左 + buffer | 提交/答辩 |

### 6.2 Go/No-Go 门槛

| 时点 | 必须满足 | 不满足的预案 |
|---|---|---|
| **D1 22:00** | split 转换可用；E0 Flickr30k R@1 在合理范围 | 全员停 Phase 0，先修数据和评测 |
| **D2 22:00** | LLM 负例就绪或明确回退规则法；E3 smoke 通过 | 回退规则法；E3 只跑规则法版 |
| **D3 22:00** | E1/E2/E2a checkpoint + Flickr30k/Winoground 数字 | 砍 E2a 或 E4b，保 E1/E2 |
| **D4 22:00** | E3 + E4b 结果落表 | 训练冻结，报告只写 E1/E2 + 负结果分析 |
| **D5 22:00** | 全部数字 + demo 可用 | demo 降级为静态 top-k，报告用表格 |

### 6.3 风险与预案

| 风险 | 概率 | 预案 |
|---|---|---|
| 数据转换/评测协议延期 | 高 | D1 全员支援；Phase 0 后移；先保 E0 |
| Colab 断连 / 配额被收回 | 高 | 每 epoch 存 checkpoint；Drive 持久化；转 Kaggle |
| fp16 不稳 | 中 | 降 lr 到 1e-6；开 grad clip；保底 fp32 |
| OOM | 中 | batch 96 + accum 3；锁塔组优先 |
| LLM 负例质量差/限流 | 中 | 词集校验 + 回退规则法；E2 降级为附录 |
| E3 没有抗遗忘效果 | 中 | 如实报告 trade-off；不强行宣称成功 |
| Winoground 提升不显著 | 中 | 以 ARO + SugarCrepe++ 为主证据 |
| 一周内做不完 | 高 | 按 §0.4 砍；D4 冻结后不再新增实验 |

---

## 7. Demo

- Gradio 单页：query 输入 + E0/E-best 切换 + top-8 结果网格。
- **盲区对照**：组合 query 下并排展示微调前后 top-5。
- 使用 500–1000 张精选图 + 预计算特征；不要在启动时编码 5000 张图。
- 预置 5–10 条组合 query（主宾颠倒、属性交换、关系词），否则用户不知道输入什么。
- 不做用户系统、数据库、Mnemo 对接；应用价值在报告第 8 节用一段话 + 一张架构图交代。

---

## 8. 报告大纲与分工

```text
1. 引言与问题定义                        成员 1
2. 相关工作                              成员 1
3. 对比预训练机理分析（0-A/0-B/0-D smoke） 成员 1
4. 失败模式诊断（0-C + 误差分类）          成员 1
5. 方法（硬负例 + 蒸馏 + 锁塔）            成员 2
6. 实验（主结果 + 消融 + 未命中分析）       成员 3
7. Demo                                  成员 3
8. 局限与展望                            全员素材，成员 1 统稿
参考文献 + 附录                          全员
```

**写作纪律**：

- 不写“首次”；改成“在统一协议下的系统复现与对比”。
- 不把 KD 防遗忘说成全新方法；贡献是 CLIP 组合微调场景下的量化 trade-off。
- 明确写：单 seed、Winoground 统计功效、LAION 数据污染、LLM 非确定性、E2a 数据量混淆。
- 主表必须同时列出 E1/E2/E2a，不只报 E-best。

---

## 9. 当前实现状态与 D1–D2 待办

> 这一节是“诚实清单”。在对应项修好之前，不要引用相关数字。

| 模块 | 现状 | 剩余缺口 | Owner | 截止 |
|---|---|---|---|---|
| 数据准备 | `download_data.py`、`prepare_flickr30k.py`、`prepare_pairwise_bench.py` 可用 | 需要实际放好 Karpathy JSON / COCO annotations / ARO·SugarCrepe 原始文件 | 成员 2 | D1 |
| Phase 0 | `notebooks/` 目前为空 | 0-B / 0-C 需成员 1 在 D1–D2 写并跑；0-A 只做精简版 | 成员 1 | D1–D2 |
| 训练入口 | full / lock_image / lora；warmup+cosine、grad clip、resume、val R@1、metrics.jsonl | 训练 seed、100-step 吞吐 pilot | 成员 2 | D1 |
| 损失 | hard-neg mask、K>1、KD 温度、λ=0 等价基线；单测覆盖 | 无 | 成员 3 | 已完成 |
| 教师缓存 | `distill.py` + `teacher.py`：特征缓存、按下标取 B×B、长度对齐检查 | 若要把 COCO 20k 混入教师缓存，需先合并成同一个 JSONL | 成员 3/2 | D2（可选） |
| 评测 | 五基准 loader、多 caption 协议、tie-aware 名次、特征缓存、单测 | ARO/SugarCrepe 的 golden 文件、COCO annotations、D5 汇总 `master.csv` | 成员 3/1 | D1–D5 |
| 硬负例 | 规则/LLM/过滤三个脚本可跑 | 输出目录、seed、LLM `--limit`/并发/断点/词集校验；对抗过滤改为 image–negative + 分位阈值 | 成员 2 | D1–D2 |
| 配置 | `e1/e2/e2a/e3/e4b/e4c` 已对齐 trainer；E2r 可选 | E2r 需要单独生成匹配子集 | 成员 2 | D2 |
| Demo | Gradio 骨架已有 | 特征缓存、精选图、预置 query | 成员 3 | D5 |
| 测试 | `tests/` 覆盖 loader / metrics / collate / loss / scheduler / teacher / lora | 每次改代码后跑 `pytest -q tests` | 成员 3 | 持续 |
| 文档 | 本版 v3.0 已重写 | 按实际结果回填数字与状态 | 成员 1 | 持续 |

**D1–D2 必须完成的最小闭环**：

1. 硬负例三脚本补齐：输出目录、规则法 seed、LLM `--limit`/词集校验、对抗过滤 image–negative + 分位阈值；E2a 至少跑通一次 smoke。
2. `prepare_flickr30k.py` 产出 train/val/test JSONL；`prepare_pairwise_bench.py` 产出 ARO/SugarCrepe golden；COCO annotations 就位；E0 五基准至少各跑一次 `--limit` 冒烟。
3. E3 教师缓存先跑 Flickr30k 版 smoke；如果要主张“抗 COCO 遗忘”，D2 前把 COCO 20k 合并进同一个 JSONL 再缓存。
4. 训练加 seed，跑 100-step 吞吐 pilot；确认 `configs/*.yaml` 与 `src/train.py` 字段一致。
5. `pytest -q tests` 全绿；demo 至少能用缓存特征跑起来。

---

## 10. Definition of Done

- [ ] clean clone 后按 README 能完成：数据准备 → E0 → E1/E2/E2a → E3/E4b → 评测 → demo。
- [ ] 所有 config 与代码字段一致；没有“写了但没生效”的超参。
- [ ] `results/master.csv` 包含 Tier 1 全部组 + Tier 2 的 E0/E2/E3/E4b。
- [ ] 未命中样本、golden 文件、特征缓存可复现。
- [ ] 报告 + 答辩 PPT + 模型卡 + 数据声明齐全。
- [ ] 明确写出局限性；不把负结果包装成正结果。
- [ ] E5 不进入主线；E4c LoRA 代码保留但标注“可选/附录”，不承诺结果。

---

## 11. 未来工作（报告一小节）

- E5：`ViT-B-16-SigLIP` 或冻结/LoRA 版 SO400M，补 sigmoid 版 hard-neg loss。
- LoRA：`src/lora.py` 已实现 q/v 行注入与 merge 导出；未来扩展到更多模块、更多层和更大模型。
- CC3M / 更大通用数据；多 seed；更大模型。
- 中文组合推理基准。
- Mnemo 接入：把 E3/E-best 封装为 `ImageTextEncoder` 服务，作为图像记忆检索的编码内核。

---

*文档版本 v3.0 · 2026-11-09 · 简化版：主线保留 E0/E1/E2/E2a/E3/E4b，E4c LoRA 降为可选；砍掉 E5、0-D、完整 0-A、CC3M 和全量五基准矩阵；重写分工、排期与 Go/No-Go。*
