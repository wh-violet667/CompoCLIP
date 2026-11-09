# 多模态图文检索课程项目 · 开发文档

> **题目**：Compositionally-Aware Image Retrieval: Diagnosing and Fixing Contrastive Vision-Language Models
> **团队**：3 人 | **周期**：约 6 周 | **算力基线**：单卡 T4 16GB（Turing 架构：仅 fp16，**不支持 bf16**，无 FlashAttention；若为 Kaggle T4 则每周 30h 配额，需排期管理）

---

## 0. 项目总览

### 0.1 一句话定位

诊断对比式视觉-语言模型（CLIP/SigLIP）的组合推理盲区，围绕**硬负例构造、损失改造、抗遗忘、微调策略**四个模型层维度做系统性实验，产出组合感知增强的检索模型与完整的消融证据链。

### 0.2 创新点（本项目的学术卖点）

| # | 创新点 | 层面 | 对标工作 | 我们的差异 |
|---|---|---|---|---|
| 1 | **硬负例来源的系统对比**：规则生成 vs LLM 生成 vs CLIP 盲区对抗过滤，三条来源在同一训练配置下公平对比 | 数据/方法 | NegCLIP（仅规则）、SCRAMBLe（仅 LLM） | 首次在同基准下横向对比负例来源，回答"负例质量 vs 数量哪个重要" |
| 2 | **硬负例加权 InfoNCE + 教师蒸馏抗遗忘**：微调时以原始 CLIP 的相似度分布为教师做 KL 正则，显式控制"组合增益 vs 通用能力遗忘"的 trade-off | 损失 | NegCLIP（无抗遗忘机制，下游指标有退化记录） | 把防遗忘从"事后补救（混数据）"变为"损失内约束"，并给出蒸馏权重消融 |
| 3 | **微调策略矩阵**：全量 / 锁图像塔 / LoRA 三策略在组合增益与遗忘代价上的帕累托分析 | 训练策略 | CLIC（单一高效微调方案） | 把"高效微调"从单点结论扩展为可选型表 |
| 4 | **基座损失形态对照**：softmax-CLIP vs sigmoid-SigLIP 在相同硬负例协议下的响应差异 | 模型对比 | 现有组合推理工作几乎全在 softmax-CLIP 上做 | SigLIP 成对 sigmoid 损失不依赖批内负例归一化，理论上硬负例梯度形态不同——验证之 |

创新点 1、2 为主线（必须完成），3 为标准消融（应该完成），4 为时间允许时的扩展（可选）。Mnemo 融合仅作为报告展望一小节，不占开发资源。

### 0.3 叙事主线（报告与答辩共用）

```
原理分析（温度/batch 消融 + 表示空间探查）
   → 基线诊断（CLIP/SigLIP 在组合推理基准上接近随机）
   → 失败归因（词袋假设的探针实验证据）
   → 修复（硬负例三来源 × 加权损失 × 蒸馏抗遗忘）
   → 量化（五基准评测矩阵 + 全消融）
   → 收尾（最小 demo + 应用展望）
```

### 0.4 交付物清单

| 交付物 | 内容 | 负责人 |
|---|---|---|
| 原理分析实验包 | 温度/batch 消融 + 表示空间探查 | 成员 1 |
| 诊断报告 | 基线评测 + 词序探针 + 误差分类体系 | 成员 1 |
| 硬负例数据集 | 规则版 / LLM 版 / 对抗过滤版三份 + 质检报告 | 成员 2 |
| 微调模型组 | E1/E2/E3（负例来源 × 蒸馏） checkpoints + 训练日志 | 成员 2 |
| 微调策略与基座实验 | E4（PEFT 矩阵）/ E5（SigLIP 对照） | 成员 3（你） |
| 评测框架与结果库 | 五基准评测器 + 全部结果 CSV + 未命中分析 | 成员 3（你） |
| 最小 demo | Gradio 检索界面（含微调前后对照） | 成员 3（你） |
| 课程报告 | 见 §7 大纲 | 全员 |

---

## 1. Phase 0：前期算法层工作（第 1 周）

> 目标：基于 open_clip 现成框架快速拿到"对比学习为什么 work、CLIP 在哪里不 work"的实验证据，支撑报告 §3–4。**原则：全部复用现成组件（open_clip / HF transformers / CLIP_benchmark），不做任何从零手写实现**，省下的时间压到模型实验上。

### 1.1 实验 0-A：温度系数与负样本池消融（成员 1，约 1 天）

直接插桩 open_clip 内置 `ClipLoss`，在 Flickr8k 上短程训练（3–5 epochs 出趋势）：

- **温度 τ 消融**：固定/可学习 τ ∈ {0.01, 0.07, 0.5}，记录收敛曲线与 R@1；
- **batch 消融**：batch ∈ {64, 256} 对比，验证"对比学习依赖大负样本池"，解释 CLIP 32768 batch 的设计动机；
- **附加对照（为创新点 4 铺垫）**：同一配置切换到 `SigLipLoss` 跑一组，观察 sigmoid 损失在小 batch 下的稳定性差异。

**产出**：`notebooks/01_temperature_batch_ablation.ipynb`。

### 1.2 实验 0-B：Embedding 空间结构探查（成员 1，约 2 天）

用预训练 CLIP ViT-B/32（不训练）分析表示空间：

1. **Modality Gap 复现**：Flickr30k 测试集图文 embedding 均值的余弦距离；真实配对 vs 随机配对相似度分布直方图；
2. **判别区间宽度**：正样本相似度均值/方差 vs 随机对，作为微调前后对比的基线；
3. **t-SNE/UMAP 语义簇可视化**。

**产出**：`notebooks/02_embedding_space_probe.ipynb`。

### 1.3 实验 0-C：词袋假设探针——失败模式的最小复现（成员 1 + 成员 2，约 2 天）

整个项目最关键的归因实验：

1. **词序扰动探针**：Flickr30k caption 四种扰动（随机打乱 / 主宾交换 / 属性交换 / 动词替换），计算扰动前后文本 embedding 相似度与图文相似度变化 Δs。预期：随机打乱后相似度仍 >0.9，主宾交换 Δs 却很小——词袋证据与盲区证据同时落袋；
2. **Winoground 零样本基线**：400 对样本，Text / Image / Group Score 三项，预期 Group Score ≈ 10–16%；
3. **误差分类**：人工审阅 50 个失败样本，按官方标签归类成表。

**产出**：`notebooks/03_word_order_probe.ipynb`、误差分类表（报告 §4 核心图表）。

### 1.4 实验 0-D：小规模端到端训练（pipeline 验证，成员 2，约 1.5 天）

Flickr8k 上用 open_clip 从头训练 ViT-B/32（8–10 epochs，T4 上约 1–1.5h），目的：

- 正式微调前验证数据管线与 fp16 训练稳定性（T4 无 bf16，重点观察 logit scale 是否发散）；
- 从零训练 vs 预训练初始化的对照曲线，定量回答"为什么不从头训 CLIP"；
- 顺便验证 `SigLipLoss` 管线（服务创新点 4）。

**产出**：训练日志与对照曲线；模型不交付。

### 1.5 Phase 0 出口标准

- [ ] 温度/batch/sigLIP 对照各有一张可用图表
- [ ] Winoground 零样本三项指标落表，与论文报告值偏差 < 5pp
- [ ] 词序扰动探针完成，词袋证据量化数字落袋
- [ ] Flickr8k 训练管线跑通

---

## 2. 数据集

### 2.1 数据集总表

| 数据集 | 规模 | 用途 | 获取方式 | 许可注意 |
|---|---|---|---|---|
| Flickr8k | 8k 图 / 40k caption | Phase 0 训练 | Kaggle 搜 `adityajn105/flickr8k` | 研究用途 |
| Flickr30k | 31.8k 图 / 158k caption | 检索主评测 + 微调训练集 | Kaggle `nunenuh/flickr30k` 或 HF `nlphuji/flickr30k` | 研究用途 |
| MS-COCO 2017 val | 5k 图 / 25k caption | 域外泛化与遗忘度量 | Kaggle `awsaf49/coco-2017-dataset` | CC-BY 4.0 |
| Winoground | 400 对图文 | 组合推理诊断与验证 | HF `facebook/winoground` | **仅作评测** |
| ARO（VL-Checklist 子集） | ~5k 组合样本 | 组合推理消融 | `github.com/ARO-benchmark/ARO` | 研究用途 |
| SugarCrepe++ | ~7.5k 组合样本 | 组合推理补充基准 | CLIC 项目仓库 | 研究用途 |
| CC3M 子集 | ~50k 采样 | 蒸馏/混合正则化的通用数据 | HF `pixparse/cc3m-wds` | 部分链接失效，需校验 |

### 2.2 数据组织约定

```text
data/
├── flickr8k/
├── flickr30k/
│   ├── images/
│   └── splits/{train,val,test}.json   # Karpathy split，与文献可比
├── coco2017/val2017/
├── winoground/        # HF datasets 加载，不落盘图片
├── aro/
├── hardneg/           # 核心产物：三来源硬负例（见 §3.2）
│   ├── rule_based.jsonl
│   ├── llm_generated.jsonl
│   └── adversarial_filtered.jsonl
└── cc3m_subset/       # 蒸馏/混合用
```

**硬性约束**：

- Flickr30k 必须用 **Karpathy split**（29k/1k/1k），否则 R@k 无法与文献对比；
- Winoground 严禁泄漏进训练数据（学术诚信，报告需声明）；
- 数据路径走环境变量 `DATA_ROOT`，脚本不硬编码。

---

## 3. 模型层实验设计（本项目核心）

### 3.1 实验矩阵总览（按 T4 单卡核算，全程约 8–10 组训练 ≈ 15–20 GPU 小时）

| 实验组 | 基座 | 负例 | 损失 | 训练策略 | 目的 | 新增训练？ |
|---|---|---|---|---|---|---|
| **E0** | CLIP ViT-B/32 | — | — | 不微调 | 对照 | 否（仅推理） |
| **E1** | 同上 | 规则法 | 硬负例加权 InfoNCE | 全量 | 负例来源对比① | 是 |
| **E2** | 同上 | LLM 法 | 同上 | 全量 | 负例来源对比② | 是 |
| **E2a** | 同上 | LLM 法 + 对抗过滤 | 同上 | 全量 | 负例来源对比③（质量 vs 数量） | 是 |
| **E3** | 同上 | 最佳来源 | + 教师蒸馏 λ=0.5 | 全量 | 创新点 2：抗遗忘 | 是（λ=0 即 E-best，复用） |
| **E4b/c** | 同上 | 最佳来源 | 最佳配置 | 锁塔 / LoRA | 创新点 3：策略矩阵 | 是（E4a 全量 = E3/E-best checkpoint，**不新增训练**） |
| **E5**（可选，**默认砍**） | SigLIP SO400M-B/16 | 最佳来源 | sigmoid 版硬负例 | 全量 | 创新点 4：损失形态对照 | 是（B/16 图像 token 数 ×4，T4 上约 4–5h，仅 W5 有余量时做） |

执行顺序：E1/E2/E2a → 定最佳负例来源 → E3 → E4b/c →（余量足够才）E5。每步结论依赖上一步，不允许并行乱序。

**T4 预算纪律**：

- 任何实验启动前先跑 100 step 实测吞吐，按实测推算总时长，超 3h 的配置必须砍 epoch 或缩 batch+累积；
- λ 消融固定为 {0, 0.5} 两组（λ=0 复用已有结果），`hard_negative_weight` 固定 1.0（仅时间富余时补 2.0 一组作附录）；
- 所有训练共用同一数据缓存与 teacher logits 缓存，避免重复前向。

### 3.2 硬负例生成 pipeline（创新点 1）

**规则法（NegCLIP 路线）**：spaCy 词性标注 + 依存解析，模板化交换主宾/名词/修饰关系；启发式过滤 + 100 条人工抽检。

**LLM 法（SCRAMBLe 轻量版）**：few-shot prompt 要求"用完全相同的词表达不同含义"，低成本模型批量生成，100 条人工质检记录合格率。

**对抗过滤（本项目特有步骤）**：用当前 CLIP 给全部生成负例打分，分层取样——
- 高相似桶（>0.85）：模型盲区，最有训练价值；
- 中相似桶（0.6–0.85）；
- 低相似桶（<0.6）：接近批内随机负例，价值低。
E2a 只用高相似桶（数量约为 E2 的 30–40%），与 E2 对比直接回答"质量 vs 数量"。

### 3.3 损失改造（创新点 2，模型层核心工作）

改造 open_clip `ClipLoss`（约 50 行）：

**① 显式硬负例加权**：正 caption 的硬负例编码后拼入文本侧分母，权重 `w_hn` 可调：

```text
L_i2t = -log( exp(s_ii/τ) / ( Σ_j exp(s_ij/τ) + w_hn · Σ_k∈HN(i) exp(s_ik/τ) ) )
```

`w_hn` 固定 1.0（T4 预算下不做消融；时间富余时补 2.0 一组进附录）。

**② 教师蒸馏抗遗忘**：预计算 E0（原始 CLIP）在 Flickr30k train + CC3M 子集上的完整相似度分布 `P_teacher`（一次性前向缓存到磁盘，训练时零显存开销）；微调时加 KL 正则：

```text
L = L_hardneg + λ · KL( P_student ‖ P_teacher )     λ ∈ {0, 0.5}（λ=1.0 仅作附录补跑）
```

蒸馏项把"通用检索能力不退化"从工程技巧（混数据、早停）变成**损失层面的显式约束**，λ 消融给出"组合增益—遗忘代价"对比——这是报告 §6 最有分量的一张图。

### 3.4 微调策略矩阵（创新点 3）

| 策略 | 可训练参数 | T4 上单组时长（估） | 预期特点 |
|---|---|---|---|
| 全量（= E3/E-best，复用不另训） | 100% | ~2h | 增益上限高，遗忘风险大 |
| 锁图像塔（只训文本塔） | ~37% | ~1.2h | 图像索引可复用；反向传播只走文本塔，显存省约 40% |
| LoRA（q/v 投影，r=8，`peft`） | ~0.3% | ~2h（前向仍全量） | 遗忘最小，checkpoint 仅几 MB，增益可能受限 |

三者跑同一最佳数据配置，在五个基准 + COCO 遗忘指标上做帕累托分析（增益 × 遗忘 × 训练成本三维）。锁塔/LoRA 两组即可构成完整矩阵，全量组零额外成本。

### 3.5 训练配置（T4 适配后的起点）

```yaml
# configs/finetune_e2.yaml
model: ViT-B-32
pretrained: laion2b_s34b_b79k
train_data: flickr30k train + hardneg/<variant>.jsonl
batch_size: 128            # T4 16GB 实测上限附近；OOM 则降 96
accum_freq: 2              # 等效 batch 256
epochs: 4                  # 短程微调；先 100 step 实测吞吐再最终敲定
lr: 1e-5
warmup: 200
wd: 0.1
lr_scheduler: cosine
precision: amp             # T4 无 bf16：fp16 + GradScaler
grad_checkpointing: false  # 仅 ViT-B/16（E5）时开启
logit_scale_clamp: 4.605   # ln(100)，防 fp16 下 logit scale 发散
hard_negative_weight: 1.0
distill_lambda: 0.0        # E3 改 0.5
distill_teacher_cache: caches/teacher_logits_f30k_cc3m.pt
save_most_recent: true
```

### 3.6 训练监控（T4 专项）

- 每 epoch：Flickr30k val R@1（拟合）+ COCO 抽 1k R@1（遗忘，下降 >2pp 触发干预）；
- **fp16 稳定性**：盯 logit scale 曲线与 GradScaler 的 scale 值，若 scale 持续下降（频繁溢出）则降 lr 至 5e-6 或对该组改用 `precision: fp32`（速度减半，仅作保底）；
- W&B 或本地 CSV，实验名与 `configs/` yaml 一一对应；
- GPU 排队：单卡串行，训练启动顺序严格按 §3.1 依赖链，禁止同时挂两组。

---

## 4. 评测方案（成员 3 负责）

### 4.1 评测器结构

```text
evals-mm/
├── golden/
│   ├── flickr30k_test.jsonl
│   ├── winoground.jsonl
│   └── aro_sugarcrepe 适配器
├── encoders.py          # 可插拔 encoder 协议（E0–E5 统一接口）
├── retrieval.py         # FAISS-FlatIP 召回
├── metrics.py           # R@1/5/10、MedR、Winoground 三项、ARO 分组、遗忘指标
├── run_eval.py          # --encoder {e0,e1,...} --benchmark {...}
└── results/             # CSV + 未命中样本（不提交 git）
```

### 4.2 评测矩阵

| 基准 | 指标 | 评测对象 |
|---|---|---|
| Flickr30k test 1k（双向） | R@1/5/10、MedR | 全部实验组 |
| MS-COCO val 5k（域外） | 同上 | 全部实验组 |
| Winoground | Text/Image/Group Score | 全部实验组 |
| ARO（关系/属性分开报） | 准确率 | 全部实验组 |
| SugarCrepe++ | 准确率 | 全部实验组 |
| 词序扰动探针复跑 | Δs 变化 | E0 vs E2 vs E3 |

### 4.3 评测纪律

- 同批次数字才可横向比较；
- 每组必出未命中样本清单，人工抽查 ≥30 条归因；
- COCO 域外下降 >2pp 必须如实呈现并归因，不许只挑有利数字；
- Winoground 仅 400 对，提升 <5pp 时必须在报告中讨论统计功效，以 ARO/SugarCrepe++ 为主要证据。

---

## 5. Demo（最简化，第 5 周用 1–2 天完成）

Gradio 单页：query 输入 + E0/E2/E3 模型切换 + top-10 结果网格。亮点功能**「盲区对照」**：组合 query 下并排展示微调前后 top-5。**不做用户系统、不做数据库、不做 Mnemo 对接**——应用价值在报告 §8 用一段文字 + 一张架构设想图交代即可。

---

## 6. 三人分工与里程碑（重排）

### 6.1 分工原则

每位成员都背负"一类模型实验 + 一块报告章节"，系统工作（评测器、demo）压缩到成员 3 的约 40% 精力，其剩余精力投入 E4/E5 模型实验。

### 6.2 分工矩阵

| 周次 | 成员 1（诊断与归因） | 成员 2（负例数据与主训练） | 成员 3/你（评测 + PEFT/基座实验） |
|---|---|---|---|
| W1 | 实验 0-A、0-B、0-C | 环境搭建、0-D、规则负例生成器 | 数据集整理、FAISS 基线、evals-mm 骨架、E0 基准跑通 |
| W2 | 误差分类、报告 §3–4 初稿 | LLM 负例生成 + 对抗过滤分桶 | 教师 logits 预计算管线、Winoground/ARO 接入评测器 |
| W3 | 报告 §4 图表定稿 | **E1/E2/E2a 训练**（创新点 1，约 5–6 GPU 时） | E1/E2/E2a 全基准评测、未命中分析；LoRA 管线调通 |
| W4 | 探针复跑（微调后） | **E3 蒸馏 λ=0.5**（创新点 2，约 2 GPU 时） | **E4b/c 锁塔+LoRA**（创新点 3，约 3 GPU 时） |
| W5 | 报告 §3–4 定稿 | 报告 §5 初稿、模型卡片 | （余量够才）**E5 SigLIP**、Gradio demo（1–2 天） |
| W6 | 报告整合 | 报告 §5 定稿 | 报告 §6 定稿、答辩 PPT（盲区对照素材） |

### 6.3 风险与预案

| 风险 | 概率 | 预案 |
|---|---|---|
| fp16 训练不稳（T4 无 bf16，logit scale 发散/梯度溢出） | 中 | logit_scale_clamp=4.605 + GradScaler 监控；仍不稳则该组降 lr 至 5e-6，保底 fp32 |
| T4 单卡 16GB OOM（batch 128 仍溢出） | 中 | 降 batch 96 + accum 3；E5（B/16）强制开 grad checkpointing |
| Kaggle 每周 30h 配额不够 | 中 | 按 §3.1 核算全程 15–20h 足够；超额时砍序：E5 → λ=1.0 附录组 → w_hn 附录组 |
| 微调后 COCO 遗忘明显 | 中 | 这正是 E3 存在的意义——作为实验结果呈现，不是 bug |
| LLM 负例质量差 | 中 | 提高过滤阈值；退化到规则法为主线，LLM 法降级为附录 |
| Winoground 提升不显著（400 对，统计功效低） | 中 | 以 ARO + SugarCrepe++ 为主要证据，Winoground 作辅助并如实讨论 |

---

## 7. 课程报告大纲（8–12 页）

```text
1. 引言与问题定义
   1.1 跨模态检索任务定义（双向）
   1.2 组合推理为何是试金石
2. 相关工作
   2.1 对比视觉-语言预训练（CLIP/SigLIP/BLIP 谱系，softmax vs sigmoid 损失）
   2.2 组合推理基准（Winoground/ARO/SugarCrepe）
   2.3 硬负例微调方法（NegCLIP/CLIC/CounterCurate/SCRAMBLe）
3. 对比预训练机理分析【实验 0-A、0-B、0-D 支撑】
   3.1 InfoNCE 目标的形式化与作用分析
   3.2 温度与负样本池消融（基于 open_clip 复现）
   3.3 表示空间几何：modality gap 观测
   3.4 从头训练为何不可行：Flickr8k 对照
4. 失败模式诊断【实验 0-C 支撑】
   4.1 Winoground 零样本：接近随机的组合推理
   4.2 词序扰动探针：词袋假设量化证据
   4.3 误差分类体系与典型案例
5. 方法
   5.1 硬负例加权 InfoNCE 的损失改造
   5.2 三来源硬负例构造与对抗过滤分桶【创新点 1】
   5.3 教师蒸馏抗遗忘【创新点 2】
   5.4 微调策略：全量 / 锁塔 / LoRA【创新点 3】
6. 实验
   6.1 主结果：实验组 × 五基准
   6.2 负例来源对比：质量 vs 数量
   6.3 蒸馏权重消融：组合增益—遗忘代价曲线
   6.4 策略矩阵帕累托分析
   6.5（可选）SigLIP 对照：损失形态的影响【创新点 4】
   6.6 未命中样本分析
7. 检索 demo 与盲区对照演示（半页）
8. 局限与展望
   数据规模、Winoground 统计功效、中文组合基准缺失；
   应用展望：作为多模态记忆检索系统（Mnemo）的 encoder 内核（一段）
参考文献 + 附录（超参表、prompt 模板、数据使用声明）
```

---

## 8. 仓库结构

```text
compositional-retrieval/
├── notebooks/              # Phase 0 分析实验
├── configs/                # 每组实验一个 yaml
├── caches/                 # 教师 logits 缓存（git 不跟踪）
├── src/
│   ├── loss.py             # 硬负例加权 + 蒸馏 KL
│   ├── hardneg/
│   │   ├── rule_based.py
│   │   ├── llm_gen.py
│   │   └── adversarial_filter.py
│   ├── distill.py          # 教师 logits 预计算
│   └── train.py            # open_clip 微调入口
├── evals-mm/               # §4 评测器
├── demo/                   # Gradio 最小 demo
├── data/                   # git 不跟踪
└── docs/
    ├── dev-doc.md          # 本文档
    └── report/
```

---

*文档版本 v2.1 · 2026-11-09 · v2.0 重构：创新点收敛为四条模型层主线（负例来源对比 / 蒸馏抗遗忘 / 策略矩阵 / 基座对照），系统与后端工作降级为最小 demo + 展望；v2.1 按单卡 T4 16GB 重新核算：全部训练改 fp16、batch 128+累积、λ 消融砍为两组、E4a 复用 checkpoint、E5 默认砍、新增 fp16 稳定性与 OOM 预案*
