# 决策日志（成员 1）

> 规则：每条决策写“背景 → 选择 → 代价/风险 → 复核方式”。只记会影响结果解释或他人复现的决策，不记日常琐事。

## D0-1 数据获取：公开镜像先行，Kaggle 原包复核

- 背景：D0 要求 Flickr30k / ARO / SugarCrepe++ / Winoground 全部到位；Kaggle 需要账号，Winoground 官方 HF 为 gated（匿名 401），COCO test2014 体积大。
- 选择：
  - Flickr30k 用 `clip-benchmark/wds_flickr30k` 的 test 分片（1000 图 × 5 caption，原图尺寸），重建为 `data/flickr30k/splits/test.jsonl`。
  - Winoground 用公开镜像 `mm-eval/Winoground`（400 对、800 图，含原图）还原为本地 `pairs.jsonl`。
  - ARO 用官方 gdown 标注 + VG 图片（`vg_relation` 23,937 对、`vg_attribution` 28,748 对）。
  - SugarCrepe++ 用 `Mayfull/sugarcrepepp_vlms`（4,757 条、1,560 张 COCO 图）。
- 代价：Flickr30k 镜像不是 Kaggle 原包；Winoground 镜像未走官方 gated 协议；两者都不能直接与论文表格逐位比较。
- 复核：成员 2 拿到 Kaggle/Winoground 权限后，用同一脚本重跑 `flickr30k` 与 `winoground`，对比 `evals-mm/results/*.csv` 的 mean/std；差异 >1pt 时以官方数据为准并在报告脚注说明。

## D0-2 SugarCrepe++ 的负例字段：不能用 caption2

- 背景：`scripts/prepare_pairwise_bench.py` 的 `NEG_KEYS` 把 `caption2` 当负例；但 SugarCrepe++ 的 `caption2` 是**正例同义改写**，`negative_caption` 才是扰动负例。
- 选择：新增 `scripts/prepare_sugarcrepepp.py`，固定 `positive=caption`、`negative=negative_caption`；另出 `sugarcrepe_paraphrase.jsonl`（positive=caption2）只做词法鲁棒性附录。
- 代价：主表与附录两套口径，报告必须写清楚主指标用哪套。
- 复核：抽 100 条人工核对，负例必须与图片语义不一致；若 `caption2` 被误用，准确率会异常接近 100%（同义句几乎不可分），这是最明显的告警信号。

## D0-3 评测器：图片/文本去重 + 宏平均

- 背景：ARO 71,679 对只对应 7,600 余张唯一图片；逐样本重复编码会让 CPU 评测从 10 分钟膨胀到 2 小时。ARO 四个子集样本量差异大，微平均会被 `vg_attribution` 主导。
- 选择：`eval_pairwise` 先按字符串去重再编码，再用索引还原；新增 `subset=macro`（各子集等权）。
- 代价：缓存文件按“去重后的唯一列表”存，换数据/换 limit 必须换 cache key；已在文件名带 `limit`。
- 复核：去重前后逐样本分数应完全一致（单元测试 `tests/test_eval_dedup.py`）。

## D0-4 Winoground 本地加载

- 背景：`loaders.load_winoground` 只走 HF `datasets`，匿名访问 gated 数据会失败；且原实现无法离线复现。
- 选择：优先读 `data/winoground/pairs.jsonl`（本地图片），不存在时回退 HF。
- 代价：本地图片格式为 PNG，体积大于原图；不影响 CLIP 预处理后的张量。
- 复核：400 对齐全、每对 2 图；`run_eval --benchmark winoground` 输出 `n_pairs=400`。

## D0-5 ARO COCO-Order 暂缺

- 背景：ARO 的 `COCO_Order` 需要 COCO **test2014** 原图（约 6GB），当前 D0 只有 Flickr30k 与 VG 图片；成员 2 的 COCO val2017 不能替代（文件名与 split 不同）。
- 选择：先报 `vg_relation`、`vg_attribution`、`flickr_order` 三个子集；`flickr_order` 用 ARO 官方 4 种扰动在 Flickr30k test 上重建（18,994 对）。COCO-Order 在结果表标 `blocked`，不伪造数字。
- 代价：ARO 宏平均目前是 3 子集口径，不是官方 4 子集。
- 复核：成员 2 提供 COCO test2014 后，跑 `prepare_aro.py --include-coco-order`（待实现）即可补齐；报告中明确写“3/4 子集”。

## D0-6 LLM 硬负例：没有 API key，不伪造 15k

- 背景：E2 需要 15,000 条 LLM 硬负例；当前环境无 `OPENAI_API_KEY`/`DASHSCOPE_API_KEY`，CPU 也无法本地跑 15k 生成。
- 选择：把 `src/hardneg/llm_gen.py` 补成可续跑、可并发、可限流的产线脚本，并加三道质检（词多重集、与 GT 重复、长度比）；用 mock 生成器跑通端到端测试，产出 `data/hardneg/llm_prompt_pack.jsonl`（前 500 条）供有 key 的成员直接开跑。**不生成假的 `llm_generated.jsonl`。**
- 代价：E2 主结果依赖成员 2/3 提供 key 或跑数；成员 1 只交付“可运行 + 质检 + 失败样例分析”。
- 复核：`python -m pytest tests/test_hardneg_pipeline.py`；有 key 后 `--limit 200` 冒烟，检查 `quality_report.json` 的 `word_multiset_pass_rate ≥ 0.98`。

## D0-7 0-B 探针：只改词序，不换词

- 背景：常见“组合性”探针会同时换词，无法区分“没学到关系”与“没学到词义”。
- 选择：5 类变换全部保持词多重集：主语/宾语交换、名词交换、形容词交换、全词打乱、三元组内打乱；另加 unrelated 对照校准任务难度。
- 代价：spaCy 依存分析失败时样本被丢弃，各 probe 的有效 n 不同；报告分别给 n 与 95% bootstrap CI。
- 复核：任意一条样本可用 `Counter(words)` 检查正负词袋完全一致；`0b_persample.jsonl` 保留原句与负例，可人工抽查。

## D0-8 0-A 降级为“线性可解码性”分析

- 背景：0-A 原计划看对比学习训练动力学，需要 GPU 与多轮训练；本机无 CUDA（Intel Iris Xe），全量训练不可行。
- 选择：用冻结 CLIP 特征做线性探针：输入 `[img, pos, neg, img*pos, img*neg, |pos-neg|]`，5 折交叉验证判断“哪句是正确描述”，与零样本余弦读数对比。若线性探针显著高于零样本，说明信息在表征里、读数方式不对，直接支撑“轻量适配器”的动机。
- 代价：不是训练动力学，不能回答“训练过程中何时学会组合”；报告限定为“推理期机制证据”。
- 复核：同一 split、同一 seed；报告同时给零样本与探针准确率，避免只报有利数字。

## 待决问题

1. Winoground 官方 gated 权限：成员 2 是否已有？有则用官方数据替换镜像结果。
2. LLM API：谁提供 key、预算多少？15k 条按 1 条 ≈ 120 tokens 估算，约 1.8M tokens。
3. COCO test2014：是否值得为 ARO 的 1/4 子集下载 6GB？若时间紧，接受 3 子集口径。
4. 主模型是否统一为 ViT-B-32/laion2b？若成员 3 的 E1 用 ViT-L/14，E0 需补一列同骨干对照。
