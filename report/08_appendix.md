# 8 附录

## A.1 复现命令

```bash
# 环境
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 数据（公开镜像，D0 临时代用；正式版换 Kaggle/官方 gated）
python scripts/prepare_flickr30k_wds.py
python scripts/prepare_winoground.py
python scripts/prepare_sugarcrepepp.py
python scripts/prepare_aro.py --include-flickr-order

# E0
python evals-mm/run_eval.py --tag e0 --benchmark winoground
python evals-mm/run_eval.py --tag e0 --benchmark aro
python evals-mm/run_eval.py --tag e0 --benchmark sugarcrepe

# Role-3 GPU：A2 4/3（E0/E3/E4b）与检索（E1–E4b）
python scripts/evaluate_role3_models.py --tags e0 e3 e4b \
  --benchmarks aro sugarcrepe coco --device cuda --image-batch 64 --text-batch 256
for tag in e1 e2 e2a e3 e4b; do
  python evals-mm/run_eval.py --tag "$tag" \
    --checkpoint "A2_models/$tag/${tag}_model.pt" --benchmark flickr30k
  python evals-mm/run_eval.py --tag "$tag" \
    --checkpoint "A2_models/$tag/${tag}_model.pt" --benchmark winoground
done
python scripts/assemble_role3_results.py
python scripts/build_full_4of3.py
python scripts/build_full_master.py
python scripts/paired_full_4of3.py
python scripts/make_failure_cases_a2.py
python scripts/make_master_md.py
python scripts/make_results_table.py

# 机制与诊断
python scripts/run_0b_probe.py --limit 5000
python scripts/run_0a_probe.py --benchmark winoground
python scripts/diagnose_failures.py

# 报告与图
python scripts/collect_results.py
python scripts/make_figures.py
python scripts/build_report.py
python scripts/build_ppt.py
```

## A.2 数据与许可

| 数据 | 来源 | 许可/状态 | 本地路径 |
|---|---|---|---|
| Flickr30k test（镜像） | `clip-benchmark/wds_flickr30k` | 研究用途；原数据需同意 Flickr 条款 | `data/flickr30k/` |
| Winoground（镜像） | `mm-eval/Winoground` | 研究用途；官方 gated | `data/winoground/` |
| ARO VG-Relation / Attribution | `mertyg/vision-language-models-are-bows` + gdown | MIT 代码；VG 图片 CC BY 4.0 | `data/aro/` |
| SugarCrepe++ | `Mayfull/sugarcrepepp_vlms` | CC BY 4.0（数据集卡） | `data/sugarcrepepp/` |
| 模型权重 | OpenCLIP `ViT-B-32/laion2b_s34b_b79k` | MIT（权重卡） | HF cache |

> 详细校验值、下载日期、负责人见 `docs/sources.md`；数据不进 git（`.gitignore` 已覆盖）。

## A.3 超参数与评测协议

| 项 | 值 |
|---|---|
| 图像预处理 | OpenCLIP 默认（Resize 224 / CenterCrop / Normalize） |
| 文本长度 | tokenizer 默认截断 77 |
| 相似度 | L2 归一化后的点积（= 余弦） |
| 二选一判定 | `sim(img, pos) > sim(img, neg)`，平局算错 |
| bootstrap | 1000 次，按样本重采样，95% 百分位 CI |
| Winoground | text/image/group 三项；group = 两个方向都正确 |
| ARO 宏平均 | 第 4 章 0-C 用 3 子集（VG-Rel / VG-Att / Flickr-Order）；第 6 章 A2 4/3 用 4 子集（含 COCO-Order） |
| SugarCrepe++ 宏平均 | 第 4 章 0-C 用 replace/swap；第 6 章 A2 4/3 用 replace/swap/add |
| 0-B | 5 类保持词袋变换；每类单独 n 与 CI |
| 0-A | 逻辑回归，5 折 GroupKFold，C=1.0，max_iter=800 |

## A.4 0-B 变换示例

| 变换 | 原句 | 负例 |
|---|---|---|
| 主宾交换 | a dog chases a cat in the park | a cat chases a dog in the park |
| 名词交换 | a man with glasses is wearing an orange hat | a hat with glasses is wearing an orange man |
| 形容词交换 | a red car next to a blue house | a blue car next to a red house |
| 关系反转 | the door is to the left of the shirt | the shirt to the left of the door |
| 全词打乱 | a man is riding a bike | bike a riding is man a |

> 所有负例在写入前用 `Counter(words)` 校验词袋完全一致；不一致的样本直接丢弃并计数。名词/形容词交换可能产生不自然句，这正是 0-A 要控制“流畅度捷径”的原因。

## A.5 0-A 特征与标签

- 每个 Winoground pair 构造两条样本：正确绑定（标签 1）与对调绑定（标签 0）。
- 特征：`[g0, g1, t0, t1, g0*t0, g0*t1, g1*t0, g1*t1, |g0-g1|, |t0-t1|]`，维度 512×10=5120。
- 分组：按 `group_id` 分 5 折，保证同一对的正反例不跨折。
- 零样本对照：`s00>s01 and s11>s10` 判 text 正确；`s00>s10 and s11>s01` 判 image 正确；两者皆真为 group 正确。

## A.6 分工与贡献

| 成员 | 主责 | 本项目实际交付 |
|---|---|---|
| 成员 1 | 诊断（0-A/0-B/0-C）+ E0 Winoground/ARO/SugarCrepe + 适配器 + 报告 1–4 章 + 统稿 | 见 `docs/decision-log.md`、`report/README.md` |
| 成员 2 | 数据转换 + 硬负例 + E0 Flickr30k/COCO + E1/E2/E2a/E3/E4b 训练 + 报告 5 章 + 模型卡 | 已交付 A2 handoff（E1–E4b 权重、负例、E0 Flickr/COCO）；第 5 章初稿 |
| 成员 3 | 评测 harness + master.csv + E1–E4b 评测 + E4b + demo + 报告 6–7 章 | 已交付：E0–E4b A2 4/3 + Flickr30k/Winoground、paired stats、failure cases、06–07 章、demo |

## A.7 已知限制

1. **数据口径**：Flickr30k/Winoground 用公开镜像，未走官方 Kaggle/gated 流程；数字不能与论文表格逐位比较。
2. **ARO 口径分两套**：第 4 章 0-C 仍按 3 子集（缺 COCO-Order）；第 6 章 A2 4/3 已补 COCO-Order，使用完整 4 子集。
3. **SugarCrepe++ 口径分两套**：第 4 章 0-C 用 replace/swap；第 6 章 A2 4/3 已补 add，使用完整 3 家族。
4. **0-A 是推理期证据**：无 CUDA，未做训练动力学；“信息不在”限定为“线性不可解码”，不排除非线性读出。
5. **LLM 硬负例未生成全量 15k**：无 API key；A2 handoff 的 E2 使用已交付的 1,739 条负例，产线与质检已就绪。
6. **Winoground 方差**：400 对、5 折 CV，单折波动约 ±5pt；报告给 CI，不报单点最优。
7. **E2 缺 COCO**：预注册的“E3 的 COCO 遗忘比 E-best 更小”尚未验证；E1/E2/E2a 的 COCO 也未同批次评测。

## A.8 图表清单

| 图/表 | 文件 | 内容 |
|---|---|---|
| 图 3-1 | `figures/fig_0b.png` | 0-B 各变换准确率（含 50% 基线） |
| 图 4-1 | `figures/fig_e0.png` | E0 三个基准的准确率 |
| 图 4-2 | `figures/fig_0c.png` | 0-C 各类型错误率 |
| 图 4-3 | `figures/fig_margin.png` | SugarCrepe++ margin 分布（近失手） |
| 表 3-1 | 正文 | 0-B 准确率、CI、margin |
| 表 4-1/2/3 | 正文 | Winoground / ARO / SugarCrepe++ E0 |
| 表 4-4/5 | 正文 | 0-C 分类与跨基准对比 |
| 表 6-1 | `results/full_4of3.md` | A2 4/3 完整 6 模型结果 |
| 表 6-2 | `results/paired_full_4of3.md` | A2 4/3 全量配对统计 |
| 表 6-3 | `results/failure_cases.md` | E3 的 ARO/Sugar 失败案例 |
| demo | `demo/README.md` | E0 vs E2 的 Gradio 盲区对照 |
