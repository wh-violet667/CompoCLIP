# 成员 1 交付说明（Diagnostics / E0 / Report 1–4）

> 本文件是成员 1 的自述交付清单，答辩/验收时先看这里。详细决策见 `docs/decision-log.md`，复现命令见 `report/08_appendix.md`。

## 一句话

完成 0-A/0-B/0-C 三项诊断、E0 三个基准（Winoground / ARO 3 子集 / SugarCrepe++）的统一评测、报告第 1–4 章与统稿模板，并把 LLM 硬负例产线补到可续跑、可质检。

## 交付物地图

| 类别 | 文件 | 说明 |
|---|---|---|
| 0-B 词序探针 | `scripts/run_0b_probe.py` | 5 类保持词袋变换 + unrelated 对照；输出 `results/0b_probe.csv` |
| 0-A 线性探针 | `scripts/run_0a_probe.py` | Winoground 零样本 vs 线性探针；输出 `results/0a_winoground.json` |
| 0-C 失败诊断 | `scripts/diagnose_failures.py` | 读逐样本 npz，按 order/attr/rel/obj 分桶；输出 `results/0c_*` |
| 结果副本 | `report/results/` | E0/0-A/0-B/0-C 的 CSV/JSON/Markdown，便于评审直接核对 |
| E0 基线 | `evals-mm/run_eval.py`（改） | 图片/文本去重、宏平均、逐样本 npz、本地 Winoground |
| 数据准备 | `scripts/prepare_*.py` | Flickr30k/Winoground/SugarCrepe++/ARO → 统一 golden |
| 硬负例产线 | `src/hardneg/llm_gen.py`（改） | 并发/限流/续跑/词袋校验/质检报告；`--dry-run` 可测 |
| 质检表 | `scripts/sample_hardneg_100.py` | 100 条人工质检 CSV + Markdown |
| 报告 | `report/01–04、08` | 第 1–4 章 + 附录；`build_report.py` 出 DOCX/PDF |
| 统稿 | `report/README.md`、`docs/handoff.md` | 合并规则、队友接口、验收口径 |
| PPT | `report/CompoCLIP_成员1汇报.pptx` | 12 页，数字自动读结果 CSV |

## 核心数字（ViT-B-32 / laion2b）

| 基准 | 指标 | 数值 |
|---|---|---:|
| Winoground | text / image / group | 35.3% / 11.3% / 7.8% |
| ARO | VG-Rel / VG-Att / Flickr-Order / 宏平均 | 50.1% / 57.9% / 68.0% / 58.7% |
| SugarCrepe++ | replace / swap / macro | 82.7% / 66.0% / 74.3% |
| 0-B | 句法变换 / 全词打乱 / 无关句 | 56.8–60.5% / 76.9% / 99.5% |
| 0-A | Winoground group：零样本 → 线性探针 | 7.8% → 50.0% |
| 0-C | SugarCrepe relation/attr/obj 错误率 | 30.3% / 24.3% / 10.4% |

## 复现（最短路径）

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
# 数据准备（公开镜像；正式数据见 docs/handoff.md）
python scripts/prepare_flickr30k_wds.py
python scripts/prepare_winoground.py
python scripts/prepare_sugarcrepepp.py
python scripts/prepare_aro.py --include-flickr-order
# 评测
python evals-mm/run_eval.py --tag e0 --benchmark winoground
python evals-mm/run_eval.py --tag e0 --benchmark aro
python evals-mm/run_eval.py --tag e0 --benchmark sugarcrepe
python scripts/run_0b_probe.py --limit 5000
python scripts/run_0a_probe.py --benchmark winoground
python scripts/diagnose_failures.py
# 出图/报告/PPT
python scripts/make_figures.py
python scripts/build_report.py --member1
python scripts/build_ppt.py
```

## 已知限制（不要藏）

1. Flickr30k / Winoground 用公开镜像，正式提交前需官方数据复核。
2. ARO 缺 COCO-Order（需 COCO test2014 原图），当前宏平均是 3 子集口径。
3. SugarCrepe++ 镜像无 add 子类，macro 是 replace/swap 二类。
4. LLM 15k 硬负例未生成（无 API key）；产线与质检已就绪。
5. 0-A 是推理期线性探针，不是训练动力学；结论限定“线性不可解码”。

## 给评审的三个问题

1. 为什么 0-B 的 word_shuffle 反而比主宾交换高？——因为全词打乱破坏所有局部搭配，模型仍可靠名词匹配；主宾交换保留搭配只改角色，正好落在盲区。
2. 为什么 0-A 线性探针在 Winoground 只有 50%？——两句都自然、词袋相同，排除了流畅度捷径；冻结表征里没有可线性解码的绑定信息。
3. 为什么 ARO 宏平均不用微平均？——子集样本量差 1.5 倍，微平均会被 VG-Attribution 主导；宏平均让每个子集等权。
