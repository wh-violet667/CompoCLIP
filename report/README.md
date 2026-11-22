# 报告统稿说明（成员 1）

## 文件与章节归属

| 文件 | 章节 | 负责人 | 状态 |
|---|---|---|---|
| `00_final_report.md` / `CompoCLIP_Report_总报告.docx` / `.pdf` | 项目总体报告（15 页） | 成员 3 汇总 | ✅ |
| `01_intro.md` | 第 1 章 引言 | 成员 1 | ✅ |
| `02_related_work.md` | 第 2 章 相关工作 | 成员 1 | ✅ |
| `03_mechanism.md` | 第 3 章 机制分析（0-A/0-B） | 成员 1 | ✅ |
| `04_diagnosis.md` | 第 4 章 E0 与失败诊断（0-C） | 成员 1 | ✅ |
| `05_data_training.md` | 第 5 章 数据与训练（E1/E2） | 成员 2/3 | 📝 初稿（C 整理 A2 材料） |
| `06_method.md` | 第 6 章 改进方法与结果（E3/E4） | 成员 3 | ✅ 正式结果；E0–E4b GPU 已补 |
| `07_conclusion.md` | 第 7 章 结论与展望 | 全体 | ✅ 定稿；含两套口径与预注册核对 |
| `08_appendix.md` | 附录（复现命令、来源、分工） | 成员 1 | ✅ |

## 统稿规则

1. **数字唯一来源**：E0 数字来自 `evals-mm/results/*.csv`；E1–E4b 的 4/3 数字来自 `A2_eval/results/*/evaluation.json`，由 `scripts/build_full_4of3.py` 汇总；共同 2/2 来自 `master_common.csv`，全量配对统计来自 `paired_full_4of3.json`。不要手抄。
2. **图表路径**：图片放 `report/figures/`，Markdown 用相对路径 `figures/xxx.png`；`build_report.py` 会以 15cm 宽插入。
3. **引用**：统一用 `[n]`，编号见 `docs/references.md`；新增文献追加编号，不重排。
4. **术语**：compositionality=组合性；hard negative=硬负例；adapter=适配器；modality gap=模态鸿沟。首次出现给英文。
5. **时态/人称**：中文报告用“本文/本项目”，不用“我们觉得”；实验结论必须带证据（表号/图号/样本数）。
6. **长度**：第 1–4 章合计约 8,000–10,000 字；第 5–7 章由负责人控制，全文不超过 20,000 字。
7. **禁止**：不得把 `--limit` 冒烟数字写成全量；不得把 `caption2` 当负例；不得引用未下载的数据集。

## 合并流程

```bash
# 1. 拉取队友分支/文件，确认 05/06/07 存在
# 2. 生成结果表与图
python scripts/collect_results.py
python scripts/make_figures.py
# 3. 合成 DOCX + PDF
python scripts/build_report.py
# 4. 检查
#    - PDF 目录页码连续
#    - 所有图表在正文被引用（“如图 x”“见表 y”）
#    - 所有 [n] 在 references.md 中存在
#    - 摘要/结论的数字与结果表一致
```

## 当前已知缺口

- E0/E3/E4b 的 ARO、Flickr30k/Winoground、COCO 已由 T4 正式补齐；结果见 `report/results/full_4of3.*` 与 `master.csv`。
- **E2 没有同批次 COCO**，预注册的“E3 的 COCO 遗忘比 E-best 更小”仍无法验证；E1/E2/E2a 的 COCO 也未在同一批次评测。
- E4b 的 Sugar 在 A2 4/3 与旧 `evals-mm` 2 家族之间差 0.439pp，主表统一采用 A2。
- 第 4 章 0-C 仍按 3/4 ARO 子集口径写；第 6 章的 A2 4/3 已补 COCO-Order。
- LLM 15k 硬负例未生成（无 API key），第 5 章 E2 结果需成员 2/3 补。
- Winoground/Flickr30k 用的是公开镜像，正式提交前需用官方数据复核一遍（差异写脚注）。
