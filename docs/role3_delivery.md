# Role 3 final delivery

Status: **C 的评测、汇总、失败分析、报告 6–7 章、demo 与答辩材料已交付。**
E2/E1/E2a 的 COCO 未跑（按要求保留为未验证），没有填猜测值。

## 1. 可直接提交的文件

| 交付物 | 文件 |
|---|---|
| 报告第 6 章 | `report/06_method.md` |
| 报告第 7 章 | `report/07_conclusion.md` |
| 完整报告 DOCX/PDF | `report/CompoCLIP_Report.docx` / `.pdf`（27 页） |
| 项目总体报告 DOCX/PDF | `report/CompoCLIP_Report_总报告.docx` / `.pdf`（15 页） |
| 角色 3 答辩 PPT | `report/CompoCLIP_成员3汇报.pptx`（9 页） |
| 主结果 4/3 表 | `report/results/full_4of3.csv` / `.md` |
| 汇总长表 | `report/results/master.csv`（90 行） |
| 共同 2/2 表 | `report/results/master_common.csv` |
| 配对统计 | `report/results/paired_stats.*`、`paired_full_4of3.*` |
| 失败案例 | `report/results/failure_cases.md`（30 条） |
| 图 6-1 / 6-2 | `report/figures/fig_role3_4of3.png`、`fig_role3_delta.png` |
| demo | `demo/app.py` + `demo/README.md` |
| 结果索引 | `report/results/README.md` |
| 大文件/Release 说明 | `docs/artifacts.md` |
| 复现记录 | `docs/role3_remaining.md`、`docs/experiment-log.md` |

## 2. 主结论（可直接用于答辩）

- E0–E4b 都完成了 A2 4/3：ARO 4 子集 + SugarCrepe 3 家族 + COCO。
- **E3**：ARO 48.195%、Sugar 79.404%、组合 63.799%、COCO t2i 41.029%。
- **E0**：45.486% / 77.539% / 61.512% / 38.794%。
- **E3 − E0**：ARO +2.709pp、Sugar +1.865pp、组合 **+2.287pp**；满足预注册
  ≥2.0pp；COCO +2.235pp，不降反升。全量配对 CI 对 ARO/Sugar 都排除 0。
- **但收益集中在 order 子集**：COCO-Order 32.431→39.244，Flickr-Order
  40.680→44.740；VG-Relation/VG-Attribution 相对 E0 基本持平。
- **E3 vs E2**：组合 +1.231pp，但 Sugar −0.910pp（CI 跨 0）；E2 仍是 E-best。
- **E4b**：组合 62.810%，与 E2 接近；共同 2/2 和检索没有稳定优势。
- 共同 2/2：E3 vs E0 ARO −0.018pp（CI 跨 0）、Sugar +2.707pp（显著）。

## 3. 答辩时必须说清的口径

1. 共同 2/2 与 A2 4/3 是两套表，不能混算。
2. E0 有 A1 归档与 A2 4/3 两个版本；主表用 A2 版本。
3. E3 的预注册成功是在 **4/3 组合指标**上成立，不是“所有子集都提升”。
4. E2 没有同批次 COCO，因此“E3 的 COCO 遗忘比 E-best 更小”未验证。
5. E4b 的 Sugar 在 A2 4/3 与旧 `evals-mm` 2 家族之间差 0.439pp；主表用 A2。

## 4. 复现

```bash
python scripts/assemble_role3_results.py
python scripts/build_full_4of3.py
python scripts/build_full_master.py
python scripts/paired_full_4of3.py
python scripts/make_failure_cases_a2.py
python scripts/make_master_md.py
python scripts/make_role3_figures.py
python scripts/make_results_table.py
python scripts/build_report.py
python scripts/build_final_report.py
python scripts/build_role3_ppt.py
```

GPU 运行命令与 GCS 路径见 `docs/role3_remaining.md`；原始结果在
`gs://demo667/results_role3/`。

## 5. Demo

```bash
python demo/app.py --finetuned A2_models/e2/e2_model.pt --limit 200
```

E0 vs E2 的 top-8 排序对照；首次运行会编码并缓存，第二次启动复用
`demo/.cache/`。详见 `demo/README.md`。
