# 实验记录（成员 1）

> 目的：记录每次正式运行的命令、数据、n、结果与耗时，便于复现与排查“数字对不上”。
> 只记 `tag=e0` 与 0-A/0-B/0-C 的正式运行；冒烟/试点单独标 `pilot`。

## 环境

- CPU：Intel Iris Xe（无 CUDA），8 逻辑核 / 4 物理核
- Python 3.x，torch CPU，open_clip_torch，spaCy `en_core_web_sm`
- 模型：`ViT-B-32` / `laion2b_s34b_b79k`

## E0

| 日期 | 命令 | 数据 | n | 结果 | 耗时 |
|---|---|---|---:|---|---|
| 2026-11-11 | `run_eval.py --tag e0 --benchmark winoground` | mm-eval 镜像 400 对 | 400 | text 0.3525 / image 0.1125 / group 0.0775 | ~10 min |
| 2026-11-11 | `run_eval.py --tag e0 --benchmark sugarcrepe` | SugarCrepe++ 4,757 | 4,757 | replace 0.8266 / swap 0.6597 / macro 0.7431 | ~8 min |
| 2026-11-11 | `run_eval.py --tag e0 --benchmark aro` | ARO 3 子集 71,679 | 71,679 | VG-Relation 0.5012 / VG-Attribution 0.5785 / Flickr-Order 0.6805 / 宏平均 0.5867 | ~70 min |

### ARO 备注

- `vg_relation` 23,937 + `vg_attribution` 28,748 + `flickr_order` 18,994。
- 图片/文本去重后编码；唯一图片 7,603 张，唯一文本约 10 万条。
- CPU 文本编码是瓶颈；与 0-B 并行时明显变慢。后续如重跑，建议单独跑并 `OMP_NUM_THREADS=8`。

## 0-B

| 日期 | 命令 | n caption | 结果 | 耗时 |
|---|---|---:|---|---|
| 2026-11-11 | `run_0b_probe.py --limit 200 --bootstrap 200`（pilot） | 200 | unrelated 0.985；word_shuffle 0.799；subj_obj 0.638；noun 0.615；adj 0.611；trigram 0.628；relation 0.589 | ~2 min |
| 2026-11-11 | `run_0b_probe.py --limit 5000 --bootstrap 1000` | 5,000 | 见 `results/0b_probe.csv` | 见下 |

## 0-A

| 日期 | 命令 | 数据 | 结果 |
|---|---|---|---|
| 2026-11-11 | `run_0a_probe.py --benchmark winoground` | 400 对 | 零样本 group 0.0775；线性探针 0.500 |
| 2026-11-11 | `run_0a_probe.py --limit 200`（pilot, Flickr） | 200 caption | 探针 0.755–1.000（流畅度捷径，仅作反例） |

## 0-C

| 日期 | 命令 | 输入 | 输出 |
|---|---|---|---|
| 2026-11-11 | `diagnose_failures.py --inputs e0_sugarcrepe_persample.npz` | 4,757 | relation 30.3% / attribute 24.3% / object 10.4% |
| 2026-11-11 | `diagnose_failures.py --inputs e0_winoground_persample.npz` | 400 | Relation group 4.7% / Object 9.2% / Both 26.9% |

## 已知坑

1. `prepare_pairwise_bench.py` 的 `NEG_KEYS` 会把 SugarCrepe++ 的 `caption2` 当负例；用 `prepare_sugarcrepepp.py` 绕开。
2. `run_eval.py` 原实现逐样本重复编码图片；ARO 71k 对会从 10 分钟膨胀到 2 小时。已加去重。
3. `bootstrap_ci` 对 7 万样本会分配 (1000, n) 索引矩阵（约 0.5GB）；内存紧时把 `--bootstrap` 降到 200。
4. Winoground 官方 gated；本地镜像用 `mm-eval/Winoground`，正式数字需复核。
