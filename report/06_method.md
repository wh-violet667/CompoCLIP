# 6 统一评测与结果

> 本章的数字以 `report/results/master_common.csv`、`report/results/paired_stats.json` 和 `evals-mm/results/*.csv` 为准。未完成的单元格不填猜测值；所有结果注明数据口径。

## 6.1 评测口径

### 6.1.1 选模与统一评测的区别

E-best 选择使用 A2 冻结的 **4 个 ARO 子集 + 3 个 SugarCrepe 家族**（VG-Relation、VG-Attribution、COCO-Order、Flickr-Order；replace、swap、add），E2 以 62.5683% 的 selection score 胜出。该结果只用于选择 E2；它同时使用 ARO/SugarCrepe，因此不能当作独立泛化检验。

成员 3 的 CPU 统一评测使用 A1/E0 已归档并可由本仓库复现的 **2/4 ARO 子集 + 2/3 SugarCrepe 家族**：

- ARO：`vg_relation`、`vg_attribution`；不含 `flickr_order` / `coco_order`。注意 A1 的 Flickr-Order 是 18,994 条、A2 的是 5,000 条，两者扰动集不同，不能混算；COCO-Order 需要 COCO2014 图像，A1 归档未覆盖。
- SugarCrepe：`replace`、`swap`；不含 `add`（A1 镜像未覆盖）。
- 检索：Flickr30k test（1,000 图 / 5,000 caption，双向）；COCO val2017（5,000 图 / 25,014 caption，双向）。
- Winoground：400 对，Text / Image / Group。

> 这是一个**共同可复现子集**，不是 B 的 4/3 冻结口径。任何跨表比较必须标明口径；不能把 2/4 的 ARO macro 与 B 的 4/4 selection score 直接比较。E1/E2/E2a 的 4/3 结果保留在 A2 交接中；本章过滤出共同的 VG 两子集和 Sugar 两家族后与 E0/E3/E4b 对齐。
>
> **ARO 的额外口径差异**：E0 的 A1 归档使用原图（无 VG crop），E1/E2/E2a 的 A2 归档使用 VG bounding-box crop；两者不能当作严格配对基线。E0 的 ARO 行只作参考，不作为 E2/E3 的因果对照。SugarCrepe 的图片源也来自两套归档，配对统计只能解释为“同一批记录”的近似比较，不能排除预处理差异。

### 6.1.2 评分规则

- ARO/SugarCrepe 二选一：正 caption 分数严格高于负 caption 才算正确；Order 子集按原 positive-first tie 规则。
- ARO 2 子集、SugarCrepe 2 家族分别等权 macro；每个子集/家族的 n 与 bootstrap 95% CI 单独报告。
- 检索：唯一图片检索；i2t 的一图多 caption 按“任一正确 caption 命中 top-k”计算；tie 用 average 名次。
- 所有比较使用相同样本 ID/顺序做配对 bootstrap；同时给 McNemar 检验。两个独立 CI 是否重叠不能代替配对检验。
- `--limit` 冒烟结果不进入主表；单 seed 结果不宣称统计显著。

## 6.2 主结果

### 6.2.1 组合推理（共同 2/4 ARO + 2/3 SugarCrepe）

| 模型 | ARO macro | SugarCrepe macro | 说明 |
|---|---:|---:|---|
| E0 | 53.987% | 74.314% | 原始 CLIP |
| E1 | 57.649% | 77.114% | 规则负例，全量 FT |
| E2 | 58.016% | 77.443% | LLM 负例，全量 FT；E-best / E4a |
| E2a | 57.434% | 77.066% | 过滤子集 |
| E3 | 未完成（CPU 超时） | 76.604% | E2 负例 + 蒸馏 λ=0.5；Sugar 取 A2 4/3 的 replace/swap 子集 |
| E4b | 未完成（CPU 超时） | 76.797% | E2 负例 + 锁图像塔 |
| E4a | = E2 | = E2 | 复用 E2 |

> ARO 的 2 子集均值与 B 的 4 子集 selection score 不可直接比较；E0 的 53.987% 也不能与 B 表中 E1/E2/E2a 的 43–44% 直接比较（后者含 Flickr/COCO-Order）。报告必须同时给子集表。

### 6.2.2 检索与 Winoground

| 模型 | Flickr30k t2i R@1 | Flickr30k i2t R@1 | Winoground group | COCO t2i R@1 |
|---|---:|---:|---:|---:|
| E0 | 66.76% | 84.20% | 7.75% | 38.79% |
| E1 | 未完成 | 未完成 | 未完成 | 未完成 |
| E2 | 未完成 | 未完成 | 未完成 | 未完成 |
| E2a | 未完成 | 未完成 | 未完成 | 未完成 |
| E3 | 未完成 | 未完成 | 未完成 | 未完成 |
| E4b | 未完成 | 未完成 | 未完成 | 未完成 |

> E0 的 Flickr30k/COCO 来自 A2 已核验归档；E1/E2/E2a/E3/E4b 由成员 3 按同一 `run_eval` 接口评测。Flickr30k test 的 1,000 张图也参与 ARO Flickr-Order 和 E-best 选择，因此检索结果不是完全独立的泛化检验。

## 6.3 配对统计与假设检验

### 6.3.1 SugarCrepe（共同 replace/swap，n=4,757）

| 对比 | macro diff | 95% CI | 结论 |
|---|---:|---|---|
| E0 vs E3 | E3 − E0 = +2.289 pp | [-0.047, +4.418] | 点估计为正，但 CI 跨 0，不能宣称显著 |
| E2 vs E3 | E2 +0.839 pp | [-0.516, +2.163] | E2 点估计更高；E3 未满足“对 E-best 下降 ≤0.5pp” |
| E2 vs E4b | E2 +0.646 pp | [-1.612, +2.786] | E4b 没有超过 E2；CI 跨 0 |

子集级：E3 的 swap 相对 E0 为 +4.501pp（McNemar p=0.0429），replace 基本不变（-0.078pp，p=0.935）；E3 相对 E2 的 replace +0.910pp（p=0.0686）、swap +0.768pp（p=0.611）。

### 6.3.2 ARO

E3/E4b 的 ARO 2 子集全量 CPU 评测超时，尚无正式配对结果；不得用 SugarCrepe 的结论替代 ARO。E0/E1/E2/E2a 的 ARO VG 两子集结果已在 `report/results/aro_common.csv`，但 E0 的 A1 原图协议与 E1/E2/E2a 的 A2 VG-crop 协议不同，不能把 E2 相对 E0 的 +4.029pp 当作严格配对效应；它只能作为参考，且 E-best 选择本身也使用了同源 ARO/Sugar 数据。

## 6.4 蒸馏分析（E3 vs E0 / E-best）

在 SugarCrepe 共同 2 家族上：

- E3 相对 E0：**+2.289pp**（E3−E0 的 95% CI [-0.047, +4.418]，跨 0）；点估计达到 +2.0pp，但不能宣称显著。
- E3 相对 E2：E2 高 **0.839pp**（CI [-0.516, +2.163]）；不满足“E3 对 E-best 下降 ≤0.5pp”的附加标准。
- COCO 遗忘：E3 的 COCO 全量评测尚未完成，不能判断是否满足“下降 ≤1.0pp”。
- ARO：E3 的 2 子集全量评测尚未完成。

因此目前只能得出“SugarCrepe 上蒸馏相对 E0 有正向点估计、但相对 E2 没有优势且不显著”；不能宣称抗遗忘或蒸馏收益成立。

## 6.5 锁图像塔分析（E4b vs E4a/E2）

在 SugarCrepe 共同 2 家族上，E4b 为 **76.797%**，E2 为 77.443%：E2 高 **0.646pp**（CI [-1.612, +2.786]）。E4b 没有表现出组合增益；Flickr30k/COCO 的遗忘与训练成本尚未完成正式评测，因此不能判断“锁塔减少遗忘”是否成立。E4b 的训练显存/时间优势只来自训练日志，不能替代 benchmark。

## 6.6 失败分析

已完成 E3 在 SugarCrepe 共同 replace/swap 子集上的 15 条最低 margin 错误，见 `report/results/failure_cases.md`；ARO 的 15 条因全量评测未完成而暂缺。案例保留图片 basename、正负 caption、分数和 margin，便于回到 `data/golden/` 与 `A2_eval/data/` 复核。人工审核的 20 条未决记录不追认、不覆盖原判断。

## 6.7 限制

1. 单训练 seed；无第二 seed，不能宣称稳定性。
2. 统一表使用 2/4 ARO（VG-Relation + VG-Attribution）+ 2/3 SugarCrepe 的公共子集；与 B 的 4/3 选模口径不同。
3. E1/E2/E2a 的 4/3 结果用于选模；本章过滤后重算，不能与 selection score 混用。
4. E2a 的 522 条是过滤子集，数量小于 E2 的 1,739 条；没有 matched-size 对照，质量与数量混淆。
5. E1 与 E2 同时改变负例来源和覆盖量；不能单独归因于“规则 vs LLM”。
6. Flickr30k/Winoground 部分图片来自公开镜像；官方数据复核、Winoground 原访问协议和图片哈希仍未完全闭合。
7. ARO 的 20 条人工反馈中仍有未决项；主结果保留原全量口径，敏感性分析另行标注。
8. 训练监控的 300 图 t2i R@1 不是 formal benchmark，不能进入主表。
9. **CPU 评测限制**：E3/E4b 的 ARO 全量、Flickr30k/Winoground 和 COCO formal 评测在本地 CPU 超时，尚未产出；表中以“未完成”标注，不填猜测值。完成这些项需要 GPU/Colab；命令见 `docs/role3_remaining.md`。
