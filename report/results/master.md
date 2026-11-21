# 结果总表（当前可用数据）

> 完整口径见 `report/06_method.md`；本表 ARO 为共同 VG-Relation + VG-Attribution 两子集；Sugar 为 replace + swap。E3/E4b 的 ARO、检索、COCO 已由 T4 补齐；E1/E2/E2a 的 COCO 未同批次评测，用“—”表示。A2 4/3 全量表见 `full_4of3.md`。

| 模型 | ARO macro | SugarCrepe macro | Flickr30k t2i R@1 | Winoground group | COCO t2i R@1 |
|---|---:|---:|---:|---:|---:|
| e0 | 54.416% | 73.897% | 66.760% | 7.750% | 38.794% |
| e1 | 57.649% | 77.114% | 73.980% | 8.250% | — |
| e2 | 58.016% | 77.443% | 72.980% | 8.000% | — |
| e2a | 57.434% | 77.066% | 73.680% | 8.250% | — |
| e3 | 54.398% | 76.604% | 73.260% | 7.500% | 41.029% |
| e4b | 57.566% | 76.358% | 71.440% | 7.750% | 38.958% |

> E4a = E2。E0 的 ARO/Sugar/COCO 来自新的 A2 4/3 baseline；Flickr30k/Winoground 来自归档。E1/E2/E2a 的 COCO 未同批次评测，用“—”表示，不填猜测值。