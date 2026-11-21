# 结果总表（当前可用数据）

> 完整口径见 `report/06_method.md`；ARO 为共同 VG-Relation + VG-Attribution 两子集；Sugar 为 replace + swap。E3/E4b 的 ARO、检索、COCO 仍待 GPU 评测。

| 模型 | ARO macro | SugarCrepe macro | Flickr30k t2i R@1 | Winoground group | COCO t2i R@1 |
|---|---:|---:|---:|---:|---:|
| e0 | 53.987% | 74.314% | 66.760% | 7.750% | 38.794% |
| e1 | 57.649% | 77.114% | — | — | — |
| e2 | 58.016% | 77.443% | — | — | — |
| e2a | 57.434% | 77.066% | — | — | — |
| e3 | — | 76.604% | — | — | — |
| e4b | — | 76.797% | — | — | — |

> E4a = E2。E0 的 Flickr30k/COCO/Winoground 来自 A2/A1 已核验归档；其他未完成项用“—”表示，不填猜测值。