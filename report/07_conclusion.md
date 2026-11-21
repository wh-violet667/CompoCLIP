# 7 结论与展望

> 本章在正式结果落表后定稿；未完成的数字不填猜测值。

## 7.1 主要结论

在共同 2/4 ARO（VG-Relation + VG-Attribution）与 2/3 SugarCrepe（replace + swap）子集上：

1. **E1/E2/E2a 在 A2 VG-crop 协议内高于 E0 的参考值**：ARO VG macro 从 E0 的 53.987% 提升到 E1/E2/E2a 的 57.4–58.0%；SugarCrepe macro 从 74.314% 提升到 77.1–77.4%。但 E0 的 ARO 是 A1 原图协议，不能当作严格配对基线；该提升只能作为提示性证据。
2. **E2 是 E1/E2/E2a 中的最佳选模结果**（B 的 4/3 selection score 62.5683%），但 E1/E2/E2a 的差别很小，且 E1/E2 同时改变来源与覆盖量，不能单独归因于负例来源。
3. **E3 在 SugarCrepe 上相对 E0 的点估计为 +2.289pp，但 E3−E0 的 95% CI [-0.047, +4.418] 跨 0**；E3 相对 E2 为 E2 +0.839pp，不满足“E3 对 E-best 下降 ≤0.5pp”的附加标准。ARO/COCO 的 E3 正式评测尚未完成，不能宣称蒸馏收益或抗遗忘成立。
4. **E4b 在 SugarCrepe 上为 76.797%，低于 E2 的 77.443%（E2 +0.646pp）**；Flickr/COCO 的遗忘与成本比较尚未完成，不能宣称锁塔更优。
5. 预注册主标准（E3 对 E0 ≥ +2.0pp、COCO 下降 ≤1.0pp）目前**未完整满足**：Sugar 点估计达到但 CI 跨 0，COCO/ARO 未完成。应按“未确认/负结果”报告，不强行包装。

## 7.2 失败模式与解释

E3 的 15 条最低 margin SugarCrepe 错误见 `report/results/failure_cases.md`；主要失败仍集中在关系与属性替换，以及少数正负 caption 本身接近同义/标注歧义的样本。ARO 的失败案例因 CPU 全量评测未完成而暂缺。结合 0-C 的分类，当前证据更支持“模型对细粒度关系/属性绑定仍不稳定”，而不是单一的数据噪声解释；但 15 条样本不足以做比例估计，必须与未决的 20 条人工反馈分开看。

## 7.3 局限

1. 单训练 seed；无第二 seed。
2. 统一表采用 2/4 ARO（VG-Relation + VG-Attribution）+ 2/3 SugarCrepe 公共子集；B 的 4/3 选择口径与此不同；Flickr-Order 在 A1/A2 中分别是 18,994/5,000 条，不能混算。E0 的 ARO 是 A1 原图协议，E1/E2/E2a 是 A2 VG-crop 协议，E0 行不能当作严格配对基线。
3. E2a 与 E2 数量不匹配；E1 与 E2 来源/覆盖同时变化。
4. E3/E4b 的 ARO、Flickr30k、Winoground、COCO formal 评测在本地 CPU 超时，尚未完成；需要 GPU/Colab 按 `docs/role3_remaining.md` 补齐。
5. Flickr30k/Winoground 部分使用公开镜像；官方数据复核未完全闭合。
6. Winoground 只有 400 对，统计功效有限；CI 跨 0 的提升不能写成结论。
7. 训练监控不是 formal benchmark。
8. 预训练数据污染、模型偏差、LLM 生成非确定性未完全排除。

## 7.4 展望

- 用官方数据复核 Flickr30k/Winoground，并补 ARO COCO-Order 与 SugarCrepe `add`。
- 补第二个训练 seed、matched-size 负例对照和更多基座（如 ViT-B/16 SigLIP）。
- 在 B 的 4/3 冻结协议上补 E3/E4b 的 GPU 全量评测，替换当前 CPU 公共子集。
- 把 E3 蒸馏扩展到 COCO/CC3M 通用数据，系统研究组合增益与遗忘的 Pareto 前沿。
- 将 E-best 封装为 `ImageTextEncoder` 服务，接入 Mnemo 的图像记忆检索。
