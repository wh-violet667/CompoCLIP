# 数据与代码来源清单

> 由成员 1 建模板，成员 2 负责逐项补全/核对。规则：**没有来源与许可的条目不得进入实验**；下载后记录 SHA256 或文件大小，防止“同名不同版本”。

## 1. 评测与训练数据

| 名称 | 用途 | 来源（URL / 命令） | 许可 | 下载日期 | 本地路径 | 校验（大小/SHA256） | 负责人 | 状态 |
|---|---|---|---|---|---|---|---|---|
| Flickr30k（公开镜像） | 0-B 探针、E0 test | `https://hf-mirror.com/datasets/clip-benchmark/wds_flickr30k` | 研究用途，原数据需同意 Flickr 条款 | 2026-11-11 | `data/flickr30k/` | 1000 图 / 5000 caption；tar 共 145,571,842 B | 成员 1（临时代用） | ✅ 已用，待 Kaggle 原包复核 |
| Flickr30k（Kaggle 原包） | 正式训练/评测 | Kaggle `huyhoangvuong/flickr30k`（以 dev-doc 为准） | 同上 | 待定 | `data/flickr30k/` | 待补 | 成员 2 | ⏳ 待提供 |
| ARO VG-Relation | E0 子集 | `gdown 1kX2iCHEv0CADL8dSO1nMdW-V0NqIAiP3` | MIT（代码）；VG 图片 CC BY 4.0 | 2026-11-11 | `data/aro/vg_relation.json` | 23,937 对；9,125,212 B | 成员 1 | ✅ |
| ARO VG-Attribution | E0 子集 | `gdown 13tWvOrNOLHxl3Rm9cR3geAdHx2qR3-Tw` | 同上 | 2026-11-11 | `data/aro/vg_attribution.json` | 28,748 对；8,707,547 B | 成员 1 | ✅ |
| ARO VG 图片 | 上述两子集 | `gdown 1qaPlrwhGNMrR3a11iopZUT_GPP_LrgP9` | 同上 | 2026-11-11 | `data/aro/images/` | 6,603 张；zip 908,654,022 B | 成员 1 | ✅ |
| ARO Flickr-Order | E0 子集（重建） | 官方 `TextShuffler` 四种扰动 + Flickr30k test | 同上 | 2026-11-11 | `data/golden/aro.jsonl`（`subset=flickr_order`） | 18,994 对 | 成员 1 | ✅ 重建口径 |
| ARO COCO-Order | E0 子集 | 需 COCO test2014 原图 | COCO 条款 | — | — | — | 成员 2 | ⛔ blocked（6GB，未提供） |
| SugarCrepe++ | E0 子集 | `https://huggingface.co/datasets/Mayfull/sugarcrepepp_vlms` | CC BY 4.0（数据集卡） | 2026-11-11 | `data/sugarcrepepp/` | 4,757 条 / 1,560 图；zip 248,964,877 B | 成员 1 | ✅ |
| Winoground（公开镜像） | E0 | `https://huggingface.co/datasets/mm-eval/Winoground` | 研究用途；官方 gated | 2026-11-11 | `data/winoground/` | 400 对 / 800 图；parquet 708,653,974 B | 成员 1 | ✅ 待官方复核 |
| COCO val2017 | E0 检索、E4 对照 | `http://images.cocodataset.org/zips/val2017.zip` | COCO 条款 | 待定 | `data/coco2017/` | 待补 | 成员 2 | ⏳ |

## 2. 代码与模型权重

| 名称 | 版本/标识 | 来源 | 许可 | 用途 | 负责人 |
|---|---|---|---|---|---|
| OpenCLIP | pip 包版本见 `requirements.txt` | https://github.com/mlfoundations/open_clip | MIT | 编码器、训练 | 成员 3 |
| ViT-B-32 / laion2b_s34b_b79k | OpenCLIP 权重 | HF `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | MIT（权重卡） | E0 基座 | 成员 3 |
| ARO 扰动代码 | commit `mertyg/vision-language-models-are-bows` | 同上 | MIT | Flickr-Order 重建 | 成员 1 |
| spaCy `en_core_web_sm` | 3.x | https://spacy.io | MIT | 依存/词性 | 成员 1 |
| NLTK | 3.x | https://www.nltk.org | Apache 2.0 | 分词对照 | 成员 1 |

## 3. 生成数据（必须可追溯）

| 名称 | 生成脚本 | 输入 | 条数 | 生成日期 | 校验 | 负责人 |
|---|---|---|---|---|---|---|
| `data/golden/sugarcrepe.jsonl` | `scripts/prepare_sugarcrepepp.py` | SugarCrepe++ examples + images | 4,757 | 2026-11-11 | 子类 replace=3,846 / swap=911 | 成员 1 |
| `data/golden/aro.jsonl` | `scripts/prepare_aro.py` | ARO 标注 + VG 图 + Flickr test | 71,679 | 2026-11-11 | 3 子集，见 decision-log D0-5 | 成员 1 |
| `data/winoground/pairs.jsonl` | `scripts/prepare_winoground.py` | mm-eval parquet | 400 | 2026-11-11 | 每对 2 图 | 成员 1 |
| `data/hardneg/llm_generated.jsonl` | `src/hardneg/llm_gen.py` | GT caption + LLM API | 15,000（目标） | 待定 | 词多重集通过率 ≥98% | 成员 2/3（需 key） |
| `data/hardneg/rule_generated.jsonl` | `src/hardneg/rule_based.py` | GT caption | 待定 | 待定 | 待补 | 成员 2 |

## 4. 禁止事项

- 不得把 `caption2`（同义改写）当负例；不得用 `--limit` 冒烟数字冒充全量结果。
- 不得提交第三方原图到 git；只提交路径与脚本（`.gitignore` 已覆盖 `data/`）。
- 任何“重跑后数字变了”的情况，必须在 `decision-log.md` 追加一条，写清数据/代码/环境差异。
