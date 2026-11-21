# Colab + VS Code 官方扩展：跑完 C 的剩余评测

> 适用：本机没有 CUDA，需要把 E3/E4b 的 ARO/SugarCrepe/COCO 和 E1–E4b 的 Flickr30k/Winoground 放到 Colab T4 上跑完。  
> 官方扩展运行的是 **Notebook kernel**，不是本地终端/SSH；本地仓库不会自动同步到 Colab。

## 0. 准备本地压缩包

在仓库根目录打包，减少 Drive 上传文件数：

```powershell
# 如果 A2_eval 里有 downloads 目录（重复的 pkg_eval_images.zip），可以先排除
tar -cf A2_eval.tar --exclude=A2_eval/downloads --exclude=A2_eval/models A2_eval
tar -cf A2_models.tar A2_models
tar -cf data_role3.tar data\golden data\flickr30k data\winoground
tar -cf role3_archive.tar evals-mm\results\e0_aro_persample.npz evals-mm\results\e0_sugarcrepe_persample.npz evals-mm\results\e0_winoground_persample.npz
```

> 四个包合计约 10GB。免费 Drive 15GB 可能刚好够；如果不够，用 Google Cloud Storage（见 `docs/gcp-gpu.md`）或只上传本次需要的 `A2_eval` 子集。

把四个 `.tar` 上传到：

```text
MyDrive/compoclip/
```

同时把当前代码也提交/推送，或者在 Colab 里上传一个小的 repo zip：

```bash
git add scripts docs report demo .gitignore
git commit -m "role3: evaluation scripts and report drafts"
git push
```

## 1. 安装 VS Code 扩展

1. 打开 VS Code；
2. 扩展市场搜索 **Google Colab**（发布者 Google）；
3. 安装；它会提示安装 **Jupyter** 依赖，一起安装；
4. 打开仓库里的任意 `.ipynb`（本仓库可新建 `notebooks/colab_role3_gpu.ipynb`）；
5. 点右上角 **Select Kernel** → **Colab**；
6. 登录 Google 账号，选择 **T4 GPU** runtime。

如果 kernel 列表里没有 T4，先在浏览器打开 [colab.research.google.com](https://colab.research.google.com/) → Runtime → Change runtime type → T4 GPU，再回到 VS Code 重新选 kernel。

## 2. 新建 Notebook 并检查 GPU

在 VS Code 新建 `.ipynb`，第一个代码 cell：

```python
!nvidia-smi
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

必须看到 `True` 和 `Tesla T4`。

## 3. 挂载 Drive、克隆仓库

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content
!git clone https://github.com/wh-violet667/CompoCLIP.git
%cd CompoCLIP
```

如果没有 push 当前 C 改动，`git clone` 得到的是旧版本；请先 commit/push，或者上传 `CompoCLIP_src.tar` 解压到 `/content/CompoCLIP`。

## 4. 解压数据并建立软链

把数据解压到 Colab 本地磁盘（不要直接在 Drive 上跑，I/O 会慢）：

```python
!mkdir -p /content/compoclip_data
!tar -xf /content/drive/MyDrive/compoclip/A2_eval.tar -C /content/compoclip_data
!tar -xf /content/drive/MyDrive/compoclip/A2_models.tar -C /content/compoclip_data
!tar -xf /content/drive/MyDrive/compoclip/data_role3.tar -C /content/compoclip_data
!tar -xf /content/drive/MyDrive/compoclip/role3_archive.tar -C /content/CompoCLIP
```

建软链：

```python
!ln -sfn /content/compoclip_data/A2_eval /content/CompoCLIP/A2_eval
!ln -sfn /content/compoclip_data/A2_models /content/CompoCLIP/A2_models
!ln -sfn /content/compoclip_data/data/golden /content/CompoCLIP/data/golden
!ln -sfn /content/compoclip_data/data/flickr30k /content/CompoCLIP/data/flickr30k
!ln -sfn /content/compoclip_data/data/winoground /content/CompoCLIP/data/winoground
```

## 5. 安装依赖

Colab 已有 CUDA/torch；不要安装 CPU 版 torch。

```python
!pip -q install "open_clip_torch==2.24.0" timm ftfy regex huggingface_hub safetensors spacy pandas tqdm pyyaml datasets faiss-cpu
!python -m spacy download en_core_web_sm
```

如果 `open_clip` 与当前 torch 冲突，先保留 Colab 自带 torch，再安装匹配版本的 open_clip；不要 `pip install torch`。

## 6. 把 A2_eval 改成 Colab 路径

`A2_eval` 的 JSONL 记录的是 Windows 绝对路径，Linux 上不可用。运行：

```python
!python scripts/make_a2_eval_portable.py \
  --root /content/CompoCLIP/A2_eval \
  --data-root /content/CompoCLIP/data
```

同时修正 Winoground 的 Windows 反斜杠路径：

```python
import json
from pathlib import Path
p = Path('/content/CompoCLIP/data/winoground/pairs.jsonl')
rows = [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
img_root = Path('/content/CompoCLIP/data/winoground/images')
for r in rows:
    r['image_0'] = str(img_root / Path(str(r['image_0']).replace('\\', '/')).name)
    r['image_1'] = str(img_root / Path(str(r['image_1']).replace('\\', '/')).name)
p.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n', encoding='utf-8')
print('winoground paths fixed', len(rows))
```

## 7. 跑 E3/E4b 的 A2 4/3 协议

```python
!python scripts/evaluate_role3_models.py \
  --tags e3 e4b \
  --benchmarks aro sugarcrepe coco \
  --device cuda --image-batch 64 --text-batch 256
```

输出在 `A2_eval/results/e3/` 和 `e4b/`，包含 `evaluation.json` 与逐样本 npz。

## 8. 跑 Flickr30k + Winoground

```python
!for tag in e1 e2 e2a e3 e4b; do \
  python evals-mm/run_eval.py --tag "$tag" \
    --checkpoint "A2_models/$tag/${tag}_model.pt" --benchmark flickr30k; \
  python evals-mm/run_eval.py --tag "$tag" \
    --checkpoint "A2_models/$tag/${tag}_model.pt" --benchmark winoground; \
done
```

E0 已有归档，E4a = E2，不需要重跑。

## 9. 生成结果表并保存

```python
!python scripts/assemble_role3_results.py
!python scripts/build_full_4of3.py
!python scripts/build_full_master.py
!python scripts/make_failure_cases_a2.py
!python scripts/make_master_md.py
```

> `build_full_4of3.py` 生成 A2 4/3 全量表；`build_full_master.py` 会把 `evals-mm/results/*.csv` 和 `A2_eval/results/*/evaluation.json` 合并进 `master.csv`；如果某些 E1/E2a 的 COCO/检索没跑，表里会保留“—”。

把结果复制回 Drive：

```python
!mkdir -p /content/drive/MyDrive/compoclip/results_role3
!cp -r report/results/* /content/drive/MyDrive/compoclip/results_role3/
!cp -r A2_eval/results/e3 /content/drive/MyDrive/compoclip/results_role3/
!cp -r A2_eval/results/e4b /content/drive/MyDrive/compoclip/results_role3/
```

之后把结果从 Drive 下载回本地，更新 `report/06_method.md`、`report/07_conclusion.md` 的“未完成”单元格。

## 10. 免费版注意事项

- 免费 Colab 最长约 12 小时，可能空闲断连；训练/评测中不要关标签页；
- `/content` 会重置，数据要从 Drive 重新解压；
- 结果必须及时复制到 Drive；
- 不要用自动点击/脚本保活，违反 Colab 条款；
- 如果 T4 配额不可用，改用 Kaggle（每周 30h T4）或 GCP（见 `docs/gcp-gpu.md`）。
