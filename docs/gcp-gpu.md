# 用 Google Cloud GPU 跑 CompoCLIP

> 目标：把 GPU 评测放到 Google Cloud 上运行。  
> 实际剩余评测已在 Colab Enterprise T4 完成，结果见 `docs/role3_remaining.md` 和
> `report/results/full_4of3.md`；本文保留为 Compute Engine 路线的操作手册。

## 0. 先选方案

| 方案 | 适合 | 成本/限制 |
|---|---|---|
| Google Colab（免费/Pro） | 一次性跑 1–2 小时评测 | 免费 T4，可能断连；数据放 Drive |
| **GCP Compute Engine + T4/L4** | 需要 SSH、VS Code Remote、长时间稳定运行 | 按小时计费；需 GPU 配额 |
| Vertex AI Workbench | 想用托管 JupyterLab + GPU | 配置更重，适合 notebook 工作流 |

下面给 **GCP Compute Engine** 的最短路径。

## 1. 安装并登录 gcloud

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable compute.googleapis.com storage.googleapis.com
```

需要已绑定结算账号。GPU 通常不在免费试用额度内。

## 2. 检查 GPU 配额

```bash
gcloud compute regions describe asia-east1 \
  --format="table(quotas.metric,quotas.limit,quotas.usage)"
```

找 `NVIDIA_T4_GPUS` 或 `GPUS_ALL_REGIONS`。如果 limit 是 0：

1. 打开 Console → IAM & Admin → Quotas；
2. 过滤 `NVIDIA T4 GPUs` + 目标 region；
3. 申请提额（可能需要几小时到一天）。

## 3. 创建 GPU 虚拟机

先确认 Deep Learning VM 的镜像 family：

```bash
gcloud compute images list \
  --project=deeplearning-platform-release \
  --filter="family~pytorch" \
  --format="table(name,family)"
```

用 T4（最便宜、够本项目）创建：

```bash
gcloud compute instances create compoclip-gpu \
  --zone=asia-east1-b \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=150GB \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

如果 `pytorch-latest-gpu` 不存在，把 `--image-family` 换成上一步列出的实际 family。  
想要更快可以用 L4：`--machine-type=g2-standard-8`（G2 自带 L4，通常不需要 `--accelerator`）。

登录并确认 GPU：

```bash
gcloud compute ssh compoclip-gpu --zone=asia-east1-b
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 4. 把代码和数据传上去

不要传 `pkg_*.zip`、完整 A2 交接包或原始训练数据；只传评测需要的部分：

- `A2_eval/`：B 的冻结基准、图片、协议（约 8GB，可排除 `downloads/` 里的重复 zip）；
- `A2_models/`：E3/E4b/E1/E2/E2a 的裸权重（约 3GB）；
- `data/golden/`、`data/flickr30k/`、`data/winoground/`：仅做检索/Winoground 时需要。

### 方式 A：GCS（推荐，可断点续传）

在本地：

```bash
gcloud storage buckets create gs://YOUR_BUCKET --location=asia-east1
gcloud storage cp -r A2_eval gs://YOUR_BUCKET/role3/
gcloud storage cp -r A2_models gs://YOUR_BUCKET/role3/
gcloud storage cp -r data/golden gs://YOUR_BUCKET/role3/
gcloud storage cp -r data/flickr30k gs://YOUR_BUCKET/role3/
gcloud storage cp -r data/winoground gs://YOUR_BUCKET/role3/
```

在 VM 上：

```bash
sudo mkdir -p /opt/compoclip && sudo chown -R "$USER" /opt/compoclip
cd /opt/compoclip
git clone https://github.com/wh-violet667/CompoCLIP.git .
gcloud storage cp -r gs://YOUR_BUCKET/role3/A2_eval .
gcloud storage cp -r gs://YOUR_BUCKET/role3/A2_models .
gcloud storage cp -r gs://YOUR_BUCKET/role3/golden data/
gcloud storage cp -r gs://YOUR_BUCKET/role3/flickr30k data/
gcloud storage cp -r gs://YOUR_BUCKET/role3/winoground data/
```

### 方式 B：直接从本机 scp

```bash
gcloud compute scp --recurse A2_eval A2_models \
  compoclip-gpu:/opt/compoclip/ --zone=asia-east1-b
```

> 所有 C 的代码改动要先 commit + push，或者在 VM 上单独 scp `scripts/`、`report/`、`demo/`。

## 5. 安装依赖

```bash
cd /opt/compoclip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import torch, open_clip; print(torch.cuda.is_available())"
```

如果 `open_clip_torch` 试图升级 torch 或装 CPU 版，先固定版本：

```bash
pip install "open_clip_torch==2.24.0" timm --no-deps
```

不要安装 CPU 版 torch；DLVM 镜像已经带 CUDA 驱动和匹配的 PyTorch。

## 6. 跑剩余评测

A2 4/3 冻结协议（E3/E4b，ARO + SugarCrepe + COCO）：

```bash
python scripts/evaluate_role3_models.py \
  --tags e3 e4b \
  --benchmarks aro sugarcrepe coco \
  --device cuda --image-batch 64 --text-batch 256
```

Flickr30k + Winoground（E1/E2/E2a/E3/E4b）：

```bash
for tag in e1 e2 e2a e3 e4b; do
  python evals-mm/run_eval.py --tag "$tag" \
    --checkpoint "A2_models/$tag/${tag}_model.pt" --benchmark flickr30k
  python evals-mm/run_eval.py --tag "$tag" \
    --checkpoint "A2_models/$tag/${tag}_model.pt" --benchmark winoground
done
```

E0 已归档，E4a = E2，不需要重跑。

## 7. 回收结果并生成报告表

```bash
python scripts/assemble_role3_results.py
python scripts/make_master_md.py
```

把 `A2_eval/results/`、`evals-mm/results/`、`report/results/` 同步回本地，或直接在 VM 上提交 report 更新。

## 8. 关机与费用

```bash
# 暂停（保留磁盘，之后可继续）
gcloud compute instances stop compoclip-gpu --zone=asia-east1-b

# 彻底删除（连磁盘）
gcloud compute instances delete compoclip-gpu \
  --zone=asia-east1-b --delete-disks=all
```

T4/L4 按小时计费；跑完一定 stop，避免闲置 GPU 费用。参考量级：T4 约 $0.35/小时、L4 约 $0.7–0.9/小时（以 Console 实际价格为准），本项目剩余评测通常 1–2 小时。

## 9. 更省事的替代

### Colab Enterprise（项目内 GPU，GCS）

如果必须使用 `cs5489-gp-121` 项目自带的 GPU 配额，走
[`docs/colab-enterprise.md`](colab-enterprise.md)：创建 Colab Enterprise
runtime，四个 tar 放 GCS，notebook 用
`notebooks/colab_enterprise_role3_gpu.ipynb`。

### Colab + VS Code

1. VS Code 安装官方 **Google Colab** 扩展；
2. 打开 `.ipynb` → Select Kernel → Colab → T4；
3. `drive.mount` 后把 `A2_eval`、`A2_models` 放 Drive，跑同样的命令；
4. 注意免费 Drive 15GB 可能不够放 A2_eval + models + 其他文件。

### Vertex AI Workbench

Console → Vertex AI → Workbench → Create New → 选 GPU（T4/L4）→ JupyterLab。适合不习惯 SSH 的情况，但启动/管理比 Compute Engine 重。
