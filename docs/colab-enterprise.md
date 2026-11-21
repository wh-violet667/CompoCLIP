# 方案 A：Colab Enterprise（项目 `cs5489-gp-121`）

> 目标：在 Colab Enterprise 的 GPU runtime 上跑完 C 的剩余评测。  
> 结论先说：**不是只上传 notebook**。notebook 只是执行入口；数据、模型权重和代码都要有来源，运行时还需要 GPU 配额（走 GCS 时还需要 bucket 权限）。

## 0. 先解释：bucket 是什么

GCS（Google Cloud Storage）里的 **bucket** 就是一个云端顶层文件夹/网盘：

- 名字全局唯一，创建时选 region（建议和 runtime 一样用 `asia-east1`）；
- 里面可以放任意文件（这里放三个 `.tar`）；
- 有独立的 IAM 权限；
- **不是** notebook 左侧文件浏览器里的目录，也不是 runtime 的本地磁盘。

创建 bucket 只是“建一个云盘文件夹”，不是买服务器，也不贵（6.25 GB 标准存储每月约 $0.15 量级，具体以项目计费为准）。

## 0.1 三种“上传”不是一回事

| 你操作的位置 | 文件实际去哪 | 适合放什么 |
|---|---|---|
| Colab Enterprise → Notebooks → Upload | Colab 的 notebook 列表 | 只放 `.ipynb` |
| Notebook 左侧文件浏览器 → Upload | runtime 本地磁盘（临时） | 小文件；6.25 GB 慢且易断，runtime 删了就没了 |
| Cloud Storage → Bucket → Upload | GCS 持久存储 | 三个 tar（推荐） |

当前 notebook 两种方式都支持：

- **GCS 模式（推荐）**：三个 tar 放 bucket，`BUCKET_URI` 填 `gs://...`；
- **本地上传模式**：三个 tar 通过 notebook 文件浏览器上传，`BUCKET_URI` 留空，notebook 会自动在常见目录里找。

## 0.2 需要准备什么

| 内容 | 放在哪 | 是否必须 |
|---|---|---|
| `notebooks/colab_enterprise_role3_gpu.ipynb` | Colab Enterprise Notebooks（上传） | 必须 |
| `A2_eval.tar`（2.67 GB） | GCS bucket，如 `gs://BUCKET/compoclip/`；或 runtime 本地 | 必须 |
| `A2_models.tar`（3.03 GB） | 同上 | 必须 |
| `data_role3.tar`（0.55 GB） | 同上 | 必须 |
| 代码（含 `A2_eval_tools/`、`scripts/`、`evals-mm/`） | GitHub `main`，notebook 里 `git clone` | 必须；仓库已推送 |
| GCS bucket + IAM | 项目 `cs5489-gp-121` | 仅 GCS 模式 |
| Colab Enterprise GPU runtime（T4/L4） | 同项目 | 必须 |

> notebook 不包含数据，也不会把本地 `colab_packages/` 自动带上去。三个 tar 要么上传到 GCS，要么通过 notebook 文件浏览器上传到 runtime。

## 1. 启用 API 和检查 GPU 配额

```bash
gcloud config set project cs5489-gp-121
gcloud services enable aiplatform.googleapis.com storage.googleapis.com compute.googleapis.com
```

然后在 Console 检查 GPU 配额：

```text
IAM & Admin → Quotas → 过滤 "NVIDIA T4 GPUs" 或 "NVIDIA L4 GPUs"
```

- limit 必须大于 0，且 region 与 runtime 一致；
- 如果是 0，先申请提额（可能几小时到几天）；
- 课程项目常见情况是 T4 为 0，但 L4 有额度；L4 也可以用。

## 2. 创建 GCS bucket 并上传三个包

> 当前这台 Windows 机器的 PATH 里没有 `gcloud`。先安装
> [Google Cloud CLI](https://cloud.google.com/sdk/docs/install)，或者直接用
> Console → Cloud Storage 上传。下面的命令假设 `gcloud` 已登录
> （`gcloud auth login`）。

### 2.1 创建 bucket

```bash
gcloud storage buckets create gs://YOUR_BUCKET \
  --project=cs5489-gp-121 \
  --location=asia-east1
```

> bucket 名全局唯一。也可以直接在 Console → Cloud Storage → Buckets 创建。

### 2.2 上传

在本地仓库根目录（PowerShell 也可以，单行执行）：

```powershell
gcloud storage cp colab_packages/A2_eval.tar colab_packages/A2_models.tar colab_packages/data_role3.tar gs://YOUR_BUCKET/compoclip/
```

`gcloud storage cp` 支持断点续传。6.25 GB 按本机上行带宽计算时间；如果上传中断，重跑同一条命令即可。

没有 `gcloud` 时，用 Console 上传到 `compoclip/` 前缀。**不要**通过 Colab 的 `files.upload()` 传 6 GB。

### 2.3 确认对象存在

```bash
gcloud storage ls -l gs://YOUR_BUCKET/compoclip/
```

应看到三个 tar，大小与 `colab_packages/SHA256SUMS.txt` 一致。

### 2.4 不想建 bucket：直接在 notebook 文件浏览器上传

可以，但有前提和代价：

1. 先创建 runtime（第 4 节），打开 notebook；
2. 在 notebook 左侧文件浏览器点 **Upload**，把三个 tar 传到当前目录（通常是 `/home/jupyter`，也可以传到 `/tmp/compoclip/downloads`）；
3. 第 1 个 cell 的 `BUCKET_URI` 留空；
4. notebook 会自动在 `DOWNLOADS`、当前目录、`/home/jupyter`、`/content`、`/tmp` 中查找三个 tar。

代价：

- 6.25 GB 走浏览器上传，速度慢且可能中断；
- 文件只在 runtime 本地磁盘，runtime 删除/重启后要重新上传；
- 结果不会自动上传 GCS，必须在删除 runtime 前从文件浏览器手动下载。

> 如果只是这次跑一遍，可以这样；如果可能重跑，建议还是建 bucket。

## 3. 给 runtime 服务账号授权（仅 GCS 模式）

Colab Enterprise runtime 默认使用项目的 Compute Engine 默认服务账号，也可能是你创建 runtime 时指定的自定义账号。**以 runtime 页面显示的 service account 为准**。

查项目号：

```bash
gcloud projects describe cs5489-gp-121 --format="value(projectNumber)"
```

授权（把 `SA_EMAIL` 换成 runtime 实际使用的账号）：

```bash
gcloud storage buckets add-iam-policy-binding gs://YOUR_BUCKET \
  --member="serviceAccount:SA_EMAIL" \
  --role="roles/storage.objectAdmin"
```

`objectAdmin` 同时允许下载三个 tar 和上传 `results_role3/`。如果只想最小权限：

- 对 bucket/前缀只读：`roles/storage.objectViewer`
- 对 `results_role3/` 前缀可写：`roles/storage.objectCreator`（或 `objectAdmin`）

> 授权后不需要重启 runtime；IAM 是即时生效的。若仍报 403，检查是不是授权给了另一个服务账号。

## 4. 创建 Colab Enterprise GPU runtime

打开你给的页面：

```text
https://console.cloud.google.com/agent-platform/colab/overview?project=cs5489-gp-121
```

然后：

1. 左侧/顶部进入 **Runtime templates**（运行时模板）；
2. **Create runtime template**：
   - Region：`asia-east1`（与 bucket 同区，出网快且免跨区流量）
   - Machine type：`n1-standard-4` 或 `n1-standard-8`
   - GPU：`NVIDIA T4 × 1`；没有 T4 配额时选 `g2-standard-4`（自带 L4）
   - Disk：`100 GB`（默认通常就是 100 GB）
   - Service account：选第 3 步授权的账号（仅 GCS 模式需要；本地上传模式可跳过）
   - Network：默认网络（需要访问 GitHub；若组织限制出网，见第 8 节）
3. 保存模板；
4. 进入 **Runtimes** → **Create runtime** → 选刚建的模板；
5. 等状态变成 **Running**。

> runtime 按小时计费，且默认不会自己停。跑完一定删除（第 7 节）。

## 5. 上传并打开 notebook

在 Colab Enterprise 的 **Notebooks** 页面：

1. **Upload**；
2. 选择本地 `notebooks/colab_enterprise_role3_gpu.ipynb`；
3. 打开后，右上角选择刚创建的 runtime（T4/L4）；
4. 第 1 个代码 cell 二选一：

**GCS 模式：**

```python
BUCKET_URI = "gs://YOUR_BUCKET/compoclip"
```

**本地上传模式：**

```python
BUCKET_URI = ""   # 三个 tar 通过左侧文件浏览器上传
```

也可以不改代码，在 runtime 模板里设置环境变量 `COMPOCLIP_BUCKET=gs://YOUR_BUCKET/compoclip`。

> 上传 notebook 到 Colab Enterprise 和把 tar 上传到 GCS 是两件事。notebook 里不会打包数据。

## 6. 运行

选择 **Run All**。notebook 会依次：

1. 检查 GPU 和 `/tmp` 空间；
2. `git clone` 仓库（含 `A2_eval_tools/`）；
3. 安装 `open_clip_torch` 等依赖（不会替换 runtime 自带的 torch）；
4. 从 GCS 下载三个 tar，或使用文件浏览器里已上传的 tar；
5. 解压并软链到仓库的 `A2_eval`、`A2_models`、`data/*`；
6. 重写 Windows 绝对路径；
7. 跑 E3/E4b 的 ARO + SugarCrepe + COCO；
8. 跑 E1/E2/E2a/E3/E4b 的 Flickr30k + Winoground；
9. 生成 `report/results/master.csv` / `master.md`；
10. GCS 模式下把结果上传到 `gs://YOUR_BUCKET/compoclip/results_role3/`；本地模式下列出需要手动下载的目录。

预计时间（T4，供参考）：

| 任务 | 预计 |
|---|---:|
| 下载 + 解压 6.25 GB | 5–15 分钟（取决于 GCS 到 runtime） |
| E3/E4b ARO + Sugar + COCO | 30–90 分钟 |
| 5 个模型 × Flickr30k + Winoground | 20–60 分钟 |
| 结果汇总 + 上传 | 2–5 分钟 |

## 7. 取回结果并删除 runtime

### GCS 模式

```bash
gcloud storage cp -r gs://YOUR_BUCKET/compoclip/results_role3 ./results_role3
```

### 本地上传模式

在删除 runtime 前，从 notebook 左侧文件浏览器下载 notebook 生成的：

```text
results_role3.tar.gz
```

（在当前工作目录，通常是 `/home/jupyter`。）解压后包含 `report_results/`、`evals_mm_results/`、`a2_eval_e3/`、`a2_eval_e4b/`。

本地解压后，把 `report/results/`、`A2_eval/results/`、`evals-mm/results/` 对应文件放回仓库，再更新 `report/06_method.md` 和 `report/07_conclusion.md`。

删除 runtime：

```text
Colab Enterprise → Runtimes → 选中 runtime → Delete
```

同时删除不再需要的 runtime template。GPU runtime 闲置也会计费。

## 8. 常见问题

### 8.1 runtime 访问不了 GitHub

说明组织网络限制出网。两个办法：

- 给 runtime 所在 VPC 配 Cloud NAT；
- 或在本地打包源码并上传到 bucket（或 notebook 文件浏览器）：

```bash
git archive --format=tar -o colab_packages/CompoCLIP_src.tar HEAD
gcloud storage cp colab_packages/CompoCLIP_src.tar gs://YOUR_BUCKET/compoclip/
```

然后在 notebook 的 clone cell 里改成从 GCS 下载并解压 `CompoCLIP_src.tar`。当前 notebook 默认走 `git clone`。

### 8.2 `403 storage.objects.get`

runtime 服务账号没有 bucket 读权限。回到第 3 步，确认授权对象是 runtime 页面显示的那个 service account。

### 8.3 `CUDA out of memory`

先把 notebook 里的 `--image-batch 64 --text-batch 256` 降到 `32 / 128`。不要改 `--image-batch` 后复用旧缓存：缓存身份包含 batch 大小，会自动新建一份。

### 8.4 任务中途断了

- `/tmp` 在 runtime 删除后消失，重新运行第 6 节即可；
- A2 评测有特征缓存，重跑会从上次完成的 benchmark 继续；
- 检索结果 CSV 会保留已完成的行；
- 结果只有在第 11 个 cell 上传到 GCS 后才安全。

### 8.5 没有 T4 配额

用 L4（`g2-standard-4`），显存更大、单价更高；或者改用 `docs/gcp-gpu.md` 的 Compute Engine + T4 路线。不要用 CPU runtime 跑正式评测。
