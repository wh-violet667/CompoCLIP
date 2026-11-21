# 方案 A：Colab Enterprise（项目 `cs5489-gp-121`）+ GCS

> 目标：在 Colab Enterprise 的 GPU runtime 上跑完 C 的剩余评测。  
> 这个版本**只支持 GCS bucket**：不使用 Google Drive 挂载，也不使用 notebook 文件浏览器本地上传。

## 0. 先解释：bucket 是什么

GCS（Google Cloud Storage）里的 **bucket** 就是一个云端顶层文件夹/网盘：

- 名字全局唯一，创建时选 region（建议和 runtime 一样用 `asia-east1`）；
- 里面可以放任意文件（这里放三个 `.tar`）；
- 有独立的 IAM 权限；
- 不是 notebook 左侧文件浏览器里的目录，也不是 runtime 的本地磁盘。

创建 bucket 只是“建一个云盘文件夹”，不需要买服务器。6.25 GB 标准存储每月不到 1 美元（以项目实际计费为准）。

## 0.1 三种“上传”不是一回事

| 你操作的位置 | 文件实际去哪 | 这个 notebook 是否使用 |
|---|---|---|
| Colab Enterprise → Notebooks → Upload | Colab 的 notebook 列表 | ✅ 只放 `.ipynb` |
| Notebook 左侧文件浏览器 → Upload | runtime 本地磁盘（临时） | ❌ 不支持 |
| Cloud Storage → Bucket → Upload | GCS 持久存储 | ✅ 三个 tar 放这里 |

`gsutil` / `gcloud storage` 操作的是 GCS bucket，不是 Google Drive；Google Drive 挂载是消费版 Colab 的功能，Colab Enterprise 不支持。

## 0.2 需要准备什么

| 内容 | 放在哪 | 是否必须 |
|---|---|---|
| `notebooks/colab_enterprise_role3_gpu.ipynb` | Colab Enterprise Notebooks（上传） | 必须 |
| `A2_eval.tar`（2.67 GB） | GCS：`gs://BUCKET/`（bucket 根目录） | 必须 |
| `A2_models.tar`（3.03 GB） | 同一 GCS 前缀 | 必须 |
| `data_role3.tar`（0.55 GB） | 同一 GCS 前缀 | 必须 |
| `role3_archive.tar`（0.25 GB） | 同一 GCS 前缀 | 必须（E0 归档结果） |
| 代码（含 `A2_eval_tools/`、`scripts/`、`evals-mm/`） | GitHub `main`，notebook 里 `git clone` | 必须；仓库已推送 |
| GCS bucket + IAM | 项目 `cs5489-gp-121` | 必须 |
| Colab Enterprise GPU runtime（T4/L4） | 同项目 | 必须 |

> notebook 不包含数据，也不会把本地 `colab_packages/` 自动带上去。四个 tar 必须上传到 GCS。

> `role3_archive.tar` 里是 E0 的逐样本结果。E0 不在 GPU 上重跑，但 `assemble_role3_results.py` 需要它来生成共同 ARO/Sugar 表和 E0-vs-E3 配对统计；少了这个包，最终表会缺少 E0 行和对应的配对结果。

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

## 2. 上传四个包到 bucket `demo667`

> 当前这台 Windows 机器的 PATH 里没有 `gcloud`。可以直接用 Console 上传；
> 下面的命令只给已安装 Cloud CLI 的情况。

### 2.1 bucket 已存在

本文使用 `demo667`，四个 tar 直接放根目录，不需要建 `compoclip/` 文件夹。
如果你换了 bucket 名，把 notebook 里的 `BUCKET_URI` 改成 `gs://新名字`。

如果还没有 bucket：Console → **Cloud Storage → Buckets → Create**，
名字全局唯一，Region 选 `asia-east1`，Storage class 用 `Standard`，禁止公开访问。

命令行等价：

```bash
gcloud storage buckets create gs://demo667 \
  --project=cs5489-gp-121 \
  --location=asia-east1
```

### 2.2 上传

把四个 tar **直接上传到 bucket 根目录**即可，不需要建 `compoclip/` 文件夹。

Console：

1. 点进 bucket；
2. 点 **Upload files**；
3. 选择本地 `colab_packages/` 下的四个 tar。

> 如果你确实建了子目录（例如 `compoclip/`），把四个 tar 放进去，并把后面的
> `BUCKET_URI` 写成 `gs://demo667/compoclip`。两种结构 notebook 都支持。

命令行（PowerShell 也可以，单行执行）：

```powershell
gcloud storage cp colab_packages/A2_eval.tar colab_packages/A2_models.tar colab_packages/data_role3.tar colab_packages/role3_archive.tar gs://demo667/
```

`gcloud storage cp` 支持断点续传。6.5 GB 按本机上行带宽计算时间；中断后重跑同一条命令即可。

### 2.3 确认对象存在

Console 里 bucket 根目录应看到四个 tar，大小与 `colab_packages/SHA256SUMS.txt` 一致。命令行：

```bash
gcloud storage ls -l gs://demo667/
```

## 3. 给 runtime 的 service account 授权

### 3.1 service account 是什么

Service account（服务账号）是 runtime 调用 Google API 时使用的“机器人账号”，
不是你的 Google 登录账号，也不是 notebook 里的用户。它长得像：

```text
123456789012-compute@developer.gserviceaccount.com
```

runtime 创建时绑定这个账号；notebook 里不需要下载 JSON key，Google 会通过
metadata server 自动提供凭据。

### 3.2 怎么找到它

- Colab Enterprise → **Runtimes** → 点开 runtime → 详情里的 **Service account**；
- 或 **Runtime templates** → 点开模板 → **Service account**。

如果创建时没有特别指定，通常是项目的 Compute Engine 默认服务账号：

```text
<项目号>-compute@developer.gserviceaccount.com
```

项目号可以在 Console 项目首页看到，或用：

```bash
gcloud projects describe cs5489-gp-121 --format="value(projectNumber)"
```

### 3.3 授权

**以 runtime 页面显示的 service account 为准**。Console：

1. Cloud Storage → Buckets → 你的 bucket；
2. **Permissions → Grant access**；
3. New principals 填 service account 邮箱；
4. Role 选 **Cloud Storage → Storage Object Admin**；
5. Save。

命令行：

```bash
gcloud storage buckets add-iam-policy-binding gs://demo667 \
  --member="serviceAccount:SA_EMAIL" \
  --role="roles/storage.objectAdmin"
```

`objectAdmin` 同时允许下载四个 tar 和上传 `results_role3/`。最小权限可以用
`objectViewer` + `objectCreator`。

> 授权后不需要重启 runtime；IAM 是即时生效的。若仍报 403，检查是不是授权给了另一个 service account。

### 3.4 可以不添加 service account 吗？

可以“不新建”，但不能“没有”：

- 表单留空/选默认 → 使用 `<项目号>-compute@developer.gserviceaccount.com`；
- GCS 模式必须给这个默认账号 bucket 权限；
- 如果项目禁用了默认 Compute Engine 服务账号，表单会强制你新建/选择一个；
- 不需要下载 JSON key。

检查默认账号：

```bash
gcloud iam service-accounts list --project=cs5489-gp-121 \
  --filter="email:compute@developer.gserviceaccount.com"
```

## 4. 创建 Colab Enterprise GPU runtime

打开：

```text
https://console.cloud.google.com/agent-platform/colab/overview?project=cs5489-gp-121
```

然后：

1. 进入 **Runtime templates**；
2. **Create runtime template**：
   - Region：`asia-east1`（与 bucket 同区）
   - Machine type：`n1-standard-4` 或 `n1-standard-8`
   - GPU：`NVIDIA T4 × 1`；没有 T4 配额时选 `g2-standard-4`（自带 L4）
   - Disk：`100 GB`
   - Service account：选第 3 步授权的账号（默认账号也可以）
   - Network：默认网络（需要访问 GitHub；若限制出网，见 8.1）
3. 保存模板；
4. 进入 **Runtimes** → **Create runtime** → 选刚建的模板；
5. 等状态变成 **Running**。

> runtime 按小时计费，且默认不会自己停。跑完一定删除（第 7 节）。

## 5. 上传并打开 notebook

1. Colab Enterprise → **Notebooks** → **Upload**；
2. 选择本地 `notebooks/colab_enterprise_role3_gpu.ipynb`；
3. 打开后，右上角选择刚创建的 runtime（T4/L4）；
4. 修改第 1 个代码 cell：

```python
# 四个 tar 在 bucket 根目录时：
BUCKET_URI = "gs://demo667"

# 如果放在子目录 compoclip/ 下，则写：
# BUCKET_URI = "gs://demo667/compoclip"
```

notebook 已固定使用 `gs://demo667`；只有当 bucket 名不是 `demo667` 时才需要改这一行。

> 只填到 bucket 根目录或前缀，不要带 `A2_eval.tar` 等文件名；notebook 会自己拼。

## 6. 运行

选择 **Run All**。notebook 会依次：

1. 检查 GPU 和 `/tmp` 空间；
2. `git clone` 仓库（含 `A2_eval_tools/`）；
3. 安装 `open_clip_torch` 等依赖（不会替换 runtime 自带的 torch）；
4. 从 GCS 下载四个 tar 到 `/tmp/compoclip/downloads`；
5. 解压主数据并软链到 `A2_eval`、`A2_models`、`data/*`，再把 `role3_archive.tar` 解压到仓库的 `evals-mm/results/`；
6. 重写 Windows 绝对路径；
7. 跑 E0 的 A2 4/3 baseline（可选但推荐；会从 HuggingFace 下载预训练权重）；
8. 跑 E3/E4b 的 ARO + SugarCrepe + COCO；
9. 跑 E1/E2/E2a/E3/E4b 的 Flickr30k + Winoground；
10. 生成 `report/results/master.csv`、`full_4of3.csv`、`paired_full_4of3.*`、`failure_cases.md` / `master.md`；
11. 把结果上传到 `BUCKET_URI` 下的 `results_role3/`。如果 `BUCKET_URI` 是 bucket 根目录，就是 `gs://demo667/results_role3/`。

预计时间（T4，供参考）：

| 任务 | 预计 |
|---|---:|
| 下载 + 解压 6.5 GB | 5–15 分钟（取决于 GCS 到 runtime） |
| E0 ARO + Sugar + COCO | 15–25 分钟 |
| E3/E4b ARO + Sugar + COCO | 30–40 分钟（本次两次共约 31 分钟） |
| 5 个模型 × Flickr30k + Winoground | 20–60 分钟 |
| 结果汇总 + 上传 | 2–5 分钟 |

## 7. 取回结果并删除 runtime

下载结果：

```bash
gcloud storage cp -r gs://demo667/results_role3 ./results_role3
```

如果 `BUCKET_URI` 用了 `compoclip/` 前缀，则把上面的路径换成
`gs://demo667/compoclip/results_role3`。

没有 `gcloud` 时，在 Console 的 bucket 里进入 `results_role3/`（或你的前缀下的
`results_role3/`）逐个下载，或安装 Cloud CLI 后执行上面的命令。

目录结构：

```text
results_role3/
├── report_results/       # master.csv / master.md / paired stats
├── evals_mm_results/     # Flickr30k / Winoground CSV 与逐样本 npz
├── a2_eval_e0/           # E0 的 evaluation.json 与 npz
├── a2_eval_e3/           # E3 的 evaluation.json 与 npz
└── a2_eval_e4b/          # E4b 的 evaluation.json 与 npz
```

把对应文件放回仓库的 `report/results/`、`evals-mm/results/`、`A2_eval/results/`，再更新
`report/06_method.md` 和 `report/07_conclusion.md`。

删除 runtime：

```text
Colab Enterprise → Runtimes → 选中 runtime → Delete
```

同时删除不再需要的 runtime template。GPU runtime 闲置也会计费。

## 8. 常见问题

### 8.1 runtime 访问不了 GitHub

说明组织网络限制出网。两个办法：

- 给 runtime 所在 VPC 配 Cloud NAT；
- 或在本地打包源码并上传到 bucket：

```bash
git archive --format=tar -o colab_packages/CompoCLIP_src.tar HEAD
gcloud storage cp colab_packages/CompoCLIP_src.tar gs://demo667/
```

然后在 notebook 的 clone cell 里改成从 GCS 下载并解压 `CompoCLIP_src.tar`。当前 notebook 默认走 `git clone`。

### 8.2 `403 storage.objects.get`

runtime 服务账号没有 bucket 读权限。回到第 3 步，确认授权对象是 runtime 页面显示的那个 service account。

### 8.3 `CUDA out of memory`

先把 notebook 里的 `--image-batch 64 --text-batch 256` 降到 `32 / 128`。不要改 `--image-batch` 后复用旧缓存：缓存身份包含 batch 大小，会自动新建一份。

### 8.4 任务中途断了

- `/tmp` 在 runtime 删除后消失，重新运行 Run All 即可；
- A2 评测有特征缓存，重跑会从上次完成的 benchmark 继续；
- 检索结果 CSV 会保留已完成的行；
- 结果只有上传到 GCS 后才安全。

### 8.5 没有 T4 配额

用 L4（`g2-standard-4`），显存更大、单价更高；或者改用 `docs/gcp-gpu.md` 的 Compute Engine + T4 路线。不要用 CPU runtime 跑正式评测。

### 8.6 能把数据放 Google Drive 再挂载吗？

**Colab Enterprise 不支持。** `google.colab.drive.mount()` 是消费版 Colab 的功能；
Colab Enterprise runtime 没有 `/content/drive` 这个 FUSE 挂载，也不保证有
`google.colab` 包。`gsutil` / `gcloud storage` 操作的是 GCS bucket，不是 Google
Drive。

如果一定要用 Google Drive，只能改回消费版 Colab（`colab.research.google.com`
或 VS Code Colab 扩展），用 `notebooks/colab_role3_gpu.ipynb`；但那条路线不使用
项目 `cs5489-gp-121` 的 GPU 配额，也不属于方案 A。

### 8.7 notebook 文件浏览器上传可以吗？

这个版本不支持。6.25 GB 走浏览器上传慢且容易断，runtime 删除后文件也没了。
请用 GCS bucket。

### 8.8 `zhuoranyu4-20261030-203332` 是 service account 吗？

不是。这个格式是 **runtime 名称/ID**（用户名 + 创建时间），不是邮箱。Service
account 一定包含 `@`，例如
`123456789012-compute@developer.gserviceaccount.com`。在 runtime 详情页的
**Service account** 一行复制真正的邮箱。
