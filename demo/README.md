# CompoCLIP demo（成员 3）

最小 Gradio demo：同一个组合 query 下，对比 **E0 原始 CLIP** 与 **E2 微调模型**
的 top-8 图片排序，用来展示关系/属性盲区是否被硬负例微调改善。

## 运行

在仓库根目录：

```bash
pip install -r requirements.txt
python demo/app.py --finetuned A2_models/e2/e2_model.pt --limit 200
```

- 不填 `--finetuned`：只展示 E0，便于先调布局；
- `--limit`：从 `data/flickr30k/images` 取前 N 张图；默认 1000，演示建议 100–300；
- `--image-dir`：换图片目录；
- `--cache-dir`：特征缓存目录，默认 `demo/.cache`。

启动后打开终端打印的本地 URL（通常 `http://127.0.0.1:7860`）。

## 预置 query

demo 内置 8 条组合 query，例如：

```text
the dog is to the left of the cat
the cat is to the left of the dog
a man in a red shirt is riding a bike
a man in a red bike is riding a shirt
```

可以在输入框里输入任意 query；文本编码和图片索引是同一套 CLIP 空间。

## 观察什么

- 左栏是 E0，右栏是 E2（E-best；E4a = E2）；
- 对关系反转/属性交换 query，比较正确图片是否进入 top-8、排名是否前移；
- 不要只看单张图：top-8 的整体排序变化才是证据；
- demo 是定性展示，不替代 `report/results/full_4of3.*` 的正式数字。

## 缓存与耗时

首次运行会把图片编码到 `demo/.cache/`，E0 和 E2 各一份；同一图片集合第二次启动
直接复用缓存。CPU 也能跑，但 1000 张图首次编码较慢；GPU 会自动使用。

快速检查：

```bash
python demo/app.py --finetuned A2_models/e2/e2_model.pt --limit 2
```

能打开页面并返回 2 张图即可；正式演示前再用 100–300 张图跑一遍。
