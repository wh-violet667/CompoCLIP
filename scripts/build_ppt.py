"""生成组会/答辩 PPT（16:9，深色简洁风），读取最新结果自动填数字。

用法：
    python scripts/make_figures.py
    python scripts/build_ppt.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "evals-mm/results"
FIG = ROOT / "report/figures"
OUT = ROOT / "report/CompoCLIP_成员1汇报.pptx"

NAVY = RGBColor(0x0B, 0x1E, 0x2D)
NAVY2 = RGBColor(0x12, 0x2E, 0x42)
ORANGE = RGBColor(0xE0, 0x8A, 0x3C)
CREAM = RGBColor(0xF3, 0xEF, 0xE6)
GREY = RGBColor(0x9A, 0xA6, 0xB2)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREEN = RGBColor(0x6F, 0xB7, 0x8E)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def bg(slide, color=NAVY):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def text(slide, x, y, w, h, s, size=18, color=CREAM, bold=False, align=PP_ALIGN.LEFT,
         font="Microsoft YaHei", spacing=1.0):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = spacing
    run = p.add_run()
    run.text = s
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return box


def bullets(slide, x, y, w, h, items, size=16, color=CREAM, gap=10):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        p.line_spacing = 1.15
        run = p.add_run()
        run.text = "· " + item
        run.font.name = "Microsoft YaHei"
        run.font.size = Pt(size)
        run.font.color.rgb = color


def rule(slide, x, y, w, color=ORANGE, h=0.06):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()


def tag(slide, s, x=0.7, y=0.45):
    text(slide, x, y, 6, 0.35, s.upper(), size=11, color=ORANGE, bold=True)


def footer(slide, n):
    text(slide, 11.9, 7.05, 0.9, 0.25, f"{n:02d}", size=10, color=GREY, align=PP_ALIGN.RIGHT)
    text(slide, 0.7, 7.05, 5, 0.25, "CompoCLIP · CS5489 · 成员 1", size=10, color=GREY)


def read_csv(name):
    p = RES / name
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def latest(rows):
    out = {}
    for r in rows:
        if r.get("tag") == "e0":
            out[(r.get("benchmark"), r.get("subset") or "")] = r
    return out


CSV = latest(read_csv("winoground.csv") + read_csv("aro.csv") + read_csv("sugarcrepe.csv"))
wg = CSV.get(("winoground", ""), {})
sc_macro = CSV.get(("sugarcrepe", "macro"), {})
aro_macro = CSV.get(("aro", "macro"), {})
probe = {}
if (RES / "0b_summary.json").exists():
    probe = {r["probe"]: r for r in json.loads((RES / "0b_summary.json").read_text(encoding="utf-8"))["probes"]}
a0 = json.loads((RES / "0a_winoground.json").read_text(encoding="utf-8")) if (RES / "0a_winoground.json").exists() else {}


def pct(x, nd=1):
    try:
        return f"{float(x) * 100:.{nd}f}%"
    except (TypeError, ValueError):
        return "—"


core_accs = [float(probe[k]["accuracy"]) for k in ("subj_obj_swap", "noun_swap", "relation_reverse") if k in probe]
core_lo = pct(min(core_accs)) if core_accs else "—"
core_hi = pct(max(core_accs)) if core_accs else "—"

# 1 封面
s = prs.slides.add_slide(BLANK); bg(s)
rule(s, 0.9, 2.1, 1.6, ORANGE, 0.08)
text(s, 0.9, 2.35, 11.5, 1.1, "CompoCLIP", size=54, bold=True)
text(s, 0.9, 3.55, 11.5, 0.7, "组合性视觉-语言表征：机制、基线与失败诊断", size=24, color=CREAM)
text(s, 0.9, 4.6, 11, 0.5, "CS5489 机器学习 · 课程项目 · 成员 1", size=15, color=ORANGE)
text(s, 0.9, 5.1, 11, 0.5, "2026 年 11 月", size=12, color=GREY)
footer(s, 1)

# 2 一页结论
s = prs.slides.add_slide(BLANK); bg(s)
tag(s, "TL;DR")
text(s, 0.7, 0.85, 12, 0.8, "三句话结论", size=32, bold=True)
rule(s, 0.7, 1.65, 2.2)
bullets(s, 0.7, 2.0, 11.9, 4.6, [
    f"CLIP 的组合性失败是“不会读”还是“没有信息”？Winoground 上零样本 group {pct(wg.get('wg_group'))}，"
    f"线性探针也只有 {pct(a0.get('linear_probe_group'))}——自然最小对立对上，信息本身不可线性解码。",
    f"词序盲区可量化：保持词袋的主宾/名词/关系变换只有 {core_lo}–{core_hi}，而无关句对照 {pct(probe.get('unrelated',{}).get('accuracy'))}。",
    f"SugarCrepe++ 的错误 58%–69% 是 margin<0.01 的近失手——校准/适配器有空间；Winoground 则是自信地错，必须改训练目标。",
], size=18, gap=18)
footer(s, 2)

# 3 问题
s = prs.slides.add_slide(BLANK); bg(s)
tag(s, "MOTIVATION")
text(s, 0.7, 0.85, 12, 0.8, "CLIP 看起来懂图，其实像词袋", size=30, bold=True)
rule(s, 0.7, 1.65, 2.2)
bullets(s, 0.7, 2.05, 6.1, 4.4, [
    "对比目标只约束“整句—整图”，不约束词序/句法角色。",
    "Winoground：两张图只差关系或属性，词袋几乎相同。",
    "ARO：VG-Relation 仅 50.1%，几乎随机；属性/词序也弱。",
    "SugarCrepe++：swap 比 replace 难得多，说明“换结构”才是核心。",
], size=15)
# 右侧数字卡
cards = [
    (f"Winoground group", pct(wg.get("wg_group"))),
    (f"ARO 宏平均（3 子集）", pct(aro_macro.get("accuracy"))),
    (f"SugarCrepe++ swap", pct(CSV.get(("sugarcrepe","swap"),{}).get("accuracy"))),
]
for i, (k, v) in enumerate(cards):
    y = 2.1 + i * 1.35
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.2), Inches(y), Inches(5.3), Inches(1.05))
    shp.fill.solid(); shp.fill.fore_color.rgb = NAVY2; shp.line.color.rgb = ORANGE; shp.line.width = Pt(1)
    text(s, 7.45, y + 0.12, 3.2, 0.35, k, size=11, color=GREY)
    text(s, 7.45, y + 0.4, 3.0, 0.5, v, size=26, bold=True, color=ORANGE)
footer(s, 3)

# 4 方法
s = prs.slides.add_slide(BLANK); bg(s)
tag(s, "METHOD")
text(s, 0.7, 0.85, 12, 0.8, "四个实验，一条证据链", size=30, bold=True)
rule(s, 0.7, 1.65, 2.2)
items = [
    ("0-B", "词序探针", "5 类保持词袋的变换\n+ unrelated 对照\n→ 敏感度与 margin"),
    ("0-A", "线性探针", "冻结特征 + 逻辑回归\n5 折 GroupKFold\n→ 信息在不在"),
    ("0-C", "失败分类", "order/attr/rel/obj\n近失手 vs 自信错误\n→ 错在哪一层"),
    ("E0", "统一基线", "Winoground / ARO /\nSugarCrepe++\n→ 可复现数字"),
]
for i, (k, t1, t2) in enumerate(items):
    x = 0.7 + i * 3.15
    shp = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(2.1), Inches(2.85), Inches(4.2))
    shp.fill.solid(); shp.fill.fore_color.rgb = NAVY2; shp.line.fill.background()
    rule(s, x, 2.1, 2.85, ORANGE, 0.07)
    text(s, x + 0.22, 2.35, 2.4, 0.5, k, size=24, bold=True, color=ORANGE)
    text(s, x + 0.22, 2.95, 2.4, 0.4, t1, size=15, bold=True)
    text(s, x + 0.22, 3.45, 2.45, 2.2, t2, size=12, color=GREY, spacing=1.25)
text(s, 0.7, 6.55, 12, 0.35, "证据链：0-B 敏感度 → 0-A 信息是否存在 → 0-C 错在哪层 → E0 可复现基线",
     size=12, color=ORANGE)
footer(s, 4)

# 5 0-B
s = prs.slides.add_slide(BLANK); bg(s)
tag(s, "0-B · WORD ORDER")
text(s, 0.7, 0.85, 12, 0.8, "只改词序，CLIP 就掉到抛硬币", size=30, bold=True)
rule(s, 0.7, 1.65, 2.2)
if (FIG / "fig_0b.png").exists():
    s.shapes.add_picture(str(FIG / "fig_0b.png"), Inches(0.7), Inches(1.95), width=Inches(7.6))
bullets(s, 8.6, 2.0, 4.2, 4.5, [
    f"无关句对照 {pct(probe.get('unrelated',{}).get('accuracy'))}：管线正常。",
    f"主宾交换 {pct(probe.get('subj_obj_swap',{}).get('accuracy'))}，名词交换 {pct(probe.get('noun_swap',{}).get('accuracy'))}。",
    f"全词打乱反而 {pct(probe.get('word_shuffle',{}).get('accuracy'))}：模型只看“词在不在”。",
    "所有句法变换 margin ≈ 0.005，几乎没有决策裕度。",
], size=13, gap=12)
footer(s, 5)

# 6 0-A
s = prs.slides.add_slide(BLANK); bg(s)
tag(s, "0-A · LINEAR PROBE")
text(s, 0.7, 0.85, 12, 0.8, "自然最小对立对上，信息读不出来", size=30, bold=True)
rule(s, 0.7, 1.65, 2.2)
bullets(s, 0.7, 2.1, 6.4, 4.4, [
    "Winoground 两句都是人写、词袋相同、语法都通；线性探针无法靠“流畅度”作弊。",
    f"零样本 group {pct(a0.get('zero_shot_group'))}，线性探针 {pct(a0.get('linear_probe_group'))}（二分类随机=50%）。",
    "对照：Flickr 合成扰动上探针 75%–100%，但那是流畅度捷径。",
    "结论：不是换一个 adapter 就能解决，必须改训练目标。",
], size=15)
# 大数字
shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.6), Inches(2.2), Inches(4.9), Inches(2.4))
shp.fill.solid(); shp.fill.fore_color.rgb = NAVY2; shp.line.color.rgb = ORANGE; shp.line.width = Pt(1)
text(s, 7.95, 2.45, 4.3, 0.4, "Winoground group accuracy", size=12, color=GREY)
text(s, 7.95, 2.85, 2.1, 0.9, pct(a0.get("zero_shot_group")), size=40, bold=True, color=GREY)
text(s, 9.9, 2.85, 2.2, 0.9, pct(a0.get("linear_probe_group")), size=40, bold=True, color=ORANGE)
text(s, 7.95, 3.85, 2.1, 0.35, "零样本", size=12, color=GREY)
text(s, 9.9, 3.85, 2.3, 0.35, "线性探针", size=12, color=ORANGE)
footer(s, 6)

# 7 E0
s = prs.slides.add_slide(BLANK); bg(s)
tag(s, "E0 · BASELINE")
text(s, 0.7, 0.85, 12, 0.8, "统一基线：三类基准，同一套协议", size=30, bold=True)
rule(s, 0.7, 1.65, 2.2)
if (FIG / "fig_e0.png").exists():
    s.shapes.add_picture(str(FIG / "fig_e0.png"), Inches(0.55), Inches(1.9), width=Inches(12.2))
text(s, 0.7, 6.35, 12, 0.4, "Winoground group 随机=16.7%；其余二选一随机=50%。ARO 为 3/4 子集口径（COCO-Order 缺原图）。",
     size=11, color=GREY)
footer(s, 7)

# 8 0-C
s = prs.slides.add_slide(BLANK); bg(s)
tag(s, "0-C · ERROR TAXONOMY")
text(s, 0.7, 0.85, 12, 0.8, "错在哪里：关系最难，且多为近失手", size=30, bold=True)
rule(s, 0.7, 1.65, 2.2)
if (FIG / "fig_0c.png").exists():
    s.shapes.add_picture(str(FIG / "fig_0c.png"), Inches(0.55), Inches(1.9), width=Inches(7.3))
bullets(s, 8.2, 2.0, 4.6, 4.5, [
    "SugarCrepe++：relation 错误 30.3%，attribute 24.3%，object 10.4%。",
    "SugarCrepe relation 错误中 69% 的 margin 在 0.01 以内——决策边界偏了。",
    "Winoground relation group 仅 4.7%，image 侧远差于 text 侧。",
    "ARO VG-Relation 49.9% = 近随机；SugarCrepe 可校准，Winoground 必须重训练。",
], size=13, gap=11)
footer(s, 8)

# 9 机制结论
s = prs.slides.add_slide(BLANK); bg(s)
tag(s, "INSIGHT")
text(s, 0.7, 0.85, 12, 0.8, "机制结论与对方法的启示", size=30, bold=True)
rule(s, 0.7, 1.65, 2.2)
items = [
    ("01", "信息不在", "自然最小对立对上线性探针=50%，仅加 adapter 上限有限。"),
    ("02", "合成探针会骗人", "打乱句不自然，线性探针可学“哪句像人话”；必须用自然负例。"),
    ("03", "硬负例有必要", "同词袋不同结构的负例，迫使模型分配维度给词序与关系。"),
    ("04", "蒸馏抗遗忘", "教师信息不超过自身表征，E3 目标是“组合↑、检索不掉”。"),
]
for i, (n, h, d) in enumerate(items):
    y = 1.95 + i * 1.18
    text(s, 0.75, y, 0.8, 0.5, n, size=22, bold=True, color=ORANGE)
    text(s, 1.6, y - 0.02, 3.0, 0.4, h, size=16, bold=True)
    text(s, 4.7, y - 0.02, 7.9, 0.7, d, size=13, color=GREY)
footer(s, 9)

# 10 对 E1-E4 的建议
s = prs.slides.add_slide(BLANK); bg(s)
tag(s, "FOR THE TEAM")
text(s, 0.7, 0.85, 12, 0.8, "给 E1–E4 的四条验收建议", size=30, bold=True)
rule(s, 0.7, 1.65, 2.2)
bullets(s, 0.7, 2.0, 11.9, 4.7, [
    "主指标：Winoground group + ARO 宏平均；SugarCrepe++ macro 辅助，不能只看 replace。",
    "E3 同时看组合增益与 Flickr30k R@1 代价；R@1 掉 >2pt 则“抗遗忘”不成立。",
    "E4b 锁图像塔的预期是“组合略降、通用保住”；若组合也降，瓶颈在文本侧。",
    "所有组必须报告近失手比例：区分“真的学会”与“边界挪了一点”。",
], size=17, gap=16)
footer(s, 10)

# 11 交付清单
s = prs.slides.add_slide(BLANK); bg(s)
tag(s, "DELIVERABLES")
text(s, 0.7, 0.85, 12, 0.8, "成员 1 交付清单", size=30, bold=True)
rule(s, 0.7, 1.65, 2.2)
bullets(s, 0.7, 2.0, 6.4, 4.8, [
    "0-A / 0-B / 0-C 脚本 + 逐样本结果",
    "E0：Winoground / ARO（3 子集）/ SugarCrepe++",
    "报告第 1–4 章 + 统稿模板 + 参考文献",
    "硬负例产线：限流/续跑/词袋校验/100 条质检表",
], size=14, gap=10)
text(s, 7.3, 2.0, 5.4, 0.35, "复现命令", size=12, color=ORANGE, bold=True)
code = ("python scripts/prepare_winoground.py\n"
        "python scripts/prepare_aro.py --include-flickr-order\n"
        "python scripts/prepare_sugarcrepepp.py\n"
        "python evals-mm/run_eval.py --tag e0 --benchmark aro\n"
        "python scripts/run_0b_probe.py --limit 5000\n"
        "python scripts/run_0a_probe.py --benchmark winoground\n"
        "python scripts/diagnose_failures.py\n"
        "python scripts/build_report.py")
text(s, 7.3, 2.45, 5.4, 3.8, code, size=9.5, color=CREAM, font="Consolas", spacing=1.25)
footer(s, 11)

# 12 Q&A
s = prs.slides.add_slide(BLANK); bg(s)
rule(s, 0.9, 2.6, 1.6, ORANGE, 0.08)
text(s, 0.9, 2.85, 11, 0.9, "Q & A", size=44, bold=True)
text(s, 0.9, 3.9, 11, 0.5, "谢谢 · 欢迎指出数据与口径问题", size=16, color=GREY)
footer(s, 12)

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print(f"PPT -> {OUT}")
