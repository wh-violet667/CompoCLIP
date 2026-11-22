"""Generate the role-3 result deck (16:9) from the final tables."""
from __future__ import annotations

import csv
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "report" / "results"
FIG = ROOT / "report" / "figures"
OUT = ROOT / "report" / "CompoCLIP_成员3汇报.pptx"

NAVY = RGBColor(0x0B, 0x1E, 0x2D)
NAVY2 = RGBColor(0x12, 0x2E, 0x42)
ORANGE = RGBColor(0xE0, 0x8A, 0x3C)
CREAM = RGBColor(0xF3, 0xEF, 0xE6)
GREY = RGBColor(0x9A, 0xA6, 0xB2)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREEN = RGBColor(0x6F, 0xB7, 0x8E)
RED = RGBColor(0xD9, 0x6B, 0x6B)
LABELS = {"e0": "E0", "e1": "E1", "e2": "E2", "e2a": "E2a", "e3": "E3", "e4b": "E4b"}

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def bg(slide, color=NAVY):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def text(slide, value, left, top, width, height, size=18, color=CREAM,
         bold=False, align=PP_ALIGN.LEFT, font="Microsoft YaHei"):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = value
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font
    return box


def title(slide, value, sub=None):
    text(slide, value, 0.65, 0.42, 12.0, 0.8, 30, CREAM, True)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.65), Inches(1.22),
                                  Inches(2.0), Inches(0.06))
    line.fill.solid()
    line.fill.fore_color.rgb = ORANGE
    line.line.fill.background()
    if sub:
        text(slide, sub, 0.65, 1.35, 12.0, 0.45, 14, GREY)


def bullets(slide, items, left, top, width, height, size=17, color=CREAM):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(8)
        run = p.add_run()
        run.text = f"• {item}"
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.font.name = "Microsoft YaHei"


def table(slide, rows, left, top, width, height, col_widths, font_size=13):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(left), Inches(top),
                                   Inches(width), Inches(height))
    tbl = shape.table
    for i, w in enumerate(col_widths):
        tbl.columns[i].width = Inches(w)
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = str(value)
            cell.margin_left = Inches(0.06)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY2 if r % 2 == 0 else NAVY
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if c else PP_ALIGN.LEFT
            for run in p.runs:
                run.font.size = Pt(font_size)
                run.font.color.rgb = ORANGE if r == 0 else CREAM
                run.font.bold = r == 0
                run.font.name = "Microsoft YaHei"


def footer(slide, idx):
    text(slide, "CompoCLIP · Role 3", 0.65, 7.08, 4.0, 0.25, 10, GREY)
    text(slide, f"{idx:02d}", 12.35, 7.08, 0.45, 0.25, 10, GREY, align=PP_ALIGN.RIGHT)


def read_full():
    with (RES / "full_4of3.csv").open(encoding="utf-8") as f:
        return {r["tag"]: r for r in csv.DictReader(f)}


def read_master():
    with (RES / "master.csv").open(encoding="utf-8") as f:
        return {(r["tag"], r["benchmark"], r["subset"], r["metric"]): r["value"]
                for r in csv.DictReader(f)}


def pct(v):
    return f"{float(v) * 100:.3f}%" if v not in ("", None) else "—"


def main():
    full = read_full()
    master = read_master()

    # 1 cover
    s = prs.slides.add_slide(BLANK)
    bg(s)
    text(s, "CompoCLIP", 0.9, 2.0, 8.0, 0.8, 44, CREAM, True)
    text(s, "角色 3：统一评测、配对统计与失败分析", 0.9, 2.95, 10.5, 0.7, 26, ORANGE, True)
    text(s, "E0–E4b · A2 4/3 + Flickr30k/Winoground · Tesla T4", 0.9, 3.75, 10.5, 0.5, 16, GREY)
    text(s, "2026-11-23", 0.9, 6.4, 4.0, 0.4, 13, GREY)

    # 2 protocol
    s = prs.slides.add_slide(BLANK)
    bg(s)
    title(s, "评测口径：两套表，不混算", "同一批模型、同一套 harness；差别在 ARO/Sugar 子集范围")
    bullets(s, [
        "共同 2/2：ARO = VG-Relation + VG-Attribution；Sugar = replace + swap。",
        "A2 4/3：ARO = 四子集（含 COCO-Order / Flickr-Order）；Sugar = replace/swap/add。",
        "E0 也重跑 A2 4/3，因此 E3-vs-E0 是严格配对比较，不再用早期 A1 归档。",
        "检索：Flickr30k test（1k 图 / 5k caption）+ Winoground 400 对。",
        "所有配对统计用相同样本顺序；bootstrap 1000 次 + McNemar。",
    ], 0.8, 1.75, 11.8, 4.8, 18)
    footer(s, 2)

    # 3 main table
    s = prs.slides.add_slide(BLANK)
    bg(s)
    title(s, "A2 4/3 主结果：完整 6 模型", "组合均值 = ARO macro 与 Sugar macro 的平均")
    rows = [["模型", "ARO 4子集", "Sugar 3家族", "组合均值", "COCO t2i"]]
    for tag in ("e0", "e1", "e2", "e2a", "e3", "e4b"):
        r = full[tag]
        rows.append([LABELS[tag],
                     pct(r["aro4_macro"]), pct(r["sugar3_macro"]),
                     pct(r["composite"]), pct(r["coco_t2i_R@1"])])
    table(s, rows, 1.2, 1.65, 10.9, 4.2, [1.5, 2.2, 2.2, 2.2, 2.2], 14)
    text(s, "E2 组合均值 62.568% ≈ B 的 selection score 62.5683%，口径校验通过。",
         1.2, 6.05, 10.8, 0.4, 14, GREY)
    footer(s, 3)

    # 4 retrieval
    s = prs.slides.add_slide(BLANK)
    bg(s)
    title(s, "检索与 Winoground", "Flickr30k 1k 图 / 5k caption；Winoground 400 对")
    rows = [["模型", "Flickr t2i", "Flickr i2t", "WG group", "COCO t2i"]]
    for tag in ("e0", "e1", "e2", "e2a", "e3", "e4b"):
        rows.append([
            LABELS[tag],
            pct(master.get((tag, "flickr30k", "", "t2i_R@1"))),
            pct(master.get((tag, "flickr30k", "", "i2t_R@1"))),
            pct(master.get((tag, "winoground", "", "wg_group"))),
            pct(full[tag]["coco_t2i_R@1"]),
        ])
    table(s, rows, 1.0, 1.65, 11.3, 3.9, [1.5, 2.4, 2.4, 2.4, 2.6], 14)
    text(s, "E4b 的 i2t 最高（88.50%），但 t2i 比 E2 低 1.54pp；Winoground 全部落在 7.50–8.25%，400 对 CI 约 ±3–4pp。",
         1.0, 5.85, 11.2, 0.6, 14, GREY)
    footer(s, 4)

    # 5 key finding
    s = prs.slides.add_slide(BLANK)
    bg(s)
    title(s, "E3：预注册主标准成立", "但在 4/3 上，收益集中在 order 子集")
    rows = [
        ["指标", "E0", "E3", "E3 − E0"],
        ["ARO 4子集", "45.486%", "48.195%", "+2.709pp"],
        ["Sugar 3家族", "77.539%", "79.404%", "+1.865pp"],
        ["组合均值", "61.512%", "63.799%", "+2.287pp"],
        ["COCO t2i", "38.794%", "41.029%", "+2.235pp"],
    ]
    table(s, rows, 0.85, 1.65, 6.5, 3.2, [1.6, 1.5, 1.5, 1.9], 14)
    bullets(s, [
        "ARO 与 Sugar 的全量配对 CI 都排除 0。",
        "COCO 不降反升，遗忘约束满足。",
        "但 E3 的 VG-Relation / VG-Attribution 相对 E0 基本持平。",
        "相对 E2：组合 +1.231pp，Sugar −0.910pp（CI 跨 0）。",
    ], 7.7, 1.65, 4.9, 4.6, 16)
    footer(s, 5)

    # 6 figure
    s = prs.slides.add_slide(BLANK)
    bg(s)
    title(s, "收益与代价：order 涨、VG 平/降", "左：子集差；右：汇总差与 COCO")
    s.shapes.add_picture(str(FIG / "fig_role3_delta.png"), Inches(0.7), Inches(1.45), width=Inches(11.9))
    footer(s, 6)

    # 7 E4b
    s = prs.slides.add_slide(BLANK)
    bg(s)
    title(s, "E4b：与 E2 接近，但没有稳定优势", "锁图像塔的组合均值 +0.242pp，检索互有胜负")
    rows = [
        ["对比", "组合", "ARO 4子集", "Sugar 3家族"],
        ["E4b 绝对值", "62.810%", "46.235%", "79.385%"],
        ["E4b − E2", "+0.242pp", "+1.412pp", "−0.929pp"],
        ["E4b − E0", "+1.298pp", "+0.749pp", "+1.846pp"],
    ]
    table(s, rows, 0.85, 1.65, 6.8, 2.6, [2.0, 1.6, 1.6, 1.6], 14)
    bullets(s, [
        "共同 2/2 上 E4b 比 E2 低：ARO −0.450pp、Sugar −1.085pp。",
        "Flickr30k t2i 71.44%（比 E2 低 1.54pp），i2t 88.50%（高 1.00pp）。",
        "Winoground group 7.75% vs E2 8.00%，差异在噪声内。",
        "不能宣称锁图像塔更优；训练成本优势另算。",
    ], 7.8, 1.65, 4.8, 4.6, 16)
    footer(s, 7)

    # 8 failures
    s = prs.slides.add_slide(BLANK)
    bg(s)
    title(s, "失败分析：30 条最接近决策边界的错误", "E3 · ARO 15 + SugarCrepe 15")
    bullets(s, [
        "ARO 关系反转：the cat is behind the wheel ↔ the wheel is behind the cat。",
        "ARO 属性错位：white tail / concrete ground ↔ concrete tail / white ground。",
        "ARO order：COCO/Flickr 短语顺序被打乱后仍被判为更匹配。",
        "Sugar：replace 近义替换、swap 对象/属性互换、add 额外对象插入。",
        "这些是边界样本，不是错误比例估计；完整表见 failure_cases.md。",
    ], 0.85, 1.75, 11.8, 4.8, 18)
    footer(s, 7)

    # 9 limits
    s = prs.slides.add_slide(BLANK)
    bg(s)
    title(s, "结论与限制", "可交付的结论要带口径；缺口不填猜测值")
    bullets(s, [
        "主结论：E3 在 A2 4/3 上满足预注册 ≥2.0pp 与 COCO 不降；机制是 order 子集收益 + VG 代价。",
        "E2 仍是 E-best；E3 vs E2 组合不降，但 Sugar 单项略低。",
        "E4b 与 E2 接近，没有稳定优势；检索 t2i/i2t 互有胜负。",
        "缺口：E2/E1/E2a 的 COCO、官方 Flickr/Winoground 复核、第二训练 seed。",
        "交付：report/06–07、full_4of3、paired stats、failure cases、demo。",
    ], 0.85, 1.75, 11.8, 4.9, 17)
    footer(s, 9)

    prs.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
