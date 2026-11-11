"""把 report/*.md 合成 DOCX，并用本机 Word 导出 PDF。

用法：
    python scripts/build_report.py
    python scripts/build_report.py --no-pdf

约定：
- 每个 `# ` 一级标题另起一页；
- `##`/`###` 映射为标题 2/3；
- `- ` / `1. ` 映射为列表；
- `| ... |` 连续行映射为表格；
- `![caption](path)` 插入图片（宽度 15cm）。
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
CHAPTERS = [
    "01_intro.md",
    "02_related_work.md",
    "03_mechanism.md",
    "04_diagnosis.md",
    "05_data_training.md",
    "06_method.md",
    "07_conclusion.md",
    "08_appendix.md",
]


def set_cjk(run, font="宋体", size=None, bold=None):
    run.font.name = font
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold


def add_runs(paragraph, text: str, size=10.5, font="宋体"):
    """支持 **bold** 的最小内联解析。"""
    parts = re.split(r"(\*\*.+?\*\*)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = paragraph.add_run(part[2:-2])
            set_cjk(r, font, size, True)
        else:
            r = paragraph.add_run(part)
            set_cjk(r, font, size)


def add_table(doc: Document, rows: list[list[str]]):
    if not rows:
        return
    ncols = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=ncols)
    table.style = "Table Grid"
    for i, row in enumerate(rows):
        for j in range(ncols):
            cell = table.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            add_runs(p, row[j] if j < len(row) else "", size=8.5)
            if i == 0:
                for run in p.runs:
                    run.bold = True


def add_markdown(doc: Document, text: str, chapter_break: bool = True):
    lines = text.splitlines()
    i = 0
    first_h1 = True
    num = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            num = 0
            i += 1
            continue
        if line.startswith("```"):
            i += 1
            code = []
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.5)
            for c in code:
                add_runs(p, c, size=8.5, font="Consolas")
                p.add_run().add_break()
            i += 1
            continue
        m = re.match(r"^(#{1,3})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            if level == 1 and chapter_break:
                if not first_h1:
                    doc.add_page_break()
                first_h1 = False
            p = doc.add_heading(level=min(level, 3))
            add_runs(p, m.group(2), size={1: 16, 2: 13, 3: 11}[level],
                     font="黑体")
            for r in p.runs:
                r.bold = True
                r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
            i += 1
            continue
        # 图片
        m = re.match(r"^!\[(.*?)\]\((.+?)\)\s*$", line)
        if m:
            path = (REPORT / m.group(2)) if not Path(m.group(2)).is_absolute() else Path(m.group(2))
            if path.exists():
                doc.add_picture(str(path), width=Cm(15))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                cap = doc.add_paragraph()
                cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                add_runs(cap, m.group(1), size=9)
            else:
                add_runs(doc.add_paragraph(), f"[缺失图片：{path}]")
            i += 1
            continue
        # 表格
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            add_table(doc, rows)
            doc.add_paragraph()
            continue
        if re.match(r"^[-*]\s+", line):
            num = 0
            p = doc.add_paragraph(style="List Bullet")
            add_runs(p, re.sub(r"^[-*]\s+", "", line))
            i += 1
            continue
        if re.match(r"^\d+\.\s+", line):
            num += 1
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.74)
            p.paragraph_format.first_line_indent = Cm(-0.35)
            p.paragraph_format.space_after = Pt(3)
            add_runs(p, f"{num}. " + re.sub(r"^\d+\.\s+", "", line))
            i += 1
            continue
        if line.startswith(">"):
            num = 0
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.6)
            add_runs(p, line.lstrip("> ").strip(), size=9.5, font="楷体")
            for r in p.runs:
                r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
            i += 1
            continue
        num = 0
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0.74)
        p.paragraph_format.space_after = Pt(3)
        add_runs(p, line)
        i += 1


def add_page_numbers(doc: Document):
    from docx.oxml import OxmlElement
    footer = doc.sections[0].footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fld_begin = OxmlElement("w:fldChar"); fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve"); instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar"); fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin); run._r.append(instr); run._r.append(fld_end)
    set_cjk(run, "宋体", 9)


def build(chapters: list[str] | None = None, suffix: str = "") -> Path:
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = sec.bottom_margin = Cm(2.2)
    sec.left_margin = sec.right_margin = Cm(2.4)

    style = doc.styles["Normal"]
    style.font.name = "宋体"
    style.font.size = Pt(10.5)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    style.paragraph_format.line_spacing = 1.25

    # 封面
    for _ in range(3):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_runs(p, "CompoCLIP", size=28, font="黑体")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_runs(p, "组合性视觉-语言表征的机制分析与轻量适配", size=18, font="黑体")
    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_runs(p, "CS5489 机器学习 课程项目", size=12)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_runs(p, "成员 1：机制分析、E0 基线、失败诊断与统稿", size=11)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_runs(p, "2026 年 11 月", size=11)
    doc.add_page_break()

    for name in (chapters or CHAPTERS):
        path = REPORT / name
        if path.exists():
            add_markdown(doc, path.read_text(encoding="utf-8"))
        else:
            doc.add_paragraph(f"[待成员 2/3 提供：{name}]")

    out = REPORT / f"CompoCLIP_Report{suffix}.docx"
    add_page_numbers(doc)
    doc.save(out)
    return out


def to_pdf(docx_path: Path) -> Path | None:
    docx_path = docx_path.resolve()
    try:
        import win32com.client as win32
        word = win32.Dispatch("Word.Application")
        word.Visible = False
        doc = word.Documents.Open(str(docx_path))
        pdf = docx_path.with_suffix(".pdf")
        doc.SaveAs(str(pdf), FileFormat=17)  # wdFormatPDF
        doc.Close(False)
        word.Quit()
        return pdf
    except Exception as e:
        print(f"PDF 导出失败（DOCX 仍可用）：{e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--member1", action="store_true", help="只合成成员 1 负责的第 1–4 章 + 附录")
    args = parser.parse_args()
    chapters = [c for c in CHAPTERS if c[:2] in ("01", "02", "03", "04", "08")] if args.member1 else None
    out = build(chapters, "_成员1分册" if args.member1 else "")
    print(f"DOCX -> {out}")
    if not args.no_pdf:
        pdf = to_pdf(out)
        if pdf:
            print(f"PDF  -> {pdf}")


if __name__ == "__main__":
    main()
