"""Build the standalone overall report from report/00_final_report.md."""
from __future__ import annotations

from build_report import build, to_pdf


def main() -> None:
    out = build(["00_final_report.md"], "_总报告", subtitle="三人小组项目总报告")
    print(f"DOCX -> {out}")
    pdf = to_pdf(out)
    if pdf:
        print(f"PDF  -> {pdf}")


if __name__ == "__main__":
    main()
