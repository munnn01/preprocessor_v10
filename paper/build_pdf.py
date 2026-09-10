"""Render the Markdown research manuscript to a polished, reviewable PDF."""

from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "manuscript.md"
OUTPUT = ROOT / "manuscript.pdf"
NAVY = colors.HexColor("#17365D")
BLUE = colors.HexColor("#2F75B5")
PALE_BLUE = colors.HexColor("#EAF2F8")
RED = colors.HexColor("#9C0006")
PALE_RED = colors.HexColor("#FCE8E6")
GRAY = colors.HexColor("#5A6570")


def register_fonts() -> tuple[str, str, str]:
    font_dir = Path(r"C:\Windows\Fonts")
    candidates = (
        ("ArialV9", "arial.ttf", "ArialV9Bold", "arialbd.ttf", "ArialV9Italic", "ariali.ttf"),
        ("DejaVuV9", "DejaVuSans.ttf", "DejaVuV9Bold", "DejaVuSans-Bold.ttf", "DejaVuV9Italic", "DejaVuSans-Oblique.ttf"),
    )
    for regular, regular_file, bold, bold_file, italic, italic_file in candidates:
        paths = [font_dir / regular_file, font_dir / bold_file, font_dir / italic_file]
        if all(path.is_file() for path in paths):
            pdfmetrics.registerFont(TTFont(regular, paths[0]))
            pdfmetrics.registerFont(TTFont(bold, paths[1]))
            pdfmetrics.registerFont(TTFont(italic, paths[2]))
            pdfmetrics.registerFontFamily(regular, normal=regular, bold=bold, italic=italic)
            return regular, bold, italic
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


FONT, FONT_BOLD, FONT_ITALIC = register_fonts()


def inline_markup(value: str) -> str:
    value = escape(value.strip())
    value = re.sub(r"\[([^]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#2F75B5">\1</link>', value)
    value = re.sub(r"&lt;(https?://[^&]+)&gt;", r'<link href="\1" color="#2F75B5">\1</link>', value)
    value = re.sub(r"`([^`]+)`", rf'<font name="{FONT}">\1</font>', value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    return value


def pipeline_figure() -> KeepTogether:
    width, height = 172 * mm, 42 * mm
    drawing = Drawing(width, height)
    labels = (
        ("Source\nvideo", 0, colors.HexColor("#F3F6F9")),
        ("QP-FiLM\nVideo Swin Lite", 31, colors.HexColor("#D9EAF7")),
        ("Frozen H.264 / H.265\nreal forward", 72, colors.HexColor("#E2F0D9")),
        ("QP-FiLM 3-D\npostprocessor", 117, colors.HexColor("#FFF2CC")),
        ("Human / machine\nconsumer", 150, colors.HexColor("#FCE4D6")),
    )
    box_y, box_h = 14 * mm, 18 * mm
    widths = (25, 35, 39, 28, 22)
    for index, (label, x_mm, fill) in enumerate(labels):
        x, box_w = x_mm * mm, widths[index] * mm
        drawing.add(Rect(x, box_y, box_w, box_h, rx=3, ry=3, fillColor=fill, strokeColor=NAVY, strokeWidth=0.8))
        parts = label.split("\n")
        for line_index, part in enumerate(parts):
            drawing.add(String(x + box_w / 2, box_y + (11 - 5 * line_index) * mm, part, textAnchor="middle", fontName=FONT_BOLD, fontSize=7.2, fillColor=NAVY))
        if index + 1 < len(labels):
            x2 = labels[index + 1][1] * mm
            drawing.add(Line(x + box_w, box_y + box_h / 2, x2, box_y + box_h / 2, strokeColor=BLUE, strokeWidth=1.2))
    drawing.add(String(width / 2, 6 * mm, "Training only: frozen predictive proxy supplies codec Jacobian; DINOv2 supplies semantic features.", textAnchor="middle", fontName=FONT_ITALIC, fontSize=7.4, fillColor=GRAY))
    caption = Paragraph("<b>Figure 1.</b> Standards-compatible deployment graph and training-only teachers.", STYLES["Caption"])
    return KeepTogether([drawing, caption, Spacer(1, 3 * mm)])


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="PaperTitle", parent=styles["Title"], fontName=FONT_BOLD, fontSize=18, leading=22, textColor=NAVY, alignment=TA_CENTER, spaceAfter=5 * mm))
    styles.add(ParagraphStyle(name="Author", parent=styles["Normal"], fontName=FONT, fontSize=9.5, leading=13, textColor=GRAY, alignment=TA_CENTER, spaceAfter=4 * mm))
    styles.add(ParagraphStyle(name="Status", parent=styles["Normal"], fontName=FONT_BOLD, fontSize=9, leading=12, textColor=RED, backColor=PALE_RED, borderColor=RED, borderWidth=0.8, borderPadding=6, alignment=TA_CENTER, spaceAfter=6 * mm))
    styles.add(ParagraphStyle(name="H1V9", parent=styles["Heading1"], fontName=FONT_BOLD, fontSize=12.5, leading=15, textColor=NAVY, spaceBefore=5 * mm, spaceAfter=2 * mm, keepWithNext=True))
    styles.add(ParagraphStyle(name="H2V9", parent=styles["Heading2"], fontName=FONT_BOLD, fontSize=10.3, leading=13, textColor=BLUE, spaceBefore=3 * mm, spaceAfter=1.5 * mm, keepWithNext=True))
    styles.add(ParagraphStyle(name="BodyV9", parent=styles["BodyText"], fontName=FONT, fontSize=8.7, leading=12, alignment=TA_JUSTIFY, textColor=colors.HexColor("#20262D"), spaceAfter=2.3 * mm))
    styles.add(ParagraphStyle(name="AbstractV9", parent=styles["BodyText"], fontName=FONT, fontSize=8.5, leading=11.5, alignment=TA_JUSTIFY, leftIndent=6 * mm, rightIndent=6 * mm, borderColor=BLUE, borderWidth=0.6, borderPadding=7, backColor=PALE_BLUE, spaceAfter=4 * mm))
    styles.add(ParagraphStyle(name="BulletV9", parent=styles["BodyText"], fontName=FONT, fontSize=8.5, leading=11.5, leftIndent=5 * mm, firstLineIndent=-3 * mm, alignment=TA_LEFT, spaceAfter=1.2 * mm))
    styles.add(ParagraphStyle(name="CodeV9", parent=styles["Code"], fontName="Courier", fontSize=7.1, leading=9.5, leftIndent=4 * mm, rightIndent=4 * mm, borderColor=colors.HexColor("#C8D1DA"), borderWidth=0.5, borderPadding=6, backColor=colors.HexColor("#F6F8FA"), spaceBefore=1 * mm, spaceAfter=3 * mm))
    styles.add(ParagraphStyle(name="Caption", parent=styles["Normal"], fontName=FONT, fontSize=7.5, leading=9.5, alignment=TA_CENTER, textColor=GRAY, spaceAfter=3 * mm))
    return styles


STYLES = make_styles()


def markdown_table(lines: list[str], doc_width: float) -> Table:
    rows = [[inline_markup(cell.strip()) for cell in line.strip().strip("|").split("|")] for line in lines]
    if len(rows) > 1 and all(set(cell.replace(" ", "")) <= {"-", ":"} for cell in rows[1]):
        rows.pop(1)
    count = max(len(row) for row in rows)
    data = []
    for row_index, row in enumerate(rows):
        row += [""] * (count - len(row))
        style = ParagraphStyle(
            name=f"TableCell{row_index}",
            parent=STYLES["BodyV9"],
            fontName=FONT_BOLD if row_index == 0 else FONT,
            fontSize=6.3 if count >= 6 else 7.1,
            leading=7.8 if count >= 6 else 8.8,
            alignment=TA_LEFT,
            textColor=colors.white if row_index == 0 else colors.HexColor("#20262D"),
            spaceAfter=0,
        )
        data.append([Paragraph(cell, style) for cell in row])
    table = Table(data, colWidths=[doc_width / count] * count, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8C2CC")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8FA")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def parse_markdown(doc_width: float):
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    title = lines[0].removeprefix("# ").strip()
    author = lines[2].strip()
    status = lines[4].strip()
    story = [Spacer(1, 8 * mm), Paragraph(inline_markup(title), STYLES["PaperTitle"]), Paragraph(inline_markup(author), STYLES["Author"]), Paragraph(inline_markup(status), STYLES["Status"])]
    index = 5
    paragraph: list[str] = []
    abstract_next = False
    inserted_figure = False

    def flush() -> None:
        nonlocal paragraph, abstract_next
        if not paragraph:
            return
        content = inline_markup(" ".join(part.strip() for part in paragraph))
        style = STYLES["AbstractV9"] if abstract_next else STYLES["BodyV9"]
        story.append(Paragraph(content, style))
        paragraph = []
        abstract_next = False

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            flush()
            index += 1
            continue
        if stripped.startswith("```"):
            flush()
            index += 1
            code: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(lines[index])
                index += 1
            story.append(Paragraph(escape("\n".join(code)).replace("\n", "<br/>"), STYLES["CodeV9"]))
            index += 1
            continue
        if stripped.startswith("|"):
            flush()
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            story.extend([markdown_table(table_lines, doc_width), Spacer(1, 3 * mm)])
            continue
        if stripped.startswith("## "):
            flush()
            heading = stripped[3:]
            if heading == "References":
                story.append(PageBreak())
            story.append(Paragraph(inline_markup(heading), STYLES["H1V9"]))
            abstract_next = heading == "Abstract"
            index += 1
            continue
        if stripped.startswith("### "):
            flush()
            story.append(Paragraph(inline_markup(stripped[4:]), STYLES["H2V9"]))
            index += 1
            continue
        if re.match(r"^(?:[-*]|\d+\.)\s+", stripped):
            flush()
            item = re.sub(r"^(?:[-*]|\d+\.)\s+", "", stripped)
            marker = "•" if stripped[0] in "-*" else stripped.split()[0]
            story.append(Paragraph(inline_markup(item), STYLES["BulletV9"], bulletText=marker))
            index += 1
            continue
        if stripped.startswith("Keywords—") and not inserted_figure:
            flush()
            story.append(Paragraph(f"<i>{inline_markup(stripped)}</i>", STYLES["BodyV9"]))
            story.append(pipeline_figure())
            inserted_figure = True
            index += 1
            continue
        paragraph.append(stripped)
        index += 1
    flush()
    return story


def decorate(canvas, document) -> None:
    canvas.saveState()
    page = canvas.getPageNumber()
    width, height = A4
    if page > 1:
        canvas.setStrokeColor(colors.HexColor("#D3DAE2"))
        canvas.line(18 * mm, height - 14 * mm, width - 18 * mm, height - 14 * mm)
        canvas.setFont(FONT, 7)
        canvas.setFillColor(GRAY)
        canvas.drawString(18 * mm, height - 11 * mm, "Adaptive Video Preprocessing for VCM on Jetson Orin NX")
        canvas.drawRightString(width - 18 * mm, height - 11 * mm, "PRE-RESULTS DRAFT")
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(GRAY)
    canvas.drawCentredString(width / 2, 11 * mm, f"{page}")
    canvas.restoreState()


def main() -> None:
    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title="Adaptive Video Preprocessing Techniques for Optimizing Video Coding for Machines (VCM) on NVIDIA Jetson Orin NX",
        author="Anonymous Author(s) — identities pending human approval",
        subject="Pre-results research manuscript",
    )
    document.build(parse_markdown(document.width), onFirstPage=decorate, onLaterPages=decorate)
    print(OUTPUT)


if __name__ == "__main__":
    main()
