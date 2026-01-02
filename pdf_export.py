import json
from io import BytesIO
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth

def _wrap_text(text: str, max_width: float, font_name: str, font_size: float) -> list[str]:
    lines = []
    for para in (text or "").split("\n"):
        words = para.split(" ")
        cur = ""
        for w in words:
            test = (cur + " " + w).strip()
            if stringWidth(test, font_name, font_size) <= max_width:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        lines.append("")  # paragraph spacing
    while lines and lines[-1] == "":
        lines.pop()
    return lines

def _make_overlay_pdf(page_w: float, page_h: float, objects: list[dict], export_scale: float) -> BytesIO:
    """
    objects: Fabric.js objects
    export_scale: 前端 canvas px 對應到 PDF points 的比例（points_per_px）
    """
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(page_w, page_h))

    font_name = "Helvetica"

    for obj in objects:
        if obj.get("type") not in ("textbox", "text"):
            continue
        if obj.get("visible") is False:
            continue
        if obj.get("deleted") is True:
            continue

        text = obj.get("text", "")
        left_px = float(obj.get("left", 0))
        top_px = float(obj.get("top", 0))
        width_px = float(obj.get("width", 200))
        font_size_px = float(obj.get("fontSize", 12))
        line_height = float(obj.get("lineHeight", 1.2))

        # Fabric 좌標：左上為原點；PDF/ReportLab：左下為原點
        x = left_px * export_scale
        y_top = top_px * export_scale
        box_w = width_px * export_scale
        font_size = font_size_px * export_scale

        # 轉成 PDF y（以左下為原點）
        y = page_h - y_top - font_size

        c.setFont(font_name, max(4, font_size))

        lines = _wrap_text(text, max_width=max(10, box_w), font_name=font_name, font_size=max(4, font_size))
        dy = max(4, font_size) * line_height
        yy = y
        for line in lines:
            c.drawString(x, yy, line)
            yy -= dy

    c.showPage()
    c.save()
    buf.seek(0)
    return buf

def export_translated_pdf(original_pdf_path: str, annotations_by_page: dict[int, dict], out_pdf_path: str):
    """
    annotations_by_page[page_no] = {
      "canvasWidth": ...px,
      "canvasHeight": ...px,
      "objects": [fabric objects...]
    }
    """
    reader = PdfReader(original_pdf_path)
    writer = PdfWriter()

    for idx, page in enumerate(reader.pages):
        page_no = idx + 1
        page_w = float(page.mediabox.width)
        page_h = float(page.mediabox.height)

        ann = annotations_by_page.get(page_no)
        if not ann:
            writer.add_page(page)
            continue

        canvas_w_px = float(ann.get("canvasWidth", 1))
        # points_per_px：PDF points / canvas px
        export_scale = page_w / max(1.0, canvas_w_px)

        overlay = _make_overlay_pdf(
            page_w=page_w,
            page_h=page_h,
            objects=ann.get("objects", []),
            export_scale=export_scale
        )
        overlay_reader = PdfReader(overlay)
        page.merge_page(overlay_reader.pages[0])
        writer.add_page(page)

    Path(out_pdf_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_pdf_path, "wb") as f:
        writer.write(f)
