# -*- coding: utf-8 -*-
"""
PaddleOCR JSON -> OpenAI 翻譯 -> 依 OCR 座標把翻譯放在原文旁 -> 輸出新 PDF

需求套件：
  pip install pymupdf reportlab openai

使用方式（範例）：
  python ocr_translate_overlay_pdf.py ^
    --input_pdf "UOC-PQR-16013_v2.pdf" ^
    --ocr_json "UOC-PQR-16013_v2_0_res.json" ^
    --output_pdf "UOC-PQR-16013_v2_bilingual.pdf" ^
    --target_lang "en" ^
    --model "gpt-4.1-mini" ^
    --dpi 150 ^
    --font_path "C:\\Windows\\Fonts\\msjh.ttc"

注意：
- OCR 座標是「影像座標(pixel, 左上原點)」，不是 PDF 原生座標。
- 本程式以 PyMuPDF 用指定 dpi 渲染 PDF 成圖片，然後把翻譯用 ReportLab 疊上去。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Tuple, Optional

import fitz  # PyMuPDF
from openai import OpenAI
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class OcrPage:
    page_index: int
    rec_texts: List[str]
    rec_boxes: List[List[int]]  # [x1,y1,x2,y2]


# ----------------------------
# OCR JSON loader
# ----------------------------

def load_ocr_pages(ocr_json_path: str) -> List[OcrPage]:
    """
    支援兩種格式：
    A) 單頁 dict（你目前的格式）：包含 page_index / rec_texts / rec_boxes
    B) 多頁 list：每個元素都是 A 的 dict
    """
    with open(ocr_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pages: List[Dict[str, Any]]
    if isinstance(data, list):
        pages = data
    elif isinstance(data, dict):
        pages = [data]
    else:
        raise ValueError("OCR JSON 格式不支援：必須是 dict 或 list[dict]")

    out: List[OcrPage] = []
    for p in pages:
        page_index = int(p.get("page_index", 0))
        rec_texts = p.get("rec_texts", []) or []
        rec_boxes = p.get("rec_boxes", []) or []
        if len(rec_texts) != len(rec_boxes):
            raise ValueError(
                f"page_index={page_index} 的 rec_texts 與 rec_boxes 長度不一致："
                f"{len(rec_texts)} vs {len(rec_boxes)}"
            )
        out.append(OcrPage(page_index=page_index, rec_texts=rec_texts, rec_boxes=rec_boxes))

    # 依 page_index 排序
    out.sort(key=lambda x: x.page_index)
    return out


def compute_ocr_extent(rec_boxes: List[List[int]]) -> Tuple[int, int]:
    """
    用 OCR 的 rec_boxes 估計 OCR 座標空間的最大範圍（max x2, max y2）。
    用來和實際渲染影像尺寸做比例校正，避免 dpi 不一致造成對不準。
    """
    if not rec_boxes:
        return 0, 0
    max_x2 = max(b[2] for b in rec_boxes)
    max_y2 = max(b[3] for b in rec_boxes)
    return int(max_x2), int(max_y2)


# ----------------------------
# OpenAI translation
# ----------------------------

def chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


class OpenAITranslator:
    """
    用 OpenAI Responses API 批次翻譯。
    - 使用 client.responses.create(...) :contentReference[oaicite:4]{index=4}
    - 取回文字用 resp.output_text :contentReference[oaicite:5]{index=5}
    """
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        target_lang: str = "en",
        source_lang: str = "auto",
        max_retries: int = 3,
        batch_size: int = 50,
        sleep_base: float = 0.8,
    ):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.target_lang = target_lang
        self.source_lang = source_lang
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.sleep_base = sleep_base

        # 小快取：同一句不重翻（整份文件層級）
        self.cache: Dict[str, str] = {}

    def translate_texts(self, texts: List[str]) -> List[str]:
        """
        回傳與 texts 等長的翻譯結果。
        """
        # 先準備需要翻譯的唯一句
        uniq: List[str] = []
        for t in texts:
            key = (t or "").strip()
            if not key:
                continue
            if key in self.cache:
                continue
            self.cache[key] = ""  # placeholder
            uniq.append(key)

        # 分批送出
        for batch in chunk_list(uniq, self.batch_size):
            self._translate_batch_inplace(batch)

        # 組回原順序
        out: List[str] = []
        for t in texts:
            key = (t or "").strip()
            out.append(self.cache.get(key, "") if key else "")
        return out

    def _translate_batch_inplace(self, batch: List[str]) -> None:
        if not batch:
            return

        if self.source_lang == "auto":
            instructions = (
                f"Translate each line to {self.target_lang}. "
                "Return ONLY the translations, one per line, preserving line count."
            )
        else:
            instructions = (
                f"Translate each line from {self.source_lang} to {self.target_lang}. "
                "Return ONLY the translations, one per line, preserving line count."
            )

        joined = "\n".join(batch)

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    instructions=instructions,
                    input=joined,
                )
                text_out = (resp.output_text or "").strip()
                lines = text_out.splitlines()

                # 容錯：行數不一致就退回逐行翻譯，保證可用
                if len(lines) != len(batch):
                    lines = [self._translate_one(t) for t in batch]

                for src, dst in zip(batch, lines):
                    self.cache[src] = (dst or "").strip()
                return

            except Exception as e:
                last_err = e
                time.sleep(self.sleep_base * attempt)

        raise RuntimeError(f"OpenAI translation failed: {last_err}") from last_err

    def _translate_one(self, text: str) -> str:
        if self.source_lang == "auto":
            instructions = f"Translate the user's text to {self.target_lang}. Output only the translation."
        else:
            instructions = f"Translate the user's text from {self.source_lang} to {self.target_lang}. Output only the translation."

        resp = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=text,
        )
        return (resp.output_text or "").strip()


# ----------------------------
# Text wrapping helpers (ReportLab)
# ----------------------------

def wrap_text_by_width(
    text: str,
    font_name: str,
    font_size: int,
    max_width_pt: float,
) -> List[str]:
    """
    以字寬估計做簡單換行，避免翻譯後太長爆框。
    """
    if not text:
        return [""]

    lines: List[str] = []
    cur = ""
    for ch in text:
        trial = cur + ch
        w = pdfmetrics.stringWidth(trial, font_name, font_size)
        if w <= max_width_pt or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


# ----------------------------
# Main overlay function
# ----------------------------

def overlay_translation_to_pdf(
    input_pdf: str,
    ocr_json: str,
    output_pdf: str,
    target_lang: str = "en",
    source_lang: str = "auto",
    model: str = "gpt-4.1-mini",
    dpi: int = 150,
    side_padding_px: int = 10,
    font_path: Optional[str] = None,
    font_name: str = "CustomFont",
    max_lines_per_item: int = 6,
    min_font_size: int = 6,
    max_font_size: int = 10,
):
    # 1) 讀 OCR pages
    pages = load_ocr_pages(ocr_json)

    # 2) 開 PDF
    doc = fitz.open(input_pdf)

    # 3) ReportLab Canvas
    c = canvas.Canvas(output_pdf)

    # 4) 字型：若要輸出中文，請務必提供字型路徑
    if font_path:
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"找不到字型檔：{font_path}")
        pdfmetrics.registerFont(TTFont(font_name, font_path))
        active_font = font_name
    else:
        active_font = "Helvetica"

    # 5) 初始化翻譯器
    translator = OpenAITranslator(
        model=model,
        target_lang=target_lang,
        source_lang=source_lang,
        batch_size=50,
        max_retries=3,
    )

    # 6) 逐頁處理（只處理 OCR JSON 中存在的頁）
    for ocr_page in pages:
        page_index = ocr_page.page_index
        if page_index < 0 or page_index >= doc.page_count:
            raise IndexError(f"OCR page_index={page_index} 超出 PDF 頁數範圍（0~{doc.page_count-1}）")

        page = doc.load_page(page_index)
        rect = page.rect
        page_w_pt = float(rect.width)
        page_h_pt = float(rect.height)

        # 渲染成底圖（確保 OCR pixel 座標可貼回）
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_w_px, img_h_px = pix.width, pix.height

        img_reader = ImageReader(BytesIO(pix.tobytes("png")))

        # 設定輸出頁面大小並畫底圖
        c.setPageSize((page_w_pt, page_h_pt))
        c.drawImage(img_reader, 0, 0, width=page_w_pt, height=page_h_pt)

        # OCR 座標空間 vs 渲染影像尺寸比例校正
        ocr_w_est, ocr_h_est = compute_ocr_extent(ocr_page.rec_boxes)
        if ocr_w_est <= 0 or ocr_h_est <= 0:
            c.showPage()
            continue

        scale_x_px = img_w_px / ocr_w_est
        scale_y_px = img_h_px / ocr_h_est

        # pixel -> point（ReportLab 用 point，左下原點）
        px_to_pt_x = page_w_pt / img_w_px
        px_to_pt_y = page_h_pt / img_h_px

        # 先批次翻譯整頁（速度、成本都會比逐條呼叫好）
        translated_texts = translator.translate_texts(ocr_page.rec_texts)

        for (src_text, box, dst_text) in zip(ocr_page.rec_texts, ocr_page.rec_boxes, translated_texts):
            if not (src_text or "").strip():
                continue
            if not (dst_text or "").strip():
                continue

            x1, y1, x2, y2 = box

            # OCR(pixel) -> rendered image pixel
            x1r = x1 * scale_x_px
            x2r = x2 * scale_x_px
            y1r = y1 * scale_y_px
            y2r = y2 * scale_y_px

            box_h_px = max(1.0, (y2r - y1r))
            font_size = int(round(box_h_px * 0.45))
            font_size = max(min_font_size, min(max_font_size, font_size))

            c.setFont(active_font, font_size)

            pad_px = float(side_padding_px)

            # 優先：放右側；不夠就放下方
            right_x_px = x2r + pad_px
            top_y_px = y1r

            available_right_px = img_w_px - right_x_px - pad_px

            def px_to_pdf(x_px: float, y_top_px: float) -> Tuple[float, float]:
                # y 軸翻轉：image 左上原點 -> PDF 左下原點
                x_pt = x_px * px_to_pt_x
                y_pt = page_h_pt - (y_top_px * px_to_pt_y)
                return x_pt, y_pt

            line_h = font_size * 1.2

            if available_right_px >= 90:
                x_pt, y_top_pt = px_to_pdf(right_x_px, top_y_px)
                max_w_pt = available_right_px * px_to_pt_x
                lines = wrap_text_by_width(dst_text, active_font, font_size, max_w_pt)

                y = y_top_pt - line_h
                for ln in lines[:max_lines_per_item]:
                    c.drawString(x_pt, y, ln)
                    y -= line_h
            else:
                below_x_px = x1r
                below_top_y_px = y2r + pad_px
                available_below_px = img_w_px - below_x_px - pad_px
                if available_below_px < 90:
                    continue

                x_pt, y_top_pt = px_to_pdf(below_x_px, below_top_y_px)
                max_w_pt = available_below_px * px_to_pt_x
                lines = wrap_text_by_width(dst_text, active_font, font_size, max_w_pt)

                y = y_top_pt - line_h
                for ln in lines[:max_lines_per_item]:
                    c.drawString(x_pt, y, ln)
                    y -= line_h

        c.showPage()

    c.save()
    doc.close()


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pdf", required=True, help="原始 PDF 路徑")
    parser.add_argument("--ocr_json", required=True, help="PaddleOCR JSON 路徑（可單頁或多頁list）")
    parser.add_argument("--output_pdf", required=True, help="輸出 PDF 路徑")
    parser.add_argument("--target_lang", default="en", help="目標語言，例如 en / zh-TW")
    parser.add_argument("--source_lang", default="auto", help="來源語言，例如 auto / zh-TW / en")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model 名稱")
    parser.add_argument("--dpi", type=int, default=150, help="渲染 PDF 用 dpi（影響對齊與輸出清晰度）")
    parser.add_argument("--side_padding_px", type=int, default=10, help="翻譯文字與原文框的間距（pixel）")
    parser.add_argument("--font_path", default=None, help="字型檔路徑（要顯示中文建議必填）")
    args = parser.parse_args()

    overlay_translation_to_pdf(
        input_pdf=args.input_pdf,
        ocr_json=args.ocr_json,
        output_pdf=args.output_pdf,
        target_lang=args.target_lang,
        source_lang=args.source_lang,
        model=args.model,
        dpi=args.dpi,
        side_padding_px=args.side_padding_px,
        font_path=args.font_path,
        font_name="CustomFont",
    )
    print(f"Done: {args.output_pdf}")


if __name__ == "__main__":
    main()
