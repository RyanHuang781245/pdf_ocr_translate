from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Tuple, List, Callable

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR


# -----------------------------
# Pipeline control
# -----------------------------
class PipelineCancelled(Exception):
    pass


# -----------------------------
# Step 1) PDF -> PNG
# -----------------------------
def pdf_to_pngs(
    pdf_path: Path,
    out_dir: Path,
    dpi: int,
    start_page: int,
    end_page: int | None,
    progress_cb: Callable[[str, int, int, str], None] | None = None,
    cancel_event: Any | None = None,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    total = doc.page_count

    if start_page < 1 or start_page > total:
        raise ValueError(f"start_page 超出範圍：1~{total}")

    if end_page is None:
        end_page = total
    if end_page < start_page or end_page > total:
        raise ValueError(f"end_page 超出範圍：{start_page}~{total}")

    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)

    outputs: List[Path] = []
    stem = pdf_path.stem

    total_pages = end_page - start_page + 1
    for idx, page_no in enumerate(range(start_page, end_page + 1), start=1):
        if cancel_event is not None and cancel_event.is_set():
            doc.close()
            raise PipelineCancelled("Cancelled during PDF render.")
        page = doc.load_page(page_no - 1)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = out_dir / f"{stem}_p{page_no:04d}.png"
        pix.save(out_path.as_posix())
        outputs.append(out_path)
        if progress_cb:
            progress_cb("render", idx, total_pages, f"Rendering page {idx}/{total_pages}")

    doc.close()
    return outputs


# -----------------------------
# Step 2) PaddleOCR (official demo style)
#    ocr.predict(input=folder) + res.save_to_json(out_folder)
# -----------------------------
def run_paddleocr_official_on_folder(
    images_dir: Path,
    json_out_dir: Path,
) -> List[Path]:
    """
    依照官方 demo 寫法：
    - ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
    - result = ocr.predict(input="pdf2img_outputs")
    - for res in result: res.save_to_json("output3")

    這裡將 input 指向 images_dir，並把 json 存到 json_out_dir。
    回傳：產生的 json 檔路徑清單（盡量依檔名排序）
    """
    json_out_dir.mkdir(parents=True, exist_ok=True)

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    result = ocr.predict(input=str(images_dir))

    # 官方 API 會把每張圖結果存成一個 json（檔名通常跟圖片相關）
    # 我們先讓它寫出來，再去掃描 json_out_dir 取得清單
    for res in result:
        # 你也可以保留 res.print() 方便看 console
        # res.print()
        res.save_to_json(str(json_out_dir))

    json_paths = sorted(json_out_dir.glob("*.json"))
    if not json_paths:
        raise RuntimeError(f"找不到官方輸出的 json：{json_out_dir}")

    return json_paths


# -----------------------------
# Step 2 alt) PaddleOCR per-image (progress + cancel)
# -----------------------------
def run_paddleocr_on_images(
    images: list[Path],
    json_out_dir: Path,
    progress_cb: Callable[[str, int, int, str], None] | None = None,
    cancel_event: Any | None = None,
) -> List[Path]:
    json_out_dir.mkdir(parents=True, exist_ok=True)

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    json_paths: list[Path] = []
    total = len(images)
    for idx, img_path in enumerate(images, start=1):
        if cancel_event is not None and cancel_event.is_set():
            raise PipelineCancelled("Cancelled during OCR.")

        result = ocr.ocr(str(img_path), cls=False)

        rec_polys: list[list[list[float]]] = []
        rec_texts: list[str] = []
        rec_scores: list[float] = []

        for item in result or []:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            poly = item[0]
            rec = item[1]
            if not isinstance(poly, (list, tuple)) or len(poly) < 4:
                continue
            if not isinstance(rec, (list, tuple)) or len(rec) < 2:
                continue
            text = rec[0]
            score = rec[1]
            rec_polys.append([[float(p[0]), float(p[1])] for p in poly[:4]])
            rec_texts.append(str(text))
            try:
                rec_scores.append(float(score))
            except Exception:
                rec_scores.append(0.0)

        data = {
            "input_path": str(img_path),
            "rec_polys": rec_polys,
            "rec_texts": rec_texts,
            "rec_scores": rec_scores,
        }
        out_path = json_out_dir / f"{img_path.stem}_res.json"
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        json_paths.append(out_path)

        if progress_cb:
            progress_cb("ocr", idx, total, f"OCR page {idx}/{total}")

    if not json_paths:
        raise RuntimeError(f"找不到 OCR 輸出的 json：{json_out_dir}")
    return json_paths

# -----------------------------
# Step 2.5) Normalize official json -> {input_path, rec_polys, rec_texts, rec_scores}
# -----------------------------
def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _norm_poly(poly: Any) -> list[list[float]] | None:
    """
    將各種可能的 poly 格式正規化為 4 點：
    [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    """
    if not isinstance(poly, (list, tuple)) or len(poly) < 4:
        return None

    out: list[list[float]] = []
    for p in poly[:4]:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            return None
        out.append([_as_float(p[0]), _as_float(p[1])])

    return out


def normalize_official_json_to_rec_format(official_json_path: Path, fallback_input_path: str | None = None) -> dict[str, Any]:
    """
    官方 save_to_json 的輸出結構在不同版本可能略有差異。
    這裡做「容錯解析」：盡可能抽出每個文字框的 poly/text/score。

    最終輸出結構：
    {
      "input_path": "...png",
      "rec_polys": [...],
      "rec_texts": [...],
      "rec_scores": [...],
    }
    """
    data = json.loads(official_json_path.read_text(encoding="utf-8"))

    input_path = data.get("input_path") or data.get("image_path") or data.get("filename") or fallback_input_path
    if not input_path:
        # 無法從 json 得知圖片路徑，就先用 json 同名 png 推測（常見情況）
        guess = official_json_path.with_suffix(".png")
        input_path = str(guess)

    rec_polys: list[list[list[float]]] = []
    rec_texts: list[str] = []
    rec_scores: list[float] = []

    # Case A：你之前那種結構（若官方剛好也是這樣）
    if isinstance(data.get("rec_polys"), list) and isinstance(data.get("rec_texts"), list):
        for poly, text, score in zip(data.get("rec_polys", []), data.get("rec_texts", []), data.get("rec_scores", [])):
            poly4 = _norm_poly(poly)
            if poly4 is None:
                continue
            rec_polys.append(poly4)
            rec_texts.append(str(text))
            rec_scores.append(_as_float(score, 0.0))
        return {"input_path": input_path, "rec_polys": rec_polys, "rec_texts": rec_texts, "rec_scores": rec_scores}

    # Case B：常見 OCR 結果列表（可能叫 results / ocr_result / det_res / rec_res / boxes 等）
    candidates = []
    for k in ("results", "result", "ocr_result", "data", "lines", "texts", "items"):
        v = data.get(k)
        if isinstance(v, list):
            candidates.append(v)

    # 也有人把內容包在 dict 裡，例如 data={"results":[...]}
    if isinstance(data.get("data"), dict):
        for k in ("results", "result", "ocr_result", "lines", "items"):
            v = data["data"].get(k)
            if isinstance(v, list):
                candidates.append(v)

    # 嘗試解析每個 item
    for items in candidates:
        for it in items:
            if not isinstance(it, dict):
                continue

            # poly/box 可能叫：poly / polygon / points / box / bbox / dt_poly / rec_poly
            poly = (
                it.get("poly")
                or it.get("polygon")
                or it.get("points")
                or it.get("box")
                or it.get("bbox")
                or it.get("dt_poly")
                or it.get("rec_poly")
            )
            poly4 = _norm_poly(poly)
            if poly4 is None:
                continue

            text = it.get("text") or it.get("rec_text") or it.get("transcription") or ""
            score = it.get("score") or it.get("rec_score") or it.get("confidence") or 0.0

            rec_polys.append(poly4)
            rec_texts.append(str(text))
            rec_scores.append(_as_float(score, 0.0))

        if rec_polys:
            break

    # Case C：再退一步，若 json 用「兩層列表」：[[poly, [text,score]], ...]
    if not rec_polys and isinstance(data.get("raw"), list):
        raw = data["raw"]
        for it in raw:
            if not isinstance(it, (list, tuple)) or len(it) < 2:
                continue
            poly4 = _norm_poly(it[0])
            if poly4 is None:
                continue
            txt_score = it[1]
            if isinstance(txt_score, (list, tuple)) and len(txt_score) >= 2:
                rec_polys.append(poly4)
                rec_texts.append(str(txt_score[0]))
                rec_scores.append(_as_float(txt_score[1], 0.0))

    return {"input_path": input_path, "rec_polys": rec_polys, "rec_texts": rec_texts, "rec_scores": rec_scores}


# -----------------------------
# Optional translation (OpenAI GPT)
# -----------------------------
def translate_texts_with_openai(
    texts: list[str],
    enabled: bool = False,
    target_lang: str = "en",
    source_lang: str = "auto",
    model: str = "gpt-4o-mini",
    batch_size: int = 40,
    max_retries: int = 3,
    cache: dict[str, str] | None = None,
    cancel_event: Any | None = None,
) -> list[str]:
    """
    OpenAI GPT translation (line-preserving).
    Keep disabled by default to avoid API usage during pipeline testing.
    """
    if not enabled:
        return list(texts)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai package is required for translation.") from exc

    client = OpenAI(api_key=api_key)
    cache = cache if cache is not None else {}

    def normalize_line(text: str) -> str:
        return " ".join(str(text).split()).strip()

    def chunk_list(items: list[str], size: int) -> list[list[str]]:
        return [items[i : i + size] for i in range(0, len(items), size)]

    def translate_one(line: str) -> str:
        if source_lang == "auto":
            instructions = f"Translate the user's text to {target_lang}. Output only the translation."
        else:
            instructions = f"Translate the user's text from {source_lang} to {target_lang}. Output only the translation."
        resp = client.responses.create(
            model=model,
            instructions=instructions,
            input=line,
        )
        return (resp.output_text or "").strip()

    def translate_batch(lines: list[str]) -> list[str]:
        if not lines:
            return []
        if source_lang == "auto":
            instructions = (
                f"Translate each line to {target_lang}. "
                "Return ONLY the translations, one per line, preserving line count."
            )
        else:
            instructions = (
                f"Translate each line from {source_lang} to {target_lang}. "
                "Return ONLY the translations, one per line, preserving line count."
            )
        joined = "\n".join(lines)

        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            if cancel_event is not None and cancel_event.is_set():
                raise PipelineCancelled("Cancelled during translation.")
            try:
                resp = client.responses.create(
                    model=model,
                    instructions=instructions,
                    input=joined,
                )
                output = (resp.output_text or "").strip()
                out_lines = output.splitlines()
                if len(out_lines) != len(lines):
                    return [translate_one(line) for line in lines]
                return out_lines
            except Exception as exc:
                last_err = exc
                time.sleep(0.5 * attempt)
        raise RuntimeError(f"OpenAI translation failed: {last_err}") from last_err

    uniq: list[str] = []
    for text in texts:
        key = normalize_line(text)
        if not key:
            continue
        if key in cache:
            continue
        cache[key] = ""
        uniq.append(key)

    for batch in chunk_list(uniq, batch_size):
        if cancel_event is not None and cancel_event.is_set():
            raise PipelineCancelled("Cancelled during translation.")
        translated = translate_batch(batch)
        for src, dst in zip(batch, translated):
            cache[src] = (dst or "").strip()

    out: list[str] = []
    for text in texts:
        key = normalize_line(text)
        out.append(cache.get(key, "") if key else "")
    return out


# -----------------------------
# Step 3) image(px) -> PyMuPDF page(pt) coord (no y-flip)
# -----------------------------
def get_pdf_page_size_pt(pdf_path: Path, page_index_0based: int) -> Tuple[float, float]:
    doc = fitz.open(pdf_path)
    if page_index_0based < 0 or page_index_0based >= doc.page_count:
        raise ValueError(f"page_index 超出範圍：0~{doc.page_count-1}")
    page = doc.load_page(page_index_0based)
    r = page.rect
    w_pt, h_pt = float(r.width), float(r.height)
    doc.close()
    return w_pt, h_pt


def px_poly_to_pymupdf_pt_poly(
    poly_px: list[list[float]],
    img_w_px: int,
    img_h_px: int,
    page_w_pt: float,
    page_h_pt: float,
) -> list[list[float]]:
    # 重要：PyMuPDF 頁面座標原點左上、y 往下，因此不做 y 翻轉
    sx = page_w_pt / img_w_px
    sy = page_h_pt / img_h_px
    return [[p[0] * sx, p[1] * sy] for p in poly_px]


def poly_to_bbox(poly: list[list[float]]) -> list[float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def add_pdf_coords_to_json(
    pdf_path: Path,
    ocr_data: dict[str, Any],
    page_index_0based: int,
) -> dict[str, Any]:
    img_path = Path(ocr_data["input_path"])
    if not img_path.exists():
        # 若 input_path 是相對路徑或不含資料夾，則以 json 所在資料夾或 images_dir 處理
        # 這裡先保持原樣，後續只需要尺寸，所以找不到時就報錯，避免 silently 造成錯位
        raise FileNotFoundError(f"找不到 input_path 指向的圖片：{img_path}")

    with Image.open(img_path) as im:
        img_w_px, img_h_px = im.size

    page_w_pt, page_h_pt = get_pdf_page_size_pt(pdf_path, page_index_0based)

    pdf_polys: list[list[list[float]]] = []
    pdf_boxes: list[list[float]] = []

    for poly_px in ocr_data.get("rec_polys", []):
        poly_pt = px_poly_to_pymupdf_pt_poly(poly_px, img_w_px, img_h_px, page_w_pt, page_h_pt)
        pdf_polys.append(poly_pt)
        pdf_boxes.append(poly_to_bbox(poly_pt))

    out = dict(ocr_data)
    out["page_index_0based"] = page_index_0based
    out["coord_transform"] = {
        "image_size_px": [img_w_px, img_h_px],
        "pdf_page_size_pt": [page_w_pt, page_h_pt],
        "method": "scale_by_page_size_no_y_flip_for_pymupdf",
    }
    out["pdf_polys"] = pdf_polys
    out["pdf_boxes"] = pdf_boxes
    return out


# -----------------------------
# Step 4) Overlay debug PDF
# -----------------------------
def overlay_debug_pdf(
    pdf_path: Path,
    per_page_json_paths: list[Path],
    out_pdf_path: Path,
    draw_boxes: bool = True,
    draw_text: bool = True,
    min_score: float = 0.0,
) -> None:
    doc = fitz.open(pdf_path)

    for js_path in per_page_json_paths:
        data = json.loads(js_path.read_text(encoding="utf-8"))

        page_idx = int(data.get("page_index_0based", 0))
        if page_idx < 0 or page_idx >= doc.page_count:
            continue

        pdf_boxes = data.get("pdf_boxes", [])
        rec_texts = data.get("rec_texts", [])
        rec_scores = data.get("rec_scores", [])

        page = doc.load_page(page_idx)
        shape = page.new_shape()

        for box, text, score in zip(pdf_boxes, rec_texts, rec_scores):
            if float(score) < min_score:
                continue

            rect = fitz.Rect(box[0], box[1], box[2], box[3])

            if draw_boxes:
                shape.draw_rect(rect)
                shape.finish(color=(1, 0, 0), width=0.7)

            if draw_text and (text or "").strip():
                box_h = max(1.0, rect.height)
                fs = min(18.0, max(5.0, box_h * 0.7))

                ok = False
                for _ in range(30):
                    rc = shape.insert_textbox(
                        rect,
                        text,
                        fontname="helv",
                        fontsize=fs,
                        color=(0, 0, 1),
                        align=0,
                    )
                    if rc >= 0:
                        ok = True
                        break
                    fs -= 0.8

                if not ok:
                    shape.insert_textbox(
                        rect,
                        text,
                        fontname="helv",
                        fontsize=4.0,
                        color=(0, 0, 1),
                        align=0,
                    )

        shape.commit()

    out_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_pdf_path.as_posix())
    doc.close()


def infer_page_no_1based_from_filename(png_path: Path) -> int | None:
    m = re.search(r"_p(\d{4,})\.png$", png_path.name, re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def run_pipeline(
    pdf_path: Path,
    out_root: Path,
    dpi: int = 300,
    start_page: int = 1,
    end_page: int | None = None,
    min_score: float = 0.0,
    draw_boxes: bool = True,
    draw_text: bool = True,
    enable_translate: bool = False,
    translate_target_lang: str = "en",
    translate_source_lang: str = "auto",
    translate_model: str = "gpt-4o-mini",
    translate_batch_size: int = 40,
    translate_max_retries: int = 3,
    progress_cb: Callable[[str, int, int, str], None] | None = None,
    cancel_event: Any | None = None,
    use_per_image_ocr: bool = False,
) -> dict[str, Any]:
    img_dir = out_root / "images"
    official_json_dir = out_root / "official_json"
    norm_json_dir = out_root / "ocr_json"

    # 1) PDF -> PNG
    png_paths = pdf_to_pngs(
        pdf_path,
        img_dir,
        dpi,
        start_page,
        end_page,
        progress_cb=progress_cb,
        cancel_event=cancel_event,
    )

    # 2) official demo style OCR
    if cancel_event is not None and cancel_event.is_set():
        raise PipelineCancelled("Cancelled before OCR.")
    if use_per_image_ocr or progress_cb or cancel_event is not None:
        official_json_paths = run_paddleocr_on_images(
            png_paths,
            official_json_dir,
            progress_cb=progress_cb,
            cancel_event=cancel_event,
        )
    else:
        official_json_paths = run_paddleocr_official_on_folder(img_dir, official_json_dir)
        if progress_cb:
            progress_cb("ocr", len(png_paths), len(png_paths), "OCR complete")

    # 2.5) normalize JSON + add per-page json
    norm_json_dir.mkdir(parents=True, exist_ok=True)
    per_page_with_coords_jsons: list[Path] = []

    png_by_page: dict[int, Path] = {}
    for p in png_paths:
        page_no_1based = infer_page_no_1based_from_filename(p)
        if page_no_1based is None:
            continue
        png_by_page[page_no_1based - 1] = p

    total_pages = len(official_json_paths)
    translate_cache: dict[str, str] = {}
    for idx, oj in enumerate(official_json_paths, start=1):
        if cancel_event is not None and cancel_event.is_set():
            raise PipelineCancelled("Cancelled during normalization.")
        fallback = None
        m = re.search(r"_p(\d{4,})", oj.stem, re.IGNORECASE)
        if m:
            page_no_1based = int(m.group(1))
            fallback_png = png_by_page.get(page_no_1based - 1)
            if fallback_png:
                fallback = str(fallback_png)

        ocr_data = normalize_official_json_to_rec_format(oj, fallback_input_path=fallback)

        page_idx_0based = None
        ip = Path(ocr_data["input_path"]).name
        m2 = re.search(r"_p(\d{4,})\.png$", ip, re.IGNORECASE)
        if m2:
            page_idx_0based = int(m2.group(1)) - 1

        if page_idx_0based is None:
            continue

        ocr_data["page_index_0based"] = page_idx_0based
        ocr_data["page_no_1based"] = page_idx_0based + 1

        if progress_cb and enable_translate:
            progress_cb("translate", idx, total_pages, f"Translate page {idx}/{total_pages}")
        edit_texts = translate_texts_with_openai(
            ocr_data.get("rec_texts", []),
            enabled=enable_translate,
            target_lang=translate_target_lang,
            source_lang=translate_source_lang,
            model=translate_model,
            batch_size=translate_batch_size,
            max_retries=translate_max_retries,
            cache=translate_cache,
            cancel_event=cancel_event,
        )
        ocr_data["edit_texts"] = list(edit_texts)

        base_json = norm_json_dir / f"{pdf_path.stem}_p{page_idx_0based+1:04d}_res.json"
        base_json.write_text(json.dumps(ocr_data, ensure_ascii=False, indent=2), encoding="utf-8")

        with_coords = add_pdf_coords_to_json(pdf_path, ocr_data, page_idx_0based)
        with_coords_json = norm_json_dir / f"{pdf_path.stem}_p{page_idx_0based+1:04d}_res_with_pdf_coords.json"
        with_coords_json.write_text(json.dumps(with_coords, ensure_ascii=False, indent=2), encoding="utf-8")
        per_page_with_coords_jsons.append(with_coords_json)
        if progress_cb:
            progress_cb("normalize", idx, total_pages, f"Normalize page {idx}/{total_pages}")

    per_page_with_coords_jsons = sorted(per_page_with_coords_jsons)
    debug_pdf = out_root / "overlay_debug.pdf"
    if progress_cb:
        progress_cb("overlay", 0, 1, "Building overlay PDF")
    overlay_debug_pdf(
        pdf_path=pdf_path,
        per_page_json_paths=per_page_with_coords_jsons,
        out_pdf_path=debug_pdf,
        draw_boxes=draw_boxes,
        draw_text=draw_text,
        min_score=min_score,
    )
    if progress_cb:
        progress_cb("overlay", 1, 1, "Overlay PDF ready")

    return {
        "images_dir": img_dir,
        "official_json_dir": official_json_dir,
        "norm_json_dir": norm_json_dir,
        "debug_pdf": debug_pdf,
        "per_page_jsons": per_page_with_coords_jsons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF→PNG→(官方 PaddleOCR predict)→座標換算→Overlay Debug PDF")
    parser.add_argument("--pdf", required=True, help="原始 PDF 路徑")
    parser.add_argument("--out", default="out", help="輸出根目錄")
    parser.add_argument("--dpi", type=int, default=300, help="PDF 轉圖 DPI（建議 200~300）")
    parser.add_argument("--start", type=int, default=1, help="起始頁（從 1 開始）")
    parser.add_argument("--end", type=int, default=None, help="結束頁（含），不填到最後")
    parser.add_argument("--min-score", type=float, default=0.0, help="Overlay 時最低信心分數門檻")
    parser.add_argument("--no-box", action="store_true", help="不畫框線")
    parser.add_argument("--no-text", action="store_true", help="不畫文字（只畫框線）")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    out_root = Path(args.out)

    result = run_pipeline(
        pdf_path=pdf_path,
        out_root=out_root,
        dpi=args.dpi,
        start_page=args.start,
        end_page=args.end,
        min_score=args.min_score,
        draw_boxes=not args.no_box,
        draw_text=not args.no_text,
        enable_translate=False,
    )

    print("Done.")
    print(f"- Images:        {result['images_dir'].resolve()}")
    print(f"- Official JSON: {result['official_json_dir'].resolve()}")
    print(f"- Normalized:    {result['norm_json_dir'].resolve()}")
    print(f"- Debug PDF:     {result['debug_pdf'].resolve()}")


if __name__ == "__main__":
    main()
