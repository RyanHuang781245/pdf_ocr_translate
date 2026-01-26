from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, List, Tuple

import fitz  # PyMuPDF
from PIL import Image
from paddlex_hps_client import triton_request, utils
from tritonclient import grpc as triton_grpc

# -----------------------------
# Font (Windows CJK default)
# -----------------------------
DEFAULT_FONTFILE = r"C:\Windows\Fonts\msjh.ttc"


# -----------------------------
# Pipeline control
# -----------------------------
class PipelineCancelled(Exception):
    pass


# -----------------------------
# Language filtering
# -----------------------------
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")
DIGIT_RE = re.compile(r"\d")
REMOVE_LATIN_RE = re.compile(r"[A-Za-z]+")
REMOVE_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")


def _filter_text_by_lang(text: str, keep_lang: str) -> str:
    if keep_lang == "all":
        return (text or "").strip()

    cleaned = str(text or "")
    if keep_lang == "zh":
        cleaned = cleaned.strip()
        has_cjk = bool(CJK_RE.search(cleaned))
        has_latin = bool(LATIN_RE.search(cleaned))
        has_digit = bool(DIGIT_RE.search(cleaned))
        if has_cjk:
            return cleaned
        if has_latin:
            return ""
        if has_digit:
            return cleaned
        return ""

    if keep_lang == "en":
        cleaned = REMOVE_CJK_RE.sub("", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not (LATIN_RE.search(cleaned) or DIGIT_RE.search(cleaned)):
            return ""
        return cleaned

    return (text or "").strip()


def filter_rec_entries_by_lang(
    rec_polys: list[list[list[float]]],
    rec_texts: list[str],
    rec_scores: list[float],
    keep_lang: str,
) -> tuple[list[list[list[float]]], list[str], list[float]]:
    if keep_lang == "all":
        return rec_polys, rec_texts, rec_scores

    new_polys: list[list[list[float]]] = []
    new_texts: list[str] = []
    new_scores: list[float] = []

    for idx, text in enumerate(rec_texts):
        if idx >= len(rec_polys):
            break
        filtered = _filter_text_by_lang(text, keep_lang)
        if not filtered:
            continue
        new_polys.append(rec_polys[idx])
        new_texts.append(filtered)
        if idx < len(rec_scores):
            new_scores.append(rec_scores[idx])
        else:
            new_scores.append(0.0)

    return new_polys, new_texts, new_scores


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
        raise ValueError(f"start_page out of range: 1~{total}")

    if end_page is None:
        end_page = total
    if end_page < start_page or end_page > total:
        raise ValueError(f"end_page out of range: {start_page}~{total}")

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
# Step 2) Triton layout-parsing -> JSON
# -----------------------------
def run_layout_parsing_predict(
    images: list[Path],
    json_out_dir: Path,
    triton_url: str = "localhost:8001",
    progress_cb: Callable[[str, int, int, str], None] | None = None,
    cancel_event: Any | None = None,
) -> list[Path]:
    json_out_dir.mkdir(parents=True, exist_ok=True)

    client = triton_grpc.InferenceServerClient(url=triton_url)

    out_jsons: list[Path] = []
    total = len(images)
    for idx, img_path in enumerate(images, start=1):
        if cancel_event is not None and cancel_event.is_set():
            raise PipelineCancelled("Cancelled during OCR.")

        input_ = {
            "file": utils.prepare_input_file(str(img_path)),
            "fileType": 1,
        }
        output = triton_request(client, "layout-parsing", input_)
        if output.get("errorCode", -1) != 0:
            msg = output.get("errorMsg") or "unknown error"
            raise RuntimeError(f"Triton OCR failed for {img_path}: {msg}")

        try:
            pruned = output["result"]["layoutParsingResults"][0]["prunedResult"]
        except Exception as exc:
            raise RuntimeError(f"Unexpected Triton output for {img_path}") from exc

        if not isinstance(pruned, dict):
            raise RuntimeError(f"Unexpected prunedResult type for {img_path}: {type(pruned)}")

        if not pruned.get("input_path"):
            pruned = dict(pruned)
            pruned["input_path"] = str(img_path)

        json_path = json_out_dir / f"{img_path.stem}.json"
        json_path.write_text(json.dumps(pruned, ensure_ascii=False, indent=2), encoding="utf-8")
        out_jsons.append(json_path)

        if progress_cb:
            progress_cb("ocr", idx, total, f"OCR page {idx}/{total}")

    uniq: list[Path] = []
    seen = set()
    for p in out_jsons:
        k = p.resolve().as_posix()
        if k not in seen:
            uniq.append(p)
            seen.add(k)

    if not uniq:
        raise RuntimeError(f"No layout-parsing JSON output: {json_out_dir}")
    return uniq


# -----------------------------
# Helpers from PPStructure flow
# -----------------------------
TEXT_BLOCK_LABELS = {"text", "doc_title", "vision_footnote", "paragraph_title", "figure_title"}
def poly_to_bbox(poly: list[list[float]]) -> list[float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def bbox_center(bbox: list[float] | tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_in_bbox(px: float, py: float, bbox: list[float]) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2


def load_table_bboxes_from_parsing(data: dict) -> list[dict]:
    tables = []
    for blk in data.get("parsing_res_list", []):
        if blk.get("block_label") == "table" and isinstance(blk.get("block_bbox"), list) and len(blk["block_bbox"]) == 4:
            tables.append(
                {
                    "bbox": blk["block_bbox"],
                    "content": blk.get("block_content"),
                    "block_id": blk.get("block_id"),
                }
            )
    return tables


def load_table_bboxes_fallback_layout(data: dict) -> list[dict]:
    tables = []
    layout = data.get("layout_det_res", {})
    for it in layout.get("boxes", []):
        if it.get("label") == "table" and isinstance(it.get("coordinate"), list) and len(it["coordinate"]) == 4:
            x1, y1, x2, y2 = it["coordinate"]
            tables.append({"bbox": [float(x1), float(y1), float(x2), float(y2)], "content": None, "block_id": None})
    return tables


def normalize_paragraph_text(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = " ".join([line.strip() for line in s.split("\n") if line.strip()])
    s = " ".join(s.split())
    return s.strip()


def infer_page_index_from_input_path(input_path: str) -> int | None:
    name = Path(input_path).name
    m = re.search(r"_p(\d{4,})\.(png|jpg|jpeg|webp)$", name, re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1)) - 1


def infer_page_index_from_stem(stem: str) -> int | None:
    m = re.search(r"_p(\d{4,})$", stem, re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1)) - 1


def resolve_image_path(input_path: str, json_path: Path, images_dir: Path) -> Path:
    p = Path(input_path)
    if p.exists():
        return p
    p2 = (json_path.parent / input_path).resolve()
    if p2.exists():
        return p2
    p3 = images_dir / Path(input_path).name
    if p3.exists():
        return p3
    return p


def bbox_to_poly(bb_px: list[float]) -> list[list[float]]:
    x1, y1, x2, y2 = bb_px
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def px_bbox_to_rect(px_bb: list[float], img_w_px: int, img_h_px: int, page: fitz.Page) -> fitz.Rect:
    page_w_pt, page_h_pt = float(page.rect.width), float(page.rect.height)
    sx = page_w_pt / img_w_px
    sy = page_h_pt / img_h_px
    x1, y1, x2, y2 = px_bb
    return fitz.Rect(x1 * sx, y1 * sy, x2 * sx, y2 * sy)


# -----------------------------
# Paragraph: autowrap + shrink-to-fit
# -----------------------------
def insert_paragraph_autowrap_shrink(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    fontfile: str | None,
    max_fs: float,
    min_fs: float,
    align: int = 0,
    clip_ellipsis: bool = False,
) -> dict[str, Any]:
    clean = normalize_paragraph_text(text)
    if not clean:
        return {"ok": True, "fontsize": max_fs, "rc": 0.0, "clipped": False}

    def _insert_box(content: str, fontsize: float) -> float:
        if fontfile:
            return page.insert_textbox(
                rect,
                content,
                fontfile=fontfile,
                fontsize=fontsize,
                color=(0, 0, 1),
                align=align,
                overlay=True,
            )
        return page.insert_textbox(
            rect,
            content,
            fontname="helv",
            fontsize=fontsize,
            color=(0, 0, 1),
            align=align,
            overlay=True,
        )

    fs = max_fs
    while fs >= min_fs:
        rc = _insert_box(clean, fs)
        if rc >= 0:
            return {"ok": True, "fontsize": fs, "rc": rc, "clipped": False}
        fs -= 0.8

    if clip_ellipsis:
        words = clean.split()
        lo, hi = 0, len(words)
        best = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = " ".join(words[:mid]) + ("..." if mid < len(words) else "")
            rc = _insert_box(cand, min_fs)
            if rc >= 0:
                best = cand
                lo = mid + 1
            else:
                hi = mid - 1

        if best:
            rc = _insert_box(best, min_fs)
            return {"ok": True, "fontsize": min_fs, "rc": rc, "clipped": True}

    rc = _insert_box(clean, min_fs)
    return {"ok": False, "fontsize": min_fs, "rc": rc, "clipped": False}


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def extract_rec_entries_from_ppstructure(
    data: dict[str, Any],
    skip_text_inside_table: bool,
    min_line_score: float,
    table_fallback_layout: bool,
    include_block_labels: set[str] | None = None,
) -> tuple[list[list[list[float]]], list[str], list[float]]:
    include_block_labels = include_block_labels or TEXT_BLOCK_LABELS
    tables = load_table_bboxes_from_parsing(data)
    if table_fallback_layout and not tables:
        tables = load_table_bboxes_fallback_layout(data)
    table_bboxes = [t["bbox"] for t in tables]

    rec_polys: list[list[list[float]]] = []
    rec_texts: list[str] = []
    rec_scores: list[float] = []

    # Paragraph blocks
    for blk in data.get("parsing_res_list", []):
        if blk.get("block_label") not in include_block_labels:
            continue
        bb_px = blk.get("block_bbox")
        if not (isinstance(bb_px, list) and len(bb_px) == 4):
            continue
        try:
            bb_px = [float(v) for v in bb_px]
        except Exception:
            continue

        content = normalize_paragraph_text(blk.get("block_content", ""))
        if not content:
            continue

        if skip_text_inside_table and table_bboxes:
            cx, cy = bbox_center((bb_px[0], bb_px[1], bb_px[2], bb_px[3]))
            if any(point_in_bbox(cx, cy, tb) for tb in table_bboxes):
                continue

        rec_polys.append(bbox_to_poly(bb_px))
        rec_texts.append(content)
        rec_scores.append(_as_float(blk.get("block_score"), 1.0))

    # Table text lines (rec_polys)
    overall = data.get("overall_ocr_res", {}) or data.get("overall_ocr", {})
    ocr_polys = overall.get("rec_polys", []) or []
    ocr_texts = overall.get("rec_texts", []) or []
    ocr_scores = overall.get("rec_scores", []) or []

    for idx, poly in enumerate(ocr_polys):
        if not (isinstance(poly, list) and len(poly) >= 4):
            continue
        if not table_bboxes:
            continue
        score = _as_float(ocr_scores[idx], 0.0) if idx < len(ocr_scores) else 0.0
        if score < min_line_score:
            continue

        poly4: list[list[float]] = []
        for p in poly[:4]:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                poly4 = []
                break
            poly4.append([_as_float(p[0]), _as_float(p[1])])
        if not poly4:
            continue

        bbox = poly_to_bbox(poly4)
        cx, cy = bbox_center(bbox)
        if not any(point_in_bbox(cx, cy, tb) for tb in table_bboxes):
            continue

        text = ocr_texts[idx] if idx < len(ocr_texts) else ""
        text = (text or "").strip()
        if not text:
            continue

        rec_polys.append(poly4)
        rec_texts.append(text)
        rec_scores.append(score)

    return rec_polys, rec_texts, rec_scores


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

    client = OpenAI(
        base_url="https://bloom-checks-frog-boost.trycloudflare.com/v1",
        api_key="ollama",
    )
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
        raise ValueError(f"page_index out of range: 0~{doc.page_count-1}")
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
    sx = page_w_pt / img_w_px
    sy = page_h_pt / img_h_px
    return [[p[0] * sx, p[1] * sy] for p in poly_px]


def add_pdf_coords_to_json(
    pdf_path: Path,
    ocr_data: dict[str, Any],
    page_index_0based: int,
) -> dict[str, Any]:
    img_path = Path(ocr_data["input_path"])
    if not img_path.exists():
        raise FileNotFoundError(f"Missing input image: {img_path}")

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
# Step 4) Overlay debug PDF (PPStructure flow)
# -----------------------------
def overlay_one_page(
    page: fitz.Page,
    data: dict[str, Any],
    img_w_px: int,
    img_h_px: int,
    fontfile: str | None,
    draw_boxes: bool,
    draw_text: bool,
    paragraph_min_fs: float,
    paragraph_clip_ellipsis: bool,
    skip_text_inside_table: bool,
    min_line_score: float,
    table_fallback_layout: bool,
) -> None:
    tables = load_table_bboxes_from_parsing(data)
    if table_fallback_layout and not tables:
        tables = load_table_bboxes_fallback_layout(data)
    table_bboxes = [t["bbox"] for t in tables]

    dbg_shape = page.new_shape() if draw_boxes else None

    for blk in data.get("parsing_res_list", []):
        if blk.get("block_label") not in TEXT_BLOCK_LABELS:
            continue
        bb_px = blk.get("block_bbox")
        if not (isinstance(bb_px, list) and len(bb_px) == 4):
            continue

        content = normalize_paragraph_text(blk.get("block_content", ""))
        if not content:
            continue

        if skip_text_inside_table and table_bboxes:
            cx, cy = bbox_center((bb_px[0], bb_px[1], bb_px[2], bb_px[3]))
            if any(point_in_bbox(cx, cy, tb) for tb in table_bboxes):
                continue

        rect = px_bbox_to_rect(bb_px, img_w_px, img_h_px, page)

        if dbg_shape is not None:
            dbg_shape.draw_rect(rect)
            dbg_shape.finish(color=(1, 0, 0), width=0.6)

        if draw_text:
            fs_max = min(14.0, max(paragraph_min_fs, rect.height * 0.28))
            info = insert_paragraph_autowrap_shrink(
                page=page,
                rect=rect,
                text=content,
                fontfile=fontfile,
                max_fs=fs_max,
                min_fs=paragraph_min_fs,
                align=0,
                clip_ellipsis=paragraph_clip_ellipsis,
            )
            if not info.get("ok", True):
                print("[WARN] paragraph overflow:", "page=", page.number, "len=", len(content), "bbox_px=", bb_px)

    overall = data.get("overall_ocr_res", {}) or data.get("overall_ocr", {})
    rec_polys = overall.get("rec_polys", [])
    rec_texts = overall.get("rec_texts", [])
    rec_scores = overall.get("rec_scores", [])

    hit = 0
    for idx, poly in enumerate(rec_polys):
        if not (isinstance(poly, list) and len(poly) >= 4):
            continue
        if not table_bboxes:
            continue

        sc = _as_float(rec_scores[idx], 0.0) if idx < len(rec_scores) else 0.0
        if sc < min_line_score:
            continue

        try:
            poly4 = [[float(p[0]), float(p[1])] for p in poly[:4]]
        except Exception:
            continue

        bbox = poly_to_bbox(poly4)
        cx, cy = bbox_center(bbox)
        if not any(point_in_bbox(cx, cy, tb) for tb in table_bboxes):
            continue

        text = rec_texts[idx] if idx < len(rec_texts) else ""
        text = (text or "").strip()
        if not text:
            continue

        bb_px = [bbox[0], bbox[1], bbox[2], bbox[3]]
        rect = px_bbox_to_rect(bb_px, img_w_px, img_h_px, page)

        if dbg_shape is not None:
            dbg_shape.draw_rect(rect)
            dbg_shape.finish(color=(0, 0.6, 0), width=0.6)

        if draw_text:
            x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
            fs = max(4.0, min(12.0, (y1 - y0) * 0.9))
            pt = fitz.Point(x0, y1 - fs * 0.2)
            page.insert_text(
                pt,
                text,
                fontname="helv",
                fontsize=fs,
                color=(0, 0, 1),
                overlay=True,
            )
            hit += 1

    if dbg_shape is not None:
        dbg_shape.commit()

    print(f"[DEBUG] page={page.number} tables={len(table_bboxes)} table_line_hits={hit}")


def overlay_debug_pdf(
    pdf_path: Path,
    pp_json_paths: list[Path],
    images_dir: Path,
    out_pdf_path: Path,
    fontfile: str | None,
    draw_boxes: bool,
    draw_text: bool,
    paragraph_min_fs: float,
    paragraph_clip_ellipsis: bool,
    skip_text_inside_table: bool,
    min_line_score: float,
    table_fallback_layout: bool,
) -> None:
    doc = fitz.open(pdf_path)

    for js_path in pp_json_paths:
        data = json.loads(js_path.read_text(encoding="utf-8"))
        input_path = data.get("input_path")
        if not input_path:
            guess = images_dir / f"{js_path.stem}.png"
            if guess.exists():
                input_path = str(guess)
            else:
                continue

        img_path = resolve_image_path(str(input_path), js_path, images_dir)
        if not img_path.exists():
            fallback_png = images_dir / f"{js_path.stem}.png"
            if fallback_png.exists():
                img_path = fallback_png
            else:
                img_path = images_dir / f"{js_path.stem}.jpg"
        page_idx = infer_page_index_from_input_path(str(img_path))
        if page_idx is None:
            page_idx = infer_page_index_from_stem(js_path.stem)
        if page_idx is None or page_idx < 0 or page_idx >= doc.page_count:
            continue

        if not img_path.exists():
            raise FileNotFoundError(f"Missing input image: {img_path}")

        with Image.open(img_path) as im:
            img_w_px, img_h_px = im.size

        page = doc.load_page(page_idx)
        overlay_one_page(
            page=page,
            data=data,
            img_w_px=img_w_px,
            img_h_px=img_h_px,
            fontfile=fontfile,
            draw_boxes=draw_boxes,
            draw_text=draw_text,
            paragraph_min_fs=paragraph_min_fs,
            paragraph_clip_ellipsis=paragraph_clip_ellipsis,
            skip_text_inside_table=skip_text_inside_table,
            min_line_score=min_line_score,
            table_fallback_layout=table_fallback_layout,
        )

    out_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_pdf_path.as_posix())
    doc.close()


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
    triton_url: str = "localhost:8001",
    keep_lang: str = "all",
    fontfile: str | None = DEFAULT_FONTFILE,
    paragraph_min_fs: float = 4.0,
    paragraph_clip_ellipsis: bool = False,
    skip_text_inside_table: bool = False,
    min_line_score: float = 0.0,
    table_fallback_layout: bool = False,
) -> dict[str, Any]:
    img_dir = out_root / "images"
    pp_json_dir = out_root / "pp_json"
    norm_json_dir = out_root / "ocr_json"

    if use_per_image_ocr:
        pass

    fontfile_path = Path(fontfile) if fontfile else None
    if fontfile_path is not None and not fontfile_path.exists():
        print(f"[WARN] fontfile not found, fallback to built-in font: {fontfile_path}")
        fontfile_path = None

    png_paths = pdf_to_pngs(
        pdf_path,
        img_dir,
        dpi,
        start_page,
        end_page,
        progress_cb=progress_cb,
        cancel_event=cancel_event,
    )

    if cancel_event is not None and cancel_event.is_set():
        raise PipelineCancelled("Cancelled before OCR.")

    pp_json_paths = run_layout_parsing_predict(
        png_paths,
        pp_json_dir,
        triton_url=triton_url,
        progress_cb=progress_cb,
        cancel_event=cancel_event,
    )

    norm_json_dir.mkdir(parents=True, exist_ok=True)
    per_page_with_coords_jsons: list[Path] = []

    total_pages = len(pp_json_paths)
    translate_cache: dict[str, str] = {}
    effective_min_line_score = max(min_score, min_line_score)

    for idx, js_path in enumerate(pp_json_paths, start=1):
        if cancel_event is not None and cancel_event.is_set():
            raise PipelineCancelled("Cancelled during normalization.")

        data = json.loads(js_path.read_text(encoding="utf-8"))
        input_path = data.get("input_path") or ""
        img_path = resolve_image_path(str(input_path), js_path, img_dir) if input_path else img_dir / f"{js_path.stem}.png"
        if not img_path.exists():
            fallback_png = img_dir / f"{js_path.stem}.png"
            if fallback_png.exists():
                img_path = fallback_png
            else:
                img_path = img_dir / f"{js_path.stem}.jpg"
        if not img_path.exists():
            raise FileNotFoundError(f"Missing input image: {img_path}")

        page_idx_0based = infer_page_index_from_input_path(str(img_path))
        if page_idx_0based is None:
            page_idx_0based = infer_page_index_from_stem(js_path.stem)
        if page_idx_0based is None:
            continue

        rec_polys, rec_texts, rec_scores = extract_rec_entries_from_ppstructure(
            data,
            skip_text_inside_table=skip_text_inside_table,
            min_line_score=effective_min_line_score,
            table_fallback_layout=table_fallback_layout,
        )
        rec_polys, rec_texts, rec_scores = filter_rec_entries_by_lang(
            rec_polys,
            rec_texts,
            rec_scores,
            keep_lang=keep_lang,
        )

        ocr_data = {
            "input_path": str(img_path),
            "page_index_0based": page_idx_0based,
            "page_no_1based": page_idx_0based + 1,
            "rec_polys": rec_polys,
            "rec_texts": rec_texts,
            "rec_scores": rec_scores,
        }

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
    if cancel_event is not None and cancel_event.is_set():
        raise PipelineCancelled("Cancelled before overlay.")
    if progress_cb:
        progress_cb("overlay", 0, 1, "Building overlay PDF")
    overlay_debug_pdf(
        pdf_path=pdf_path,
        pp_json_paths=pp_json_paths,
        images_dir=img_dir,
        out_pdf_path=debug_pdf,
        fontfile=str(fontfile_path) if fontfile_path else None,
        draw_boxes=draw_boxes,
        draw_text=draw_text,
        paragraph_min_fs=paragraph_min_fs,
        paragraph_clip_ellipsis=paragraph_clip_ellipsis,
        skip_text_inside_table=skip_text_inside_table,
        min_line_score=effective_min_line_score,
        table_fallback_layout=table_fallback_layout,
    )
    if progress_cb:
        progress_cb("overlay", 1, 1, "Overlay PDF ready")

    return {
        "images_dir": img_dir,
        "official_json_dir": pp_json_dir,
        "norm_json_dir": norm_json_dir,
        "debug_pdf": debug_pdf,
        "per_page_jsons": per_page_with_coords_jsons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF -> PNG -> Triton layout-parsing -> overlay debug PDF")
    parser.add_argument("--pdf", required=True, help="Input PDF path")
    parser.add_argument("--out", default="out", help="Output directory")
    parser.add_argument("--dpi", type=int, default=300, help="PDF render DPI")
    parser.add_argument("--start", type=int, default=1, help="Start page (1-based)")
    parser.add_argument("--end", type=int, default=None, help="End page (inclusive)")
    parser.add_argument("--url", type=str, default="localhost:8001", help="Triton gRPC URL")
    parser.add_argument(
        "--keep-lang",
        type=str,
        choices=["all", "zh", "en"],
        default="all",
        help="Keep only target language in OCR results",
    )
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum table line score for overlay")
    parser.add_argument("--no-box", action="store_true", help="Do not draw debug boxes")
    parser.add_argument("--no-text", action="store_true", help="Do not draw debug text")
    parser.add_argument("--fontfile", default=DEFAULT_FONTFILE, help="Font file path for paragraph text")
    parser.add_argument("--paragraph-min-fs", type=float, default=4.0, help="Paragraph min font size")
    parser.add_argument("--paragraph-clip-ellipsis", action="store_true", help="Paragraph ellipsis fallback")
    parser.add_argument("--skip-text-inside-table", action="store_true", help="Skip paragraph text inside tables")
    parser.add_argument("--min-line-score", type=float, default=0.0, help="Minimum table line score")
    parser.add_argument("--table-fallback-layout", action="store_true", help="Use layout table boxes if parsing is empty")
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
        triton_url=args.url,
        keep_lang=args.keep_lang,
        fontfile=args.fontfile,
        paragraph_min_fs=args.paragraph_min_fs,
        paragraph_clip_ellipsis=args.paragraph_clip_ellipsis,
        skip_text_inside_table=args.skip_text_inside_table,
        min_line_score=args.min_line_score,
        table_fallback_layout=args.table_fallback_layout,
    )

    print("Done.")
    print(f"- Images:        {result['images_dir'].resolve()}")
    print(f"- PP JSON:       {result['official_json_dir'].resolve()}")
    print(f"- Normalized:    {result['norm_json_dir'].resolve()}")
    print(f"- Debug PDF:     {result['debug_pdf'].resolve()}")


if __name__ == "__main__":
    main()
