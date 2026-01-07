from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Tuple, List

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR


# -----------------------------
# Step 1) PDF -> PNG
# -----------------------------
def pdf_to_pngs(pdf_path: Path, out_dir: Path, dpi: int, start_page: int, end_page: int | None) -> List[Path]:
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

    for page_no in range(start_page, end_page + 1):
        page = doc.load_page(page_no - 1)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = out_dir / f"{stem}_p{page_no:04d}.png"
        pix.save(out_path.as_posix())
        outputs.append(out_path)

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

    img_dir = out_root / "images"
    official_json_dir = out_root / "official_json"
    norm_json_dir = out_root / "ocr_json"

    # 1) PDF -> PNG
    png_paths = pdf_to_pngs(pdf_path, img_dir, args.dpi, args.start, args.end)

    # 2) 官方 demo style OCR：對資料夾 predict，並 save_to_json
    official_json_paths = run_paddleocr_official_on_folder(img_dir, official_json_dir)

    # 2.5) 將官方 json 正規化成你後續用的 rec_* 結構，並建立 per-page json（單頁一個）
    norm_json_dir.mkdir(parents=True, exist_ok=True)
    per_page_with_coords_jsons: list[Path] = []

    # 讓 png 檔名與頁碼對應（以你的 pdf_to_pngs 命名規則）
    png_by_page: dict[int, Path] = {}
    for p in png_paths:
        page_no_1based = infer_page_no_1based_from_filename(p)
        if page_no_1based is None:
            continue
        png_by_page[page_no_1based - 1] = p  # key = 0-based page index

    # 依檔名排序處理 json（若官方 json 名稱不同，也不影響；我們用內容的 input_path / fallback 推測）
    for oj in official_json_paths:
        # 先猜測 input_path：用「同頁 png」當 fallback（若官方 json 沒寫 input_path）
        # 這裡用檔名中含 p0001 的規則推測，推不到就不給 fallback
        fallback = None
        m = re.search(r"_p(\d{4,})", oj.stem, re.IGNORECASE)
        if m:
            page_no_1based = int(m.group(1))
            fallback_png = png_by_page.get(page_no_1based - 1)
            if fallback_png:
                fallback = str(fallback_png)

        ocr_data = normalize_official_json_to_rec_format(oj, fallback_input_path=fallback)

        # 決定頁碼：優先用 input_path 檔名的 p0001
        page_idx_0based = None
        ip = Path(ocr_data["input_path"]).name
        m2 = re.search(r"_p(\d{4,})\.png$", ip, re.IGNORECASE)
        if m2:
            page_idx_0based = int(m2.group(1)) - 1

        if page_idx_0based is None:
            # 若推不到頁碼，就跳過（避免誤畫到錯頁）
            continue

        ocr_data["page_index_0based"] = page_idx_0based
        ocr_data["page_no_1based"] = page_idx_0based + 1

        base_json = norm_json_dir / f"{pdf_path.stem}_p{page_idx_0based+1:04d}_res.json"
        base_json.write_text(json.dumps(ocr_data, ensure_ascii=False, indent=2), encoding="utf-8")

        # 3) 加入 pdf 座標（PyMuPDF 座標系，不翻 y）
        with_coords = add_pdf_coords_to_json(pdf_path, ocr_data, page_idx_0based)
        with_coords_json = norm_json_dir / f"{pdf_path.stem}_p{page_idx_0based+1:04d}_res_with_pdf_coords.json"
        with_coords_json.write_text(json.dumps(with_coords, ensure_ascii=False, indent=2), encoding="utf-8")
        per_page_with_coords_jsons.append(with_coords_json)

    # 4) Overlay debug PDF（整份輸出）
    per_page_with_coords_jsons = sorted(per_page_with_coords_jsons)
    debug_pdf = out_root / "overlay_debug.pdf"
    overlay_debug_pdf(
        pdf_path=pdf_path,
        per_page_json_paths=per_page_with_coords_jsons,
        out_pdf_path=debug_pdf,
        draw_boxes=not args.no_box,
        draw_text=not args.no_text,
        min_score=args.min_score,
    )

    print("Done.")
    print(f"- Images:        {img_dir.resolve()}")
    print(f"- Official JSON: {official_json_dir.resolve()}")
    print(f"- Normalized:    {norm_json_dir.resolve()}")
    print(f"- Debug PDF:     {debug_pdf.resolve()}")


if __name__ == "__main__":
    main()
