from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

import fitz
from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from pipeline_ocr_overlay import run_pipeline

BASE_DIR = Path(__file__).resolve().parent
OUT_ROOT = BASE_DIR / "out"
JOB_ROOT = OUT_ROOT / "jobs"
UPLOAD_ROOT = OUT_ROOT / "uploads"
ALLOWED_EXTENSIONS = {".pdf"}
app = Flask(__name__)


def _safe_job_id(job_id: str) -> bool:
    return bool(re.fullmatch(r"[a-f0-9]{32}", job_id))


def _job_dir(job_id: str) -> Path:
    return JOB_ROOT / job_id


def _load_page_data(page_json_path: Path) -> dict[str, Any]:
    data = json.loads(page_json_path.read_text(encoding="utf-8"))
    rec_polys = data.get("rec_polys", []) or []
    rec_texts = data.get("rec_texts", []) or []
    edit_texts = data.get("edit_texts", []) or []
    rec_scores = data.get("rec_scores", []) or []
    count = len(rec_polys)

    if not edit_texts:
        edit_texts = list(rec_texts)
    if len(edit_texts) < count:
        edit_texts = list(edit_texts) + list(rec_texts[len(edit_texts) : count])
    if len(rec_scores) < count:
        rec_scores = list(rec_scores) + [0.0] * (count - len(rec_scores))

    image_size = data.get("coord_transform", {}).get("image_size_px", None)
    return {
        "page_index_0based": int(data.get("page_index_0based", 0)),
        "input_image": Path(data.get("input_path", "")).name,
        "image_size_px": image_size,
        "rec_polys": rec_polys,
        "rec_texts": rec_texts,
        "edit_texts": edit_texts,
        "rec_scores": rec_scores,
    }


def _hex_to_rgb(value: str | None, default: tuple[float, float, float] = (0.1, 0.2, 0.3)) -> tuple[float, float, float]:
    if not value:
        return default
    value = value.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        return default
    try:
        r = int(value[0:2], 16) / 255.0
        g = int(value[2:4], 16) / 255.0
        b = int(value[4:6], 16) / 255.0
        return (r, g, b)
    except ValueError:
        return default


def _load_page_transforms(job_dir: Path) -> dict[int, tuple[float, float, float, float]]:
    json_dir = job_dir / "ocr_json"
    mapping: dict[int, tuple[float, float, float, float]] = {}
    for path in json_dir.glob("*_res_with_pdf_coords.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        page_idx = int(data.get("page_index_0based", 0))
        transform = data.get("coord_transform", {})
        img_size = transform.get("image_size_px") or []
        pdf_size = transform.get("pdf_page_size_pt") or []
        if len(img_size) == 2 and len(pdf_size) == 2:
            mapping[page_idx] = (float(img_size[0]), float(img_size[1]), float(pdf_size[0]), float(pdf_size[1]))
    return mapping


def _apply_edits_to_pdf(job_id: str, job_dir: Path, edits: dict[str, Any]) -> Path:
    pdf_path = job_dir / f"{job_id}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing PDF: {pdf_path}")

    page_transforms = _load_page_transforms(job_dir)
    if not page_transforms:
        raise RuntimeError("Missing OCR coord transform data.")

    pages_by_index = {int(p.get("page_index_0based", 0)): p for p in edits.get("pages", []) if isinstance(p, dict)}

    doc = fitz.open(pdf_path)
    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)
        page_edits = pages_by_index.get(page_idx)
        if not page_edits:
            continue
        transform = page_transforms.get(page_idx)
        if not transform:
            continue
        img_w, img_h, page_w, page_h = transform
        if img_w <= 0 or img_h <= 0:
            continue
        sx = page_w / img_w
        sy = page_h / img_h

        shape = page.new_shape()
        for box in page_edits.get("boxes", []):
            if not isinstance(box, dict) or box.get("deleted"):
                continue
            bbox = box.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            text = str(box.get("text", "")).strip()
            if not text:
                continue
            x, y, w, h = [float(v) for v in bbox]
            rect = fitz.Rect(x * sx, y * sy, (x + w) * sx, (y + h) * sy)
            if rect.is_empty:
                continue

            font_size_px = float(box.get("font_size") or 0.0)
            font_size_pt = font_size_px * sy if font_size_px > 0 else max(5.0, rect.height * 0.7)
            color = _hex_to_rgb(box.get("color"))

            ok = False
            current = font_size_pt
            for _ in range(20):
                rc = shape.insert_textbox(
                    rect,
                    text,
                    fontname="helv",
                    fontsize=current,
                    color=color,
                    align=0,
                )
                if rc >= 0:
                    ok = True
                    break
                current -= max(0.5, current * 0.1)
                if current < 4.0:
                    break

            if not ok:
                shape.insert_textbox(
                    rect,
                    text,
                    fontname="helv",
                    fontsize=max(4.0, current),
                    color=color,
                    align=0,
                )

        shape.commit()

    out_path = job_dir / "edited.pdf"
    doc.save(out_path.as_posix())
    doc.close()
    return out_path


@app.route("/", methods=["GET"])
def index() -> str:
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload() -> str:
    file = request.files.get("pdf")
    if not file or file.filename == "":
        abort(400, "Missing PDF file.")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        abort(400, "Only PDF files are supported.")

    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

    job_id = uuid.uuid4().hex
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    pdf_filename = secure_filename(f"{job_id}.pdf")
    pdf_path = job_dir / pdf_filename
    file.save(pdf_path)

    dpi = int(request.form.get("dpi", 300))
    start_page = int(request.form.get("start", 1))
    end_page_raw = request.form.get("end", "").strip()
    end_page = int(end_page_raw) if end_page_raw else None

    run_pipeline(
        pdf_path=pdf_path,
        out_root=job_dir,
        dpi=dpi,
        start_page=start_page,
        end_page=end_page,
        min_score=0.0,
        draw_boxes=True,
        draw_text=True,
        enable_translate=False,
    )

    return redirect(url_for("editor", job_id=job_id))


@app.route("/job/<job_id>", methods=["GET"])
def editor(job_id: str) -> str:
    if not _safe_job_id(job_id):
        abort(404)
    job_dir = _job_dir(job_id)
    if not job_dir.exists():
        abort(404)
    return render_template(
        "editor.html",
        job_id=job_id,
        debug_pdf_url=url_for("job_file", job_id=job_id, filename="overlay_debug.pdf"),
    )


@app.route("/api/job/<job_id>", methods=["GET"])
def job_data(job_id: str):
    if not _safe_job_id(job_id):
        abort(404)
    job_dir = _job_dir(job_id)
    json_dir = job_dir / "ocr_json"
    if not json_dir.exists():
        abort(404)

    json_paths = sorted(json_dir.glob("*_res_with_pdf_coords.json"))
    pages = []
    for path in json_paths:
        page = _load_page_data(path)
        if not page["input_image"]:
            continue
        page["image_url"] = url_for("job_file", job_id=job_id, filename=f"images/{page['input_image']}")
        pages.append(page)

    edited_pdf_path = job_dir / "edited.pdf"
    payload = {
        "job_id": job_id,
        "debug_pdf_url": url_for("job_file", job_id=job_id, filename="overlay_debug.pdf"),
        "edited_pdf_url": url_for("job_file", job_id=job_id, filename="edited.pdf") if edited_pdf_path.exists() else None,
        "pages": pages,
    }
    return jsonify(payload)


@app.route("/api/job/<job_id>/save", methods=["POST"])
def save_job(job_id: str):
    if not _safe_job_id(job_id):
        abort(404)
    job_dir = _job_dir(job_id)
    if not job_dir.exists():
        abort(404)

    payload = request.get_json(force=True)
    edits_path = job_dir / "edits.json"
    edits_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        edited_pdf = _apply_edits_to_pdf(job_id, job_dir, payload)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True, "edited_pdf_url": url_for("job_file", job_id=job_id, filename=edited_pdf.name)})


@app.route("/jobs/<job_id>/<path:filename>", methods=["GET"])
def job_file(job_id: str, filename: str):
    if not _safe_job_id(job_id):
        abort(404)
    job_dir = _job_dir(job_id)
    if not job_dir.exists():
        abort(404)
    return send_from_directory(job_dir, filename)


if __name__ == "__main__":
    app.run(debug=True)
