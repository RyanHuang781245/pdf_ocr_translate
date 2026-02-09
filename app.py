from __future__ import annotations

import datetime
import functools
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import fitz
from flask import Flask, Response, abort, jsonify, redirect, render_template, request, send_from_directory, stream_with_context, url_for
from werkzeug.utils import secure_filename

from pipeline_ocr_overlay import PipelineCancelled, run_pipeline, px_point_to_pdf_pt

BASE_DIR = Path(__file__).resolve().parent
OUT_ROOT = BASE_DIR / "out"
JOB_ROOT = OUT_ROOT / "jobs"
UPLOAD_ROOT = OUT_ROOT / "uploads"
TRITON_URL = os.getenv("TRITON_URL", "https://racks-editing-norm-timber.trycloudflare.com/table-recognition")
AZURE_BASE_URL = os.getenv("AZURE_OPENAI_BASE_URL", "https://uocp-azure-openai.openai.azure.com/openai/v1/")
AZURE_API_KEY_ENV = os.getenv("AZURE_OPENAI_API_KEY_ENV", "UO_AZURE_OPENAI_API_KEY")
AZURE_BATCH_MODEL = os.getenv("AZURE_BATCH_MODEL", "gpt-4o-mini-global-batch")
AZURE_BATCH_POLL_SECONDS = float(os.getenv("AZURE_BATCH_POLL_SECONDS", "60"))
AZURE_BATCH_COMPLETION_WINDOW = os.getenv("AZURE_BATCH_COMPLETION_WINDOW", "24h")
GLOSSARY_INSPECTION_PATH = os.getenv("GLOSSARY_INSPECTION_PATH", str((Path(__file__).resolve().parent / "glossary" / "inspection_terminology.json")))
GLOSSARY_PROCESS_PATH = os.getenv("GLOSSARY_PROCESS_PATH", str((Path(__file__).resolve().parent / "glossary" / "process_terminology.json")))
GLOBAL_GLOSSARY_PATH = os.getenv("GLOBAL_GLOSSARY_PATH", str((Path(__file__).resolve().parent / "glossary" / "global_glossary.json")))
AZURE_BATCH_SYSTEM_PROMPT = os.getenv(
    "AZURE_BATCH_SYSTEM_PROMPT",
    "\n".join(
        [
            "You are a professional medical device regulatory translator.",
            "Translate the text from Chinese to English accurately and literally.",
            "Do NOT summarize, paraphrase, explain, or add content.",
            "Preserve all numbers, codes, references, and formatting.",
            "Output only the translated English text.",
        ]
    ),
).strip()
BATCH_INPUT_NAME = "azure_batch_input.jsonl"
BATCH_OUTPUT_NAME = "azure_batch_output.jsonl"
BATCH_STATUS_NAME = "batch_status.json"
ALLOWED_EXTENSIONS = {".pdf"}
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msjh.ttf",
    r"C:\Windows\Fonts\msjhbd.ttf",
    r"C:\Windows\Fonts\msjhl.ttf",
    r"C:\Windows\Fonts\msjh.ttc",
    r"C:\Windows\Fonts\msjhbd.ttc",
    r"C:\Windows\Fonts\msjhl.ttc",
    r"C:\Windows\Fonts\mingliu.ttc",
    r"C:\Windows\Fonts\simsun.ttc",
]
app = Flask(__name__)
logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

ACTIVE_UPLOAD: dict[str, Any] | None = None
ACTIVE_UPLOAD_LOCK = threading.Lock()
JOBS_EVENT = threading.Condition()
JOBS_VERSION = 0


def _safe_job_id(job_id: str) -> bool:
    return bool(re.fullmatch(r"[a-f0-9]{32}", job_id))


def _job_dir(job_id: str) -> Path:
    return JOB_ROOT / job_id


def _job_timestamp(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def _notify_jobs_update() -> None:
    global JOBS_VERSION
    with JOBS_EVENT:
        JOBS_VERSION += 1
        JOBS_EVENT.notify_all()


def _build_jobs_list() -> list[dict[str, Any]]:
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for job_dir in sorted(JOB_ROOT.iterdir()):
        if not job_dir.is_dir():
            continue
        job_id = job_dir.name
        if not _safe_job_id(job_id):
            continue

        pdf_path = job_dir / f"{job_id}.pdf"
        debug_pdf_path = job_dir / "overlay_debug.pdf"
        edited_pdf_path = job_dir / "edited.pdf"

        created_at = _job_timestamp(pdf_path) or _job_timestamp(job_dir)
        updated_at = max(_job_timestamp(debug_pdf_path), _job_timestamp(edited_pdf_path), created_at)

        debug_ready = debug_pdf_path.exists()
        batch_status = _load_batch_status(job_dir)
        batch_config = _load_batch_config(job_dir)
        job_meta = _load_job_meta(job_dir) or {}
        job_name = job_meta.get("job_name")
        if isinstance(job_name, str):
            job_name = job_name.strip() or None
        else:
            job_name = None
        if not debug_ready:
            status_code = "ocr"
            status_label = "OCR"
        elif batch_config:
            batch_state = str((batch_status or {}).get("status") or "").lower()
            if batch_state in {"failed", "canceled", "cancelled"}:
                status_code = "translate_failed"
                status_label = "翻譯失敗"
            elif batch_state == "completed":
                status_code = "completed"
                status_label = "完成"
            else:
                status_code = "translate"
                status_label = "翻譯中"
        else:
            status_code = "completed"
            status_label = "完成"

        jobs.append(
            {
                "job_id": job_id,
                "created_at": created_at,
                "updated_at": updated_at,
                "status_code": status_code,
                "status_label": status_label,
                "status": status_label,
                "job_name": job_name,
                "editor_url": url_for("editor", job_id=job_id),
                "debug_pdf_url": url_for("job_file", job_id=job_id, filename="overlay_debug.pdf")
                if debug_ready
                else None,
                "edited_pdf_url": url_for("job_file", job_id=job_id, filename="edited.pdf")
                if edited_pdf_path.exists()
                else None,
            })
    jobs.sort(key=lambda item: item["updated_at"], reverse=True)
    return jobs


def _batch_status_path(job_dir: Path) -> Path:
    return job_dir / BATCH_STATUS_NAME


def _batch_config_path(job_dir: Path) -> Path:
    return job_dir / "batch_config.json"

def _job_meta_path(job_dir: Path) -> Path:
    return job_dir / "job_meta.json"

def _global_glossary_path() -> Path:
    return Path(GLOBAL_GLOSSARY_PATH)

def _write_job_meta(job_dir: Path, meta: dict[str, Any]) -> None:
    _job_meta_path(job_dir).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def _load_job_meta(job_dir: Path) -> dict[str, Any] | None:
    path = _job_meta_path(job_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_batch_config(job_dir: Path, config: dict[str, Any]) -> None:
    _batch_config_path(job_dir).write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_batch_config(job_dir: Path) -> dict[str, Any] | None:
    path = _batch_config_path(job_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_batch_status(job_dir: Path, status: str, **meta: Any) -> None:
    payload = {
        "status": status,
        "updated_at": time.time(),
        **meta,
    }
    _batch_status_path(job_dir).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _notify_jobs_update()


def _load_batch_status(job_dir: Path) -> dict[str, Any] | None:
    path = _batch_status_path(job_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_global_glossary() -> list[dict[str, str]]:
    path = _global_glossary_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    cleaned: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        cn = str(item.get("cn") or "").strip()
        en = str(item.get("en") or "").strip()
        if not cn or not en:
            continue
        cleaned.append({"cn": cn, "en": en})
    return cleaned


def _write_global_glossary(items: list[dict[str, str]]) -> None:
    payload: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        cn = str(item.get("cn") or "").strip()
        en = str(item.get("en") or "").strip()
        if not cn or not en:
            continue
        key = (cn, en)
        if key in seen:
            continue
        payload.append({"cn": cn, "en": en})
        seen.add(key)
    _global_glossary_path().write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_edits_map(job_dir: Path) -> dict[int, list[dict[str, Any]]]:
    edits_path = job_dir / "edits.json"
    if not edits_path.exists():
        return {}
    data = json.loads(edits_path.read_text(encoding="utf-8"))
    pages: dict[int, list[dict[str, Any]]] = {}
    for page in data.get("pages", []):
        if not isinstance(page, dict):
            continue
        page_idx = int(page.get("page_index_0based", 0))
        boxes = page.get("boxes", [])
        if not isinstance(boxes, list):
            boxes = []
        pages[page_idx] = [box for box in boxes if isinstance(box, dict)]
    return pages


def _bbox_to_poly(bbox: dict[str, Any] | list[float] | tuple[float, float, float, float]) -> list[list[float]]:
    if isinstance(bbox, dict):
        x = float(bbox.get("x", 0.0))
        y = float(bbox.get("y", 0.0))
        w = float(bbox.get("w", 0.0))
        h = float(bbox.get("h", 0.0))
    elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x, y, w, h = [float(v) for v in bbox]
    else:
        return []
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def _load_page_data(
    page_json_path: Path,
    edits_boxes: list[dict[str, Any]] | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if data is None:
        data = json.loads(page_json_path.read_text(encoding="utf-8"))
    if edits_boxes is not None:
        rec_polys: list[list[list[float]]] = []
        rec_texts: list[str] = []
        edit_texts: list[str] = []
        rec_scores: list[float] = []
        font_sizes: list[float] = []
        colors: list[str] = []
        box_ids: list[int] = []
        for box in edits_boxes:
            if not isinstance(box, dict):
                continue
            if box.get("deleted"):
                continue
            poly = _bbox_to_poly(box.get("bbox"))
            if not poly:
                continue
            text = str(box.get("text", ""))
            rec_polys.append(poly)
            rec_texts.append(text)
            edit_texts.append(text)
            rec_scores.append(1.0)
            font_sizes.append(float(box.get("font_size") or 0.0))
            colors.append(str(box.get("color") or "#0000ff"))
            box_ids.append(int(box.get("id") or len(box_ids)))
        count = len(rec_polys)
    else:
        rec_polys = data.get("rec_polys", []) or []
        rec_texts = data.get("rec_texts", []) or []
        edit_texts = data.get("edit_texts", []) or []
        rec_scores = data.get("rec_scores", []) or []
        font_sizes = []
        colors = []
        box_ids = []
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
        "font_sizes": font_sizes,
        "colors": colors,
        "box_ids": box_ids,
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


def _resolve_fontfile() -> str | None:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _parse_batch_custom_id(custom_id: str) -> tuple[int, int] | None:
    m = re.match(r"p(\d+)-l(\d+)$", custom_id or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _extract_batch_translation(item: dict[str, Any]) -> str:
    body = item.get("response", {}).get("body", {}) or {}
    if "output_text" in body:
        return str(body.get("output_text") or "").strip()
    choices = body.get("choices", []) or []
    if choices:
        return str(choices[0].get("message", {}).get("content", "")).strip()
    return ""


def _load_glossary_entries() -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_path in (GLOSSARY_INSPECTION_PATH, GLOSSARY_PROCESS_PATH, GLOBAL_GLOSSARY_PATH):
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            cn = str(item.get("cn") or "").strip()
            en = str(item.get("en") or "").strip()
            if not cn or not en:
                continue
            key = (cn, en)
            if key in seen:
                continue
            entries.append(key)
            seen.add(key)
    entries.sort(key=lambda pair: len(pair[0]), reverse=True)
    return entries


def _load_combined_glossary() -> list[tuple[str, str]]:
    return _load_glossary_entries()


def _apply_glossary(text: str, entries: list[tuple[str, str]] | None = None) -> str:
    if not text:
        return text
    entries = entries or _load_glossary_entries()
    if not entries:
        return text
    out = text
    hits: list[tuple[str, str]] = []
    for cn, en in entries:
        if cn in out:
            out = out.replace(cn, en)
            hits.append((cn, en))
    if hits:
        preview = ", ".join([f"{cn}->{en}" for cn, en in hits[:6]])
        more = f" (+{len(hits) - 6})" if len(hits) > 6 else ""
        print(f"[GLOSSARY] hits={len(hits)} {preview}{more}")
    return out


def _poly_to_bbox(poly: list[list[float]] | None) -> dict[str, float] | None:
    if not poly or len(poly) < 4:
        return None
    xs = [float(p[0]) for p in poly if isinstance(p, (list, tuple)) and len(p) >= 2]
    ys = [float(p[1]) for p in poly if isinstance(p, (list, tuple)) and len(p) >= 2]
    if not xs or not ys:
        return None
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}


def _get_azure_client():
    try:
        from dotenv import load_dotenv
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai and python-dotenv are required for Azure batch translation.") from exc

    load_dotenv()
    api_key = os.getenv(AZURE_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Environment variable {AZURE_API_KEY_ENV} is not set.")
    return OpenAI(base_url=AZURE_BASE_URL, api_key=api_key)


def _resolve_batch_prompt(target_lang: str, override: str | None = None) -> str:
    if override:
        return override.strip()
    normalized = (target_lang or "").strip().lower()
    if normalized in {"en", "english", "en-us", "en-gb"}:
        return AZURE_BATCH_SYSTEM_PROMPT
    return "\n".join(
        [
            "You are a professional translator.",
            f"Translate the text to {target_lang} accurately and literally.",
            "Do NOT summarize, paraphrase, explain, or add content.",
            "Preserve all numbers, codes, references, and formatting.",
            "Output only the translated text.",
        ]
    ).strip()


def _build_batch_items(
    ocr_pages: list[dict[str, Any]],
    model_name: str,
    system_prompt: str,
    glossary_entries: list[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for page in ocr_pages:
        page_idx = int(page.get("page_index_0based", 0))
        texts = page.get("rec_texts", []) or []
        for idx, text in enumerate(texts):
            clean = _normalize_text(text)
            if not clean:
                continue
            clean = _apply_glossary(clean, glossary_entries)
            custom_id = f"p{page_idx:04d}-l{idx:04d}"
            items.append(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": clean},
                        ],
                    },
                }
            )
    return items


def _write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _load_ocr_pages(job_dir: Path) -> list[dict[str, Any]]:
    json_dir = job_dir / "ocr_json"
    if not json_dir.exists():
        raise FileNotFoundError(f"Missing OCR JSON directory: {json_dir}")
    page_paths = sorted(json_dir.glob("*_res_with_pdf_coords.json"))
    if not page_paths:
        raise RuntimeError("No OCR JSON pages found.")
    return [json.loads(path.read_text(encoding="utf-8")) for path in page_paths]


def _build_translations_from_jsonl_text(raw_text: str) -> dict[str, str]:
    translations: dict[str, str] = {}
    for line in raw_text.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        custom_id = item.get("custom_id", "")
        translated = _extract_batch_translation(item)
        if translated:
            translations[custom_id] = translated
    return translations


def _build_edits_payload_from_translations(
    ocr_pages: list[dict[str, Any]],
    translations: dict[str, str],
) -> dict[str, Any]:
    pages_payload: list[dict[str, Any]] = []
    for page in ocr_pages:
        page_idx = int(page.get("page_index_0based", 0))
        rec_polys = page.get("rec_polys", []) or []
        rec_texts = page.get("rec_texts", []) or []
        edit_texts = page.get("edit_texts", []) or []
        boxes: list[dict[str, Any]] = []
        for idx, poly in enumerate(rec_polys):
            custom_id = f"p{page_idx:04d}-l{idx:04d}"
            text = translations.get(custom_id)
            if not text:
                text = edit_texts[idx] if idx < len(edit_texts) else rec_texts[idx] if idx < len(rec_texts) else ""
            text = _normalize_text(text)
            if not text:
                continue
            bbox = _poly_to_bbox(poly)
            if not bbox:
                continue
            boxes.append({"id": idx, "bbox": bbox, "text": text, "deleted": False})

        pages_payload.append({"page_index_0based": page_idx, "boxes": boxes})

    return {"pages": pages_payload}


def _run_batch_translate_job(job_id: str, job_dir: Path, config: dict[str, Any] | None = None) -> None:
    config = config or _load_batch_config(job_dir) or {}
    target_lang = str(config.get("target_lang") or "en")
    model_name = str(config.get("model") or AZURE_BATCH_MODEL)
    system_prompt = _resolve_batch_prompt(target_lang, config.get("system_prompt"))
    status_meta = {
        "job_id": job_id,
        "started_at": time.time(),
        "model": model_name,
        "target_lang": target_lang,
    }
    logger.info("Batch translate start job_id=%s target_lang=%s model=%s", job_id, target_lang, model_name)
    _write_batch_status(job_dir, "running", **status_meta)
    try:
        ocr_pages = _load_ocr_pages(job_dir)
        glossary_entries = _load_combined_glossary()
        batch_items = _build_batch_items(
            ocr_pages,
            model_name=model_name,
            system_prompt=system_prompt,
            glossary_entries=glossary_entries,
        )
        logger.info("Batch translate collected pages=%s lines=%s", len(ocr_pages), len(batch_items))
        if not batch_items:
            raise RuntimeError("No OCR text lines found to translate.")

        batch_input_path = job_dir / BATCH_INPUT_NAME
        _write_jsonl(batch_input_path, batch_items)
        logger.info("Batch translate wrote input jsonl=%s", batch_input_path.resolve())

        client = _get_azure_client()
        file_obj = client.files.create(
            file=open(batch_input_path, "rb"),
            purpose="batch",
            extra_body={"expires_after": {"seconds": 1209600, "anchor": "created_at"}},
        )
        logger.info("Batch translate uploaded file_id=%s", file_obj.id)
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="chat/completions",
            completion_window=AZURE_BATCH_COMPLETION_WINDOW,
        )

        batch_id = batch.id
        logger.info("Batch translate created batch_id=%s", batch_id)
        _write_batch_status(job_dir, "running", **status_meta, batch_id=batch_id)

        status = batch.status
        while status not in ("completed", "failed", "canceled", "cancelled"):
            time.sleep(AZURE_BATCH_POLL_SECONDS)
            batch = client.batches.retrieve(batch_id)
            status = batch.status
            logger.info("Batch translate poll job_id=%s batch_id=%s status=%s", job_id, batch_id, status)
            _write_batch_status(
                job_dir,
                status,
                **status_meta,
                batch_id=batch_id,
                last_check=datetime.datetime.now().isoformat(timespec="seconds"),
            )

        if status != "completed":
            raise RuntimeError(f"Batch status = {status}")

        output_file_id = batch.output_file_id or batch.error_file_id
        if not output_file_id:
            raise RuntimeError("Batch has no output_file_id/error_file_id.")

        file_response = client.files.content(output_file_id)
        raw_text = file_response.text or ""
        (job_dir / BATCH_OUTPUT_NAME).write_text(raw_text, encoding="utf-8")
        logger.info("Batch translate downloaded output file_id=%s", output_file_id)

        translations = _build_translations_from_jsonl_text(raw_text)
        edits_payload = _build_edits_payload_from_translations(ocr_pages, translations)
        edits_path = job_dir / "edits.json"
        edits_path.write_text(json.dumps(edits_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Batch translate wrote edits.json=%s", edits_path.resolve())
        _apply_edits_to_pdf(job_id, job_dir, edits_payload)
        logger.info("Batch translate wrote edited.pdf job_id=%s", job_id)
        _write_batch_status(job_dir, "completed", **status_meta, batch_id=batch_id)
        logger.info("Batch translate completed job_id=%s", job_id)
    except Exception as exc:
        logger.exception("Batch translate failed job_id=%s error=%s", job_id, exc)
        _write_batch_status(job_dir, "failed", **status_meta, error=str(exc))


def _run_ocr_pipeline_job(
    job_id: str,
    job_dir: Path,
    pdf_path: Path,
    dpi: int,
    start_page: int,
    end_page: int | None,
    translate_target_lang: str,
    translate_model: str,
    keep_lang: str,
    enable_translate: bool,
    cancel_event: threading.Event,
) -> None:
    logger.info("OCR pipeline start job_id=%s", job_id)
    try:
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
            translate_target_lang=translate_target_lang,
            translate_model=translate_model,
            triton_url=TRITON_URL,
            keep_lang=keep_lang,
            cancel_event=cancel_event,
        )
    except PipelineCancelled:
        logger.info("OCR pipeline cancelled job_id=%s", job_id)
        try:
            shutil.rmtree(job_dir)
        except Exception as exc:
            logger.warning("Failed to delete cancelled job_dir=%s error=%s", job_dir, exc)
        _notify_jobs_update()
        return
    except Exception as exc:
        logger.exception("OCR pipeline failed job_id=%s error=%s", job_id, exc)
        _notify_jobs_update()
        return
    finally:
        global ACTIVE_UPLOAD
        with ACTIVE_UPLOAD_LOCK:
            if ACTIVE_UPLOAD and ACTIVE_UPLOAD.get("job_id") == job_id:
                ACTIVE_UPLOAD = None

    logger.info("OCR pipeline completed job_id=%s", job_id)
    _notify_jobs_update()
    if enable_translate:
        batch_config = {
            "target_lang": translate_target_lang,
            "model": translate_model,
        }
        _write_batch_config(job_dir, batch_config)
        _notify_jobs_update()
        threading.Thread(target=_run_batch_translate_job, args=(job_id, job_dir, batch_config), daemon=True).start()


def _load_page_transforms(job_dir: Path) -> dict[int, dict[str, Any]]:
    json_dir = job_dir / "ocr_json"
    mapping: dict[int, dict[str, Any]] = {}
    for path in json_dir.glob("*_res_with_pdf_coords.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        page_idx = int(data.get("page_index_0based", 0))
        transform = data.get("coord_transform", {})
        img_size = transform.get("image_size_px") or []
        pdf_size = transform.get("pdf_page_size_pt") or []
        rotation = transform.get("page_rotation")
        if len(img_size) == 2 and len(pdf_size) == 2:
            mapping[page_idx] = {
                "img_w": float(img_size[0]),
                "img_h": float(img_size[1]),
                "page_w": float(pdf_size[0]),
                "page_h": float(pdf_size[1]),
                "rotation": int(rotation) if rotation is not None else 0,
            }
    return mapping


def _apply_edits_to_pdf(job_id: str, job_dir: Path, edits: dict[str, Any]) -> Path:
    pdf_path = job_dir / f"{job_id}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing PDF: {pdf_path}")

    page_transforms = _load_page_transforms(job_dir)
    if not page_transforms:
        raise RuntimeError("Missing OCR coord transform data.")

    pages_by_index = {int(p.get("page_index_0based", 0)): p for p in edits.get("pages", []) if isinstance(p, dict)}

    fontfile = _resolve_fontfile()
    debug_boxes = os.getenv("DEBUG_EDIT_BOXES") == "1"
    doc = fitz.open(pdf_path)
    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)
        page_edits = pages_by_index.get(page_idx)
        if not page_edits:
            continue
        transform = page_transforms.get(page_idx)
        if not transform:
            continue
        img_w = float(transform.get("img_w", 0.0))
        img_h = float(transform.get("img_h", 0.0))
        page_w = float(transform.get("page_w", 0.0))
        page_h = float(transform.get("page_h", 0.0))
        rotation = int(transform.get("rotation", 0))
        if img_w <= 0 or img_h <= 0:
            continue
        sx = page_w / img_w
        sy = page_h / img_h

        shape = page.new_shape()
        dbg_shape = page.new_shape() if debug_boxes else None
        for box in page_edits.get("boxes", []):
            if not isinstance(box, dict) or box.get("deleted"):
                continue
            bbox = box.get("bbox")
            if isinstance(bbox, dict):
                x = float(bbox.get("x", 0.0))
                y = float(bbox.get("y", 0.0))
                w = float(bbox.get("w", 0.0))
                h = float(bbox.get("h", 0.0))
            elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x, y, w, h = [float(v) for v in bbox]
            else:
                continue
            text = str(box.get("text", "")).strip()
            if not text:
                continue

            p1 = px_point_to_pdf_pt(x, y, img_w, img_h, page_w, page_h, rotation)
            p2 = px_point_to_pdf_pt(x + w, y, img_w, img_h, page_w, page_h, rotation)
            p3 = px_point_to_pdf_pt(x + w, y + h, img_w, img_h, page_w, page_h, rotation)
            p4 = px_point_to_pdf_pt(x, y + h, img_w, img_h, page_w, page_h, rotation)
            xs = [p1[0], p2[0], p3[0], p4[0]]
            ys = [p1[1], p2[1], p3[1], p4[1]]
            rect = fitz.Rect(min(xs), min(ys), max(xs), max(ys))
            if rect.is_empty:
                continue
            if dbg_shape is not None:
                dbg_shape.draw_rect(rect)
                dbg_shape.finish(color=(1, 0, 0), width=0.6)

            font_size_px = float(box.get("font_size") or 0.0)
            font_size_pt = font_size_px * (page_h / img_h) if font_size_px > 0 else max(5.0, rect.height * 0.7)
            color = _hex_to_rgb(box.get("color"))
            rotate = rotation if rotation else 0

            ok = False
            current = font_size_pt
            for _ in range(20):
                if fontfile:
                    rc = shape.insert_textbox(
                        rect,
                        text,
                        fontfile=fontfile,
                        fontsize=current,
                        color=color,
                        align=0,
                        rotate=rotate,
                    )
                else:
                    rc = shape.insert_textbox(
                        rect,
                        text,
                        fontname="helv",
                        fontsize=current,
                        color=color,
                        align=0,
                        rotate=rotate,
                    )
                if rc >= 0:
                    ok = True
                    break
                current -= max(0.5, current * 0.1)
                if current < 4.0:
                    break

            if not ok:
                if fontfile:
                    shape.insert_textbox(
                        rect,
                        text,
                        fontfile=fontfile,
                        fontsize=max(4.0, current),
                        color=color,
                        align=0,
                        rotate=rotate,
                    )
                else:
                    shape.insert_textbox(
                        rect,
                        text,
                        fontname="helv",
                        fontsize=max(4.0, current),
                        color=color,
                        align=0,
                        rotate=rotate,
                    )

        shape.commit()
        if dbg_shape is not None:
            dbg_shape.commit()

    out_path = job_dir / "edited.pdf"
    doc.save(out_path.as_posix())
    doc.close()
    return out_path


@app.route("/", methods=["GET"])
def index() -> str:
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload() -> str:
    files = request.files.getlist("pdf")
    folder_path = (request.form.get("folder_path") or "").strip()
    if (not files or all(f.filename == "" for f in files)) and not folder_path:
        abort(400, "Missing PDF file or folder path.")

    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

    dpi = 200
    start_page = int(request.form.get("start", 1))
    end_page_raw = request.form.get("end", "").strip()
    end_page = int(end_page_raw) if end_page_raw else None
    enable_translate = request.form.get("translate") == "on"
    translate_target_lang = request.form.get("target_lang", "en").strip() or "en"
    translate_model = request.form.get("model", AZURE_BATCH_MODEL).strip() or AZURE_BATCH_MODEL
    keep_lang = request.form.get("keep_lang", "all").strip().lower() or "all"
    if keep_lang not in {"all", "zh", "en"}:
        keep_lang = "all"

    def enqueue_job(source_pdf: Path, display_name: str) -> None:
        job_id = uuid.uuid4().hex
        job_dir = _job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        job_name = f"{display_name}_{job_id[:8]}"
        _write_job_meta(job_dir, {"job_name": job_name})

        pdf_filename = secure_filename(f"{job_id}.pdf")
        pdf_path = job_dir / pdf_filename
        if source_pdf.exists():
            shutil.copy2(source_pdf, pdf_path)
        else:
            raise FileNotFoundError(f"Missing PDF: {source_pdf}")

        cancel_event = threading.Event()
        global ACTIVE_UPLOAD
        with ACTIVE_UPLOAD_LOCK:
            ACTIVE_UPLOAD = {"event": cancel_event, "job_id": job_id, "started_at": time.time()}

        threading.Thread(
            target=_run_ocr_pipeline_job,
            args=(
                job_id,
                job_dir,
                pdf_path,
                dpi,
                start_page,
                end_page,
                translate_target_lang,
                translate_model,
                keep_lang,
                enable_translate,
                cancel_event,
            ),
            daemon=True,
        ).start()

    if folder_path:
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            abort(400, "Folder path not found.")
        for pdf in sorted(folder.glob("*.pdf")):
            display_name = secure_filename(pdf.stem) or "job"
            enqueue_job(pdf, display_name)

    for file in files:
        if not file or file.filename == "":
            continue
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue
        tmp_path = UPLOAD_ROOT / secure_filename(file.filename)
        file.save(tmp_path)
        display_name = secure_filename(Path(file.filename).stem) or "job"
        enqueue_job(tmp_path, display_name)
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    _notify_jobs_update()

    return redirect(url_for("index"))


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

    edits_map = _load_edits_map(job_dir)
    json_paths = sorted(json_dir.glob("*_res_with_pdf_coords.json"))
    pages = []
    for path in json_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        page_idx_guess = int(data.get("page_index_0based", 0))
        edits_boxes = edits_map.get(page_idx_guess) if page_idx_guess in edits_map else None
        page = _load_page_data(path, edits_boxes=edits_boxes, data=data)
        if not page["input_image"]:
            continue
        page["image_url"] = url_for("job_file", job_id=job_id, filename=f"images/{page['input_image']}")
        pages.append(page)

    edited_pdf_path = job_dir / "edited.pdf"
    config = _load_batch_config(job_dir) or {}
    target_lang = str(config.get("target_lang") or "en")
    system_prompt = config.get("system_prompt") or _resolve_batch_prompt(target_lang)
    payload = {
        "job_id": job_id,
        "pdf_url": url_for("job_file", job_id=job_id, filename=f"{job_id}.pdf"),
        "debug_pdf_url": url_for("job_file", job_id=job_id, filename="overlay_debug.pdf"),
        "edited_pdf_url": url_for("job_file", job_id=job_id, filename="edited.pdf") if edited_pdf_path.exists() else None,
        "batch_status": _load_batch_status(job_dir),
        "glossary": _load_global_glossary(),
        "system_prompt": system_prompt,
        "pages": pages,
    }
    return jsonify(payload)


@app.route("/api/job/<job_id>/batch-translate", methods=["POST"])
def batch_translate(job_id: str):
    if not _safe_job_id(job_id):
        abort(404)
    job_dir = _job_dir(job_id)
    if not job_dir.exists():
        abort(404)
    status = _load_batch_status(job_dir)
    if status and status.get("status") in {"running", "queued"}:
        return jsonify({"ok": True, "status": status})
    config = _load_batch_config(job_dir) or {}
    threading.Thread(target=_run_batch_translate_job, args=(job_id, job_dir, config), daemon=True).start()
    return jsonify({"ok": True, "status": {"status": "running"}})


@app.route("/api/job/<job_id>/batch-status", methods=["GET"])
def batch_status(job_id: str):
    if not _safe_job_id(job_id):
        abort(404)
    job_dir = _job_dir(job_id)
    if not job_dir.exists():
        abort(404)
    status = _load_batch_status(job_dir) or {"status": "not_started"}
    return jsonify({"ok": True, "status": status})


@app.route("/api/job/<job_id>/batch-restore", methods=["POST"])
def batch_restore(job_id: str):
    if not _safe_job_id(job_id):
        abort(404)
    job_dir = _job_dir(job_id)
    if not job_dir.exists():
        abort(404)
    output_path = job_dir / BATCH_OUTPUT_NAME
    if not output_path.exists():
        return jsonify({"ok": False, "error": "Batch output not found."}), 400

    try:
        raw_text = output_path.read_text(encoding="utf-8")
        translations = _build_translations_from_jsonl_text(raw_text)
        ocr_pages = _load_ocr_pages(job_dir)
        edits_payload = _build_edits_payload_from_translations(ocr_pages, translations)
        edits_path = job_dir / "edits.json"
        edits_path.write_text(json.dumps(edits_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _apply_edits_to_pdf(job_id, job_dir, edits_payload)
        logger.info("Batch translate restored edits.json job_id=%s", job_id)
        _notify_jobs_update()
    except Exception as exc:
        logger.exception("Batch translate restore failed job_id=%s error=%s", job_id, exc)
        return jsonify({"ok": False, "error": str(exc)}), 500

    return jsonify({"ok": True})


@app.route("/api/job/<job_id>/system-prompt", methods=["POST"])
def save_system_prompt(job_id: str):
    if not _safe_job_id(job_id):
        abort(404)
    job_dir = _job_dir(job_id)
    if not job_dir.exists():
        abort(404)
    payload = request.get_json(force=True) or {}
    system_prompt = str(payload.get("system_prompt") or "").strip()
    config = _load_batch_config(job_dir) or {}
    if system_prompt:
        config["system_prompt"] = system_prompt
    else:
        config.pop("system_prompt", None)
    _write_batch_config(job_dir, config)
    _notify_jobs_update()
    return jsonify({"ok": True, "system_prompt": config.get("system_prompt")})


@app.route("/api/glossary", methods=["GET", "POST"])
def global_glossary():
    if request.method == "GET":
        return jsonify({"ok": True, "glossary": _load_global_glossary()})
    payload = request.get_json(force=True) or {}
    items = payload.get("glossary", [])
    if not isinstance(items, list):
        return jsonify({"ok": False, "error": "Invalid glossary payload."}), 400
    _write_global_glossary(items)
    _notify_jobs_update()
    return jsonify({"ok": True, "glossary": _load_global_glossary()})


@app.route("/api/jobs", methods=["GET"])
def list_jobs():
    jobs = _build_jobs_list()
    return jsonify({"jobs": jobs})


@app.route("/api/jobs/stream", methods=["GET"])
def jobs_stream():
    @stream_with_context
    def generate():
        last_version = -1
        while True:
            with JOBS_EVENT:
                if last_version == JOBS_VERSION:
                    JOBS_EVENT.wait(timeout=15)
                current_version = JOBS_VERSION
            if current_version == last_version:
                yield ": ping\n\n"
                continue
            last_version = current_version
            payload = {"jobs": _build_jobs_list()}
            data = json.dumps(payload, ensure_ascii=False)
            yield f"event: jobs\ndata: {data}\n\n"

    resp = Response(generate(), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


@app.route("/api/job/<job_id>", methods=["DELETE"])
def delete_job(job_id: str):
    if not _safe_job_id(job_id):
        abort(404)
    job_dir = _job_dir(job_id)
    if not job_dir.exists():
        return jsonify({"ok": True, "deleted": False})
    try:
        shutil.rmtree(job_dir)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    _notify_jobs_update()
    return jsonify({"ok": True, "deleted": True})


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
    _notify_jobs_update()
    return jsonify({"ok": True, "edited_pdf_url": url_for("job_file", job_id=job_id, filename=edited_pdf.name)})


@app.route("/jobs/<job_id>/<path:filename>", methods=["GET"])
def job_file(job_id: str, filename: str):
    if not _safe_job_id(job_id):
        abort(404)
    job_dir = _job_dir(job_id)
    if not job_dir.exists():
        abort(404)
    return send_from_directory(job_dir, filename)


@app.route("/api/upload-cancel", methods=["POST"])
def cancel_upload():
    global ACTIVE_UPLOAD
    with ACTIVE_UPLOAD_LOCK:
        active = ACTIVE_UPLOAD
    if not active:
        return jsonify({"ok": False, "status": "idle"})
    event = active.get("event")
    if event is not None:
        event.set()
    _notify_jobs_update()
    return jsonify({"ok": True, "job_id": active.get("job_id")})


if __name__ == "__main__":
    app.run(port=5001, debug=True, threaded=True)
