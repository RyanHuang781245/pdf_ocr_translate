from __future__ import annotations

import io
import json
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

from flask import url_for
from werkzeug.utils import secure_filename

from . import state


def safe_job_id(job_id: str) -> bool:
    return bool(re.fullmatch(r"[a-f0-9]{32}", job_id))


def job_dir(job_id: str) -> Path:
    return state.JOB_ROOT / job_id


def job_timestamp(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def notify_jobs_update() -> None:
    with state.JOBS_EVENT:
        state.JOBS_VERSION += 1
        state.JOBS_EVENT.notify_all()


def build_jobs_list() -> list[dict[str, Any]]:
    state.JOB_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for job_dir_path in sorted(state.JOB_ROOT.iterdir()):
        if not job_dir_path.is_dir():
            continue
        job_id = job_dir_path.name
        if not safe_job_id(job_id):
            continue

        pdf_path = job_dir_path / f"{job_id}.pdf"
        debug_pdf_path = job_dir_path / "overlay_debug.pdf"
        edited_pdf_path = job_dir_path / "edited.pdf"

        created_at = job_timestamp(pdf_path) or job_timestamp(job_dir_path)
        debug_ts = job_timestamp(debug_pdf_path)
        edited_ts = job_timestamp(edited_pdf_path)
        updated_at = max(debug_ts, edited_ts, created_at)
        job_meta = load_job_meta(job_dir_path) or {}
        started_at = job_meta.get("processing_started_at") or created_at
        completed_at = job_meta.get("processing_completed_at")
        if not isinstance(completed_at, (int, float)):
            if debug_ts:
                completed_at = debug_ts
            elif edited_ts:
                completed_at = edited_ts
        if isinstance(completed_at, (int, float)) and isinstance(started_at, (int, float)):
            duration_seconds = max(0.0, float(completed_at) - float(started_at))
        elif isinstance(started_at, (int, float)):
            duration_seconds = max(0.0, time.time() - float(started_at))
        else:
            duration_seconds = max(0.0, updated_at - created_at)
        ocr_started_at = job_meta.get("ocr_started_at") or created_at
        ocr_completed_at = job_meta.get("ocr_completed_at") or debug_ts or None
        if isinstance(ocr_completed_at, (int, float)) and isinstance(ocr_started_at, (int, float)):
            ocr_duration_seconds = max(0.0, float(ocr_completed_at) - float(ocr_started_at))
        else:
            ocr_duration_seconds = None
        translate_started_at = job_meta.get("translate_started_at")
        translate_completed_at = job_meta.get("translate_completed_at") or edited_ts or None
        if isinstance(translate_completed_at, (int, float)) and isinstance(translate_started_at, (int, float)):
            translate_duration_seconds = max(0.0, float(translate_completed_at) - float(translate_started_at))
        else:
            translate_duration_seconds = None

        debug_ready = debug_pdf_path.exists()
        batch_status = load_batch_status(job_dir_path)
        batch_config = load_batch_config(job_dir_path)
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
                "duration_seconds": duration_seconds,
                "ocr_duration_seconds": ocr_duration_seconds,
                "translate_duration_seconds": translate_duration_seconds,
                "status_code": status_code,
                "status_label": status_label,
                "status": status_label,
                "job_name": job_name,
                "editor_url": url_for("editor.editor", job_id=job_id),
                "debug_pdf_url": url_for(
                    "jobs.job_file", job_id=job_id, filename="overlay_debug.pdf"
                )
                if debug_ready
                else None,
                "edited_pdf_url": url_for(
                    "jobs.job_file", job_id=job_id, filename="edited.pdf"
                )
                if edited_pdf_path.exists()
                else None,
            }
        )
    jobs.sort(key=lambda item: item["updated_at"], reverse=True)
    return jobs


def batch_status_path(job_dir_path: Path) -> Path:
    return job_dir_path / state.BATCH_STATUS_NAME


def batch_config_path(job_dir_path: Path) -> Path:
    return job_dir_path / "batch_config.json"


def batch_alias_path(job_dir_path: Path) -> Path:
    return job_dir_path / state.BATCH_ALIAS_NAME


def batch_prefill_path(job_dir_path: Path) -> Path:
    return job_dir_path / state.BATCH_PREFILL_NAME


def job_meta_path(job_dir_path: Path) -> Path:
    return job_dir_path / "job_meta.json"


def write_job_meta(job_dir_path: Path, meta: dict[str, Any]) -> None:
    job_meta_path(job_dir_path).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_job_meta(job_dir_path: Path) -> dict[str, Any] | None:
    path = job_meta_path(job_dir_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def update_job_meta(job_dir_path: Path, **updates: Any) -> None:
    meta = load_job_meta(job_dir_path) or {}
    meta.update({k: v for k, v in updates.items() if v is not None})
    write_job_meta(job_dir_path, meta)


def write_batch_config(job_dir_path: Path, config: dict[str, Any]) -> None:
    batch_config_path(job_dir_path).write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_batch_config(job_dir_path: Path) -> dict[str, Any] | None:
    path = batch_config_path(job_dir_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_batch_status(job_dir_path: Path, status: str, **meta: Any) -> None:
    payload = {
        "status": status,
        "updated_at": time.time(),
        **meta,
    }
    batch_status_path(job_dir_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    notify_jobs_update()


def load_batch_status(job_dir_path: Path) -> dict[str, Any] | None:
    path = batch_status_path(job_dir_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_batch_alias_map(job_dir_path: Path, alias_map: dict[str, str]) -> None:
    batch_alias_path(job_dir_path).write_text(
        json.dumps(alias_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_batch_alias_map(job_dir_path: Path) -> dict[str, str]:
    path = batch_alias_path(job_dir_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    cleaned: dict[str, str] = {}
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        cleaned[k] = v
    return cleaned


def write_batch_prefill_map(job_dir_path: Path, prefill: dict[str, str]) -> None:
    batch_prefill_path(job_dir_path).write_text(
        json.dumps(prefill, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_batch_prefill_map(job_dir_path: Path) -> dict[str, str]:
    path = batch_prefill_path(job_dir_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    cleaned: dict[str, str] = {}
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        cleaned[k] = v
    return cleaned


def load_edits_map(job_dir_path: Path) -> dict[int, list[dict[str, Any]]]:
    edits_path = job_dir_path / "edits.json"
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


def build_translated_zip(job_ids: set[str] | None) -> tuple[io.BytesIO, int]:
    state.JOB_ROOT.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    names: set[str] = set()
    count = 0
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for job_dir_path in sorted(state.JOB_ROOT.iterdir()):
            if not job_dir_path.is_dir():
                continue
            job_id = job_dir_path.name
            if not safe_job_id(job_id):
                continue
            if job_ids is not None and job_id not in job_ids:
                continue
            edited_path = job_dir_path / "edited.pdf"
            if not edited_path.exists():
                continue
            job_meta = load_job_meta(job_dir_path) or {}
            job_name = job_meta.get("job_name")
            if isinstance(job_name, str):
                job_name = job_name.strip() or None
            else:
                job_name = None
            base = job_name or job_id
            safe_name = secure_filename(base) or job_id
            filename = f"{safe_name}.pdf"
            if filename in names:
                filename = f"{safe_name}_{job_id[:8]}.pdf"
            names.add(filename)
            zf.write(edited_path, arcname=filename)
            count += 1
    buf.seek(0)
    return buf, count


def delete_job_dir(job_id: str) -> tuple[bool, str | None]:
    job_dir_path = job_dir(job_id)
    if not job_dir_path.exists():
        return False, None
    try:
        shutil.rmtree(job_dir_path)
    except Exception as exc:
        return False, str(exc)
    notify_jobs_update()
    return True, None


def get_active_upload() -> dict[str, object] | None:
    with state.ACTIVE_UPLOAD_LOCK:
        return state.ACTIVE_UPLOAD


def set_active_upload(payload: dict[str, object] | None) -> None:
    with state.ACTIVE_UPLOAD_LOCK:
        state.ACTIVE_UPLOAD = payload


def clear_active_upload(job_id: str) -> None:
    with state.ACTIVE_UPLOAD_LOCK:
        if state.ACTIVE_UPLOAD and state.ACTIVE_UPLOAD.get("job_id") == job_id:
            state.ACTIVE_UPLOAD = None
