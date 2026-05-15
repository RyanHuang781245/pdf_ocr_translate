from __future__ import annotations

import json
import time
import uuid
from typing import Any

from . import state


def _clamp_ratio(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric < 0:
        return 0.0
    if numeric > 1:
        return 1.0
    return numeric


def _normalize_box(box: Any) -> dict[str, Any] | None:
    if not isinstance(box, dict):
        return None
    text = str(box.get("text") or "").strip()
    if not text:
        return None
    try:
        font_size = float(box.get("font_size") or state.DEFAULT_FONT_SIZE_PX)
    except (TypeError, ValueError):
        font_size = float(state.DEFAULT_FONT_SIZE_PX)
    normalized = {
        "x_ratio": _clamp_ratio(box.get("x_ratio")),
        "y_ratio": _clamp_ratio(box.get("y_ratio")),
        "w_ratio": _clamp_ratio(box.get("w_ratio")),
        "h_ratio": _clamp_ratio(box.get("h_ratio")),
        "text": text,
        "font_size": max(1.0, font_size),
        "color": str(box.get("color") or state.DEFAULT_TEXT_COLOR),
        "text_align": str(box.get("text_align") or "").strip().lower(),
        "rotation": int(box.get("rotation") or 0) % 360,
        "no_clip": bool(box.get("no_clip", False)),
    }
    if normalized["text_align"] not in {"left", "center", "right"}:
        normalized["text_align"] = "left"
    return normalized


def _normalize_page(page: Any) -> dict[str, Any] | None:
    if not isinstance(page, dict):
        return None
    try:
        page_index = int(page.get("page_index_0based") or 0)
    except (TypeError, ValueError):
        page_index = 0
    boxes = []
    for raw_box in page.get("boxes", []):
        normalized_box = _normalize_box(raw_box)
        if normalized_box:
            boxes.append(normalized_box)
    return {
        "page_index_0based": max(0, page_index),
        "boxes": boxes,
    }


def _normalize_template(template: Any) -> dict[str, Any] | None:
    if not isinstance(template, dict):
        return None
    template_id = str(template.get("id") or "").strip() or uuid.uuid4().hex
    source_job_id = str(template.get("source_job_id") or "").strip()
    template_name = str(template.get("name") or "").strip()
    display_name = str(template.get("display_name") or template_name or source_job_id or template_id).strip()
    status = str(template.get("status") or "draft").strip().lower()
    if status not in {"draft", "saved"}:
        status = "draft"
    now_ts = float(time.time())
    try:
        created_at = float(template.get("created_at") or now_ts)
    except (TypeError, ValueError):
        created_at = now_ts
    try:
        updated_at = float(template.get("updated_at") or now_ts)
    except (TypeError, ValueError):
        updated_at = now_ts
    pages = []
    for raw_page in template.get("pages", []):
        normalized_page = _normalize_page(raw_page)
        if normalized_page:
            pages.append(normalized_page)
    pages.sort(key=lambda item: item["page_index_0based"])
    saved_name = template_name if template_name else ""
    if status == "saved" and not saved_name:
        return None
    return {
        "id": template_id,
        "name": saved_name,
        "display_name": display_name or saved_name or template_id,
        "status": status,
        "source_job_id": source_job_id,
        "created_at": created_at,
        "updated_at": updated_at,
        "pages": pages,
    }


def load_document_templates() -> list[dict[str, Any]]:
    path = state.DOCUMENT_TEMPLATES_PATH
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    cleaned: list[dict[str, Any]] = []
    changed = False
    for item in data:
        normalized = _normalize_template(item)
        if not normalized:
            changed = True
            continue
        cleaned.append(normalized)
        if normalized != item:
            changed = True
    cleaned.sort(key=lambda item: (item["updated_at"], item["created_at"]), reverse=True)
    if changed:
        write_document_templates(cleaned)
    return cleaned


def write_document_templates(templates: list[dict[str, Any]]) -> None:
    path = state.DOCUMENT_TEMPLATES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(templates, ensure_ascii=False, indent=2), encoding="utf-8")


def get_document_template(template_id: str) -> dict[str, Any] | None:
    cleaned_id = str(template_id or "").strip()
    if not cleaned_id:
        return None
    for template in load_document_templates():
        if template["id"] == cleaned_id:
            return template
    return None


def get_document_template_by_job(job_id: str) -> dict[str, Any] | None:
    cleaned_job_id = str(job_id or "").strip()
    if not cleaned_job_id:
        return None
    for template in load_document_templates():
        if template.get("source_job_id") == cleaned_job_id:
            return template
    return None


def create_template_draft(*, source_job_id: str, display_name: str) -> dict[str, Any]:
    existing = get_document_template_by_job(source_job_id)
    if existing:
        return existing
    now_ts = float(time.time())
    draft = {
        "id": uuid.uuid4().hex,
        "name": "",
        "display_name": str(display_name or source_job_id).strip() or source_job_id,
        "status": "draft",
        "source_job_id": str(source_job_id or "").strip(),
        "created_at": now_ts,
        "updated_at": now_ts,
        "pages": [],
    }
    templates = load_document_templates()
    templates.append(draft)
    write_document_templates(templates)
    return draft


def save_document_template(payload: dict[str, Any]) -> dict[str, Any]:
    template_id = str(payload.get("id") or "").strip()
    source_job_id = str(payload.get("source_job_id") or "").strip()
    templates = load_document_templates()
    existing = None
    existing_index = None
    for index, item in enumerate(templates):
        if template_id and item["id"] == template_id:
            existing = item
            existing_index = index
            break
    if existing is None and source_job_id:
        for index, item in enumerate(templates):
            if item.get("source_job_id") == source_job_id:
                existing = item
                existing_index = index
                break

    normalized = _normalize_template(
        {
          "id": template_id or (existing or {}).get("id"),
          "name": payload.get("name") or (existing or {}).get("name") or "",
          "display_name": payload.get("display_name") or (existing or {}).get("display_name") or payload.get("name") or "",
          "status": "saved",
          "source_job_id": source_job_id or (existing or {}).get("source_job_id") or "",
          "created_at": (existing or {}).get("created_at"),
          "updated_at": time.time(),
          "pages": payload.get("pages", []),
        }
    )
    if not normalized or not normalized["pages"]:
        raise ValueError("Invalid document template payload.")
    if existing_index is not None:
        templates[existing_index] = normalized
    else:
        templates.append(normalized)
    templates.sort(key=lambda item: (item["updated_at"], item["created_at"]), reverse=True)
    write_document_templates(templates)
    return normalized


def delete_document_template(template_id: str) -> bool:
    cleaned_id = str(template_id or "").strip()
    if not cleaned_id:
        return False
    templates = load_document_templates()
    remaining = [item for item in templates if item["id"] != cleaned_id]
    if len(remaining) == len(templates):
        return False
    write_document_templates(remaining)
    return True
