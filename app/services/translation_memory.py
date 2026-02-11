from __future__ import annotations

import json
import time
from typing import Any

from . import state


def _normalize_tm_entry(value: Any, now_ts: float) -> dict[str, Any] | None:
    if isinstance(value, str):
        return {"text": value, "last_used": now_ts}
    if not isinstance(value, dict):
        return None
    text = value.get("text")
    if not isinstance(text, str):
        return None
    last_used = value.get("last_used")
    try:
        last_used_ts = float(last_used) if last_used is not None else now_ts
    except (TypeError, ValueError):
        last_used_ts = now_ts
    return {"text": text, "last_used": last_used_ts}


def load_translation_memory() -> dict[str, dict[str, Any]]:
    path = state.TRANSLATION_MEMORY_PATH
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    now_ts = time.time()
    ttl_seconds = state.TRANSLATION_MEMORY_TTL_SECONDS
    cleaned: dict[str, dict[str, Any]] = {}
    changed = False
    for k, v in data.items():
        if not isinstance(k, str):
            changed = True
            continue
        entry = _normalize_tm_entry(v, now_ts)
        if not entry:
            changed = True
            continue
        last_used = entry.get("last_used", now_ts)
        if ttl_seconds and (now_ts - float(last_used) > ttl_seconds):
            changed = True
            continue
        cleaned[k] = entry
        if entry != v:
            changed = True
    if changed:
        write_translation_memory(cleaned)
    return cleaned


def write_translation_memory(memory: dict[str, dict[str, Any]]) -> None:
    path = state.TRANSLATION_MEMORY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
