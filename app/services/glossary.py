from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from . import state

_PROTECTED_TERM_PREFIX = "[[[GLOSSARY_TERM_"
_PROTECTED_TERM_PATTERN = re.compile(r"\[\[\[GLOSSARY_TERM_\d+::(.*?)\]\]\]")
_GLOSSARY_CACHE_LOCK = threading.Lock()
_GLOBAL_GLOSSARY_CACHE: tuple[Path, float | None, list[dict[str, str]]] | None = None
_COMBINED_GLOSSARY_CACHE: tuple[
    tuple[tuple[str, float | None], ...],
    list[tuple[str, str]],
] | None = None


def global_glossary_path() -> Path:
    return Path(state.GLOBAL_GLOSSARY_PATH)


def _resolve_glossary_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = state.BASE_DIR / path
    return path


def _path_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def invalidate_glossary_cache() -> None:
    global _GLOBAL_GLOSSARY_CACHE, _COMBINED_GLOSSARY_CACHE
    with _GLOSSARY_CACHE_LOCK:
        _GLOBAL_GLOSSARY_CACHE = None
        _COMBINED_GLOSSARY_CACHE = None


def _clean_glossary_items(data: object) -> list[dict[str, str]]:
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


def load_global_glossary() -> list[dict[str, str]]:
    global _GLOBAL_GLOSSARY_CACHE
    path = global_glossary_path()
    current_mtime = _path_mtime(path)
    with _GLOSSARY_CACHE_LOCK:
        cached = _GLOBAL_GLOSSARY_CACHE
        if cached and cached[0] == path and cached[1] == current_mtime:
            return list(cached[2])

    if current_mtime is None:
        cleaned: list[dict[str, str]] = []
    else:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cleaned = []
        else:
            cleaned = _clean_glossary_items(data)

    with _GLOSSARY_CACHE_LOCK:
        _GLOBAL_GLOSSARY_CACHE = (path, current_mtime, cleaned)
    return list(cleaned)


def write_global_glossary(items: list[dict[str, str]]) -> None:
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
    path = global_glossary_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    invalidate_glossary_cache()


def load_glossary_entries() -> list[tuple[str, str]]:
    global _COMBINED_GLOSSARY_CACHE
    paths = tuple(
        _resolve_glossary_path(raw_path)
        for raw_path in (
            state.GLOSSARY_INSPECTION_PATH,
            state.GLOSSARY_PROCESS_PATH,
            state.GLOBAL_GLOSSARY_PATH,
        )
        if raw_path
    )
    cache_key = tuple((str(path), _path_mtime(path)) for path in paths)
    with _GLOSSARY_CACHE_LOCK:
        cached = _COMBINED_GLOSSARY_CACHE
        if cached and cached[0] == cache_key:
            return list(cached[1])

    entries: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in _clean_glossary_items(data):
            cn = item["cn"]
            en = item["en"]
            key = (cn, en)
            if key in seen:
                continue
            entries.append(key)
            seen.add(key)
    entries.sort(key=lambda pair: len(pair[0]), reverse=True)
    with _GLOSSARY_CACHE_LOCK:
        _COMBINED_GLOSSARY_CACHE = (cache_key, entries)
    return list(entries)


def load_combined_glossary() -> list[tuple[str, str]]:
    return load_glossary_entries()


def apply_glossary(text: str, entries: list[tuple[str, str]] | None = None) -> str:
    if not text:
        return text
    entries = entries or load_glossary_entries()
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


def apply_glossary_with_protection(
    text: str,
    entries: list[tuple[str, str]] | None = None,
) -> str:
    if not text:
        return text
    entries = entries or load_glossary_entries()
    if not entries:
        return text

    out_parts: list[str] = []
    hits: list[tuple[str, str]] = []
    i = 0
    term_index = 1
    while i < len(text):
        matched = False
        for cn, en in entries:
            if not cn or not en:
                continue
            if text.startswith(cn, i):
                protected = f"{_PROTECTED_TERM_PREFIX}{term_index:04d}::{en}]]]"
                out_parts.append(protected)
                hits.append((cn, en))
                i += len(cn)
                term_index += 1
                matched = True
                break
        if matched:
            continue
        out_parts.append(text[i])
        i += 1

    if hits:
        preview = ", ".join([f"{cn}->{en}" for cn, en in hits[:6]])
        more = f" (+{len(hits) - 6})" if len(hits) > 6 else ""
        print(f"[GLOSSARY] protected_hits={len(hits)} {preview}{more}")
    return "".join(out_parts)


def restore_protected_glossary_terms(text: str) -> str:
    if not text or _PROTECTED_TERM_PREFIX not in text:
        return text
    return _PROTECTED_TERM_PATTERN.sub(lambda match: match.group(1), text)
