import json
import os
import threading
from datetime import datetime, timezone
from typing import Dict, Optional

from src.config import MEDIA_CATALOG_PATH

_LOCK = threading.RLock()
_STATE: Optional[Dict] = None
_STORE_PATH = MEDIA_CATALOG_PATH


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _default_state() -> Dict:
    return {
        "web_images": {},
        "web_image_sources": {},
        "web_image_raw_hashes": {},
    }


def _ensure_loaded():
    global _STATE
    if _STATE is not None:
        return
    os.makedirs(os.path.dirname(_STORE_PATH), exist_ok=True)
    if not os.path.exists(_STORE_PATH):
        _STATE = _default_state()
        return
    try:
        with open(_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            _STATE = _default_state()
            return
        base = _default_state()
        for k in base.keys():
            v = data.get(k)
            base[k] = v if isinstance(v, dict) else {}
        legacy_images = data.get("image_descriptions")
        if isinstance(legacy_images, dict):
            for media_hash, entry in legacy_images.items():
                if not isinstance(entry, dict):
                    continue
                key = str(media_hash or "").strip().lower()
                if not key or key in base["web_images"]:
                    continue
                base["web_images"][key] = {
                    "file_path": str(entry.get("file_path") or "").strip(),
                    "description": str(entry.get("description") or "").strip(),
                    "created_at": str(entry.get("created_at") or _now_iso()).strip() or _now_iso(),
                    "last_used_at": str(entry.get("last_used_at") or "").strip(),
                    "use_count": int(entry.get("use_count", 0) or 0),
                }
        _STATE = base
    except Exception:
        _STATE = _default_state()


def _save():
    os.makedirs(os.path.dirname(_STORE_PATH), exist_ok=True)
    tmp = f"{_STORE_PATH}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_STATE, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _STORE_PATH)


def _touch(entry: Dict):
    entry["last_used_at"] = _now_iso()
    entry["use_count"] = int(entry.get("use_count", 0) or 0) + 1


def get_web_image_asset(media_hash: str) -> Optional[Dict]:
    key = str(media_hash or "").strip().lower()
    if not key:
        return None
    with _LOCK:
        _ensure_loaded()
        entry = (_STATE["web_images"]).get(key)
        if not isinstance(entry, dict):
            return None
        _touch(entry)
        return {
            "description": str(entry.get("description") or "").strip(),
            "file_path": str(entry.get("file_path") or "").strip(),
        }


def upsert_web_image_asset(media_hash: str, file_path: str = "", description: str = ""):
    key = str(media_hash or "").strip().lower()
    if not key:
        return
    with _LOCK:
        _ensure_loaded()
        bucket = _STATE["web_images"]
        entry = bucket.get(key) or {"created_at": _now_iso(), "use_count": 0}
        path = str(file_path or "").strip()
        desc = str(description or "").strip()
        if path:
            entry["file_path"] = path
        if desc:
            entry["description"] = desc
        _touch(entry)
        bucket[key] = entry
        _save()


def get_web_image_source(source_url: str) -> Optional[Dict]:
    key = str(source_url or "").strip()
    if not key:
        return None
    with _LOCK:
        _ensure_loaded()
        entry = (_STATE["web_image_sources"]).get(key)
        if not isinstance(entry, dict):
            return None
        _touch(entry)
        return {
            "media_hash": str(entry.get("media_hash") or "").strip().lower(),
            "description": str(entry.get("description") or "").strip(),
        }


def upsert_web_image_source(source_url: str, media_hash: str, description: str = ""):
    key = str(source_url or "").strip()
    h = str(media_hash or "").strip().lower()
    if not key or not h:
        return
    with _LOCK:
        _ensure_loaded()
        bucket = _STATE["web_image_sources"]
        entry = bucket.get(key) or {"created_at": _now_iso(), "use_count": 0}
        entry["media_hash"] = h
        desc = str(description or "").strip()
        if desc:
            entry["description"] = desc
        _touch(entry)
        bucket[key] = entry
        _save()


def get_web_raw_hash(raw_hash: str) -> Optional[str]:
    key = str(raw_hash or "").strip().lower()
    if not key:
        return None
    with _LOCK:
        _ensure_loaded()
        entry = (_STATE["web_image_raw_hashes"]).get(key)
        if not isinstance(entry, dict):
            return None
        _touch(entry)
        return str(entry.get("media_hash") or "").strip().lower() or None


def upsert_web_raw_hash(raw_hash: str, media_hash: str):
    key = str(raw_hash or "").strip().lower()
    h = str(media_hash or "").strip().lower()
    if not key or not h:
        return
    with _LOCK:
        _ensure_loaded()
        bucket = _STATE["web_image_raw_hashes"]
        entry = bucket.get(key) or {"created_at": _now_iso(), "use_count": 0}
        entry["media_hash"] = h
        _touch(entry)
        bucket[key] = entry
        _save()
