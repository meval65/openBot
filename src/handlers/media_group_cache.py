import time
from typing import Any, Dict, List, Optional

from telegram import Message

_CACHE_KEY = "media_group_cache"
_TTL_SECONDS = 900
_MAX_GROUPS = 200
_MAX_ITEMS_PER_GROUP = 20


def _normalize_desc(msg: Message) -> Optional[Dict[str, Any]]:
    if msg.photo:
        p = msg.photo[-1]
        return {
            "kind": "image",
            "file_id": getattr(p, "file_id", ""),
            "file_name": "",
            "mime_type": "image/jpeg",
            "ext_fallback": ".jpg",
        }

    if msg.sticker:
        st = msg.sticker
        if getattr(st, "is_animated", False):
            return {
                "kind": "unsupported_tgs",
                "file_id": getattr(st, "file_id", ""),
                "file_name": "",
                "mime_type": "",
                "ext_fallback": "",
            }
        if getattr(st, "is_video", False):
            return {
                "kind": "video",
                "file_id": getattr(st, "file_id", ""),
                "file_name": "",
                "mime_type": "video/webm",
                "ext_fallback": ".webm",
            }
        return {
            "kind": "image",
            "file_id": getattr(st, "file_id", ""),
            "file_name": "",
            "mime_type": "image/webp",
            "ext_fallback": ".webp",
        }

    if msg.video:
        v = msg.video
        return {
            "kind": "video",
            "file_id": getattr(v, "file_id", ""),
            "file_name": str(getattr(v, "file_name", "") or ""),
            "mime_type": str(getattr(v, "mime_type", "") or "video/mp4"),
            "ext_fallback": ".mp4",
        }

    if msg.animation:
        a = msg.animation
        return {
            "kind": "video",
            "file_id": getattr(a, "file_id", ""),
            "file_name": str(getattr(a, "file_name", "") or ""),
            "mime_type": str(getattr(a, "mime_type", "") or "video/mp4"),
            "ext_fallback": ".mp4",
        }

    if msg.document:
        d = msg.document
        mime = str(getattr(d, "mime_type", "") or "").lower()
        name = str(getattr(d, "file_name", "") or "")
        if mime.startswith("image/"):
            kind = "image"
            fallback = ".jpg"
        elif mime.startswith("video/"):
            kind = "video"
            fallback = ".mp4"
        else:
            return None
        return {
            "kind": kind,
            "file_id": getattr(d, "file_id", ""),
            "file_name": name,
            "mime_type": mime,
            "ext_fallback": fallback,
        }

    return None


def register_media_group_message(context, message: Optional[Message]):
    if not message:
        return
    mgid = str(getattr(message, "media_group_id", "") or "").strip()
    if not mgid:
        return
    desc = _normalize_desc(message)
    if not desc:
        return

    chat_id = getattr(getattr(message, "chat", None), "id", None)
    if chat_id is None:
        return
    key = f"{chat_id}:{mgid}"

    cache = context.application.bot_data.setdefault(_CACHE_KEY, {})
    if not isinstance(cache, dict):
        cache = {}
        context.application.bot_data[_CACHE_KEY] = cache

    now = time.time()
    # cleanup old
    stale_keys = [k for k, v in cache.items() if now - float(v.get("ts", 0)) > _TTL_SECONDS]
    for k in stale_keys:
        cache.pop(k, None)

    bucket = cache.setdefault(key, {"ts": now, "items": []})
    bucket["ts"] = now
    items = bucket.setdefault("items", [])
    if not isinstance(items, list):
        items = []
        bucket["items"] = items

    # anti-duplicate by file_id
    file_id = str(desc.get("file_id") or "")
    if file_id and any(str(it.get("file_id") or "") == file_id for it in items):
        return

    if len(items) < _MAX_ITEMS_PER_GROUP:
        items.append(desc)

    if len(cache) > _MAX_GROUPS:
        # drop oldest entries
        ordered = sorted(cache.items(), key=lambda kv: float(kv[1].get("ts", 0)))
        for k, _ in ordered[: max(0, len(cache) - _MAX_GROUPS)]:
            cache.pop(k, None)


def get_media_group_items(context, message: Optional[Message]) -> List[Dict[str, Any]]:
    if not message:
        return []
    mgid = str(getattr(message, "media_group_id", "") or "").strip()
    chat_id = getattr(getattr(message, "chat", None), "id", None)
    if not mgid or chat_id is None:
        return []

    key = f"{chat_id}:{mgid}"
    cache = context.application.bot_data.get(_CACHE_KEY, {})
    if not isinstance(cache, dict):
        return []
    bucket = cache.get(key, {})
    items = bucket.get("items", [])
    if not isinstance(items, list):
        return []
    return [it for it in items if isinstance(it, dict)]
