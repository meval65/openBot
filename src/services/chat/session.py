import atexit
import json
import os
import logging
import threading
import datetime
import re
import time
from collections import deque
from typing import List, Optional, Dict, Any, Callable

from src.config import (
    SESSION_DIR,
    MAX_HISTORY_LEN,
    HISTORY_TOKEN_BUDGET,
    HISTORY_SUMMARY_TRIGGER_RATIO,
    HISTORY_SUMMARY_TARGET_RATIO,
    HISTORY_SUMMARY_MIN_EXTRACT,
    HISTORY_EST_CHARS_PER_TOKEN,
)
from src.utils.time_utils import format_human_time, now_local

logger = logging.getLogger(__name__)

_LEADING_INTERNAL_TIME_TAG_RE = re.compile(r"^\s*(?:\[(?:t:\d{6,}|time:[^\]]+)\]\s*)+")
_SYSTEM_BLOCK_RE = re.compile(
    r"\[SYSTEM\].*?\[END SYSTEM\]",
    re.IGNORECASE | re.DOTALL,
)
_INTERNAL_HEADER_LINE_RE = re.compile(
    r"^\s*\[INTERNAL:[^\]]+\]\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_INLINE_TOOL_CALL_RE = re.compile(
    r"(?:`?\s*(?:ai_personal_computer|search_web|create_schedule|list_schedules|cancel_schedule|save_memory|list_memories|forget_memory|update_memory|inspect_images_from_ai_personal_computer|send_files_from_ai_personal_computer|announce_action)\s*`?)\s*\([^`]*\)\s*`?",
    re.IGNORECASE,
)
_PERSISTED_EXTRA_METADATA_KEYS = {
    "user_profile_context",
    "visual_token_factor",
    "user_profile_summary",
    "user_profile_update_score",
    "user_profile_summary_updated_at",
}


def _is_sticker_media_path(path: str) -> bool:
    norm = str(path or "").replace("\\", "/").lower()
    base = os.path.basename(norm)
    return (
        base.startswith("sticker_")
        or base.startswith("stickerthumb_")
        or base.startswith("sticker_thumb_")
        or "/stickers/" in norm
    )


class SessionManager:
    def __init__(self, max_history: int = MAX_HISTORY_LEN):
        self.session_data: deque = deque()
        self.meta_data: Dict[str, Any] = {}
        self.lock = threading.Lock()

        self.MAX_HISTORY = max_history
        self.HISTORY_TOKEN_BUDGET = max(1000, int(HISTORY_TOKEN_BUDGET))
        self.HISTORY_SUMMARY_TRIGGER_RATIO = min(0.99, max(0.5, float(HISTORY_SUMMARY_TRIGGER_RATIO)))
        self.HISTORY_SUMMARY_TARGET_RATIO = min(
            self.HISTORY_SUMMARY_TRIGGER_RATIO - 0.01,
            max(0.3, float(HISTORY_SUMMARY_TARGET_RATIO)),
        )
        self.HISTORY_SUMMARY_MIN_EXTRACT = max(2, int(HISTORY_SUMMARY_MIN_EXTRACT))
        self.HISTORY_EST_CHARS_PER_TOKEN = max(2, int(HISTORY_EST_CHARS_PER_TOKEN))
        self.SESSION_DIR = SESSION_DIR
        self.SESSION_FILE = os.path.join(SESSION_DIR, "session.json")
        self._last_save_time = 0.0
        self._save_debounce_seconds = 0.75
        self._save_timer: Optional[threading.Timer] = None

        atexit.register(self.flush)

        os.makedirs(self.SESSION_DIR, exist_ok=True)
        self._is_loaded = False
        try:
            self._load_session()
        except Exception as e:
            logger.warning(f"[SESSION] Eager load failed, fallback to lazy load: {e}")

    def get_lock(self) -> threading.Lock:
        return self.lock

    def get_session(self) -> deque:
        if not self._is_loaded:
            self._load_session()
        return self.session_data

    def get_metadata(self, key: str, default: Any = None) -> Any:
        if not self._is_loaded:
            self._load_session()
        return self.meta_data.get(key, default)

    def set_metadata(self, key: str, value: Any, persist: bool = True):
        with self.get_lock():
            if not self._is_loaded:
                self._load_session()
            self.meta_data[key] = value
            if persist:
                self._save_session_to_disk()

    def update_session(
        self,
        user_text: str,
        ai_text: str,
        image_path: str = None,
        video_path: str = None,
        ai_workspace_image_path: str = None,
        ai_workspace_video_path: str = None,
        interaction_source: str = "user",
    ):
        with self.get_lock():
            if not self._is_loaded:
                self._load_session()

            now_time = format_human_time(now_local())
            user_message = self._build_user_message(
                user_text=user_text,
                image_path=image_path,
                video_path=video_path,
                ai_workspace_image_path=ai_workspace_image_path,
                ai_workspace_video_path=ai_workspace_video_path,
                now_time=now_time,
            )
            model_text = self._sanitize_model_text(ai_text)
            self.session_data.append(user_message)
            self.session_data.append({"role": "model", "parts": [model_text], "time": now_time, "media_refs": []})

            extracted = self._handle_history_limit()
            self._update_interaction_time(interaction_source=interaction_source)
            self._save_session_to_disk(force=True)
            return extracted

    def append_model_message(
        self,
        ai_text: str,
        interaction_source: str = "proactive",
    ):
        with self.get_lock():
            if not self._is_loaded:
                self._load_session()

            text = self._sanitize_model_text(ai_text)
            if not text:
                return []

            now_time = format_human_time(now_local())
            self.session_data.append({"role": "model", "parts": [text], "time": now_time, "media_refs": []})
            extracted = self._handle_history_limit()
            self._update_interaction_time(interaction_source=interaction_source)
            self._save_session_to_disk(force=True)
            return extracted

    def attach_latest_model_image_paths(self, image_paths: List[Any]) -> bool:
        with self.get_lock():
            if not self._is_loaded:
                self._load_session()

            clean_refs: List[Dict[str, str]] = []
            for item in image_paths or []:
                if isinstance(item, dict):
                    kind = str(item.get("kind") or "image").strip().lower()
                    host_path = str(item.get("host_path") or "").strip()
                    ai_workspace_path = str(item.get("ai_workspace_path") or "").strip()
                    role = str(item.get("role") or "model").strip().lower() or "model"
                    if kind in {"image", "video"} and (host_path or ai_workspace_path):
                        clean_refs.append(
                            {
                                "kind": kind,
                                "host_path": host_path,
                                "ai_workspace_path": ai_workspace_path,
                                "role": role,
                            }
                        )
                    continue
                path = str(item or "").strip()
                if path and os.path.exists(path):
                    clean_refs.append(
                        {
                            "kind": "image",
                            "host_path": path,
                            "ai_workspace_path": "",
                            "role": "model",
                        }
                    )
            if not clean_refs:
                return False

            for msg in reversed(self.session_data):
                if not isinstance(msg, dict) or msg.get("role") != "model":
                    continue
                media_refs = self._normalize_media_refs(msg)
                existing_keys = {
                    (
                        str(ref.get("kind") or "").strip().lower(),
                        str(ref.get("host_path") or "").strip(),
                        str(ref.get("ai_workspace_path") or "").strip(),
                    )
                    for ref in media_refs
                    if isinstance(ref, dict)
                }
                changed = False
                for ref in clean_refs[:5]:
                    dedupe_key = (
                        str(ref.get("kind") or "").strip().lower(),
                        str(ref.get("host_path") or "").strip(),
                        str(ref.get("ai_workspace_path") or "").strip(),
                    )
                    if dedupe_key in existing_keys:
                        continue
                    media_refs.append(ref)
                    existing_keys.add(dedupe_key)
                    changed = True
                if changed:
                    msg["media_refs"] = media_refs
                    self._save_session_to_disk(force=True)
                    return True
                return False
        return False

    def _build_user_message(
        self,
        user_text: str,
        image_path: Optional[str],
        now_time: str,
        video_path: Optional[str] = None,
        ai_workspace_image_path: Optional[str] = None,
        ai_workspace_video_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        base_text = str(user_text or "").strip()
        u_parts = [base_text] if base_text else []
        media_refs = []
        if image_path and os.path.exists(image_path) and not _is_sticker_media_path(image_path):
            media_refs.append(
                {
                    "kind": "image",
                    "host_path": image_path,
                    "ai_workspace_path": str(ai_workspace_image_path or "").strip(),
                    "role": "user",
                }
            )
        if video_path and os.path.exists(video_path) and not _is_sticker_media_path(video_path):
            media_refs.append(
                {
                    "kind": "video",
                    "host_path": video_path,
                    "ai_workspace_path": str(ai_workspace_video_path or "").strip(),
                    "role": "user",
                }
            )
        return {"role": "user", "parts": u_parts, "time": now_time, "media_refs": media_refs}

    @staticmethod
    def _sanitize_model_text(text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        cleaned = _LEADING_INTERNAL_TIME_TAG_RE.sub("", raw).strip()
        cleaned = _SYSTEM_BLOCK_RE.sub("", cleaned).strip()
        cleaned = _INTERNAL_HEADER_LINE_RE.sub("", cleaned).strip()
        cleaned = _INLINE_TOOL_CALL_RE.sub("", cleaned).strip()
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    def _sanitize_history_records(self) -> bool:
        changed = False
        for msg in self.session_data:
            if not isinstance(msg, dict):
                continue
            normalized_refs = self._normalize_media_refs(msg)
            if normalized_refs != msg.get("media_refs"):
                msg["media_refs"] = normalized_refs
                changed = True
            parts = msg.get("parts", [])
            if not isinstance(parts, list):
                continue
            new_parts = []
            local_changed = False
            for p in parts:
                if not isinstance(p, str):
                    new_parts.append(p)
                    continue
                cleaned = self._sanitize_model_text(p)
                new_parts.append(cleaned)
                if cleaned != p:
                    local_changed = True
            if local_changed:
                msg["parts"] = new_parts
                changed = True
        return changed

    def _normalize_media_refs(self, msg: Dict[str, Any]) -> List[Dict[str, str]]:
        refs = []
        raw_refs = msg.get("media_refs", [])
        if isinstance(raw_refs, list):
            for item in raw_refs:
                if not isinstance(item, dict):
                    continue
                kind = str(item.get("kind") or "").strip().lower()
                if kind not in {"image", "video"}:
                    continue
                host_path = str(item.get("host_path") or "").strip()
                ai_workspace_path = str(item.get("ai_workspace_path") or "").strip()
                role = str(item.get("role") or msg.get("role") or "").strip().lower() or "user"
                refs.append(
                    {
                        "kind": kind,
                        "host_path": host_path,
                        "ai_workspace_path": ai_workspace_path,
                        "role": role,
                    }
                )
        deduped = []
        seen = set()
        for ref in refs:
            key = (
                ref.get("kind", ""),
                ref.get("host_path", ""),
                ref.get("ai_workspace_path", ""),
                ref.get("role", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ref)
        return deduped

    def _handle_history_limit(self) -> List[Dict]:
        extracted = []
        if len(self.session_data) >= self.MAX_HISTORY:
            for _ in range(self.HISTORY_SUMMARY_MIN_EXTRACT):
                if self.session_data:
                    extracted.append(self.session_data.popleft())
        while len(self.session_data) > self.MAX_HISTORY:
            extracted.append(self.session_data.popleft())

        return extracted

    def _estimate_history_tokens_fallback(self, history: deque) -> int:
        total_tokens = 0
        for msg in history:
            if not isinstance(msg, dict):
                continue
            parts = msg.get("parts", [])
            media_refs = self._normalize_media_refs(msg)
            for ref in media_refs:
                if str(ref.get("kind") or "").strip().lower() == "image":
                    role = str(ref.get("role") or msg.get("role") or "").strip().lower()
                    total_tokens += 258 if role == "user" else 24
                elif str(ref.get("kind") or "").strip().lower() == "video":
                    role = str(ref.get("role") or msg.get("role") or "").strip().lower()
                    total_tokens += 512 if role == "user" else 24
            for p in parts:
                if not isinstance(p, str):
                    continue
                total_tokens += max(1, len(str(p)) // self.HISTORY_EST_CHARS_PER_TOKEN)
            if msg.get("time") is not None:
                total_tokens += 8
        return max(1, total_tokens)

    def trim_history_by_token_budget(self, token_counter: Optional[Callable[[deque], int]] = None) -> List[Dict]:
        with self.get_lock():
            if not self._is_loaded:
                self._load_session()

            counter = token_counter or self._estimate_history_tokens_fallback
            trigger_tokens = int(self.HISTORY_TOKEN_BUDGET * self.HISTORY_SUMMARY_TRIGGER_RATIO)
            target_tokens = int(self.HISTORY_TOKEN_BUDGET * self.HISTORY_SUMMARY_TARGET_RATIO)

            try:
                estimated_tokens = int(counter(self.session_data))
            except Exception as e:
                logger.warning(f"[SESSION] Native token counter failed, fallback to char estimate: {e}")
                estimated_tokens = self._estimate_history_tokens_fallback(self.session_data)

            if estimated_tokens <= trigger_tokens:
                return []

            extracted: List[Dict] = []
            extracted_count = 0
            while self.session_data and (
                estimated_tokens > target_tokens or extracted_count < self.HISTORY_SUMMARY_MIN_EXTRACT
            ):
                extracted.append(self.session_data.popleft())
                extracted_count += 1
                try:
                    estimated_tokens = int(counter(self.session_data))
                except Exception:
                    estimated_tokens = self._estimate_history_tokens_fallback(self.session_data)

            if extracted:
                self._save_session_to_disk(force=True)
            return extracted

    def _update_interaction_time(self, interaction_source: str = "user"):
        now = now_local()
        source = (interaction_source or "user").strip().lower()
        if source == "user":
            self.meta_data["last_user_interaction"] = now

    def clear_session(self):
        with self.get_lock():
            self._init_empty_session()
            if os.path.exists(self.SESSION_FILE):
                try:
                    os.remove(self.SESSION_FILE)
                except OSError as e:
                    logger.error(f"Clear error: {e}")

    def _load_session(self):
        with self.get_lock():
            if self._is_loaded:
                return

            if os.path.exists(self.SESSION_FILE):
                try:
                    with open(self.SESSION_FILE, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    self.session_data = deque(data.get("history", []))
                    self.meta_data = self._parse_metadata(data)
                    changed = False
                    if self._sanitize_history_records():
                        changed = True
                        logger.info("[SESSION] Sanitized internal time tags from model history.")
                    if changed:
                        self._save_session_to_disk(force=True)
                    self._is_loaded = True
                    return

                except (json.JSONDecodeError, OSError) as e:
                    logger.error(f"Load error: {e}")

            self._init_empty_session()
            self._is_loaded = True

    def _parse_metadata(self, data: Dict) -> Dict[str, Any]:
        meta = {
            "last_user_interaction": None,
            "last_proactive_sent_ts": None,
            "last_proactive_trigger_context": "",
            "last_proactive_trigger_ts": None,
            "rolling_summary": data.get("rolling_summary", ""),
        }

        ts = data.get("last_user_interaction_ts")
        if ts:
            try:
                parsed = datetime.datetime.fromisoformat(ts)
                meta["last_user_interaction"] = parsed
            except ValueError:
                pass

        proactive_ts = data.get("last_proactive_sent_ts")
        if proactive_ts:
            try:
                meta["last_proactive_sent_ts"] = datetime.datetime.fromisoformat(proactive_ts)
            except ValueError:
                pass

        meta["last_proactive_trigger_context"] = str(
            data.get("last_proactive_trigger_context", "") or ""
        )
        trigger_ts = data.get("last_proactive_trigger_ts")
        if trigger_ts:
            try:
                meta["last_proactive_trigger_ts"] = datetime.datetime.fromisoformat(trigger_ts)
            except ValueError:
                pass

        extra_meta = data.get("meta", {})
        if isinstance(extra_meta, dict):
            for k, v in extra_meta.items():
                if not isinstance(k, str) or k not in _PERSISTED_EXTRA_METADATA_KEYS:
                    continue
                meta[k] = v

        return meta

    def _init_empty_session(self):
        self.session_data.clear()
        self.meta_data = {
            "last_user_interaction": None,
            "last_proactive_sent_ts": None,
            "last_proactive_trigger_context": "",
            "last_proactive_trigger_ts": None,
            "rolling_summary": "",
        }

    def _schedule_delayed_save(self, delay_seconds: float):
        delay_seconds = max(0.05, float(delay_seconds))
        if self._save_timer and self._save_timer.is_alive():
            return

        def _flush_later():
            try:
                self.flush()
            except Exception as e:
                logger.error(f"Delayed save error: {e}")

        self._save_timer = threading.Timer(delay_seconds, _flush_later)
        self._save_timer.daemon = True
        self._save_timer.start()

    def _save_session_to_disk(self, force: bool = False):
        now = time.time()
        if not force:
            debounce_seconds = float(getattr(self, "_save_debounce_seconds", 0.0) or 0.0)
            elapsed = now - getattr(self, "_last_save_time", 0.0)
            if elapsed < debounce_seconds:
                self._schedule_delayed_save(debounce_seconds - elapsed)
                return
        elif self._save_timer and self._save_timer.is_alive():
            self._save_timer.cancel()
            self._save_timer = None

        try:
            last_user_int = self.meta_data.get("last_user_interaction")
            if isinstance(last_user_int, str):
                try:
                    last_user_int = datetime.datetime.fromisoformat(last_user_int)
                    self.meta_data["last_user_interaction"] = last_user_int
                except ValueError:
                    last_user_int = None

            last_proactive_sent = self.meta_data.get("last_proactive_sent_ts")
            if isinstance(last_proactive_sent, str):
                try:
                    last_proactive_sent = datetime.datetime.fromisoformat(last_proactive_sent)
                    self.meta_data["last_proactive_sent_ts"] = last_proactive_sent
                except ValueError:
                    last_proactive_sent = None

            last_trigger_ts = self.meta_data.get("last_proactive_trigger_ts")
            if isinstance(last_trigger_ts, str):
                try:
                    last_trigger_ts = datetime.datetime.fromisoformat(last_trigger_ts)
                    self.meta_data["last_proactive_trigger_ts"] = last_trigger_ts
                except ValueError:
                    last_trigger_ts = None

            core_keys = {
                "last_user_interaction",
                "last_proactive_sent_ts",
                "last_proactive_trigger_context",
                "last_proactive_trigger_ts",
                "rolling_summary",
            }
            extra_meta = {}
            for k, v in self.meta_data.items():
                if k in core_keys:
                    continue
                if k not in _PERSISTED_EXTRA_METADATA_KEYS:
                    continue
                if isinstance(v, datetime.datetime):
                    extra_meta[k] = v.isoformat()
                elif isinstance(v, (str, int, float, bool)) or v is None:
                    extra_meta[k] = v

            data = {
                "last_user_interaction_ts": last_user_int.isoformat() if isinstance(last_user_int, datetime.datetime) else None,
                "last_proactive_sent_ts": (
                    last_proactive_sent.isoformat() if isinstance(last_proactive_sent, datetime.datetime) else None
                ),
                "last_proactive_trigger_context": self.meta_data.get("last_proactive_trigger_context", ""),
                "last_proactive_trigger_ts": (
                    last_trigger_ts.isoformat() if isinstance(last_trigger_ts, datetime.datetime) else None
                ),
                "rolling_summary": self.meta_data.get("rolling_summary", ""),
                "meta": extra_meta,
                "history": list(self.session_data),
            }

            self._atomic_write(self.SESSION_FILE, data)
            self._last_save_time = time.time()
            self._save_timer = None

        except Exception as e:
            logger.error(f"Save error: {e}")

    def flush(self):
        with self.get_lock():
            if self._is_loaded:
                if self._save_timer and self._save_timer.is_alive():
                    self._save_timer.cancel()
                    self._save_timer = None
                self._save_session_to_disk(force=True)

    def _atomic_write(self, path: str, data: Dict):
        tmp_path = f"{path}.tmp"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise

    def update_rolling_summary(self, new_summary: str):
        with self.get_lock():
            self.meta_data["rolling_summary"] = new_summary
            self._save_session_to_disk()

    def mark_proactive_sent(self):
        with self.get_lock():
            self.meta_data["last_proactive_sent_ts"] = now_local()
            self._save_session_to_disk()

    def record_proactive_trigger_context(self, context: str):
        with self.get_lock():
            self.meta_data["last_proactive_trigger_context"] = str(context or "").strip()
            self.meta_data["last_proactive_trigger_ts"] = now_local()
            self._save_session_to_disk()
