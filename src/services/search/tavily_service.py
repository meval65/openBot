import time
import datetime
import logging
import threading
import os
import json
import hashlib
from typing import Any, Dict, List, Union

from tavily import TavilyClient

from src.config import TAVILY_API_KEYS, HEALTH_DIR

logger = logging.getLogger(__name__)

VALID_TOPICS = {"general", "news", "finance"}
VALID_SEARCH_DEPTH = {"basic", "advanced", "fast", "ultra-fast"}
VALID_TIME_RANGE = {"none", "day", "week", "month", "year"}


def _tool_error(code: str, message: str) -> str:
    safe_code = str(code or "unknown").strip().lower().replace(" ", "_")
    safe_message = str(message or "Terjadi kegagalan pada web search.").strip()
    return f"[TOOL_ERROR search_web code={safe_code}] {safe_message}"


def _seconds_until_next_month() -> float:
    now = datetime.datetime.now()
    if now.month == 12:
        first_next_month = datetime.datetime(now.year + 1, 1, 1)
    else:
        first_next_month = datetime.datetime(now.year, now.month + 1, 1)
    return max(0.0, (first_next_month - now).total_seconds())


def _classify_tavily_error(error_text: str) -> str:
    msg = (error_text or "").lower()
    if "429" in msg or "too many requests" in msg or "rate limit" in msg:
        return "rate_limit"
    if "quota" in msg or "usage limit" in msg or "credit" in msg:
        return "quota"
    if "401" in msg or "403" in msg or "unauthorized" in msg or "forbidden" in msg or "invalid api key" in msg:
        return "auth"
    if "timeout" in msg or "timed out" in msg or "connection" in msg or "dns" in msg or "temporary" in msg:
        return "transient"
    return "unknown"


class TavilySearchService:
    def __init__(self):
        self.api_keys = TAVILY_API_KEYS
        self._lock = threading.Lock()
        self._key_index = 0
        self._failures: dict[int, int] = {i: 0 for i in range(len(self.api_keys))}
        self._blacklisted_until: dict[int, float] = {i: 0.0 for i in range(len(self.api_keys))}

        os.makedirs(HEALTH_DIR, exist_ok=True)
        self._state_path = os.path.join(HEALTH_DIR, "tavily_health.json")
        joined = "|".join(self.api_keys)
        self._keys_hash = hashlib.sha256(joined.encode("utf-8")).hexdigest() if joined else ""
        self._load_state()

    def _load_state(self):
        if not self.api_keys or not os.path.exists(self._state_path):
            return
        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if str(payload.get("keys_hash") or "") != self._keys_hash:
                logger.info("[TAVILY] API keys changed; health state reset.")
                return

            key_index = int(payload.get("key_index", 0) or 0)
            if 0 <= key_index < len(self.api_keys):
                self._key_index = key_index

            failures = payload.get("failures", {})
            blacklisted = payload.get("blacklisted_until", {})
            for i in range(len(self.api_keys)):
                self._failures[i] = int(failures.get(str(i), failures.get(i, 0)) or 0)
                self._blacklisted_until[i] = float(blacklisted.get(str(i), blacklisted.get(i, 0.0)) or 0.0)
            logger.info("[TAVILY] Loaded persisted key health state.")
        except Exception as e:
            logger.warning(f"[TAVILY] Failed to load persisted state: {e}")

    def _save_state(self):
        if not self.api_keys:
            return
        try:
            payload = {
                "keys_hash": self._keys_hash,
                "key_index": int(self._key_index),
                "failures": {str(k): int(v) for k, v in self._failures.items()},
                "blacklisted_until": {str(k): float(v) for k, v in self._blacklisted_until.items()},
                "updated_at": time.time(),
            }
            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception as e:
            logger.warning(f"[TAVILY] Failed to save state: {e}")

    def _current_key(self) -> str | None:
        with self._lock:
            if not self.api_keys:
                return None
            return self.api_keys[self._key_index]

    def _rotate(self) -> bool:
        with self._lock:
            if not self.api_keys:
                return False

            now = time.time()
            total = len(self.api_keys)
            for offset in range(1, total + 1):
                candidate = (self._key_index + offset) % total
                if now >= self._blacklisted_until.get(candidate, 0.0):
                    self._key_index = candidate
                    logger.info(f"[TAVILY] Rotated to key #{candidate + 1}")
                    self._save_state()
                    return True

            best = min(self._blacklisted_until, key=lambda k: self._blacklisted_until[k])
            wait = max(0, self._blacklisted_until[best] - now)
            logger.warning(f"[TAVILY] All keys are in penalty window. Forcing key #{best + 1} (wait ~{wait:.0f}s)")
            self._key_index = best
            self._save_state()
            return False

    def _mark_failure(self, key_index: int, error_kind: str):
        with self._lock:
            if key_index not in self._failures:
                return

            self._failures[key_index] = self._failures.get(key_index, 0) + 1
            key_preview = self.api_keys[key_index][:8] + "..." if self.api_keys else "?"

            if error_kind == "rate_limit":
                penalty = 60
                reason = "Rate limit"
            elif error_kind == "quota":
                penalty = _seconds_until_next_month()
                days = penalty / 86400 if penalty else 0
                reason = f"Quota exhausted (reset next month, ~{days:.1f} days)"
            elif error_kind == "auth":
                penalty = 6 * 3600
                reason = "Authentication/authorization error"
            elif error_kind == "transient":
                penalty = min(30 * self._failures[key_index], 300)
                reason = "Transient upstream/network error"
            else:
                penalty = min(45 * self._failures[key_index], 300)
                reason = "Unknown API error"

            self._blacklisted_until[key_index] = time.time() + penalty
            logger.warning(
                f"[TAVILY] Key #{key_index + 1} ({key_preview}) blacklisted. "
                f"Reason: {reason}. Cooldown: {penalty:.0f}s"
            )
            self._save_state()

    def search(
        self,
        query: str,
        topic: str = "general",
        max_results: int = 5,
        search_depth: str = "basic",
        time_range: str = "none",
        include_image: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        if not self.api_keys:
            return _tool_error("not_configured", "Web search tidak tersedia karena TAVILY_API_KEY belum dikonfigurasi.")

        query = str(query or "").strip()
        if not query:
            return _tool_error("empty_query", "Query web search kosong.")

        topic = (topic or "general").strip().lower()
        if topic not in VALID_TOPICS:
            logger.warning(f"[TAVILY] Invalid topic '{topic}', fallback to 'general'")
            topic = "general"

        depth = str(search_depth or "basic").strip().lower()
        if depth not in VALID_SEARCH_DEPTH:
            logger.warning(f"[TAVILY] Invalid search_depth '{search_depth}', fallback to 'basic'")
            depth = "basic"

        normalized_range = str(time_range or "none").strip().lower()
        if normalized_range not in VALID_TIME_RANGE:
            logger.warning(f"[TAVILY] Invalid time_range '{time_range}', fallback to 'none'")
            normalized_range = "none"

        include_images = bool(include_image)

        try:
            safe_max_results = int(max_results or 5)
        except (TypeError, ValueError):
            safe_max_results = 5
        safe_max_results = max(1, min(safe_max_results, 10))

        max_attempts = max(3, len(self.api_keys) + 1)
        for attempt in range(max_attempts):
            key = self._current_key()
            if not key:
                return _tool_error("no_key", "Web search tidak tersedia karena tidak ada API key yang bisa dipakai.")

            with self._lock:
                current_idx = self._key_index

            try:
                client = TavilyClient(api_key=key)
                search_kwargs = dict(
                    query=query,
                    topic=topic,
                    include_answer=True,
                    search_depth=depth,
                    max_results=safe_max_results,
                )
                if normalized_range != "none":
                    search_kwargs["time_range"] = normalized_range
                if include_images:
                    search_kwargs["include_images"] = True
                    search_kwargs["include_image_descriptions"] = True
                result = client.search(**search_kwargs)

                lines: List[str] = []
                answer = result.get("answer", "")
                if answer:
                    lines.append(f"**Ringkasan:** {answer}\n")

                results_list = result.get("results", [])
                if not results_list and not answer:
                    return "Tidak ada hasil web yang relevan ditemukan untuk query tersebut."

                for r in results_list:
                    title = r.get("title", "")
                    url = r.get("url", "")
                    content = r.get("content", "")[:400]
                    lines.append(f"**{title}**\n{url}\n{content}")

                image_items: List[Dict[str, str]] = []
                if include_images and isinstance(result, dict):
                    for img in (result.get("images", []) or [])[:8]:
                        if isinstance(img, str):
                            image_items.append({"url": img, "description": ""})
                        elif isinstance(img, dict):
                            image_items.append(
                                {
                                    "url": str(img.get("url") or img.get("image_url") or "").strip(),
                                    "description": str(
                                        img.get("description")
                                        or img.get("image_description")
                                        or ""
                                    ).strip(),
                                }
                            )

                with self._lock:
                    self._failures[current_idx] = 0
                    self._blacklisted_until[current_idx] = 0.0
                    self._save_state()

                return {
                    "text": "\n\n".join(lines),
                    "images": [x for x in image_items if x.get("url")],
                }

            except Exception as e:
                error_kind = _classify_tavily_error(str(e))
                logger.warning(f"[TAVILY] Key #{current_idx + 1} attempt {attempt + 1} failed ({error_kind}): {e}")
                self._mark_failure(current_idx, error_kind=error_kind)
                self._rotate()
                time.sleep(1.0)

        return _tool_error("upstream_failed", "Web search gagal setelah beberapa percobaan ke layanan pencarian.")
