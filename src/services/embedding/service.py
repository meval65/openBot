import os
import json
import logging
import hashlib
import functools
import threading
import re
from typing import List, Dict, Optional
from collections import deque

import time
from google import genai
from google.genai import types

from src.config import (
    GOOGLE_API_KEYS,
    EMBEDDING_MODEL,
    EMBEDDING_MAX_TEXT_LENGTH,
    EMBEDDING_OUTPUT_DIM,
    EMBEDDING_CACHE_MAX_SIZE,
    EMBEDDING_CACHE_TTL_SECONDS,
    EMBEDDING_INFLIGHT_WAIT_SECONDS,
    EMBEDDING_MIN_TEXT_LEN,
    EMBEDDING_SOFT_TPM_LIMIT,
    EMBEDDING_SOFT_RPM_LIMIT,
    EMBEDDING_CACHE_DIR,
    EMBEDDING_CACHE_PATH,
)
from src.utils.api_error_policy import (
    classify_api_error,
    handle_api_error_retry,
)
from src.utils.api_key_monitor import get_shared_api_key_monitor

logger = logging.getLogger(__name__)


class MemoryAnalyzer:
    def __init__(self):
        self.api_keys = GOOGLE_API_KEYS
        if not self.api_keys:
            raise ValueError("[EMBED] No API keys configured")
        
        self.current_key_index = 0
        self.health_monitor = get_shared_api_key_monitor(self.api_keys, monitor_id="gemini")
        self._client_state_lock = threading.Lock()
        self._initialize_client()

        self.MAX_TEXT_LENGTH = EMBEDDING_MAX_TEXT_LENGTH
        self.expected_dimension = None
        self.embedding_cache: Dict[str, tuple[float, List[float]]] = {}
        self._cache_ttl_sec = max(60, int(EMBEDDING_CACHE_TTL_SECONDS))
        self._cache_max_size = max(100, int(EMBEDDING_CACHE_MAX_SIZE))
        self._cache_lock = threading.RLock()
        self._inflight_lock = threading.Lock()
        self._inflight_events: Dict[str, threading.Event] = {}
        self._disk_cache_path = EMBEDDING_CACHE_PATH
        self._last_disk_flush = 0.0
        self._soft_tpm_limit = max(1000, int(EMBEDDING_SOFT_TPM_LIMIT))
        self._soft_rpm_limit = max(1, int(EMBEDDING_SOFT_RPM_LIMIT))
        self._rate_lock = threading.Lock()
        self._rate_window = deque()  # (ts, tokens)
        self._load_disk_cache()

    def _initialize_client(self):
        with self._client_state_lock:
            self.client = genai.Client(api_key=self.api_keys[self.current_key_index])
            active_index = int(self.current_key_index)
        logger.info("Embedding siap menggunakan API key #%d", active_index + 1)

    def _get_client_snapshot(self) -> tuple[int, genai.Client]:
        with self._client_state_lock:
            key_index = int(self.current_key_index)
            api_key = str(self.api_keys[key_index])
        return key_index, genai.Client(api_key=api_key)

    def _rotate_api_key(self) -> bool:
        with self._client_state_lock:
            current_index = int(self.current_key_index)
            total_keys = len(self.api_keys)
        new_key_index = self.health_monitor.get_healthy_key(
            current_index, total_keys
        )
        if new_key_index is None:
            logger.warning("Tidak ada API key embedding yang sehat untuk dipakai saat ini.")
            return False

        try:
            with self._client_state_lock:
                self.current_key_index = int(new_key_index)
                self.client = genai.Client(api_key=self.api_keys[self.current_key_index])
                active_index = int(self.current_key_index)
            logger.warning("Embedding berpindah ke API key #%d", active_index + 1)
            return True
        except Exception as e:
            logger.error("Gagal mengganti API key embedding: %s", e)
            return False

    @staticmethod
    @functools.lru_cache(maxsize=1024)
    def _cached_text_embedding_key(text: str) -> str:
        """LRU-cached full SHA256 hash of text content â€” avoids re-hashing identical inputs."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    @staticmethod
    def _looks_low_signal_text(text: str) -> bool:
        cleaned = (text or "").strip()
        if not cleaned:
            return True
        if len(cleaned) < EMBEDDING_MIN_TEXT_LEN:
            # Allow very short but meaningful numeric/alpha strings only if they have >=2 alnum chars
            alnum_count = len(re.findall(r"[A-Za-z0-9]", cleaned))
            return alnum_count < 2
        if len(cleaned) <= 12:
            low_signal_tokens = {
                "ok", "oke", "sip", "yes", "no", "ya", "gk", "ga", "hmm", "hmmm",
                "wkwk", "haha", "lol", "hi", "halo", "test", "tes",
            }
            if cleaned.lower() in low_signal_tokens:
                return True
        return False

    def _load_disk_cache(self):
        try:
            os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
            if not os.path.exists(self._disk_cache_path):
                return
            if os.path.getsize(self._disk_cache_path) == 0:
                # Empty file can happen after interrupted write; treat as cold start.
                return
            with open(self._disk_cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return
            now_ts = time.time()
            loaded = 0
            for k, v in data.items():
                if not isinstance(v, dict):
                    continue
                ts = float(v.get("ts", 0.0) or 0.0)
                vec = v.get("vec")
                if not isinstance(vec, list) or not vec:
                    continue
                if (now_ts - ts) > self._cache_ttl_sec:
                    continue
                self.embedding_cache[str(k)] = (ts, vec)
                loaded += 1
                if loaded >= self._cache_max_size:
                    break
            if loaded:
                logger.info("[EMBED] Loaded %d cache item(s) from disk.", loaded)
        except json.JSONDecodeError as e:
            # Auto-heal corrupted/partial cache file so warning doesn't repeat forever.
            logger.warning(f"[EMBED] Disk cache corrupt, resetting cache file: {e}")
            try:
                bad_path = f"{self._disk_cache_path}.bad"
                if os.path.exists(bad_path):
                    os.remove(bad_path)
                os.replace(self._disk_cache_path, bad_path)
            except Exception:
                try:
                    os.remove(self._disk_cache_path)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"[EMBED] Failed to load disk cache: {e}")

    def _flush_disk_cache(self, force: bool = False):
        now_ts = time.time()
        if not force and (now_ts - self._last_disk_flush) < 5.0:
            return
        try:
            os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
            with self._cache_lock:
                payload = {
                    k: {"ts": ts, "vec": vec}
                    for k, (ts, vec) in self.embedding_cache.items()
                }
            tmp_path = f"{self._disk_cache_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._disk_cache_path)
            self._last_disk_flush = now_ts
        except Exception as e:
            logger.warning(f"[EMBED] Failed to flush disk cache: {e}")

    def _get_cached_embedding(self, cache_key: str) -> Optional[List[float]]:
        if not cache_key:
            return None
        with self._cache_lock:
            item = self.embedding_cache.get(cache_key)
            if not item:
                return None
            ts, vec = item
            if (time.time() - ts) > self._cache_ttl_sec:
                self.embedding_cache.pop(cache_key, None)
                return None
            return vec

    def get_cached_text_embedding(self, text: str) -> Optional[List[float]]:
        clean_text = self._preprocess_text(str(text or ""))
        if not clean_text or self._looks_low_signal_text(clean_text):
            return None
        cache_key = self._cached_text_embedding_key(clean_text)
        return self._get_cached_embedding(cache_key)

    def _set_cached_embedding(self, cache_key: str, vector: List[float]):
        if not cache_key or not vector:
            return
        now_ts = time.time()
        with self._cache_lock:
            if len(self.embedding_cache) >= self._cache_max_size:
                # Evict ~15% oldest entries
                oldest_keys = sorted(
                    self.embedding_cache.items(),
                    key=lambda kv: kv[1][0]
                )[: max(1, int(self._cache_max_size * 0.15))]
                for key, _ in oldest_keys:
                    self.embedding_cache.pop(key, None)
            self.embedding_cache[cache_key] = (now_ts, vector)
            
        # Debounce disk flush
        if hasattr(self, '_flush_timer') and self._flush_timer:
            self._flush_timer.cancel()
            
        def _do_flush():
            try:
                self._flush_disk_cache()
            except Exception as e:
                logger.error(f"[EMB-CACHE] Flush error: {e}")
                
        self._flush_timer = threading.Timer(5.0, _do_flush)
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def _preprocess_text(self, text: str) -> str:
        clean_text = text.replace("\n", " ").replace("\r", " ")
        clean_text = " ".join(clean_text.split()).strip()
        if len(clean_text) > self.MAX_TEXT_LENGTH:
            clean_text = clean_text[:self.MAX_TEXT_LENGTH]
        return clean_text

    def _build_content_parts(self, content, content_type: str = "text") -> Optional[list]:
        try:
            if content_type == "text":
                if not content or not str(content).strip():
                    return None
                clean = self._preprocess_text(str(content))
                return [clean] if clean else None

        except Exception as e:
            logger.error(f"[EMBED] Failed to build content parts for {content_type}: {e}")
            return None
        return None

    @staticmethod
    def _estimate_embed_tokens_for_text(text: str) -> int:
        clean = (text or "").strip()
        if not clean:
            return 0
        # Simple conservative estimate for TPM budgeting.
        return max(1, len(clean) // 4)

    def _estimate_embed_tokens_for_contents(self, contents: list) -> int:
        total = 0
        for c in contents or []:
            if isinstance(c, str):
                total += self._estimate_embed_tokens_for_text(c)
            else:
                # Non-text content (image/video/pdf bytes part): fixed conservative budget unit.
                total += 800
        return max(1, total)

    def _throttle_embed_budget(self, est_tokens: int):
        if self._soft_rpm_limit <= 0 or self._soft_tpm_limit <= 0:
            return
        est_tokens = max(1, int(est_tokens))
        while True:
            now_ts = time.time()
            with self._rate_lock:
                while self._rate_window and (now_ts - self._rate_window[0][0]) >= 60.0:
                    self._rate_window.popleft()

                req_count = len(self._rate_window)
                tok_count = sum(t for _, t in self._rate_window)

                can_req = req_count < self._soft_rpm_limit
                can_tok = (tok_count + est_tokens) <= self._soft_tpm_limit
                if can_req and can_tok:
                    self._rate_window.append((now_ts, est_tokens))
                    return

                wait_candidates = []
                if self._rate_window:
                    oldest_ts = self._rate_window[0][0]
                    wait_candidates.append(max(0.05, 60.0 - (now_ts - oldest_ts)))
                else:
                    wait_candidates.append(0.2)

                if not can_tok:
                    running = tok_count
                    for ts, tok in self._rate_window:
                        running -= tok
                        if (running + est_tokens) <= self._soft_tpm_limit:
                            wait_candidates.append(max(0.05, 60.0 - (now_ts - ts)))
                            break
                sleep_for = max(0.05, min(wait_candidates))
            time.sleep(sleep_for)

    def get_embedding(self, content, content_type: str = "text", use_cache: bool = True) -> List[float]:
        if content is None:
            return []
        if content_type != "text":
            logger.info("[EMBED] Multimodal embedding dinonaktifkan. content_type=%s", content_type)
            return []

        if content_type == "text" and isinstance(content, str) and not content.strip():
            return []

        cache_key = None
        clean_text = None
        owns_inflight_key = False
        if content_type == "text" and isinstance(content, str):
            clean_text = self._preprocess_text(content)
            if self._looks_low_signal_text(clean_text):
                logger.debug("[EMBED] Skip low-signal text input.")
                return []

        if use_cache and content_type == "text" and isinstance(content, str):
            cache_key = self._cached_text_embedding_key((clean_text or content).strip())
            cached_vec = self._get_cached_embedding(cache_key)
            if cached_vec is not None:
                return cached_vec

            # In-flight dedup: if same key is already computing, wait briefly and reuse.
            leader = False
            with self._inflight_lock:
                inflight_event = self._inflight_events.get(cache_key)
                if inflight_event is None:
                    inflight_event = threading.Event()
                    self._inflight_events[cache_key] = inflight_event
                    leader = True
                    owns_inflight_key = True

            if not leader:
                inflight_event.wait(timeout=max(1, EMBEDDING_INFLIGHT_WAIT_SECONDS))
                cached_after_wait = self._get_cached_embedding(cache_key)
                if cached_after_wait is not None:
                    return cached_after_wait

        try:
            parts = self._build_content_parts(content, content_type)
            if not parts:
                return []

            config = types.EmbedContentConfig(output_dimensionality=EMBEDDING_OUTPUT_DIM)
            max_attempts = max(3, len(self.api_keys) + 1)
            for attempt in range(max_attempts):
                request_key_index = int(getattr(self, "current_key_index", 0) or 0)
                try:
                    self._throttle_embed_budget(self._estimate_embed_tokens_for_contents(parts))
                    request_key_index, request_client = self._get_client_snapshot()
                    response = request_client.models.embed_content(
                        model=EMBEDDING_MODEL,
                        contents=parts,
                        config=config,
                    )
                    self.health_monitor.mark_success(request_key_index)

                    embedding = response.embeddings[0].values

                    if not embedding:
                        logger.error("[EMBED] Empty embedding returned")
                        return []

                    if self.expected_dimension is None:
                        self.expected_dimension = len(embedding)
                    elif len(embedding) != self.expected_dimension:
                        logger.error(
                            f"[EMBED] Dimension mismatch: expected {self.expected_dimension}, got {len(embedding)}"
                        )
                        return []

                    result_list = list(embedding)
                    if cache_key is not None:
                        self._set_cached_embedding(cache_key, result_list)

                    return result_list

                except Exception as e:
                    classification = classify_api_error(e)
                    reason_code = str(classification.get("reason_code") or "other")
                    logger.error(
                        f"[API-KEY-{request_key_index+1}] [EMBED-FAIL] Error ({content_type}) Attempt {attempt+1}: {e}",
                        exc_info=(attempt == 1),
                    )

                    if reason_code == "quota_exhausted":
                        logger.warning("[EMBED] Model embedding kena quota/resource exhausted, coba ulang sebentar lagi.")
                    if handle_api_error_retry(
                        self,
                        reason_code=reason_code,
                        key_index=request_key_index,
                        attempt=attempt,
                        base_retry_delay=1.0,
                        rotate_sleep_seconds=1.0,
                        quota_retry_delay=1.0,
                    ):
                        continue
                    time.sleep(1.0 + attempt)
            return []
        finally:
            if owns_inflight_key:
                self._release_inflight_key(cache_key)

    def _release_inflight_key(self, cache_key: Optional[str]):
        if not cache_key:
            return
        with self._inflight_lock:
            ev = self._inflight_events.pop(cache_key, None)
            if ev:
                ev.set()


