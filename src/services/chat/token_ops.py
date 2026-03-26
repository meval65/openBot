import datetime
import json
import logging
import os
import time
from typing import Dict, Optional, Tuple

from google.genai import types

from src.config import CHAT_MAX_RETRIES, HEALTH_DIR
from src.config import (
    HISTORY_VISUAL_CALIBRATION_ALPHA,
    HISTORY_VISUAL_FACTOR_MAX,
    HISTORY_VISUAL_FACTOR_MIN,
    HISTORY_VISUAL_PART_TOKEN_BASE,
)
from src.utils.api_error_policy import (
    all_chat_models_in_penalty,
    classify_api_error,
    handle_api_error_retry,
)

logger = logging.getLogger(__name__)


def preload_selective(self):
    """
    Lightweight preload at boot:
    - load session/history into RAM
    - read a few core metadata values
    """
    t0 = time.perf_counter()
    try:
        _ = self.session_manager.get_session()
        _ = self.session_manager.get_metadata("last_user_interaction")
        _ = self.session_manager.get_metadata("rolling_summary")
        try:
            self.bot_config.load()
        except Exception:
            pass

        dt = (time.perf_counter() - t0) * 1000.0
        logger.info("[WARMUP] Lightweight preload completed in %.1fms", dt)
    except Exception as e:
        logger.warning(f"[WARMUP] Lightweight preload failed: {e}")


def load_token_usage_state(self) -> Dict:
    try:
        os.makedirs(HEALTH_DIR, exist_ok=True)
        if not os.path.exists(self._token_usage_path):
            return _empty_token_usage_state()
        with open(self._token_usage_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data.setdefault("monitor", "chat")
            data.setdefault("total_requests", 0)
            data.setdefault("total_input_tokens", 0)
            data.setdefault("total_output_tokens", 0)
            data.setdefault("total_tokens", 0)
            data.setdefault("last_updated", None)
            data.setdefault("requests", [])
            return data
    except Exception as e:
        logger.warning(f"[TOKEN] Failed to load token usage state: {e}")
    return _empty_token_usage_state()


def save_token_usage_state(self):
    try:
        os.makedirs(os.path.dirname(self._token_usage_path), exist_ok=True)
        with open(self._token_usage_path, "w", encoding="utf-8") as f:
            json.dump(self._token_usage, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"[TOKEN] Failed to persist token usage state: {e}")


def extract_response_usage_tokens(response) -> Tuple[int, int, int]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return 0, 0, 0

    in_tokens = int(
        getattr(usage, "prompt_token_count", 0)
        or getattr(usage, "input_token_count", 0)
        or 0
    )
    out_tokens = int(
        getattr(usage, "candidates_token_count", 0)
        or getattr(usage, "output_token_count", 0)
        or 0
    )
    total_tokens = int(
        getattr(usage, "total_token_count", 0)
        or getattr(usage, "total_tokens", 0)
        or 0
    )
    if total_tokens <= 0:
        total_tokens = in_tokens + out_tokens
    return max(0, in_tokens), max(0, out_tokens), max(0, total_tokens)


def record_token_usage(
    self,
    model: str,
    mode: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    latency_ms: Optional[float] = None,
):
    input_tokens = int(max(0, input_tokens or 0))
    output_tokens = int(max(0, output_tokens or 0))
    total_tokens = int(max(0, total_tokens or (input_tokens + output_tokens)))
    if input_tokens <= 0 and output_tokens <= 0 and total_tokens <= 0:
        return

    entry = {
        "ts": datetime.datetime.now().isoformat(),
        "model": str(model or ""),
        "mode": str(mode or "unknown"),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    if latency_ms is not None:
        entry["latency_ms"] = round(float(latency_ms), 1)

    with self._token_usage_lock:
        self._token_usage["total_requests"] = int(self._token_usage.get("total_requests", 0)) + 1
        self._token_usage["total_input_tokens"] = int(self._token_usage.get("total_input_tokens", 0)) + input_tokens
        self._token_usage["total_output_tokens"] = int(self._token_usage.get("total_output_tokens", 0)) + output_tokens
        self._token_usage["total_tokens"] = int(self._token_usage.get("total_tokens", 0)) + total_tokens
        self._token_usage["last_updated"] = entry["ts"]

        reqs = self._token_usage.get("requests", [])
        if not isinstance(reqs, list):
            reqs = []
        reqs.append(entry)
        self._token_usage["requests"] = reqs[-200:]
        self._save_token_usage_state()

    logger.debug(
        "[TOKEN] mode=%s model=%s in=%d out=%d total=%d",
        mode,
        model,
        input_tokens,
        output_tokens,
        total_tokens,
    )


def count_history_tokens_native(self, history_deque) -> int:
    contents = []
    for msg in list(history_deque or []):
        if not isinstance(msg, dict):
            continue
        role = "user" if msg.get("role") == "user" else "model"
        parts = msg.get("parts", [])
        media_refs = msg.get("media_refs", [])
        tokens_text = []
        time_tag = self._get_compact_msg_time_tag(msg)
        if time_tag:
            tokens_text.append(time_tag)
        has_structured_media_refs = isinstance(media_refs, list) and any(isinstance(ref, dict) for ref in media_refs)
        if isinstance(media_refs, list):
            for ref in media_refs:
                if not isinstance(ref, dict):
                    continue
                kind = str(ref.get("kind") or "").strip().lower()
                ai_path = str(ref.get("ai_workspace_path") or "").strip()
                if kind == "image":
                    tokens_text.append("[ai-workspace-image]" if ai_path else "[image]")
                elif kind == "video":
                    tokens_text.append("[ai-workspace-video]" if ai_path else "[video]")
        for p in parts:
            if not isinstance(p, str):
                continue
            if not has_structured_media_refs:
                tokens_text.append(p)
            else:
                tokens_text.append(p)
        if not tokens_text:
            continue
        contents.append(types.Content(role=role, parts=[types.Part(text="\n".join(tokens_text))]))

    if not contents:
        return 0

    max_attempts = min(2, max(1, CHAT_MAX_RETRIES))
    for attempt in range(max_attempts):
        active_model = self._select_chat_model_for_attempt()
        try:
            _, request_client = self._get_client_snapshot()
            resp = request_client.models.count_tokens(
                model=active_model,
                contents=contents,
            )
            total = int(
                getattr(resp, "total_tokens", 0)
                or getattr(resp, "total_token_count", 0)
                or 0
            )
            if total <= 0:
                raise RuntimeError("count_tokens returned 0 tokens")
            return total
        except Exception as e:
            classification = classify_api_error(e)
            reason_code = str(classification.get("reason_code") or "other")

            if handle_api_error_retry(
                self,
                reason_code=reason_code,
                attempt=attempt,
                base_retry_delay=0.5,
                rotate_sleep_seconds=0.5,
                quota_retry_delay=0.5,
                model_name=active_model,
                set_model_penalty_seconds=60.0,
                set_model_penalty_fn=self._set_model_penalty,
                all_models_in_penalty_fn=lambda: all_chat_models_in_penalty(self),
                all_models_penalty_log="Semua model chat sedang cooldown saat hitung token. Coba ganti API key.",
                rotate_api_key_fn=self._rotate_api_key,
                high_demand_backoff_fn=self._high_demand_backoff,
            ):
                continue
            if attempt == max_attempts - 1:
                raise
    return 0


def update_visual_token_calibration(self, input_tokens: int):
    try:
        units = float(getattr(self, "_last_request_visual_units", 0.0) or 0.0)
        if units <= 0:
            return
        in_tok = int(max(0, input_tokens or 0))
        if in_tok <= 0:
            return

        base = max(200, int(HISTORY_VISUAL_PART_TOKEN_BASE))
        observed_per_part = float(in_tok) / float(units)
        observed_factor = observed_per_part / float(base)

        alpha = max(0.01, min(0.8, float(HISTORY_VISUAL_CALIBRATION_ALPHA)))
        fmin = float(HISTORY_VISUAL_FACTOR_MIN)
        fmax = float(HISTORY_VISUAL_FACTOR_MAX)
        if fmin > fmax:
            fmin, fmax = fmax, fmin

        old_factor = float(getattr(self, "_visual_token_factor", 1.0) or 1.0)
        new_factor = ((1.0 - alpha) * old_factor) + (alpha * observed_factor)
        new_factor = max(fmin, min(fmax, new_factor))
        self._visual_token_factor = float(new_factor)

        self.session_manager.set_metadata(
            "visual_token_factor",
            round(float(new_factor), 4),
            persist=True,
        )
        logger.info(
            "[VISUAL-CAL] units=%.2f in=%d base=%d factor %.3f->%.3f",
            units,
            in_tok,
            base,
            old_factor,
            new_factor,
        )
    except Exception as e:
        logger.warning(f"[VISUAL-CAL] update failed: {e}")
    finally:
        self._last_request_visual_units = 0.0


def _empty_token_usage_state() -> Dict:
    return {
        "monitor": "chat",
        "total_requests": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "last_updated": None,
        "requests": [],
    }
