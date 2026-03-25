import datetime
import logging
import re
import time
from typing import Any, Callable, Dict, Iterable, Optional
from zoneinfo import ZoneInfo

_PACIFIC_TZ = ZoneInfo("America/Los_Angeles")
logger = logging.getLogger(__name__)


def seconds_until_next_pacific_midnight() -> int:
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_pt = now_utc.astimezone(_PACIFIC_TZ)
    target_pt = now_pt.replace(hour=0, minute=0, second=0, microsecond=0)
    if now_pt >= target_pt:
        target_pt = target_pt + datetime.timedelta(days=1)
    target_utc = target_pt.astimezone(datetime.timezone.utc)
    seconds = int((target_utc - now_utc).total_seconds())
    return max(60, seconds)


def _extract_status_code(error_obj: Any, text: str) -> int:
    status_code = int(getattr(error_obj, "status_code", 0) or 0)
    if status_code:
        return status_code
    match = re.search(r"\b(4\d{2}|5\d{2})\b", text or "")
    if match:
        try:
            return int(match.group(1))
        except Exception:
            return 0
    return 0


def _walk_strings(node: Any) -> Iterable[str]:
    if isinstance(node, str):
        yield node
        return
    if isinstance(node, dict):
        for v in node.values():
            yield from _walk_strings(v)
        return
    if isinstance(node, (list, tuple, set)):
        for v in node:
            yield from _walk_strings(v)


def _extract_response_json(error_obj: Any) -> Dict[str, Any]:
    payload = getattr(error_obj, "response_json", None)
    if isinstance(payload, dict):
        return payload
    alt = getattr(error_obj, "response", None)
    if isinstance(alt, dict):
        return alt
    return {}


def classify_api_error(error_obj: Any) -> Dict[str, Any]:
    raw_text = str(error_obj or "")
    response_json = _extract_response_json(error_obj)
    extra_text = " ".join(_walk_strings(response_json))
    text = f"{raw_text} {extra_text}".lower()
    status_code = _extract_status_code(error_obj, text)
    api_status = str(
        ((response_json.get("error") or {}) if isinstance(response_json, dict) else {}).get("status")
        or ""
    ).upper()

    has = lambda *parts: any(p in text for p in parts)

    is_invalid_argument = status_code == 400 or has("invalid argument")
    is_auth = status_code in (401, 403) or api_status in {"PERMISSION_DENIED", "UNAUTHENTICATED"} or has(
        "permission denied",
        "unauthorized",
        "forbidden",
        "invalid api key",
        "api key was reported as leaked",
    )
    is_rpd = status_code == 429 and has(
        "requests per day",
        "request per day",
        "rpd",
        "daily limit",
    )
    is_tpm_rpm = status_code == 429 and has(
        "tokens per minute",
        "requests per minute",
        "tpm",
        "rpm",
        "quota metric",
        "rate_limit_exceeded",
        "rate limit",
        "too many requests",
    )
    is_quota = (
        has("resource exhausted", "quota")
        or (status_code == 429 and api_status == "RESOURCE_EXHAUSTED" and not (is_rpd or is_tpm_rpm))
    )
    is_high_demand = (
        (status_code == 503 and has("unavailable"))
        or has("high demand", "overloaded", "model is currently experiencing high demand")
    )
    is_rate_limit = (status_code == 429) and not (is_rpd or is_tpm_rpm or is_quota)

    if is_invalid_argument:
        reason_code = "invalid_argument"
    elif is_auth:
        reason_code = "auth_key"
    elif is_rpd:
        reason_code = "rpd_limit"
    elif is_tpm_rpm:
        reason_code = "tpm_rpm_limit"
    elif is_quota:
        reason_code = "quota_exhausted"
    elif is_high_demand:
        reason_code = "high_demand"
    elif is_rate_limit:
        reason_code = "rate_limit"
    else:
        reason_code = "other"

    return {
        "reason_code": reason_code,
        "status_code": status_code,
        "text": text,
        "raw_text": raw_text,
    }


def apply_key_penalty_and_rotate(
    self,
    *,
    recovery_window: float,
    reason_text: str,
    reason_code: str,
    force_unhealthy: bool = True,
    rotate_sleep_seconds: float = 1.0,
) -> bool:
    self.health_monitor.mark_failure(
        self.current_key_index,
        force_unhealthy=force_unhealthy,
        recovery_window=recovery_window,
        reason_text=reason_text,
        reason_code=reason_code,
    )
    if self._rotate_api_key():
        time.sleep(max(0.0, float(rotate_sleep_seconds)))
        return True
    return False


def ordered_chat_models(self) -> list[str]:
    primary = str(getattr(self, "primary_chat_model", "") or "").strip()
    current = str(getattr(self, "chat_model_name", "") or "").strip()
    models: list[str] = []
    if primary:
        models.append(primary)
    for item in getattr(self, "chat_model_candidates", []) or []:
        clean = str(item or "").strip()
        if clean and clean not in models:
            models.append(clean)
    if current and current not in models:
        models.append(current)
    return models


def all_chat_models_in_penalty(self) -> bool:
    models = ordered_chat_models(self)
    if not models:
        return False
    return all(self._get_model_penalty_remaining(m) > 0 for m in models)


def handle_api_error_retry(
    self,
    *,
    reason_code: str,
    attempt: int = 0,
    base_retry_delay: float = 1.0,
    rotate_sleep_seconds: float = 1.0,
    quota_retry_delay: Optional[float] = None,
    model_name: str = "",
    set_model_penalty_seconds: float = 0.0,
    high_demand_penalty_seconds: float = 0.0,
    set_model_penalty_fn: Optional[Callable[[str, float], None]] = None,
    all_models_in_penalty_fn: Optional[Callable[[], bool]] = None,
    all_models_penalty_log: str = "",
    rotate_api_key_fn: Optional[Callable[[], bool]] = None,
    high_demand_backoff_fn: Optional[Callable[[int], float]] = None,
) -> bool:
    reason = str(reason_code or "other")
    rotate_fn = rotate_api_key_fn or getattr(self, "_rotate_api_key", None)

    if reason == "rpd_limit":
        return apply_key_penalty_and_rotate(
            self,
            recovery_window=seconds_until_next_pacific_midnight(),
            reason_text="RPD limit (sehat kembali setelah reset harian Pacific Time)",
            reason_code=reason,
            force_unhealthy=True,
            rotate_sleep_seconds=rotate_sleep_seconds,
        )

    if reason == "tpm_rpm_limit":
        return apply_key_penalty_and_rotate(
            self,
            recovery_window=60,
            reason_text="TPM/RPM limit (cooldown 60 detik)",
            reason_code=reason,
            force_unhealthy=True,
            rotate_sleep_seconds=rotate_sleep_seconds,
        )

    if reason == "quota_exhausted":
        if model_name and callable(set_model_penalty_fn) and float(set_model_penalty_seconds) > 0:
            set_model_penalty_fn(model_name, float(set_model_penalty_seconds))
        if callable(all_models_in_penalty_fn) and all_models_in_penalty_fn():
            if all_models_penalty_log:
                logger.warning("%s", all_models_penalty_log)
            if callable(rotate_fn) and rotate_fn():
                time.sleep(max(0.0, min(1.0, float(base_retry_delay))))
                return True
        sleep_for = quota_retry_delay
        if sleep_for is None:
            sleep_for = min(1.0, float(base_retry_delay))
        time.sleep(max(0.0, float(sleep_for)))
        return True

    if reason in {"auth_key", "rate_limit"}:
        penalty = 86400 if reason == "auth_key" else 60
        return apply_key_penalty_and_rotate(
            self,
            recovery_window=penalty,
            reason_text=("Auth/API key tidak valid" if reason == "auth_key" else "Rate limit API key"),
            reason_code=reason,
            force_unhealthy=(reason == "auth_key"),
            rotate_sleep_seconds=rotate_sleep_seconds,
        )

    if reason == "high_demand":
        if model_name and callable(set_model_penalty_fn) and float(high_demand_penalty_seconds) > 0:
            set_model_penalty_fn(model_name, float(high_demand_penalty_seconds))
        if callable(high_demand_backoff_fn):
            sleep_for = high_demand_backoff_fn(attempt)
        else:
            sleep_for = float(base_retry_delay) + max(0, int(attempt))
        time.sleep(max(0.0, float(sleep_for)))
        return True

    return False
