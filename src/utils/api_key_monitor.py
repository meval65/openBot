import time
import threading
import logging
from typing import Optional, Dict

from src.config import API_KEY_FAILURE_THRESHOLD, API_KEY_RECOVERY_TIME, API_KEY_SUCCESS_THRESHOLD

logger = logging.getLogger(__name__)

import os
import json
import hashlib
from src.config import STORAGE_DIR, HEALTH_DIR

_MONITOR_REGISTRY_LOCK = threading.Lock()
_MONITOR_REGISTRY: Dict[str, tuple[tuple[str, ...], "APIKeyHealthMonitor"]] = {}


class APIKeyHealthMonitor:
    def __init__(self, api_keys: list, monitor_id: str = "default"):
        self.api_keys = api_keys
        self.lock = threading.Lock()
        self.FAILURE_THRESHOLD = API_KEY_FAILURE_THRESHOLD
        self.RECOVERY_TIME = API_KEY_RECOVERY_TIME
        self.SUCCESS_THRESHOLD = API_KEY_SUCCESS_THRESHOLD

        self.persist_path = os.path.join(HEALTH_DIR, f"api_health_{monitor_id}.json")
        self.health: Dict[int, Dict] = {}

        os.makedirs(STORAGE_DIR, exist_ok=True)
        os.makedirs(HEALTH_DIR, exist_ok=True)

        # Use a separator to avoid accidental hash ambiguity.
        combined_keys = "|".join(api_keys)
        self.keys_hash = hashlib.sha256(combined_keys.encode()).hexdigest()

        if os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                    # Load only when key set is unchanged.
                    if saved_data.get('keys_hash') == self.keys_hash:
                        health_data = saved_data.get('health', {})
                        self.health = {int(k): v for k, v in health_data.items()}
                    else:
                        logger.info("Daftar API key berubah untuk %s, status health di-reset.", monitor_id)
            except Exception as e:
                logger.warning("Gagal memuat status health API key: %s", e)

        if not self.health:
            self.health = {
                i: {
                    'healthy': True,
                    'failures': 0,
                    'recovery_until': 0.0,
                    'consecutive_success': 0,
                    'last_reason_code': "",
                    'last_failure_ts': 0.0,
                } for i in range(len(api_keys))
            }

    def _save_state(self):
        try:
            with open(self.persist_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "keys_hash": self.keys_hash,
                    "health": self.health,
                }, f)
        except Exception:
            pass

    def mark_failure(
        self,
        key_index: int,
        force_unhealthy: bool = False,
        recovery_window: float = None,
        reason_text: str = "",
        reason_code: str = "",
    ):
        with self.lock:
            if key_index not in self.health:
                return

            data = self.health[key_index]
            data['failures'] += 1
            data['consecutive_success'] = 0
            data['last_reason_code'] = str(reason_code or "").strip()

            now = time.time()
            data['last_failure_ts'] = now
            if recovery_window is not None:
                data['recovery_until'] = now + recovery_window
            else:
                data['recovery_until'] = now + self.RECOVERY_TIME

            if force_unhealthy or data['failures'] >= self.FAILURE_THRESHOLD:
                data['healthy'] = False
                key_preview = self.api_keys[key_index][:8] + "..." if self.api_keys else "Unknown"
                wait_seconds = max(0, data['recovery_until'] - now)
                reason = str(reason_text or "").strip()
                if not reason:
                    reason = (
                        "Unhealthy key"
                        if force_unhealthy
                        else f"Failure threshold reached ({data['failures']})"
                    )
                logger.warning(
                    "API key #%d (%s) dinonaktifkan sementara. Alasan: %s. Tunggu %.0f detik.",
                    key_index + 1,
                    key_preview,
                    reason,
                    wait_seconds,
                )
            self._save_state()

    def mark_success(self, key_index: int):
        with self.lock:
            if key_index not in self.health:
                return

            data = self.health[key_index]
            data['consecutive_success'] += 1

            if data['consecutive_success'] >= self.SUCCESS_THRESHOLD:
                data['failures'] = 0
                if not data['healthy']:
                    data['healthy'] = True
                    logger.info("API key #%d kembali sehat.", key_index + 1)
                data['last_reason_code'] = ""
                self._save_state()

    def get_healthy_key(self, current_index: int, total_keys: int) -> Optional[int]:
        with self.lock:
            if total_keys <= 0 or not self.health:
                return None

            now = time.time()

            for offset in range(total_keys):
                candidate = (current_index + offset) % total_keys
                data = self.health.get(candidate)

                if not data:
                    continue

                recovery_until = float(data.get('recovery_until', 0.0) or 0.0)
                in_penalty_window = now < recovery_until
                if in_penalty_window:
                    continue

                if not data.get('healthy', True):
                    data['healthy'] = True
                    data['failures'] = 0
                    data['consecutive_success'] = 0
                    data['last_reason_code'] = ""
                    logger.info("API key #%d kembali aktif setelah masa tunggu selesai.", candidate + 1)
                    self._save_state()
                return candidate

            best_cand = min(self.health.keys(), key=lambda k: self.health[k]['recovery_until'])
            wait_time = max(0, self.health[best_cand]['recovery_until'] - now)
            logger.warning(
                "Semua %d API key masih masa tunggu. Key tercepat pulih adalah #%d (sisa ~%.0f detik).",
                total_keys,
                best_cand + 1,
                wait_time,
            )
            return None


def get_shared_api_key_monitor(api_keys: list, monitor_id: str = "default") -> APIKeyHealthMonitor:
    key_tuple = tuple(str(k or "").strip() for k in (api_keys or []))
    with _MONITOR_REGISTRY_LOCK:
        existing = _MONITOR_REGISTRY.get(monitor_id)
        if existing is not None:
            existing_keys, existing_monitor = existing
            if existing_keys == key_tuple:
                return existing_monitor

        monitor = APIKeyHealthMonitor(list(key_tuple), monitor_id=monitor_id)
        _MONITOR_REGISTRY[monitor_id] = (key_tuple, monitor)
        return monitor
