import json
import logging
import os
import threading
from datetime import datetime
from typing import Dict, Tuple

from src.config import CACHE_DIR
from src.utils.time_utils import now_local, to_local_aware

logger = logging.getLogger(__name__)

_STORE_PATH = os.path.join(CACHE_DIR, "proactive_learning.json")
_PROACTIVE_REPLY_WINDOW_SECONDS = 2 * 3600
_NEUTRAL_SCORE = 0.5


def _empty_hour_stats() -> Dict[str, float]:
    return {
        "user_chats": 0.0,
        "proactive_sent": 0.0,
        "proactive_replied": 0.0,
        "reply_latency_total_sec": 0.0,
        "reply_latency_count": 0.0,
    }


class ProactiveLearning:
    def __init__(self):
        self._lock = threading.RLock()
        self._state = {
            "hours": {str(i): _empty_hour_stats() for i in range(24)},
            "pending_proactive": None,
            "recent_ignored_count": 0,
        }
        self._load()

    def _load(self):
        try:
            if not os.path.exists(_STORE_PATH):
                return
            with open(_STORE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return
            hours = data.get("hours", {})
            if isinstance(hours, dict):
                for i in range(24):
                    raw = hours.get(str(i), {})
                    if isinstance(raw, dict):
                        stat = _empty_hour_stats()
                        for key in stat.keys():
                            try:
                                stat[key] = float(raw.get(key, 0.0) or 0.0)
                            except Exception:
                                stat[key] = 0.0
                        self._state["hours"][str(i)] = stat
            pending = data.get("pending_proactive")
            if isinstance(pending, dict):
                self._state["pending_proactive"] = pending
            self._state["recent_ignored_count"] = int(data.get("recent_ignored_count", 0) or 0)
        except Exception as e:
            logger.warning("[PROACTIVE-LEARNING] Failed loading state: %s", e)

    def _save(self):
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            tmp_path = f"{_STORE_PATH}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._state, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, _STORE_PATH)
        except Exception as e:
            logger.warning("[PROACTIVE-LEARNING] Failed saving state: %s", e)

    def _hour_bucket(self, dt: datetime) -> Dict[str, float]:
        return self._state["hours"][str(int(dt.hour))]

    def _resolve_pending_locked(self, dt: datetime, treat_as_reply: bool) -> Tuple[bool, float]:
        pending = self._state.get("pending_proactive")
        if not isinstance(pending, dict):
            return False, 0.0

        sent_ts = to_local_aware(pending.get("sent_ts"))
        replied = bool(pending.get("replied"))
        if not sent_ts or replied or dt < sent_ts:
            return False, 0.0

        latency_sec = max(0.0, (dt - sent_ts).total_seconds())
        sent_hour = self._state["hours"].get(str(int(sent_ts.hour)))
        if not isinstance(sent_hour, dict):
            self._state["pending_proactive"] = None
            return False, latency_sec

        if treat_as_reply and latency_sec <= _PROACTIVE_REPLY_WINDOW_SECONDS:
            sent_hour["proactive_replied"] += 1.0
            sent_hour["reply_latency_total_sec"] += float(latency_sec)
            sent_hour["reply_latency_count"] += 1.0
            self._state["recent_ignored_count"] = 0
            self._state["pending_proactive"] = None
            return True, latency_sec

        if latency_sec > _PROACTIVE_REPLY_WINDOW_SECONDS:
            self._state["recent_ignored_count"] = int(self._state.get("recent_ignored_count", 0) or 0) + 1
            self._state["pending_proactive"] = None
        return False, latency_sec

    def record_user_message(self, ts=None, counts_as_reply: bool = True):
        dt = to_local_aware(ts) or now_local()
        with self._lock:
            bucket = self._hour_bucket(dt)
            bucket["user_chats"] += 1.0
            self._resolve_pending_locked(dt, treat_as_reply=bool(counts_as_reply))
            self._save()

    def record_proactive_sent(self, ts=None):
        dt = to_local_aware(ts) or now_local()
        with self._lock:
            self._resolve_pending_locked(dt, treat_as_reply=False)
            bucket = self._hour_bucket(dt)
            bucket["proactive_sent"] += 1.0
            self._state["pending_proactive"] = {
                "sent_ts": dt.isoformat(),
                "hour": int(dt.hour),
                "replied": False,
            }
            self._save()

    def get_score_snapshot(self, ts=None) -> Dict[str, float]:
        dt = to_local_aware(ts) or now_local()
        with self._lock:
            self._resolve_pending_locked(dt, treat_as_reply=False)
            hours = self._state["hours"]
            current = hours[str(int(dt.hour))]
            total_user = sum(float(v.get("user_chats", 0.0) or 0.0) for v in hours.values())
            total_sent = sum(float(v.get("proactive_sent", 0.0) or 0.0) for v in hours.values())
            has_learning_data = (total_user > 0.0) or (total_sent > 0.0)
            max_user = max((float(v.get("user_chats", 0.0) or 0.0) for v in hours.values()), default=0.0)
            active_score = (
                float(current.get("user_chats", 0.0) or 0.0) / max(1.0, max_user)
                if max_user > 0
                else _NEUTRAL_SCORE
            )

            sent = float(current.get("proactive_sent", 0.0) or 0.0)
            replied = float(current.get("proactive_replied", 0.0) or 0.0)
            reply_score = (replied / sent) if sent > 0 else _NEUTRAL_SCORE

            latency_count = float(current.get("reply_latency_count", 0.0) or 0.0)
            if latency_count > 0:
                avg_latency = float(current.get("reply_latency_total_sec", 0.0) or 0.0) / latency_count
                speed_score = max(0.0, 1.0 - min(avg_latency, 6 * 3600.0) / (6 * 3600.0))
            else:
                speed_score = _NEUTRAL_SCORE

            ignore_penalty = max(0.0, 1.0 - reply_score) if sent > 0 else 0.0
            recent_ignored = min(1.0, float(int(self._state.get("recent_ignored_count", 0) or 0)) / 3.0)

            score = (active_score * 0.35) + (reply_score * 0.35) + (speed_score * 0.20) - (ignore_penalty * 0.20) - (recent_ignored * 0.15)
            if not has_learning_data:
                score = _NEUTRAL_SCORE
            score = max(0.0, min(1.0, score))

            return {
                "hour": float(dt.hour),
                "active_score": round(active_score, 4),
                "reply_score": round(reply_score, 4),
                "speed_score": round(speed_score, 4),
                "ignore_penalty": round(ignore_penalty, 4),
                "recent_ignored_penalty": round(recent_ignored, 4),
                "final_score": round(score, 4),
            }

    def get_prompt_summary(self, ts=None) -> str:
        snap = self.get_score_snapshot(ts=ts)
        return (
            f"jam={int(snap['hour'])} "
            f"aktif={snap['active_score']:.2f} "
            f"balas={snap['reply_score']:.2f} "
            f"cepat={snap['speed_score']:.2f} "
            f"abaikan={snap['ignore_penalty']:.2f} "
            f"skor={snap['final_score']:.2f}"
        )
