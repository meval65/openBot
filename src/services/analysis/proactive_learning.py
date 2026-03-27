import json
import logging
import os
import re
import threading
from datetime import datetime
from typing import Dict, Optional, Tuple

from src.config import CACHE_DIR
from src.database.connection import DBConnection
from src.utils.time_utils import now_local, to_local_aware

logger = logging.getLogger(__name__)

_STORE_PATH = os.path.join(CACHE_DIR, "proactive_learning.json")
_PROACTIVE_REPLY_WINDOW_SECONDS = 2 * 3600
_AUTO_REPLY_WINDOW_SECONDS = 15 * 60
_SOFT_REPLY_WINDOW_SECONDS = 45 * 60
_NEUTRAL_SCORE = 0.5
_ACK_TOKENS = {
    "ya",
    "iya",
    "iyah",
    "yup",
    "yaps",
    "ok",
    "oke",
    "okay",
    "sip",
    "siap",
    "noted",
    "makasih",
    "terima kasih",
    "thanks",
    "thank you",
    "boleh",
    "enggak",
    "nggak",
    "gak",
    "tidak",
    "bisa",
    "gabisa",
}
_WORD_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)


def _empty_hour_stats() -> Dict[str, float]:
    return {
        "user_chats": 0.0,
        "proactive_sent": 0.0,
        "proactive_replied": 0.0,
        "reply_latency_total_sec": 0.0,
        "reply_latency_count": 0.0,
    }


class ProactiveLearning:
    def __init__(self, db: Optional[DBConnection] = None, legacy_store_path: Optional[str] = None):
        self._lock = threading.RLock()
        self.db = db or DBConnection()
        self._legacy_store_path = legacy_store_path or _STORE_PATH
        self._state = {
            "hours": {str(i): _empty_hour_stats() for i in range(24)},
            "pending_proactive": None,
            "recent_ignored_count": 0,
        }
        self._load()

    def _load(self):
        try:
            loaded = self._load_from_db()
            if loaded and self._has_learning_data():
                return
            if self._migrate_legacy_file():
                self._save()
                self._cleanup_legacy_file()
        except Exception as e:
            logger.warning("[PROACTIVE-LEARNING] Failed loading state: %s", e)

    def _load_from_db(self) -> bool:
        cursor = self.db.get_cursor()
        try:
            cursor.execute(
                """
                SELECT hour, user_chats, proactive_sent, proactive_replied,
                       reply_latency_total_sec, reply_latency_count
                FROM proactive_learning_hourly
                ORDER BY hour ASC
                """
            )
            rows = cursor.fetchall()
            if not rows:
                return False
            for row in rows:
                hour = str(int(row[0]))
                self._state["hours"][hour] = {
                    "user_chats": float(row[1] or 0.0),
                    "proactive_sent": float(row[2] or 0.0),
                    "proactive_replied": float(row[3] or 0.0),
                    "reply_latency_total_sec": float(row[4] or 0.0),
                    "reply_latency_count": float(row[5] or 0.0),
                }

            cursor.execute(
                """
                SELECT pending_sent_ts, pending_hour, recent_ignored_count
                FROM proactive_learning_state
                WHERE id = 1
                """
            )
            row = cursor.fetchone()
            if row:
                pending_sent_ts = row[0]
                pending_hour = row[1]
                self._state["recent_ignored_count"] = int(row[2] or 0)
                if pending_sent_ts:
                    sent_dt = to_local_aware(pending_sent_ts)
                    self._state["pending_proactive"] = {
                        "sent_ts": sent_dt.isoformat() if sent_dt else str(pending_sent_ts),
                        "hour": int(pending_hour or 0),
                        "replied": False,
                    }
                else:
                    self._state["pending_proactive"] = None
            return True
        finally:
            cursor.close()

    def _migrate_legacy_file(self) -> bool:
        try:
            if not os.path.exists(self._legacy_store_path):
                return False
            with open(self._legacy_store_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return False
            hours = data.get("hours", {})
            if isinstance(hours, dict):
                for i in range(24):
                    raw = hours.get(str(i), {})
                    if not isinstance(raw, dict):
                        continue
                    stat = _empty_hour_stats()
                    for key in stat.keys():
                        try:
                            stat[key] = float(raw.get(key, 0.0) or 0.0)
                        except Exception:
                            stat[key] = 0.0
                    self._state["hours"][str(i)] = stat
            pending = data.get("pending_proactive")
            if isinstance(pending, dict):
                self._state["pending_proactive"] = {
                    "sent_ts": str(pending.get("sent_ts") or "").strip(),
                    "hour": int(pending.get("hour", 0) or 0),
                    "replied": bool(pending.get("replied")),
                }
            self._state["recent_ignored_count"] = int(data.get("recent_ignored_count", 0) or 0)
            return self._has_learning_data()
        except Exception as e:
            logger.warning("[PROACTIVE-LEARNING] Failed migrating legacy file: %s", e)
            return False

    def _cleanup_legacy_file(self):
        try:
            if os.path.exists(self._legacy_store_path):
                os.remove(self._legacy_store_path)
        except OSError as e:
            logger.warning("[PROACTIVE-LEARNING] Failed removing legacy state file: %s", e)

    def _has_learning_data(self) -> bool:
        if int(self._state.get("recent_ignored_count", 0) or 0) > 0:
            return True
        if isinstance(self._state.get("pending_proactive"), dict):
            return True
        for bucket in self._state["hours"].values():
            if any(float(bucket.get(key, 0.0) or 0.0) > 0.0 for key in _empty_hour_stats().keys()):
                return True
        return False

    def _save(self):
        conn = self.db.get_connection()
        with self.db._write_lock:
            cursor = conn.cursor()
            try:
                for i in range(24):
                    bucket = self._state["hours"][str(i)]
                    cursor.execute(
                        """
                        INSERT INTO proactive_learning_hourly (
                            hour,
                            user_chats,
                            proactive_sent,
                            proactive_replied,
                            reply_latency_total_sec,
                            reply_latency_count
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(hour) DO UPDATE SET
                            user_chats = excluded.user_chats,
                            proactive_sent = excluded.proactive_sent,
                            proactive_replied = excluded.proactive_replied,
                            reply_latency_total_sec = excluded.reply_latency_total_sec,
                            reply_latency_count = excluded.reply_latency_count
                        """,
                        (
                            i,
                            float(bucket.get("user_chats", 0.0) or 0.0),
                            float(bucket.get("proactive_sent", 0.0) or 0.0),
                            float(bucket.get("proactive_replied", 0.0) or 0.0),
                            float(bucket.get("reply_latency_total_sec", 0.0) or 0.0),
                            float(bucket.get("reply_latency_count", 0.0) or 0.0),
                        ),
                    )

                pending = self._state.get("pending_proactive")
                pending_sent_ts = None
                pending_hour = None
                if isinstance(pending, dict):
                    sent_dt = to_local_aware(pending.get("sent_ts"))
                    if sent_dt:
                        pending_sent_ts = sent_dt
                        pending_hour = int(pending.get("hour", sent_dt.hour) or sent_dt.hour)

                cursor.execute(
                    """
                    INSERT INTO proactive_learning_state (
                        id,
                        pending_sent_ts,
                        pending_hour,
                        recent_ignored_count
                    ) VALUES (1, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        pending_sent_ts = excluded.pending_sent_ts,
                        pending_hour = excluded.pending_hour,
                        recent_ignored_count = excluded.recent_ignored_count
                    """,
                    (
                        pending_sent_ts,
                        pending_hour,
                        int(self._state.get("recent_ignored_count", 0) or 0),
                    ),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cursor.close()

    def _hour_bucket(self, dt: datetime) -> Dict[str, float]:
        return self._state["hours"][str(int(dt.hour))]

    @staticmethod
    def _looks_like_reply_text(message_text: str) -> bool:
        clean = " ".join(str(message_text or "").split()).strip().lower()
        if not clean:
            return False
        if clean.endswith("?") and len(clean) <= 120:
            return True
        if clean in _ACK_TOKENS:
            return True
        if any(token in clean for token in ("makasih", "terima kasih", "thank", "noted")):
            return True
        words = _WORD_RE.findall(clean)
        if len(words) <= 8 and len(clean) <= 60:
            compact = " ".join(words)
            if compact in _ACK_TOKENS:
                return True
            if any(token in clean for token in ("ya", "iya", "oke", "ok", "siap", "gak", "nggak", "tidak", "boleh", "bisa")):
                return True
        return False

    def _should_treat_as_reply(self, latency_sec: float, counts_as_reply: bool, message_text: str) -> bool:
        if not counts_as_reply or latency_sec < 0:
            return False
        if latency_sec <= _AUTO_REPLY_WINDOW_SECONDS:
            return True
        if latency_sec <= _SOFT_REPLY_WINDOW_SECONDS and self._looks_like_reply_text(message_text):
            return True
        return False

    def _resolve_pending_locked(
        self,
        dt: datetime,
        treat_as_reply: bool,
        message_text: str = "",
    ) -> Tuple[bool, float]:
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

        if self._should_treat_as_reply(latency_sec, treat_as_reply, message_text):
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

    def record_user_message(
        self,
        ts=None,
        counts_as_reply: bool = True,
        message_text: str = "",
    ):
        dt = to_local_aware(ts) or now_local()
        with self._lock:
            bucket = self._hour_bucket(dt)
            bucket["user_chats"] += 1.0
            self._resolve_pending_locked(
                dt,
                treat_as_reply=bool(counts_as_reply),
                message_text=str(message_text or ""),
            )
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

            score = (
                (active_score * 0.35)
                + (reply_score * 0.35)
                + (speed_score * 0.20)
                - (ignore_penalty * 0.20)
                - (recent_ignored * 0.15)
            )
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
