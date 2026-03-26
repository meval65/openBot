import hashlib
import json
import logging
import os
import random
import re
import threading
import time
from typing import Optional, TYPE_CHECKING, Tuple

import requests
from pymeteosource.api import Meteosource
from pymeteosource.types import sections, tiers, units

from src.config import (
    BOT_DISPLAY_NAME,
    METEOSOURCE_API_KEY,
    PROACTIVE_ANALYSIS_MODEL,
    PROACTIVE_MIN_SEND_GAP_SECONDS,
)
from src.utils.api_utils import with_retry
from src.utils.time_utils import now_local as _now_local, to_local_aware

if TYPE_CHECKING:
    from src.services.chat.handler import ChatHandler
    from src.services.memory import MemoryManager
    from src.services.scheduling import SchedulerService

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
BOT_SLUG = re.sub(r"[^a-z0-9]+", "-", BOT_DISPLAY_NAME.lower()).strip("-") or "ai-bot"

# Simplified deterministic policy.
NIGHT_START_HOUR = 23
NIGHT_END_HOUR = 6
MIN_INTERACTION_GAP_SECONDS = 30 * 60
MIN_PROACTIVE_GAP_SECONDS = int(PROACTIVE_MIN_SEND_GAP_SECONDS)
TICK_INTERVAL_MIN = 45 * 60
TICK_INTERVAL_MAX = 90 * 60
REPEAT_WINDOW_SECONDS = 3600
REPEAT_SIMILARITY_THRESHOLD = 0.72
MIN_LEARNING_SCORE = 0.35


def _is_night_mode() -> bool:
    now = _now_local()
    h = now.hour
    if NIGHT_START_HOUR <= NIGHT_END_HOUR:
        return NIGHT_START_HOUR <= h < NIGHT_END_HOUR
    return h >= NIGHT_START_HOUR or h < NIGHT_END_HOUR

def _gap_seconds_from(value) -> Optional[int]:
    dt = to_local_aware(value)
    if not dt:
        return None
    return int((_now_local() - dt).total_seconds())


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _jaccard_similarity(a: str, b: str) -> float:
    ta = set(_normalize_text(a).split())
    tb = set(_normalize_text(b).split())
    if not ta or not tb:
        return 0.0
    return float(len(ta.intersection(tb)) / max(1, len(ta.union(tb))))


def _is_repeated_context(chat_handler: "ChatHandler", candidate_context: str) -> Tuple[bool, str]:
    candidate = _normalize_text(candidate_context)
    if not candidate:
        return False, ""
    try:
        last_context = str(chat_handler.session_manager.get_metadata("last_proactive_trigger_context", "") or "").strip()
        last_ts = chat_handler.session_manager.get_metadata("last_proactive_trigger_ts")
        if not last_context or not last_ts:
            return False, ""

        elapsed = _gap_seconds_from(last_ts)
        if elapsed is None or elapsed > REPEAT_WINDOW_SECONDS:
            return False, ""

        last_norm = _normalize_text(last_context)
        if not last_norm:
            return False, ""

        if hashlib.sha256(candidate.encode("utf-8")).hexdigest() == hashlib.sha256(last_norm.encode("utf-8")).hexdigest():
            return True, "same_hash"

        sim = _jaccard_similarity(candidate, last_norm)
        if sim >= REPEAT_SIMILARITY_THRESHOLD:
            return True, f"similarity={sim:.2f}"
    except Exception as e:
        logger.warning(f"[PROACTIVE-ENGINE] Repeat-check failed: {e}")
    return False, ""


def _get_weather_summary() -> str:
    try:
        if not METEOSOURCE_API_KEY:
            return "Data cuaca tidak tersedia."

        meteosource = Meteosource(METEOSOURCE_API_KEY, tiers.FREE)
        forecast = meteosource.get_point_forecast(
            place_id="jakarta",
            sections=[sections.CURRENT],
            units=units.METRIC,
        )
        current = forecast.current
        temp = getattr(current, "temperature", None)
        summary = getattr(current, "summary", None)
        if temp is not None and summary:
            return f"{summary}, {temp} C"
    except Exception as e:
        logger.warning(f"[PROACTIVE-ENGINE] Weather fetch failed: {e}")
    return "Data cuaca tidak tersedia."


def _build_decision_prompt(
    chat_handler: "ChatHandler",
    memory_manager: "MemoryManager",
    scheduler_service: "SchedulerService",
    gap_seconds: Optional[int],
) -> str:
    now = _now_local()
    now_str = now.strftime("%A, %d %B %Y - %H:%M (%Z)")
    gap_text = "tidak diketahui"
    if gap_seconds is not None:
        h, rem = divmod(max(0, gap_seconds), 3600)
        m = rem // 60
        gap_text = f"{h} jam {m} menit" if h > 0 else f"{m} menit"

    rolling_summary = str(chat_handler.session_manager.get_metadata("rolling_summary", "") or "").strip() or "Belum ada ringkasan."

    top_mem_lines = []
    try:
        for mem in memory_manager.get_top_memories(limit=8):
            top_mem_lines.append(f"- [{mem.get('type')}] {mem.get('summary')}")
    except Exception:
        pass
    memories_text = "\n".join(top_mem_lines) if top_mem_lines else "Tidak ada memori penting."

    sched_lines = []
    try:
        pending = scheduler_service.get_pending_schedules(lookahead_minutes=60 * 24, max_results=8)
        for s in pending:
            sched_lines.append(f"- [{s.get('scheduled_at')}] {s.get('context', '')}")
    except Exception:
        pass
    schedules_text = "\n".join(sched_lines) if sched_lines else "Tidak ada jadwal aktif."

    persona = str(chat_handler.get_effective_instruction() or "").strip() or "Tidak ada instruksi khusus."
    learning_summary = "Belum ada data ritme."
    try:
        learning_summary = str(chat_handler.proactive_learning.get_prompt_summary() or learning_summary).strip()
    except Exception:
        pass

    return f"""Kamu modul internal pengambil keputusan proactive.
Putuskan apakah saat ini AI perlu mengirim pesan spontan ke user.

Konteks:
- Waktu lokal: {now_str}
- Jeda sejak interaksi user terakhir: {gap_text}
- Cuaca: {_get_weather_summary()}
- Sinyal ritme interaksi user: {learning_summary}

Ringkasan percakapan:
{rolling_summary}

Memori penting:
{memories_text}

Jadwal aktif:
{schedules_text}

Persona AI:
{persona}

Aturan:
- Prioritaskan natural dan tidak spam.
- Jika sinyal lemah, pilih false.
- Jika true, proactive_context harus singkat, konkret, dan bisa dipakai AI utama untuk menyapa.

Output hanya JSON:
{{
  "should_chat": true atau false,
  "reasoning": "alasan singkat",
  "proactive_context": "isi konteks (kosong jika false)"
}}"""


def _call_openrouter_decision(api_key: str, prompt: str) -> Optional[dict]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": f"https://{BOT_SLUG}",
        "X-Title": f"{BOT_DISPLAY_NAME} Proactive Engine",
    }
    payload = {
        "model": PROACTIVE_ANALYSIS_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 320,
        "response_format": {"type": "json_object"},
    }

    @with_retry(max_retries=3, base_delay=1.2, max_delay=8.0)
    def _execute():
        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=45)
        if 500 <= resp.status_code < 600:
            raise requests.HTTPError(f"{resp.status_code} server error", response=resp)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content) if isinstance(content, list) else str(content)
        try:
            return json.loads(content)
        except Exception:
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if not m:
                raise ValueError("No JSON object found in response")
            return json.loads(m.group())

    try:
        return _execute()
    except Exception as e:
        logger.warning(f"[PROACTIVE-ENGINE] OpenRouter decision failed: {e}")
        return None


class ProactiveEngine:
    def __init__(self):
        self._next_fire_at: float = time.time() + random.uniform(TICK_INTERVAL_MIN, TICK_INTERVAL_MAX)
        self._state_lock = threading.Lock()
        self._tick_running = False

    def _load_key(self) -> Optional[str]:
        key = str(os.getenv("OPENROUTER_API_KEY") or "").strip()
        if not key:
            logger.warning("[PROACTIVE-ENGINE] OPENROUTER_API_KEY not set. Engine disabled.")
            return None
        return key

    def _reschedule(self):
        interval = random.uniform(TICK_INTERVAL_MIN, TICK_INTERVAL_MAX)
        self._next_fire_at = time.time() + interval
        logger.info("[PROACTIVE-ENGINE] Next tick in ~%d minutes.", int(interval / 60))

    def try_acquire_tick(self) -> bool:
        with self._state_lock:
            if self._tick_running:
                return False
            if time.time() < self._next_fire_at:
                return False
            self._tick_running = True
            return True

    def finish_tick(self) -> None:
        with self._state_lock:
            self._tick_running = False

    def should_allow_trigger(
        self,
        chat_handler: "ChatHandler",
        memory_manager: "MemoryManager",
        scheduler_service: "SchedulerService",
        candidate_context: str,
        source: str = "schedule",
    ) -> Tuple[bool, str]:
        if _is_night_mode() and source != "schedule":
            return False, "night_mode"

        proactive_gap = _gap_seconds_from(chat_handler.session_manager.get_metadata("last_proactive_sent_ts"))
        if proactive_gap is not None and proactive_gap < MIN_PROACTIVE_GAP_SECONDS:
            return False, f"cooldown:{proactive_gap}s"

        repeated, repeat_reason = _is_repeated_context(chat_handler, candidate_context)
        if repeated:
            return False, f"repeat:{repeat_reason}"
        try:
            learning_score = float(chat_handler.proactive_learning.get_score_snapshot().get("final_score", 0.0) or 0.0)
            if source != "schedule" and learning_score < MIN_LEARNING_SCORE:
                return False, f"learning_score:{learning_score:.2f}"
        except Exception as e:
            logger.warning("[PROACTIVE-ENGINE] Learning score check failed: %s", e)

        return True, "allow"

    def run_tick(
        self,
        chat_handler: "ChatHandler",
        memory_manager: "MemoryManager",
        scheduler_service: "SchedulerService",
    ) -> Optional[str]:
        api_key = self._load_key()
        if not api_key:
            self._reschedule()
            return None

        if _is_night_mode():
            logger.info("[PROACTIVE-ENGINE] Night mode active. Skip.")
            self._reschedule()
            return None

        last_interaction = chat_handler.session_manager.get_metadata("last_user_interaction")
        gap_seconds = _gap_seconds_from(last_interaction)
        if gap_seconds is not None and gap_seconds < MIN_INTERACTION_GAP_SECONDS:
            logger.info("[PROACTIVE-ENGINE] Interaction too recent (%ss). Skip.", gap_seconds)
            self._reschedule()
            return None

        proactive_gap = _gap_seconds_from(chat_handler.session_manager.get_metadata("last_proactive_sent_ts"))
        if proactive_gap is not None and proactive_gap < MIN_PROACTIVE_GAP_SECONDS:
            logger.info("[PROACTIVE-ENGINE] Cooldown active (%ss). Skip.", proactive_gap)
            self._reschedule()
            return None
        try:
            learning_score = float(chat_handler.proactive_learning.get_score_snapshot().get("final_score", 0.0) or 0.0)
            if learning_score < MIN_LEARNING_SCORE:
                logger.info("[PROACTIVE-ENGINE] Learning score too low (%.2f). Skip.", learning_score)
                self._reschedule()
                return None
        except Exception as e:
            logger.warning("[PROACTIVE-ENGINE] Learning score read failed: %s", e)

        prompt = _build_decision_prompt(chat_handler, memory_manager, scheduler_service, gap_seconds)
        decision = _call_openrouter_decision(api_key, prompt)
        self._reschedule()
        if not decision:
            return None

        should_chat = decision.get("should_chat", False)
        if isinstance(should_chat, str):
            should_chat = should_chat.strip().lower() in {"true", "1", "yes"}
        else:
            should_chat = bool(should_chat)

        proactive_context = str(decision.get("proactive_context", "") or "").strip()
        reasoning = str(decision.get("reasoning", "") or "").strip()
        logger.info(
            "[PROACTIVE-ENGINE] Decision should_chat=%s reasoning=%s",
            should_chat,
            reasoning[:160],
        )
        if not should_chat or not proactive_context:
            return None

        repeated, repeat_reason = _is_repeated_context(chat_handler, proactive_context)
        if repeated:
            logger.info("[PROACTIVE-ENGINE] Repeat blocked (%s).", repeat_reason)
            return None

        return proactive_context
