import logging
import datetime
import pytz
from typing import Dict, List, Optional

from src.config import TIMEZONE, MAX_MEMORIES_DISPLAY
from src.utils.time_utils import get_local_tz, now_local, format_human_time

logger = logging.getLogger(__name__)


def _part_of_day(hour_24: int) -> str:
    if 5 <= hour_24 < 11:
        return "pagi"
    if 11 <= hour_24 < 15:
        return "siang"
    if 15 <= hour_24 < 18:
        return "sore"
    return "malam"


class ContextBuilder:
    def __init__(self):
        pass

    def build_context(
        self,
        relevant_memories: List[Dict],
        last_interaction: Optional[str],
        schedule_context: Optional[str],
        mood_context: Optional[str] = None,
        web_context: Optional[str] = None,
        user_profile_context: Optional[str] = None,
    ) -> str:
        sections = []
        time_anchor = self._build_time_anchor()

        sections.append(self._build_system_section(last_interaction))

        if user_profile_context:
            sections.append(f"[USER PROFILE]\n{user_profile_context}")

        if mood_context:
            sections.append(f"[MOOD STATE]\n{mood_context}")

        if relevant_memories:
            sections.append(self._build_memories_section(relevant_memories))

        if schedule_context:
            sections.append(f"[TRIGGERED REMINDER]\n{schedule_context}")

        if web_context:
            sections.append(f"[REAL-TIME INFORMATION]\n{web_context}")

        # Repeat the exact time anchor at the bottom so it stays salient.
        sections.append(time_anchor)
        return "\n\n".join(filter(None, sections))

    def _build_time_anchor(self) -> str:
        local_now = now_local()
        hour_24 = local_now.hour
        day_phase = _part_of_day(hour_24)
        return (
            "[TIME ANCHOR - WAJIB PATUHI]\n"
            f"Jam acuan final: {hour_24:02d}:00 ({day_phase}).\n"
            "Jika menyebut pagi/siang/sore/malam, WAJIB cocokkan ke jam acuan ini.\n"
            "Dilarang menebak waktu dari ingatan percakapan lama."
        )

    def _build_system_section(self, last_interaction: Optional[str]) -> str:
        local_now = now_local()
        # Include 24h + AM/PM format so model avoids wrong time assumptions.
        time_str = local_now.strftime("%A, %d %B %Y - %H:%M (%I:%M %p)")
        hour_24 = local_now.hour
        day_phase = _part_of_day(hour_24)
        tz_name = get_local_tz().zone

        lines = [
            "[SYSTEM CONTEXT]",
            f"Waktu Sekarang Saat Ini (Sangat Akurat): {time_str} ({tz_name})",
            f"Label Periode Waktu Akurat: {day_phase} (jam {hour_24:02d})",
            "Gunakan label waktu di atas sebagai sumber utama. Jangan menebak waktu dari konteks lama.",
            "ATURAN KERAS: Saat menyebut periode waktu (pagi/siang/sore/malam), hitung dari jam di atas.",
        ]

        if last_interaction:
            try:
                if isinstance(last_interaction, str):
                    last_dt = datetime.datetime.fromisoformat(last_interaction)
                    if last_dt.tzinfo is None:
                        last_dt = get_local_tz().localize(last_dt)
                else:
                    last_dt = last_interaction
                delta = local_now - last_dt.astimezone(get_local_tz())
                hours = int(delta.total_seconds() / 3600)
                if hours < 1:
                    gap = "kurang dari 1 jam lalu"
                elif hours < 24:
                    gap = f"{hours} jam lalu"
                else:
                    days = hours // 24
                    gap = f"{days} hari lalu"
                lines.append(f"Last Interaction: {gap}")
            except Exception:
                pass

        return "\n".join(lines)

    def _build_memories_section(self, memories: List[Dict]) -> str:
        if not memories:
            return ""

        lines = ["[RELEVANT MEMORIES]"]
        for mem in memories[:MAX_MEMORIES_DISPLAY]:
            summary = mem.get("summary", "")
            mem_type = mem.get("type", "")
            if not summary:
                continue
            type_tag = f"[{mem_type}] " if mem_type else ""
            lines.append(f"- {type_tag}{summary}")

        return "\n".join(lines)
