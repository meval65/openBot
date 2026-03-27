import logging
import os
import threading
import time
import datetime
from typing import Dict, List, Optional

import numpy as np
from google.genai import types

from src.config import (
    BACKGROUND_SUMMARY_MODEL,
    BOT_DISPLAY_NAME,
    VISUAL_UNIT_WEIGHT_IMAGE,
    VISUAL_UNIT_WEIGHT_STICKER,
)
from src.utils.error_types import LLMGenerationError
from src.services.media.pipeline import (
    ingest_local_image,
    ingest_local_video,
    is_sticker_path,
)
from src.services.chat.tool_prompt import build_tool_usage_directive
from src.services.chat.workspace_context import build_workspace_snapshot
from src.services.chat.media_parts import part_from_bytes_with_resolution

logger = logging.getLogger(__name__)

BACKGROUND_ANALYSIS_DELAY_SECONDS = 30.0
BACKGROUND_ANALYSIS_WINDOW = 30
USER_PROFILE_SUMMARY_THRESHOLD = 5.0
USER_PROFILE_RECENT_HISTORY_LIMIT = 24


def _get_user_profile_summary_text(self) -> str:
    return " ".join(
        str(self.session_manager.get_metadata("user_profile_summary", "") or "").split()
    ).strip()


def _combine_user_profile_context(
    external_context: Optional[str],
    generated_summary: Optional[str],
) -> str:
    blocks: List[str] = []
    ext = " ".join(str(external_context or "").split()).strip()
    gen = " ".join(str(generated_summary or "").split()).strip()
    if ext:
        blocks.append(ext)
    if gen:
        blocks.append(f"[Ringkasan profil user saat ini]\n{gen}")
    return "\n\n".join(blocks).strip()


def bump_user_profile_update_score(self, amount: float, reason: str = "") -> float:
    try:
        current = float(self.session_manager.get_metadata("user_profile_update_score", 0.0) or 0.0)
    except Exception:
        current = 0.0
    new_score = max(0.0, current + float(amount or 0.0))
    self.session_manager.set_metadata(
        "user_profile_update_score",
        round(new_score, 2),
        persist=True,
    )
    if reason:
        logger.info(
            "[USER-PROFILE] score += %.2f -> %.2f | reason=%s",
            float(amount or 0.0),
            new_score,
            reason,
        )
    return new_score


def _build_user_profile_recent_history(self, history: List[Dict]) -> str:
    lines: List[str] = []
    recent = list(history or [])[-USER_PROFILE_RECENT_HISTORY_LIMIT:]
    for msg in recent:
        if not isinstance(msg, dict):
            continue
        role = "User" if msg.get("role") == "user" else BOT_DISPLAY_NAME
        parts = msg.get("parts", [])
        text = " ".join(str(p or "").strip() for p in parts if isinstance(p, str)).strip()
        if not text:
            continue
        lines.append(f"{role}: {text}")
    return "\n".join(lines).strip()


def _start_user_profile_refresh(self):
    with self._user_profile_refresh_lock:
        if self._user_profile_refresh_running:
            return False
        self._user_profile_refresh_running = True

    def _run():
        try:
            generate_user_profile_summary(self)
        finally:
            with self._user_profile_refresh_lock:
                self._user_profile_refresh_running = False

    threading.Thread(target=_run, name="user-profile-refresh", daemon=True).start()
    return True


def maybe_schedule_user_profile_refresh(self):
    try:
        score = float(self.session_manager.get_metadata("user_profile_update_score", 0.0) or 0.0)
    except Exception:
        score = 0.0
    if score < USER_PROFILE_SUMMARY_THRESHOLD:
        return
    started = _start_user_profile_refresh(self)
    if started:
        logger.info(
            "[USER-PROFILE] refresh scheduled | score=%.2f threshold=%.2f",
            score,
            USER_PROFILE_SUMMARY_THRESHOLD,
        )


def generate_user_profile_summary(self):
    try:
        session_data = self._gather_session_data()
        rolling_summary = " ".join(
            str(self.session_manager.get_metadata("rolling_summary", "") or "").split()
        ).strip()
        previous_profile = _get_user_profile_summary_text(self)
        recent_history = _build_user_profile_recent_history(self, list(session_data.get("history", [])))
        top_memories = []
        try:
            for mem in self.memory_manager.get_top_memories(limit=12):
                top_memories.append(f"- [{mem.get('type')}] {mem.get('summary')}")
        except Exception as e:
            logger.warning("[USER-PROFILE] failed reading top memories: %s", e)

        prompt = f"""
You are updating a concise user profile summary for a personal AI companion.
Use only the information provided. Do not invent facts.
Keep it concise, stable, and useful as future context.

Output in English section headers exactly like this:
#Work context
#Personal context
#Top of mind
#Brief history
#Recent months

Write the content in natural Indonesian.
If a section is unknown, write "Belum jelas".
Keep each section short and meaningful.

Meaning of sections:
- #Work context: identitas kerja/sekolah/peran user yang relevan.
- #Personal context: gambaran umum personal, minat, gaya, atau kecenderungan user.
- #Top of mind: hal yang paling aktif, penting, atau sedang dikerjakan user saat ini.
- #Brief history: riwayat singkat yang lebih stabil tentang user.
- #Recent months: hal yang relatif baru dalam beberapa bulan terakhir.

Previous Profile:
{previous_profile or 'Belum ada profile sebelumnya.'}

STM / Recent Conversation:
{recent_history or 'Belum ada percakapan terbaru yang cukup.'}

MTM / Rolling Summary:
{rolling_summary or 'Belum ada rolling summary.'}

LTM / Important Memories:
{chr(10).join(top_memories).strip() or 'Belum ada.'}
""".strip()

        response = self.call_gemini(
            model=BACKGROUND_SUMMARY_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=600,
            ),
        )
        text = " ".join(str(getattr(response, "text", "") or "").split()).strip()
        if not text:
            return
        self.session_manager.set_metadata("user_profile_summary", text, persist=True)
        self.session_manager.set_metadata("user_profile_update_score", 0.0, persist=True)
        self.session_manager.set_metadata(
            "user_profile_summary_updated_at",
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            persist=True,
        )
        logger.info("[USER-PROFILE] summary regenerated successfully.")
    except Exception as e:
        logger.error("[USER-PROFILE] regenerate failed: %s", e)


def _stage_media_for_ai_workspace(self, local_path: Optional[str], media_kind: str) -> Dict[str, str]:
    path = str(local_path or "").strip()
    if not path or not os.path.isfile(path):
        return {}
    try:
        staged = self.terminal_service.stage_local_file_to_workspace(path, media_kind=media_kind)
    except Exception as e:
        logger.warning("[AI-WORKSPACE] Failed staging %s '%s': %s", media_kind, path, e)
        return {}
    if not isinstance(staged, dict) or not staged.get("ok"):
        err = str((staged or {}).get("error") or "unknown")
        logger.warning("[AI-WORKSPACE] Failed staging %s '%s': %s", media_kind, path, err)
        return {}
    return {
        "host_path": str(staged.get("host_path") or "").strip(),
        "container_path": str(staged.get("container_path") or "").strip(),
    }


def _canonical_media_paths(fallback_host_path: Optional[str], staged: Optional[Dict[str, str]]) -> tuple[str, str]:
    fallback = str(fallback_host_path or "").strip()
    staged_host = str((staged or {}).get("host_path") or "").strip()
    staged_container = str((staged or {}).get("container_path") or "").strip()
    return (staged_host or fallback, staged_container)


def build_generation_state(
    self,
    query_text: str,
    schedule_context: Optional[str],
    user_profile_context: Optional[str] = None,
) -> Dict:
    session_data = self._gather_session_data()
    relevant_memories = self._retrieve_memories(query_text)
    mood_context = self._extract_mood_context(relevant_memories)
    combined_user_profile = _combine_user_profile_context(
        user_profile_context,
        _get_user_profile_summary_text(self),
    )

    system_context = self.context_builder.build_context(
        relevant_memories,
        session_data["last_interaction"],
        schedule_context,
        mood_context=mood_context,
        user_profile_context=combined_user_profile,
    )

    rolling_summary = self.session_manager.get_metadata("rolling_summary")
    if rolling_summary:
        system_context += f"\n\n[Previous Conversation Summary]\n{rolling_summary}"

    return {
        "session_data": session_data,
        "relevant_memories": relevant_memories,
        "mood_context": mood_context,
        "system_context": system_context,
    }


def build_full_system_prompt(
    self,
    system_context: str,
    style: str = "default",
    extra_instruction: str = "",
) -> str:
    tool_usage_directive = build_tool_usage_directive(
        style=style,
        available_tools=getattr(self, "_tool_names", None),
    )
    workspace_snapshot = build_workspace_snapshot(self._workspace_dir)
    extra = str(extra_instruction or "")
    return (
        f"{self.get_effective_instruction()}{tool_usage_directive}\n\n"
        f"{system_context}\n\n"
        f"{workspace_snapshot}"
        f"{extra}"
    )


def execute_flow(
    self,
    user_text: str,
    image_path: str,
    video_path: str = None,
    user_profile_context: Optional[str] = None,
) -> str:
    flow_t0 = time.perf_counter()
    stage_marks: Dict[str, float] = {}

    session_data = self._gather_session_data()
    stage_marks["session"] = time.perf_counter()
    schedule_context = self._process_pending_schedule()
    stage_marks["schedule"] = time.perf_counter()
    prompt_state = build_generation_state(
        self,
        query_text=user_text,
        schedule_context=schedule_context,
        user_profile_context=user_profile_context,
    )
    relevant_memories = prompt_state["relevant_memories"]
    system_context = prompt_state["system_context"]
    stage_marks["context"] = time.perf_counter()
    stage_marks["memory"] = stage_marks["context"]
    full_system = build_full_system_prompt(self, system_context, style="default")

    permanent_image_path = None
    primary_image_part: Optional[types.Part] = None
    permanent_video_path = None
    primary_video_part: Optional[types.Part] = None
    video_media_ctx: Optional[Dict] = None

    if image_path and os.path.exists(image_path):
        try:
            image_ctx = ingest_local_image(image_path)
            data = image_ctx["analysis_data"]
            mime_type = image_ctx["analysis_mime"]
            used_collage = bool(image_ctx["used_collage"])
            frame_count = int(image_ctx["frame_count"] or 0)
            image_level = image_ctx["media_resolution"]
            primary_image_part = part_from_bytes_with_resolution(
                data=data,
                mime_type=mime_type,
                level=image_level,
                model_name=self.chat_model_name,
            )
            permanent_image_path = image_ctx["stored_path"]
            if used_collage:
                anim_ctx = (
                    f"[Konteks: ini media animasi; frame sudah diringkas sebagai kolase berurutan"
                    f"{f' ({frame_count} frame)' if frame_count else ''}. Baca urut dari kiri ke kanan lalu atas ke bawah.]"
                )
                user_text = f"{(user_text or '').strip()}\n\n{anim_ctx}".strip() if (user_text or "").strip() else anim_ctx
        except Exception as e:
            logger.error(f"[IMG] Failed to read image: {e}")

    if video_path and os.path.exists(video_path):
        try:
            video_media_ctx = ingest_local_video(video_path)
            permanent_video_path = video_media_ctx["stored_path"]
            vdata = video_media_ctx["analysis_data"]
            vmime = video_media_ctx["analysis_mime"]
            primary_video_part = types.Part.from_bytes(data=vdata, mime_type=vmime)
        except Exception as e:
            logger.error(f"[VIDEO] Failed to process video: {e}")

    gemini_history = self._build_gemini_history(session_data["history"])

    history_image_path = permanent_image_path if (permanent_image_path and not is_sticker_path(permanent_image_path)) else None
    history_video_path = permanent_video_path if (permanent_video_path and not is_sticker_path(permanent_video_path)) else None
    staged_image = _stage_media_for_ai_workspace(self, history_image_path, "image") if history_image_path else {}
    staged_video = _stage_media_for_ai_workspace(self, history_video_path, "video") if history_video_path else {}
    if history_image_path and not staged_image:
        raise LLMGenerationError("AI workspace image staging failed.")
    if history_video_path and not staged_video:
        raise LLMGenerationError("AI workspace video staging failed.")
    history_image_path, workspace_image_path = _canonical_media_paths(history_image_path, staged_image)
    history_video_path, workspace_video_path = _canonical_media_paths(history_video_path, staged_video)

    combined_text = user_text or ""
    media_path_lines: List[str] = []
    if workspace_image_path:
        media_path_lines.append(
            f"[AI_WORKSPACE_IMAGE] {workspace_image_path}\n"
            "File gambar yang sudah dinormalisasi ini tersedia di komputer pribadimu dan boleh kamu akses langsung via terminal/tool."
        )
    elif history_image_path:
        media_path_lines.append(f"[IMG_PATH] {history_image_path}")
    if workspace_video_path:
        media_path_lines.append(
            f"[AI_WORKSPACE_VIDEO] {workspace_video_path}\n"
            "File video yang sudah dinormalisasi ini tersedia di komputer pribadimu dan boleh kamu akses langsung via terminal/tool."
        )
    elif history_video_path:
        media_path_lines.append(f"[VID_PATH] {history_video_path}")
    if media_path_lines:
        path_block = "\n".join(media_path_lines)
        if combined_text:
            combined_text = f"{combined_text}\n\n{path_block}".strip()
        else:
            combined_text = path_block
    if primary_video_part:
        video_prompt_ctx = "[Konteks: user mengirim video.]"
        combined_text = f"{combined_text}\n\n{video_prompt_ctx}".strip() if combined_text else video_prompt_ctx

    user_parts: List[types.Part] = [types.Part(text=combined_text)]
    if primary_image_part:
        user_parts.append(primary_image_part)
    if primary_video_part:
        user_parts.append(primary_video_part)
    history_units = float(getattr(self, "_last_history_visual_units_used", 0.0) or 0.0)
    image_units = 0.0
    if primary_image_part:
        image_units = (
            max(0.1, float(VISUAL_UNIT_WEIGHT_STICKER))
            if is_sticker_path(permanent_image_path or image_path)
            else max(0.1, float(VISUAL_UNIT_WEIGHT_IMAGE))
        )
    video_units = 0.0
    if primary_video_part and permanent_video_path:
        video_units = max(0.1, float(video_media_ctx.get("visual_units") if isinstance(video_media_ctx, dict) else 1.0))
    total_units = history_units + image_units + video_units
    self._last_request_visual_units = float(total_units)

    response_text = self._generate_with_tools(
        full_system,
        gemini_history,
        user_parts,
    )
    clean_response_text = self.session_manager._sanitize_model_text(response_text)
    stage_marks["generate"] = time.perf_counter()

    if clean_response_text:
        history_user_text = (user_text or "").strip()
        self._post_process_response(
            history_user_text,
            clean_response_text,
            history_image_path,
            history_video_path,
            ai_workspace_image_path=workspace_image_path,
            ai_workspace_video_path=workspace_video_path,
        )

    total_ms = (time.perf_counter() - flow_t0) * 1000.0
    sess_ms = (stage_marks.get("session", flow_t0) - flow_t0) * 1000.0
    sched_ms = (stage_marks.get("schedule", flow_t0) - stage_marks.get("session", flow_t0)) * 1000.0
    mem_ms = (stage_marks.get("memory", flow_t0) - stage_marks.get("schedule", flow_t0)) * 1000.0
    ctx_ms = (stage_marks.get("context", flow_t0) - stage_marks.get("memory", flow_t0)) * 1000.0
    gen_ms = (stage_marks.get("generate", flow_t0) - stage_marks.get("context", flow_t0)) * 1000.0
    logger.info(
        "[LATENCY] total=%.1fms | session=%.1fms schedule=%.1fms memory=%.1fms context=%.1fms generate=%.1fms | history=%d memories=%d has_image=%s has_video=%s",
        total_ms,
        sess_ms,
        sched_ms,
        mem_ms,
        ctx_ms,
        gen_ms,
        len(session_data.get("history", [])),
        len(relevant_memories),
        bool(primary_image_part),
        bool(primary_video_part),
    )

    return clean_response_text or "Unable to generate response. Please try again."




def retrieve_memories(self, query_text: str) -> List[Dict]:
    try:
        t0 = time.perf_counter()
        normalized_query = " ".join(str(query_text or "").split()).strip()
        if not normalized_query:
            return []

        query_embedding = None
        if normalized_query == str(getattr(self, "_last_query_text", "") or ""):
            query_embedding = getattr(self, "_last_query_embedding", None)

        if query_embedding is None:
            query_embedding = self.analyzer.get_cached_text_embedding(normalized_query)

        if query_embedding is None:
            query_embedding = self.analyzer.get_embedding(normalized_query)
        t1 = time.perf_counter()
        if query_embedding is None:
            logger.info("[LATENCY] memory_retrieval skipped: embedding_none in %.1fms", (t1 - t0) * 1000.0)
            return []
        if isinstance(query_embedding, np.ndarray) and query_embedding.size == 0:
            logger.info("[LATENCY] memory_retrieval skipped: embedding_empty_np in %.1fms", (t1 - t0) * 1000.0)
            return []
        if isinstance(query_embedding, (list, tuple)) and len(query_embedding) == 0:
            logger.info("[LATENCY] memory_retrieval skipped: embedding_empty_list in %.1fms", (t1 - t0) * 1000.0)
            return []

        self._last_query_text = normalized_query
        self._last_query_embedding = query_embedding

        embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        memories = self.memory_manager.get_relevant_memories(
            query_embedding=embedding_list,
            max_results=10,
            embedding_namespaces=["memory"],
        )
        t2 = time.perf_counter()

        logger.info(
            "[MEMORY-RETRIEVAL] Found %d relevant memories | embed=%.1fms retrieval=%.1fms total=%.1fms",
            len(memories),
            (t1 - t0) * 1000.0,
            (t2 - t1) * 1000.0,
            (t2 - t0) * 1000.0,
        )
        return memories
    except Exception as e:
        logger.error(f"[MEMORY-RETRIEVAL] Failed: {e}")
        return []


def extract_mood_context(memories: List[Dict]) -> Optional[str]:
    mood_memories = [m for m in memories if m.get("type") == "mood_state"]
    if not mood_memories:
        return None
    lines = [m.get("summary", "") for m in mood_memories if m.get("summary")]
    return "\n".join(lines) if lines else None


def gather_session_data(self) -> Dict:
    return {
        "history": self.session_manager.get_session(),
        "last_interaction": self.session_manager.get_metadata("last_user_interaction"),
    }


def process_pending_schedule(self) -> Optional[str]:
    claim = self.scheduler_service.claim_pending_schedules(
        lookahead_minutes=2,
        include_overdue=True,
        max_results=200,
        owner="interaction",
    )
    pending_schedules = claim.get("items", []) if isinstance(claim, dict) else []
    claim_note = str((claim or {}).get("claim_note") or "")
    if not pending_schedules or not claim_note:
        return None

    contexts = []
    schedule_ids = []
    for s in pending_schedules:
        sid = s.get("id")
        if sid is None:
            continue
        schedule_ids.append(int(sid))
        contexts.append(s.get("context", ""))

    if not schedule_ids:
        self.scheduler_service.release_claim(claim_note, schedule_ids=schedule_ids)
        return None

    staged = int(self.stage_pending_schedule_claim(claim_note, schedule_ids) or 0)
    if staged <= 0:
        self.scheduler_service.release_claim(claim_note, schedule_ids=schedule_ids)
        return None

    return " & ".join(filter(None, contexts))


def post_process_response(
    self,
    user_text: str,
    response_text: str,
    image_path: Optional[str],
    video_path: Optional[str] = None,
    ai_workspace_image_path: Optional[str] = None,
    ai_workspace_video_path: Optional[str] = None,
):
    history_trimmed = False
    extracted = self.session_manager.update_session(
        user_text,
        response_text,
        image_path=image_path,
        video_path=video_path,
        ai_workspace_image_path=ai_workspace_image_path,
        ai_workspace_video_path=ai_workspace_video_path,
    )
    token_extracted = self.session_manager.trim_history_by_token_budget(token_counter=self._count_history_tokens_native)
    if token_extracted:
        extracted.extend(token_extracted)
        history_trimmed = True
    if extracted and not history_trimmed:
        # update_session may also evict oldest turns when history grows too long.
        history_trimmed = True
    if extracted:
        threading.Thread(
            target=self._generate_rolling_summary,
            args=(extracted,),
            daemon=True,
        ).start()

    score_delta = 1.0
    if len(str(user_text or "").split()) >= 16 or len(str(user_text or "").strip()) >= 100:
        score_delta += 1.0
    if image_path:
        score_delta += 1.0
    if video_path:
        score_delta += 1.0
    if history_trimmed:
        score_delta += 1.0
    bump_user_profile_update_score(self, score_delta, reason="post_process_response")
    maybe_schedule_user_profile_refresh(self)


def generate_rolling_summary(self, extracted_messages: list):
    try:
        old_summary = self.session_manager.get_metadata("rolling_summary", "")

        text_lines = []
        for m in extracted_messages:
            role = "User" if m.get("role") == "user" else BOT_DISPLAY_NAME
            parts = m.get("parts", [])
            text_part = " ".join(
                p
                for p in parts
                if isinstance(p, str)
            )
            if text_part:
                text_lines.append(f"{role}: {text_part}")

        conversation_text = "\n".join(text_lines)

        prompt = "Summarize the following old conversation to keep context alive.\n"
        if old_summary:
            prompt += f"\nPrevious Summary:\n{old_summary}\n"
        prompt += f"\nNew Conversation to add to summary:\n{conversation_text}\n"
        prompt += "\nOutput ONLY the new concise combined summary."

        response = self.call_gemini(
            model=BACKGROUND_SUMMARY_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=512,
            ),
        )
        text = " ".join(str(getattr(response, "text", "") or "").split()).strip()
        if text:
            self.session_manager.update_rolling_summary(text)

    except Exception as e:
        logger.error(f"Rolling summary process failed: {e}")


def trigger_proactive_message(self, context: str) -> Optional[str]:
    if not self._acquire_processing_lock():
        self._last_proactive_failure_reason = "busy_lock"
        logger.info("[PROACTIVE] Skipped trigger because chat handler is busy.")
        return None
    try:
        self._last_proactive_failure_reason = ""
        prompt_state = build_generation_state(
            self,
            query_text=context,
            schedule_context=None,
            user_profile_context=self.session_manager.get_metadata("user_profile_context", ""),
        )
        session_data = prompt_state["session_data"]
        system_context = prompt_state["system_context"]

        proactive_instruction = (
            "\n\n[PROACTIVE TRIGGER]\n"
            "You are initiating this message yourself based on a scheduled reminder.\n"
            "Deliver the reminder naturally and in character. Do NOT start with phrases like "
            "'Reminder:' or '[System]'. Just speak naturally as yourself."
        )
        full_system = build_full_system_prompt(
            self,
            system_context,
            style="default",
            extra_instruction=proactive_instruction,
        )
        gemini_history = self._build_gemini_history(session_data["history"])
        user_parts = [types.Part(text=f"[Scheduled Reminder] {context}")]
        hist_units = float(getattr(self, "_last_history_visual_units_used", 0.0) or 0.0)
        self._last_request_visual_units = hist_units

        response = self._generate_with_tools(full_system, gemini_history, user_parts)
        if response:
            self._last_proactive_failure_reason = ""
            return response
        self._last_proactive_failure_reason = "empty_response"
        logger.warning("[PROACTIVE] trigger_proactive_message produced empty response.")
    except Exception as e:
        self._last_proactive_failure_reason = f"exception:{type(e).__name__}"
        logger.error(f"Proactive message failed: {e}")
    finally:
        self._release_processing_lock()


def finalize_proactive_delivery(self, proactive_context: str, response_text: str):
    text = (response_text or "").strip()
    if not text:
        return
    self.session_manager.append_model_message(text, interaction_source="proactive")
    self.session_manager.record_proactive_trigger_context(proactive_context)
    self.session_manager.mark_proactive_sent()
    try:
        self.proactive_learning.record_proactive_sent()
    except Exception as e:
        logger.warning("[PROACTIVE] Failed recording proactive learning event: %s", e)



