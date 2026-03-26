import datetime
import io
import logging
import mimetypes
import os
import time
from typing import Dict, List, Optional

import pytz
from PIL import Image
from google.genai import types

from src.config import (
    HISTORY_IMAGE_MAX_SIDE,
    HISTORY_RECENT_MEDIA_WINDOW,
    HISTORY_TOKEN_BUDGET,
    HISTORY_VISUAL_PART_TOKEN_BASE,
    HISTORY_VISUAL_TOKEN_MIN,
    HISTORY_VISUAL_TOKEN_RATIO,
    INPUT_IMAGE_MEDIA_RESOLUTION,
    STICKER_MEDIA_RESOLUTION,
    TIMEZONE,
    VISUAL_UNIT_WEIGHT_IMAGE,
    VISUAL_UNIT_WEIGHT_STICKER,
)
from src.services.media.pipeline import (
    build_video_sticker_payload,
    is_sticker_path,
    load_video_analysis,
)
from src.services.media.video_service import (
    estimate_video_visual_units,
)
from src.services.chat.tool_prompt import build_tool_usage_directive
from src.services.chat.workspace_context import build_workspace_snapshot

logger = logging.getLogger(__name__)


def _to_media_resolution(level: str):
    key = f"MEDIA_RESOLUTION_{str(level or '').strip().upper()}"
    return getattr(types.MediaResolution, key, None)


def _should_use_media_resolution(model_name: str) -> bool:
    return "gemini" in str(model_name or "").lower()


def _part_from_bytes_with_resolution(data: bytes, mime_type: str, level: str, model_name: str):
    if not _should_use_media_resolution(model_name):
        return types.Part.from_bytes(data=data, mime_type=mime_type)
    resolved = _to_media_resolution(level)
    if resolved is None:
        return types.Part.from_bytes(data=data, mime_type=mime_type)
    try:
        return types.Part.from_bytes(
            data=data,
            mime_type=mime_type,
            media_resolution=resolved,
        )
    except Exception:
        return types.Part.from_bytes(data=data, mime_type=mime_type)


def _visual_units_for_image_path(path: str) -> float:
    if is_sticker_path(path):
        return max(0.1, float(VISUAL_UNIT_WEIGHT_STICKER))
    return max(0.1, float(VISUAL_UNIT_WEIGHT_IMAGE))


def _local_tz():
    try:
        return pytz.timezone(TIMEZONE)
    except Exception:
        return pytz.timezone("Asia/Jakarta")


def _format_human_time(dt: datetime.datetime) -> str:
    tz = _local_tz()
    if dt.tzinfo is None:
        dt = tz.localize(dt)
    local_dt = dt.astimezone(tz)
    return local_dt.strftime("%Y-%m-%d %H:%M %Z")


def _generate_video_sticker_response(
    self,
    user_text: str,
    video_data: bytes,
    mime_type: str,
    user_profile_context: Optional[str],
    video_visual_units: float = 1.0,
) -> str:
    session_data = self._gather_session_data()
    schedule_context = self._process_pending_schedule()
    relevant_memories = self._retrieve_memories(user_text)
    mood_context = self._extract_mood_context(relevant_memories)

    system_context = self.context_builder.build_context(
        relevant_memories,
        session_data["last_interaction"],
        schedule_context,
        mood_context=mood_context,
        user_profile_context=user_profile_context,
    )
    rolling_summary = self.session_manager.get_metadata("rolling_summary")
    if rolling_summary:
        system_context += f"\n\n[Previous Conversation Summary]\n{rolling_summary}"

    tool_usage_directive = build_tool_usage_directive(
        style="strict",
        available_tools=getattr(self, "_tool_names", None),
    )
    workspace_snapshot = build_workspace_snapshot(self._workspace_dir)
    full_system = (
        f"{self.get_effective_instruction()}{tool_usage_directive}\n\n"
        f"{system_context}\n\n"
        f"{workspace_snapshot}"
    )

    gemini_history = self._build_gemini_history(session_data["history"])
    user_parts = [
        types.Part(text=(user_text or "[Konteks: user mengirim sticker video.]")),
        _part_from_bytes_with_resolution(
            data=video_data,
            mime_type=mime_type,
            level=STICKER_MEDIA_RESOLUTION,
            model_name=self.chat_model_name,
        ),
    ]
    history_units = float(getattr(self, "_last_history_visual_units_used", 0.0) or 0.0)
    current_video_units = max(0.1, float(video_visual_units))
    total_units = history_units + current_video_units
    self._last_request_visual_units = float(total_units)
    return self._generate_with_tools(
        full_system,
        gemini_history,
        user_parts,
    )



def process_video_sticker_message(
    self,
    video_file_path: str,
    user_text: str = "",
    user_profile_context: Optional[str] = None,
) -> str:
    req_t0 = time.perf_counter()
    success = False
    err_text = ""
    if not self._acquire_processing_lock():
        self._record_request_perf(
            latency_ms=(time.perf_counter() - req_t0) * 1000.0,
            success=False,
            error_text="busy_lock",
        )
        return "Please wait, I'm still processing your previous message..."
    try:
        self.finalize_pending_schedule_claim(
            delivered=False,
            note="Superseded by new interaction",
        )
        self.clear_pending_outbound_media()
        self.clear_pending_outbound_files()
        self.clear_pending_outbound_messages()
        self._tool_call_local.last_terminal_cwd = ""
        self.bot_config.load()
        if user_profile_context:
            self.session_manager.set_metadata(
                "user_profile_context",
                str(user_profile_context).strip(),
                persist=True,
            )

        video_ctx = build_video_sticker_payload(self.cache_db, video_file_path)
        video_data = video_ctx["analysis_data"]
        mime_type = video_ctx["analysis_mime"]
        video_hash = video_ctx["hash"]
        if not mime_type.startswith("video/"):
            mime_type = "video/webm"
        collage_frame_count = int(video_ctx.get("frame_count", 0) or 0)
        response_media_data = video_data
        response_media_mime = mime_type
        response_prompt_ctx = ""
        if bool(video_ctx.get("used_collage")):
            response_prompt_ctx = (
                f"[Konteks: ini sticker video pendek yang sudah diringkas sebagai kolase frame berurutan"
                f"{f' ({collage_frame_count} frame)' if collage_frame_count else ''}. "
                "Baca urut dari kiri ke kanan lalu atas ke bawah.]"
            )
        logger.info(f"[STICKER-VIDEO] visual-only path for {video_hash[:12]}")
        base_prompt_text = (
            f"{(user_text or '').strip()}\n\n"
            f"{response_prompt_ctx}\n\n"
            "[Konteks sticker video: user mengirim sticker video.]"
        ).strip()
        response_text = _generate_video_sticker_response(
            self,
            base_prompt_text,
            response_media_data,
            response_media_mime,
            user_profile_context,
            float(video_ctx.get("visual_units") or 1.0),
        )

        clean_response_text = self.session_manager._sanitize_model_text(response_text)
        history_user_text = (user_text or "").strip()
        if clean_response_text:
            self._post_process_response(
                history_user_text,
                clean_response_text,
                None,
                None,
            )
            success = True
            return clean_response_text
        return "Unable to generate response. Please try again."
    except Exception as e:
        logger.error(f"[STICKER-VIDEO] Failed: {e}")
        err_text = str(e)
        return "Sticker video processing failed. Please try again."
    finally:
        self._record_request_perf(
            latency_ms=(time.perf_counter() - req_t0) * 1000.0,
            success=success,
            error_text=err_text,
        )
        self._release_processing_lock()



def is_visual_followup(user_text: str) -> bool:
    text = (user_text or "").strip().lower()
    if not text:
        return False
    keywords = (
        "gambar", "foto", "image", "img", "pic", "sticker",
        "yang tadi", "yang sebelumnya", "gambar tadi", "foto tadi",
        "lihat gambar", "lihat foto"
    )
    return any(k in text for k in keywords)


def extract_history_image_paths(history) -> set:
    paths = set()
    try:
        for msg in history or []:
            for part in msg.get("parts", []):
                if isinstance(part, str) and part.startswith(":::IMG_PATH:::"):
                    paths.add(part.replace(":::IMG_PATH:::", "", 1).strip())
    except Exception:
        return set()
    return paths


def extract_recent_history_image_paths(history, window: int = HISTORY_RECENT_MEDIA_WINDOW) -> set:
    paths = set()
    try:
        recent = list(history or [])[-max(1, int(window)):]
        for msg in recent:
            for part in msg.get("parts", []):
                if isinstance(part, str) and part.startswith(":::IMG_PATH:::"):
                    paths.add(part.replace(":::IMG_PATH:::", "", 1).strip())
    except Exception:
        return set()
    return paths


def extract_recent_history_video_paths(history, window: int = HISTORY_RECENT_MEDIA_WINDOW) -> set:
    paths = set()
    try:
        recent = list(history or [])[-max(1, int(window)):]
        for msg in recent:
            for part in msg.get("parts", []):
                if isinstance(part, str) and part.startswith(":::VID_PATH:::"):
                    paths.add(part.replace(":::VID_PATH:::", "", 1).strip())
    except Exception:
        return set()
    return paths


def build_gemini_history(self, history) -> List[types.Content]:
    visual_budget_tokens = max(
        int(HISTORY_VISUAL_TOKEN_MIN),
        int(max(1000, int(HISTORY_TOKEN_BUDGET)) * max(0.05, min(0.8, float(HISTORY_VISUAL_TOKEN_RATIO)))),
    )
    used_visual_tokens = 0
    used_visual_units = 0.0
    used_visual_parts = 0
    gemini_history = []
    seen_history_images: set = set()
    seen_history_videos: set = set()
    base_per_part = max(200, int(HISTORY_VISUAL_PART_TOKEN_BASE))
    factor = float(getattr(self, "_visual_token_factor", 1.0) or 1.0)
    factor = max(0.2, min(5.0, factor))
    # Keep per-image estimate aligned with Gemini-like patch base cost behavior:
    # avoid dropping too low even when calibrated factor decreases.
    est_per_part = max(258, int(base_per_part * factor))
    history_img_max_side = max(512, int(HISTORY_IMAGE_MAX_SIDE))
    recent_image_paths = extract_recent_history_image_paths(history, window=HISTORY_RECENT_MEDIA_WINDOW)
    recent_video_paths = extract_recent_history_video_paths(history, window=HISTORY_RECENT_MEDIA_WINDOW)

    def _read_history_image_bytes(path: str, max_side: int = 480) -> tuple[bytes, str]:
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                if max(im.size) > max_side:
                    im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=82, optimize=True)
                return buf.getvalue(), "image/jpeg"
        except Exception:
            # History rebuild only needs raw bytes + mime type.
            with open(path, "rb") as f:
                data = f.read()
            mime_type, _ = mimetypes.guess_type(path)
            if not mime_type:
                mime_type = "image/jpeg"
            return data, mime_type

    for msg in history:
        role = msg.get("role", "user")
        parts = msg.get("parts", [])
        time_tag = self._get_compact_msg_time_tag(msg)
        time_tag_used = False

        content_parts: List[types.Part] = []
        for p in parts:
            if not isinstance(p, str):
                continue
            if p.startswith(":::IMG_PATH:::"):
                img_path = p.replace(":::IMG_PATH:::", "", 1)
                if is_sticker_path(img_path):
                    seen_history_images.add(img_path)
                    continue
                if img_path in seen_history_images:
                    continue
                content_parts.append(types.Part(text=f"[IMG_PATH] {img_path}"))
                if (
                    img_path in recent_image_paths
                    and
                    used_visual_tokens < visual_budget_tokens
                    and os.path.exists(img_path)
                ):
                    # Keep history images reasonably detailed for better OCR/detail retention.
                    data, mime_type = _read_history_image_bytes(img_path, max_side=history_img_max_side)
                    unit_weight = _visual_units_for_image_path(img_path)
                    est_tokens = max(258, int(est_per_part * unit_weight))
                    if (used_visual_tokens + est_tokens) > visual_budget_tokens:
                        seen_history_images.add(img_path)
                        continue
                    try:
                        level = STICKER_MEDIA_RESOLUTION if is_sticker_path(img_path) else INPUT_IMAGE_MEDIA_RESOLUTION
                        content_parts.append(
                            _part_from_bytes_with_resolution(
                                data=data,
                                mime_type=mime_type,
                                level=level,
                                model_name=self.chat_model_name,
                            )
                        )
                        used_visual_tokens += est_tokens
                        used_visual_units += float(unit_weight)
                        used_visual_parts += 1
                    except Exception as e:
                        logger.warning(f"[IMG-HISTORY] Failed to load recent image {img_path}: {e}")
                seen_history_images.add(img_path)
            elif p.startswith(":::VID_PATH:::"):
                vid_path = p.replace(":::VID_PATH:::", "", 1).strip()
                if is_sticker_path(vid_path):
                    seen_history_videos.add(vid_path)
                    continue
                if vid_path in seen_history_videos:
                    continue
                content_parts.append(types.Part(text=f"[VID_PATH] {vid_path}"))
                if (
                    vid_path in recent_video_paths
                    and
                    used_visual_tokens < visual_budget_tokens
                    and os.path.exists(vid_path)
                ):
                    try:
                        video_analysis = load_video_analysis(vid_path)
                        vdata = video_analysis["analysis_data"]
                        vmime = video_analysis["analysis_mime"]
                        unit_weight = max(0.1, float(video_analysis.get("visual_units") or estimate_video_visual_units(vid_path)))
                        est_tokens = max(258, int(est_per_part * unit_weight))
                        if (used_visual_tokens + est_tokens) <= visual_budget_tokens:
                            level = STICKER_MEDIA_RESOLUTION if is_sticker_path(vid_path) else INPUT_IMAGE_MEDIA_RESOLUTION
                            content_parts.append(
                                _part_from_bytes_with_resolution(
                                    data=vdata,
                                    mime_type=vmime,
                                    level=level,
                                    model_name=self.chat_model_name,
                                )
                            )
                            used_visual_tokens += est_tokens
                            used_visual_units += float(unit_weight)
                            used_visual_parts += 1
                    except Exception as e:
                        logger.warning(f"[VID-HISTORY] Failed to load recent video {vid_path}: {e}")
                seen_history_videos.add(vid_path)
            else:
                if time_tag and not time_tag_used:
                    p = f"{time_tag} {p}"
                    time_tag_used = True
                content_parts.append(types.Part(text=p))

        if time_tag and not time_tag_used:
            content_parts.insert(0, types.Part(text=time_tag))

        if not content_parts:
            continue

        gemini_role = "user" if role == "user" else "model"
        gemini_history.append(types.Content(role=gemini_role, parts=content_parts))

    self._last_history_visual_parts_used = int(used_visual_parts)
    self._last_history_visual_units_used = float(used_visual_units)

    return gemini_history


def get_compact_msg_time_tag(msg: Dict) -> str:
    try:
        raw_time = msg.get("time")
        if isinstance(raw_time, str) and raw_time.strip():
            return f"[time:{raw_time.strip()}]"

        raw_t = msg.get("t")
        if isinstance(raw_t, (int, float)):
            t = int(raw_t)
            if t > 0:
                dt = datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc)
                return f"[time:{_format_human_time(dt)}]"

        raw_ts = msg.get("ts")
        if isinstance(raw_ts, str) and raw_ts.strip():
            parsed = datetime.datetime.fromisoformat(raw_ts.strip())
            return f"[time:{_format_human_time(parsed)}]"
    except Exception:
        return ""
    return ""



