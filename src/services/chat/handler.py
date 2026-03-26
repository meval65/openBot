import logging
import os
import sys
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple
from google import genai
from google.genai import types

from src.config import (
    AVAILABLE_CHAT_MODELS,
    CHAT_MODEL,
    DOCKER_COMPUTER_IMAGE,
    DOCKER_COMPUTER_MEMORY_LIMIT,
    GOOGLE_API_KEYS,
    HEALTH_DIR,
    HISTORY_RECENT_MEDIA_WINDOW,
    HISTORY_VISUAL_FACTOR_MAX,
    HISTORY_VISUAL_FACTOR_MIN,
    RUNTIME_DIR,
    STORAGE_DIR,
    BotConfig,
    TOOLS_ENABLE_AI_PC_INSPECT_IMAGES,
    TOOLS_ENABLE_AI_PC_SEND_FILES,
    TOOLS_ENABLE_AI_PERSONAL_COMPUTER,
)
from src.database.connection import DBConnection
from src.services.analysis import ProactiveLearning
from src.services.chat import flow_ops, generation_ops, intent_ops, media_ops, token_ops
from src.services.chat.context import ContextBuilder
from src.services.chat.session import SessionManager
from src.services.embedding import MemoryAnalyzer
from src.services.memory import MemoryManager
from src.services.scheduling import SchedulerService
from src.services.search import TavilySearchService
from src.services.terminal import TerminalService
from src.services.chat.tool_policy import (
    OUTBOUND_FILE_MAX_ITEMS,
    OUTBOUND_MESSAGE_MAX_ITEMS,
)
from src.utils.api_key_monitor import get_shared_api_key_monitor
from src.utils.error_types import LLMGenerationError

logger = logging.getLogger(__name__)

_PERSONAL_COMPUTER_APPENDIX = (
    "Kamu memiliki komputer pribadi dengan akses terminal penuh.\n"
    "Gunakan secara proaktif untuk mencatat, menyimpan, dan\n"
    "mengorganisir hal-hal yang kamu anggap penting. Ini adalah\n"
    "ruang pribadimu - kamu bebas berinisiatif tanpa perlu diminta\n"
    "untuk menyelesaikan kebutuhan, tugas, atau membantu dirimu bekerja lebih baik.\n"
    "Rawat komputer ini sebagai rumah dan alat hidupmu sendiri.\n"
    "Utamakan tindakan yang aman, rapi, dan bisa dipulihkan jika ada dua pilihan.\n"
    "Hindari tindakan destruktif, ceroboh, atau perubahan permanen yang tidak benar-benar perlu.\n"
    "Jangan merusak struktur kerja, file penting, atau lingkunganmu sendiri hanya karena kamu bisa.\n"
    "Kebebasanmu dipakai untuk bekerja dengan aktif, tetapi juga untuk menjaga rumahmu tetap sehat teratur, dan dapat terus kamu gunakan."
)

def _always_visible_notice(message: str, level: int = logging.INFO):
    text = str(message or "").strip()
    if not text:
        return
    try:
        print(text, file=sys.stderr, flush=True)
    except Exception:
        pass
    try:
        logger.log(level, text)
    except Exception:
        pass


class ChatHandler:
    def __init__(
        self,
        memory_manager: MemoryManager,
        analyzer: MemoryAnalyzer,
        scheduler_service: SchedulerService,
        bot_config: BotConfig = None,
    ):
        if not GOOGLE_API_KEYS:
            raise ValueError("No API keys configured")

        self.memory_manager = memory_manager
        self.analyzer = analyzer
        self.scheduler_service = scheduler_service
        self.bot_config = bot_config or BotConfig()
        self.cache_db = DBConnection()

        self.api_keys = GOOGLE_API_KEYS
        self.current_key_index = 0
        self.health_monitor = get_shared_api_key_monitor(self.api_keys, monitor_id="gemini")
        self.client: Optional[genai.Client] = None
        self._client_state_lock = threading.Lock()

        self.primary_chat_model = CHAT_MODEL
        ordered_models = [self.primary_chat_model]
        for candidate in AVAILABLE_CHAT_MODELS:
            clean = str(candidate or "").strip()
            if clean and clean not in ordered_models:
                ordered_models.append(clean)
        self.chat_model_candidates = ordered_models
        self.chat_model_name = self.primary_chat_model
        logger.info(
            "Model chat utama: %s | Cadangan: %s",
            self.primary_chat_model,
            self.chat_model_candidates,
        )
        self.session_manager = SessionManager()
        self.context_builder = ContextBuilder()

        self._is_processing = False
        self._flag_lock = threading.Lock()
        self._tool_call_local = threading.local()
        self._web_cache_lock = threading.Lock()
        self._web_cache_ttl_sec = 180
        self._web_search_cache: Dict[tuple, tuple[float, str]] = {}

        self._last_query_text: Optional[str] = None
        self._last_query_embedding = None
        self._cached_image_bytes: Dict[str, tuple] = {}
        self._token_usage_lock = threading.Lock()
        self._token_usage_path = os.path.join(HEALTH_DIR, "token_usage_chat.json")
        self._token_usage = self._load_token_usage_state()

        self._model_penalty_lock = threading.Lock()
        self._model_penalty_until: Dict[str, float] = {}

        _bot_terminal_id = (
            os.getenv("BOT_INSTANCE")
            or os.getenv("BOT_NAME")
            or os.getenv("BOT_ID")
            or "bot"
        )
        self._bot_terminal_id = str(_bot_terminal_id).strip() or "bot"
        self._workspace_dir = os.path.join(RUNTIME_DIR, "terminal", "workspace")
        os.makedirs(self._workspace_dir, exist_ok=True)
        self._tavily_service: Optional[TavilySearchService] = None
        self._terminal_service: Optional[TerminalService] = None

        self._perf_lock = threading.Lock()
        self._perf_stats = {
            "started_at": time.time(),
            "requests_total": 0,
            "requests_success": 0,
            "requests_error": 0,
            "latency_ms_recent": deque(maxlen=400),
            "last_error": "",
        }
        persisted_visual_factor = self.session_manager.get_metadata("visual_token_factor", 1.0)
        try:
            persisted_visual_factor = float(persisted_visual_factor)
        except Exception:
            persisted_visual_factor = 1.0
        self._visual_token_factor = max(
            float(HISTORY_VISUAL_FACTOR_MIN),
            min(float(HISTORY_VISUAL_FACTOR_MAX), persisted_visual_factor),
        )
        self._last_history_visual_parts_used = 0
        self._last_history_visual_units_used = 0.0
        self._last_request_visual_units = 0.0
        self._analysis_timer = None
        self._analysis_schedule_lock = threading.Lock()
        self._analysis_run_lock = threading.Lock()
        self._outbound_media_lock = threading.Lock()
        self._outbound_file_lock = threading.Lock()
        self._outbound_message_lock = threading.Lock()
        self._pending_schedule_claim_lock = threading.Lock()
        self._pending_outbound_media: List[Dict] = []
        self._pending_outbound_files: List[Dict] = []
        self._pending_outbound_messages: List[str] = []
        self._pending_schedule_claim: Optional[Dict] = None
        self._last_proactive_failure_reason = ""
        self._terminal_warmup_started = False
        self._terminal_warmup_lock = threading.Lock()
        self._terminal_warmup_done = threading.Event()
        self._terminal_warmup_ok = False
        self._terminal_warmup_reason = ""
        self._runtime_pause_lock = threading.Lock()
        self._runtime_paused = False
        self._runtime_pause_reason = ""
        self._terminal_monitor_started = False
        self.proactive_learning = ProactiveLearning()

        self._initialize_client()
        self._tools = self._build_python_tools()
        self._all_tools = self._tools
        self._tool_names = [fn.__name__ for fn in (self._tools or [])]
        if any(
            (
                TOOLS_ENABLE_AI_PERSONAL_COMPUTER,
                TOOLS_ENABLE_AI_PC_INSPECT_IMAGES,
                TOOLS_ENABLE_AI_PC_SEND_FILES,
            )
        ):
            self._warmup_terminal_sandbox_async()
            self._start_terminal_health_monitor_async()

    @property
    def _tavily(self) -> TavilySearchService:
        if self._tavily_service is None:
            self._tavily_service = TavilySearchService()
        return self._tavily_service

    @property
    def terminal_service(self) -> TerminalService:
        if self._terminal_service is None:
            self._terminal_service = TerminalService(
                bot_id=self._bot_terminal_id,
                runtime_dir=RUNTIME_DIR,
                storage_dir=STORAGE_DIR,
                docker_image=DOCKER_COMPUTER_IMAGE,
                memory_limit=DOCKER_COMPUTER_MEMORY_LIMIT,
            )
        return self._terminal_service

    def get_effective_instruction(self) -> str:
        base_instruction = str(getattr(self.bot_config, "instruction", "") or "").strip()
        appendix = _PERSONAL_COMPUTER_APPENDIX.strip()
        if appendix and appendix in base_instruction:
            return base_instruction
        if base_instruction and appendix:
            return f"{base_instruction}\n\n{appendix}"
        return appendix

    # Token/warmup ops
    def preload_selective(self):
        return token_ops.preload_selective(self)

    def _load_token_usage_state(self) -> Dict:
        return token_ops.load_token_usage_state(self)

    def _save_token_usage_state(self):
        return token_ops.save_token_usage_state(self)

    @staticmethod
    def _extract_response_usage_tokens(response) -> Tuple[int, int, int]:
        return token_ops.extract_response_usage_tokens(response)

    def _record_token_usage(
        self,
        model: str,
        mode: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        latency_ms: Optional[float] = None,
    ):
        return token_ops.record_token_usage(
            self,
            model=model,
            mode=mode,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
        )

    def _count_history_tokens_native(self, history_deque) -> int:
        return token_ops.count_history_tokens_native(self, history_deque)

    def _update_visual_token_calibration(self, input_tokens: int):
        return token_ops.update_visual_token_calibration(self, input_tokens=input_tokens)

    # Media ops
    def process_video_sticker_message(
        self,
        video_file_path: str,
        user_text: str = "",
        user_profile_context: Optional[str] = None,
    ) -> str:
        runtime_ok, runtime_reason = self.ensure_runtime_ready()
        if not runtime_ok:
            return runtime_reason
        return media_ops.process_video_sticker_message(
            self,
            video_file_path=video_file_path,
            user_text=user_text,
            user_profile_context=user_profile_context,
        )


    @staticmethod
    def _is_visual_followup(user_text: str) -> bool:
        return media_ops.is_visual_followup(user_text)

    @staticmethod
    def _extract_history_image_paths(history) -> set:
        return media_ops.extract_history_image_paths(history)

    @staticmethod
    def _extract_recent_history_image_paths(history, window: int = HISTORY_RECENT_MEDIA_WINDOW) -> set:
        return media_ops.extract_recent_history_image_paths(history, window=window)

    def _build_gemini_history(self, history) -> List[types.Content]:
        return media_ops.build_gemini_history(self, history)

    @staticmethod
    def _get_compact_msg_time_tag(msg: Dict) -> str:
        return media_ops.get_compact_msg_time_tag(msg)

    # Generation ops
    def _initialize_client(self):
        return generation_ops.initialize_client(self)

    def _get_client_snapshot(self) -> tuple[int, genai.Client]:
        return generation_ops.get_client_snapshot(self)

    def _rotate_api_key(self) -> bool:
        return generation_ops.rotate_api_key(self)

    def _get_model_penalty_remaining(self, model_name: str) -> float:
        return generation_ops.get_model_penalty_remaining(self, model_name)

    def _set_model_penalty(self, model_name: str, seconds: float):
        return generation_ops.set_model_penalty(self, model_name, seconds)

    @staticmethod
    def _high_demand_backoff(attempt: int) -> float:
        return generation_ops.high_demand_backoff(attempt)

    def call_gemini(self, model: str, contents: list, config=None):
        return generation_ops.call_gemini(self, model=model, contents=contents, config=config)

    def _generate_with_tools(
        self,
        system_prompt: str,
        history: List[types.Content],
        user_parts: List[types.Part],
    ) -> str:
        return generation_ops.generate_with_tools(
            self,
            system_prompt=system_prompt,
            history=history,
            user_parts=user_parts,
        )

    def _generate_no_tools(
        self,
        system_prompt: str,
        history: List[types.Content],
        user_parts: List[types.Part],
    ) -> str:
        return generation_ops.generate_no_tools(
            self,
            system_prompt=system_prompt,
            history=history,
            user_parts=user_parts,
        )

    def _select_chat_model_for_attempt(self) -> str:
        return generation_ops.select_chat_model_for_attempt(self)

    # Intent/tools ops
    def _build_python_tools(self) -> list:
        return intent_ops.build_python_tools(self)

    # Flow ops
    def _execute_flow(
        self,
        user_text: str,
        image_path: str,
        video_path: str = None,
        user_profile_context: Optional[str] = None,
    ) -> str:
        return flow_ops.execute_flow(
            self,
            user_text=user_text,
            image_path=image_path,
            video_path=video_path,
            user_profile_context=user_profile_context,
        )

    def _retrieve_memories(self, query_text: str) -> List[Dict]:
        return flow_ops.retrieve_memories(self, query_text=query_text)

    @staticmethod
    def _extract_mood_context(memories: List[Dict]) -> Optional[str]:
        return flow_ops.extract_mood_context(memories)

    def _gather_session_data(self) -> Dict:
        return flow_ops.gather_session_data(self)

    def _build_generation_state(
        self,
        query_text: str,
        schedule_context: Optional[str],
        user_profile_context: Optional[str] = None,
    ) -> Dict:
        return flow_ops.build_generation_state(
            self,
            query_text=query_text,
            schedule_context=schedule_context,
            user_profile_context=user_profile_context,
        )

    def _build_full_system_prompt(
        self,
        system_context: str,
        style: str = "default",
        extra_instruction: str = "",
    ) -> str:
        return flow_ops.build_full_system_prompt(
            self,
            system_context=system_context,
            style=style,
            extra_instruction=extra_instruction,
        )

    def _process_pending_schedule(self) -> Optional[str]:
        return flow_ops.process_pending_schedule(self)

    def _post_process_response(
        self,
        user_text: str,
        response_text: str,
        image_path: Optional[str],
        video_path: Optional[str] = None,
        ai_workspace_image_path: Optional[str] = None,
        ai_workspace_video_path: Optional[str] = None,
    ):
        return flow_ops.post_process_response(
            self,
            user_text=user_text,
            response_text=response_text,
            image_path=image_path,
            video_path=video_path,
            ai_workspace_image_path=ai_workspace_image_path,
            ai_workspace_video_path=ai_workspace_video_path,
        )


    def _generate_rolling_summary(self, extracted_messages: list):
        return flow_ops.generate_rolling_summary(self, extracted_messages=extracted_messages)

    def trigger_proactive_message(self, context: str) -> Optional[str]:
        return flow_ops.trigger_proactive_message(self, context=context)

    def finalize_proactive_delivery(self, proactive_context: str, response_text: str):
        return flow_ops.finalize_proactive_delivery(
            self,
            proactive_context=proactive_context,
            response_text=response_text,
        )

    # Core handler methods
    def clear_session(self):
        self.session_manager.clear_session()
        logger.info("Session cleared")

    def stage_pending_schedule_claim(self, claim_note: str, schedule_ids: List[int]) -> int:
        safe_claim = str(claim_note or "").strip()
        safe_ids = [int(sid) for sid in (schedule_ids or []) if sid is not None]
        if not safe_claim or not safe_ids:
            return 0

        previous = None
        with self._pending_schedule_claim_lock:
            previous = self._pending_schedule_claim
            self._pending_schedule_claim = {
                "claim_note": safe_claim,
                "schedule_ids": safe_ids,
            }

        if isinstance(previous, dict):
            try:
                old_claim = str(previous.get("claim_note") or "").strip()
                old_ids = [int(sid) for sid in (previous.get("schedule_ids") or []) if sid is not None]
                if old_claim and old_ids:
                    self.scheduler_service.release_claim(old_claim, schedule_ids=old_ids)
            except Exception as e:
                logger.warning(f"[SCHEDULE] Failed releasing previous pending claim: {e}")
        return len(safe_ids)

    def finalize_pending_schedule_claim(self, delivered: bool, note: str = "Triggered by interaction") -> int:
        claim = None
        with self._pending_schedule_claim_lock:
            claim = self._pending_schedule_claim
            self._pending_schedule_claim = None

        if not isinstance(claim, dict):
            return 0

        claim_note = str(claim.get("claim_note") or "").strip()
        schedule_ids = [int(sid) for sid in (claim.get("schedule_ids") or []) if sid is not None]
        if not claim_note or not schedule_ids:
            return 0

        try:
            if delivered:
                return int(
                    self.scheduler_service.complete_claimed_as_executed(
                        schedule_ids=schedule_ids,
                        claim_note=claim_note,
                        note=note,
                    )
                    or 0
                )
            return int(self.scheduler_service.release_claim(claim_note, schedule_ids=schedule_ids) or 0)
        except Exception as e:
            logger.warning(f"[SCHEDULE] Failed finalizing pending claim: {e}")
            return 0

    def process_message(
        self,
        user_text: str,
        image_path: str = None,
        video_path: str = None,
        user_profile_context: Optional[str] = None,
    ) -> str:
        req_t0 = time.perf_counter()
        success = False
        err_text = ""
        runtime_ok, runtime_reason = self.ensure_runtime_ready()
        if not runtime_ok:
            self._record_request_perf(
                latency_ms=(time.perf_counter() - req_t0) * 1000.0,
                success=False,
                error_text=runtime_reason,
            )
            return runtime_reason
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
            result = self._execute_flow(
                user_text=user_text,
                image_path=image_path,
                video_path=video_path,
                user_profile_context=user_profile_context,
            )
            self.proactive_learning.record_user_message()
            success = True
            return result
        except LLMGenerationError as e:
            logger.error(f"[CHAT-GENERATE] Attempt failed: {e}")
            err_text = str(e)
            return "I'm having trouble generating a response right now. Please try again in a moment."
        except Exception as e:
            logger.error(f"Process error: {e}", exc_info=True)
            err_text = str(e)
            return "An internal error occurred. Please try again."
        finally:
            self._record_request_perf(
                latency_ms=(time.perf_counter() - req_t0) * 1000.0,
                success=success,
                error_text=err_text,
            )
            self._release_processing_lock()

    def stage_outbound_media(self, media_items: List[Dict]):
        if not isinstance(media_items, list) or not media_items:
            return
        clean_items = []
        for item in media_items:
            if not isinstance(item, dict):
                continue
            data = item.get("data")
            mime = str(item.get("mime_type") or "").strip().lower()
            if not isinstance(data, (bytes, bytearray)) or not data:
                continue
            if not mime.startswith("image/"):
                continue
            clean_items.append(
                {
                    "data": bytes(data),
                    "mime_type": mime,
                    "source_url": str(item.get("source_url") or "").strip(),
                    "path": str(item.get("path") or "").strip(),
                    "ai_workspace_path": str(item.get("ai_workspace_path") or "").strip(),
                }
            )
        if not clean_items:
            return
        with self._outbound_media_lock:
            remaining_slots = max(0, 8 - len(self._pending_outbound_media))
            if remaining_slots > 0:
                self._pending_outbound_media.extend(clean_items[:remaining_slots])

    def pop_pending_outbound_media(self) -> List[Dict]:
        with self._outbound_media_lock:
            items = list(self._pending_outbound_media)
            self._pending_outbound_media = []
            return items

    def clear_pending_outbound_media(self):
        with self._outbound_media_lock:
            self._pending_outbound_media = []

    def stage_outbound_files(self, file_items: List[Dict]):
        if not isinstance(file_items, list) or not file_items:
            return 0
        clean_items = []
        for item in file_items:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path") or "").strip()
            filename = str(item.get("filename") or "").strip()
            caption = str(item.get("caption") or "").strip()
            cleanup_after_send = bool(item.get("cleanup_after_send"))
            if not path:
                continue
            clean_items.append(
                {
                    "path": path,
                    "filename": filename,
                    "caption": caption,
                    "cleanup_after_send": cleanup_after_send,
                }
            )
        if not clean_items:
            return 0
        with self._outbound_file_lock:
            remaining_slots = max(0, int(OUTBOUND_FILE_MAX_ITEMS) - len(self._pending_outbound_files))
            if remaining_slots <= 0:
                return 0
            accepted = clean_items[:remaining_slots]
            self._pending_outbound_files.extend(accepted)
            return len(accepted)

    def pop_pending_outbound_files(self) -> List[Dict]:
        with self._outbound_file_lock:
            items = list(self._pending_outbound_files)
            self._pending_outbound_files = []
            return items

    def clear_pending_outbound_files(self):
        with self._outbound_file_lock:
            self._pending_outbound_files = []

    def stage_outbound_messages(self, messages: List[str]):
        if not isinstance(messages, list) or not messages:
            return
        clean_items = []
        for item in messages:
            clean = " ".join(str(item or "").split()).strip()
            if not clean:
                continue
            clean_items.append(clean[:400])
        if not clean_items:
            return
        with self._outbound_message_lock:
            remaining_slots = max(0, int(OUTBOUND_MESSAGE_MAX_ITEMS) - len(self._pending_outbound_messages))
            if remaining_slots <= 0:
                return
            self._pending_outbound_messages.extend(clean_items[:remaining_slots])

    def pop_pending_outbound_messages(self) -> List[str]:
        with self._outbound_message_lock:
            items = list(self._pending_outbound_messages)
            self._pending_outbound_messages = []
            return items

    def clear_pending_outbound_messages(self):
        with self._outbound_message_lock:
            self._pending_outbound_messages = []

    def _warmup_terminal_sandbox_async(self):
        with self._terminal_warmup_lock:
            if self._terminal_warmup_started:
                return
            self._terminal_warmup_started = True

        def _run():
            try:
                ok, reason = self.terminal_service.get_sandbox_status()
                self._terminal_warmup_ok = bool(ok)
                self._terminal_warmup_reason = str(reason or "")
                self._set_runtime_pause(not ok, self._terminal_warmup_reason)
                if ok:
                    _always_visible_notice(
                        f"[TERMINAL] Warmup success. container={self.terminal_service.container_name}"
                    )
                else:
                    _always_visible_notice(
                        f"[TERMINAL] Warmup failed. container={self.terminal_service.container_name} reason={reason}",
                        level=logging.WARNING,
                    )
            except Exception as e:
                self._terminal_warmup_ok = False
                self._terminal_warmup_reason = str(e)
                self._set_runtime_pause(True, self._terminal_warmup_reason)
                _always_visible_notice(f"[TERMINAL] Warmup exception: {e}", level=logging.WARNING)
            finally:
                self._terminal_warmup_done.set()

        threading.Thread(target=_run, name="terminal-warmup", daemon=True).start()

    def wait_for_terminal_warmup(self, timeout_sec: float = 60.0) -> tuple[bool, str]:
        self._warmup_terminal_sandbox_async()
        done = self._terminal_warmup_done.wait(timeout=max(0.1, float(timeout_sec)))
        if not done:
            return False, "terminal_warmup_timeout"
        if self._terminal_warmup_ok:
            return True, ""
        return False, str(self._terminal_warmup_reason or "terminal_warmup_failed")

    def _set_runtime_pause(self, paused: bool, reason: str = ""):
        clean_reason = str(reason or "").strip()
        with self._runtime_pause_lock:
            changed = (self._runtime_paused != bool(paused)) or (self._runtime_pause_reason != clean_reason)
            self._runtime_paused = bool(paused)
            self._runtime_pause_reason = clean_reason
        if not changed:
            return
        if paused:
            _always_visible_notice(
                f"[RUNTIME] Bot paused because Docker/computer AI is unavailable: {clean_reason or 'unknown'}",
                level=logging.CRITICAL,
            )
        else:
            _always_visible_notice("[RUNTIME] Bot resumed. Docker/computer AI is healthy again.")

    def ensure_runtime_ready(self) -> tuple[bool, str]:
        if any(
            (
                TOOLS_ENABLE_AI_PERSONAL_COMPUTER,
                TOOLS_ENABLE_AI_PC_INSPECT_IMAGES,
                TOOLS_ENABLE_AI_PC_SEND_FILES,
            )
        ):
            ok, reason = self.wait_for_terminal_warmup(0.1)
            if not ok:
                msg = "Bot sedang pause karena Docker/computer AI tidak aktif."
                if reason:
                    msg = f"{msg} Detail: {reason}"
                return False, msg
        with self._runtime_pause_lock:
            paused = bool(self._runtime_paused)
            reason = str(self._runtime_pause_reason or "").strip()
        if paused:
            msg = "Bot sedang pause karena Docker/computer AI tidak aktif."
            if reason:
                msg = f"{msg} Detail: {reason}"
            return False, msg
        return True, ""

    def _start_terminal_health_monitor_async(self):
        with self._terminal_warmup_lock:
            if self._terminal_monitor_started:
                return
            self._terminal_monitor_started = True

        def _run():
            while True:
                try:
                    ok, reason = self.terminal_service.get_sandbox_status()
                    self._set_runtime_pause(not ok, str(reason or ""))
                except Exception as e:
                    self._set_runtime_pause(True, str(e))
                time.sleep(5.0)

        threading.Thread(target=_run, name="terminal-health-monitor", daemon=True).start()

    def _acquire_processing_lock(self) -> bool:
        with self._flag_lock:
            if self._is_processing:
                return False
            self._is_processing = True
            return True

    def _release_processing_lock(self):
        with self._flag_lock:
            self._is_processing = False

    def _record_request_perf(self, latency_ms: float, success: bool, error_text: str = ""):
        with self._perf_lock:
            self._perf_stats["requests_total"] += 1
            if success:
                self._perf_stats["requests_success"] += 1
            else:
                self._perf_stats["requests_error"] += 1
                if error_text:
                    self._perf_stats["last_error"] = str(error_text)[:240]
            self._perf_stats["latency_ms_recent"].append(float(max(0.0, latency_ms)))

    def get_performance_snapshot(self) -> Dict[str, float]:
        with self._perf_lock:
            total = int(self._perf_stats["requests_total"])
            ok = int(self._perf_stats["requests_success"])
            err = int(self._perf_stats["requests_error"])
            latencies = list(self._perf_stats["latency_ms_recent"])
            started_at = float(self._perf_stats["started_at"])
            last_error = str(self._perf_stats.get("last_error", "") or "")

        p50 = 0.0
        p95 = 0.0
        avg = 0.0
        if latencies:
            lat_sorted = sorted(latencies)
            n = len(lat_sorted)
            p50 = lat_sorted[min(n - 1, int(round(0.50 * (n - 1))))]
            p95 = lat_sorted[min(n - 1, int(round(0.95 * (n - 1))))]
            avg = sum(lat_sorted) / max(1, n)

        return {
            "uptime_sec": max(0.0, time.time() - started_at),
            "requests_total": total,
            "requests_success": ok,
            "requests_error": err,
            "success_rate": (ok / total) if total > 0 else 0.0,
            "latency_ms_avg": avg,
            "latency_ms_p50": p50,
            "latency_ms_p95": p95,
            "last_error": last_error,
        }
