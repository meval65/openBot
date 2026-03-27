import logging
import random
import time
from typing import List, Optional

from google import genai
from google.genai import types

from src.config import (
    CHAT_BASE_RETRY_DELAY,
    CHAT_MAX_RETRIES,
    HIGH_DEMAND_BASE_DELAY,
    HIGH_DEMAND_MAX_DELAY,
    HIGH_DEMAND_MODEL_COOLDOWN,
    TOOL_MAX_REMOTE_CALLS,
)
from src.utils.api_error_policy import (
    all_chat_models_in_penalty,
    classify_api_error,
    handle_api_error_retry,
    ordered_chat_models,
)
from src.services.chat.tool_runtime import (
    NON_BUDGETED_TOOL_NAMES,
    build_tool_registry,
    execute_tool_calls,
)

logger = logging.getLogger(__name__)


def get_client_snapshot(self) -> tuple[int, genai.Client]:
    with self._client_state_lock:
        key_index = int(self.current_key_index)
        api_key = str(self.api_keys[key_index])
    return key_index, genai.Client(api_key=api_key)


def select_chat_model_for_attempt(self) -> str:
    primary = str(getattr(self, "primary_chat_model", "") or "").strip()
    models = ordered_chat_models(self)
    if not primary:
        primary = str(models[0] if models else (getattr(self, "chat_model_name", "") or "")).strip()
    chosen = primary
    high_demand_remaining = 0.0
    try:
        high_demand_remaining = float(self._get_model_high_demand_remaining(primary) or 0.0)
    except Exception:
        high_demand_remaining = 0.0
    if high_demand_remaining > 0.0:
        for candidate in models:
            clean = str(candidate or "").strip()
            if not clean or clean == primary:
                continue
            try:
                if float(self._get_model_high_demand_remaining(clean) or 0.0) <= 0.0:
                    chosen = clean
                    break
            except Exception:
                chosen = clean
                break
    self.chat_model_name = chosen
    return chosen


def _extract_function_calls(response) -> List[types.FunctionCall]:
    calls = []

    direct_calls = getattr(response, "function_calls", None)
    if isinstance(direct_calls, list):
        for item in direct_calls:
            if item is not None:
                calls.append(item)
    if calls:
        return calls

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            fc = getattr(part, "function_call", None)
            if fc is not None:
                calls.append(fc)
    return calls


def _extract_function_call_names(response) -> List[str]:
    names: List[str] = []
    for fc in _extract_function_calls(response):
        name = str(getattr(fc, "name", "") or "").strip()
        if name:
            names.append(name)
    # Keep unique order.
    seen = set()
    deduped = []
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        deduped.append(n)
    return deduped


def _summarize_empty_response(response) -> str:
    try:
        if response is None:
            return "response=None"

        direct_text = getattr(response, "text", None)
        direct_calls = getattr(response, "function_calls", None)
        candidates = getattr(response, "candidates", None) or []
        chunks = [
            f"response_type={type(response).__name__}",
            f"has_text={bool(str(direct_text or '').strip())}",
            f"direct_function_calls={len(direct_calls) if isinstance(direct_calls, list) else 0}",
            f"candidate_count={len(candidates)}",
        ]

        candidate_summaries = []
        for idx, candidate in enumerate(candidates[:3], start=1):
            finish_reason = getattr(candidate, "finish_reason", None)
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            part_types = []
            has_part_text = False
            has_part_function_call = False
            for part in parts[:6]:
                if getattr(part, "text", None):
                    has_part_text = True
                    part_types.append("text")
                elif getattr(part, "function_call", None) is not None:
                    has_part_function_call = True
                    part_types.append("function_call")
                elif getattr(part, "function_response", None) is not None:
                    part_types.append("function_response")
                elif getattr(part, "inline_data", None) is not None:
                    part_types.append("inline_data")
                elif getattr(part, "file_data", None) is not None:
                    part_types.append("file_data")
                else:
                    part_types.append(type(part).__name__)
            candidate_summaries.append(
                f"cand{idx}(finish={finish_reason}, parts={len(parts)}, has_part_text={has_part_text}, has_part_function_call={has_part_function_call}, part_types={part_types})"
            )
        if candidate_summaries:
            chunks.append("candidates=" + "; ".join(candidate_summaries))
        return " | ".join(chunks)
    except Exception as e:
        return f"failed_to_summarize_response={type(e).__name__}: {e}"


def _has_malformed_function_call_response(response) -> bool:
    try:
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            finish_reason = str(getattr(candidate, "finish_reason", "") or "").upper()
            if "MALFORMED_FUNCTION_CALL" in finish_reason:
                return True
    except Exception:
        return False
    return False


def _is_empty_response_error(error_msg: str) -> bool:
    msg = str(error_msg or "").strip().lower()
    return ("empty response received" in msg) or (msg == "empty_response")


def _get_pending_outbound_snapshot(self) -> dict:
    media_count = 0
    file_count = 0
    message_count = 0
    try:
        with self._outbound_media_lock:
            media_count = len(getattr(self, "_pending_outbound_media", []) or [])
    except Exception:
        media_count = 0
    try:
        with self._outbound_file_lock:
            file_count = len(getattr(self, "_pending_outbound_files", []) or [])
    except Exception:
        file_count = 0
    try:
        with self._outbound_message_lock:
            message_count = len(getattr(self, "_pending_outbound_messages", []) or [])
    except Exception:
        message_count = 0
    return {
        "media_count": int(media_count),
        "file_count": int(file_count),
        "message_count": int(message_count),
    }


def _build_tool_failure_fallback_system(
    system_prompt: str,
    reason_text: str,
    called_tool_names: Optional[List[str]] = None,
    outbound_snapshot: Optional[dict] = None,
) -> str:
    tool_names = [str(name or "").strip() for name in (called_tool_names or []) if str(name or "").strip()]
    outbound = outbound_snapshot if isinstance(outbound_snapshot, dict) else {}
    artifact_notes: List[str] = []
    if int(outbound.get("media_count", 0) or 0) > 0:
        artifact_notes.append(f"{int(outbound.get('media_count', 0) or 0)} media sudah berhasil disiapkan")
    if int(outbound.get("file_count", 0) or 0) > 0:
        artifact_notes.append(f"{int(outbound.get('file_count', 0) or 0) or 0} file sudah berhasil disiapkan")
    if int(outbound.get("message_count", 0) or 0) > 0:
        artifact_notes.append("ada status proses yang sudah terkirim")
    artifact_text = "; ".join(artifact_notes) if artifact_notes else "tidak ada artefak tool yang berhasil disiapkan"
    tools_text = ", ".join(tool_names) if tool_names else "tidak diketahui"
    return (
        f"{system_prompt}\n\n"
        "[SYSTEM - TOOL FAILURE FALLBACK]\n"
        "Percobaan memakai tool atau data live gagal setelah beberapa percobaan.\n"
        "Dalam jawaban finalmu, jika user meminta data live, verifikasi, atau aksi tool, kamu HARUS bilang dengan jelas bahwa pengecekan atau eksekusi barusan gagal, tidak tersedia, atau tidak berhasil diselesaikan.\n"
        "Jangan mengklaim hasil tool, data real-time, atau aksi file/sistem seolah-olah sudah berhasil kalau memang belum berhasil.\n"
        "Kamu tetap boleh membantu dengan pengetahuan umum, penalaran biasa, saran, atau langkah manual selama kamu jujur soal keterbatasan ini.\n"
        "Jika ada artefak yang memang sudah berhasil disiapkan untuk user, kamu boleh menyebut itu secara singkat tanpa berpura-pura semua tool berhasil.\n"
        f"Tool yang sempat terlibat: {tools_text}.\n"
        f"Status artefak: {artifact_text}.\n"
        f"Alasan internal singkat: {reason_text}.\n"
        "[END SYSTEM]"
    )


def _consume_staged_tool_image_parts(self) -> List[types.Part]:
    staged = getattr(self._tool_call_local, "web_image_inputs", None)
    self._tool_call_local.web_image_inputs = []
    if not isinstance(staged, list) or not staged:
        return []
    parts: List[types.Part] = []
    for item in staged[:3]:
        if not isinstance(item, dict):
            continue
        data = item.get("data")
        mime = str(item.get("mime_type") or "image/jpeg").strip().lower()
        if not isinstance(data, (bytes, bytearray)) or not data:
            continue
        if not mime.startswith("image/"):
            mime = "image/jpeg"
        try:
            parts.append(types.Part.from_bytes(data=bytes(data), mime_type=mime))
        except Exception:
            continue
    return parts


def _consume_staged_inspect_image_parts(self) -> List[types.Part]:
    staged = getattr(self._tool_call_local, "inspect_image_inputs", None)
    self._tool_call_local.inspect_image_inputs = []
    if not isinstance(staged, list) or not staged:
        return []
    parts: List[types.Part] = []
    for item in staged[:3]:
        if not isinstance(item, dict):
            continue
        data = item.get("data")
        mime = str(item.get("mime_type") or "image/jpeg").strip().lower()
        if not isinstance(data, (bytes, bytearray)) or not data:
            continue
        if not mime.startswith("image/"):
            mime = "image/jpeg"
        try:
            parts.append(types.Part.from_bytes(data=bytes(data), mime_type=mime))
        except Exception:
            continue
    return parts


def initialize_client(self):
    try:
        with self._client_state_lock:
            self.client = genai.Client(api_key=self.api_keys[self.current_key_index])
            active_index = int(self.current_key_index)
        logger.info("Chat siap menggunakan API key #%d", active_index + 1)
    except Exception as e:
        logger.critical("Inisialisasi client chat gagal: %s", e)
        raise


def rotate_api_key(self) -> bool:
    with self._client_state_lock:
        total_keys = len(self.api_keys)
        current_index = int(self.current_key_index)
    if total_keys <= 0:
        return False
    start_index = (current_index + 1) % total_keys
    new_key_index = self.health_monitor.get_healthy_key(
        start_index, total_keys
    )
    if new_key_index is None:
        logger.warning("Tidak ada API key chat yang sehat untuk dipakai saat ini.")
        return False

    try:
        with self._client_state_lock:
            self.current_key_index = int(new_key_index)
            self.client = genai.Client(api_key=self.api_keys[self.current_key_index])
            active_index = int(self.current_key_index)
        logger.warning("Chat berpindah ke API key #%d", active_index + 1)
        return True
    except Exception as e:
        logger.error("Gagal mengganti API key chat: %s", e)
        return False


def get_model_penalty_remaining(self, model_name: str) -> float:
    with self._model_penalty_lock:
        until = self._model_penalty_until.get(model_name, 0.0)
    return max(0.0, until - time.time())


def set_model_penalty(self, model_name: str, seconds: float):
    if not model_name or seconds <= 0:
        return
    now = time.time()
    with self._model_penalty_lock:
        old_until = self._model_penalty_until.get(model_name, 0.0)
        new_until = max(old_until, now + seconds)
        self._model_penalty_until[model_name] = new_until
    logger.warning(
        "Model %s masuk masa tunggu selama %.1f detik",
        model_name,
        max(0.0, new_until - now),
    )


def get_model_high_demand_remaining(self, model_name: str) -> float:
    with self._model_penalty_lock:
        until = self._model_high_demand_until.get(model_name, 0.0)
    return max(0.0, until - time.time())


def set_model_high_demand_penalty(self, model_name: str, seconds: float):
    if not model_name or seconds <= 0:
        return
    now = time.time()
    with self._model_penalty_lock:
        old_until = self._model_high_demand_until.get(model_name, 0.0)
        new_until = max(old_until, now + seconds)
        self._model_high_demand_until[model_name] = new_until
    logger.warning(
        "Model %s masuk cooldown high-demand selama %.1f detik",
        model_name,
        max(0.0, new_until - now),
    )


def high_demand_backoff(attempt: int) -> float:
    base = max(0.2, HIGH_DEMAND_BASE_DELAY)
    cap = max(base, HIGH_DEMAND_MAX_DELAY)
    exp_delay = min(cap, base * (2 ** max(0, attempt)))
    jitter = random.uniform(0, min(1.0, base))
    return min(cap, exp_delay + jitter)


def call_gemini(self, model: str, contents: list, config=None):
    max_attempts = max(CHAT_MAX_RETRIES, len(self.api_keys) + 1)
    req_t0 = time.perf_counter()
    for attempt in range(max_attempts):
        request_key_index = int(self.current_key_index)
        remaining = self._get_model_penalty_remaining(model)
        if remaining > 0:
            sleep_for = min(remaining, HIGH_DEMAND_MAX_DELAY)
            logger.warning(
                "Menunggu %.1f detik sebelum mencoba lagi model %s",
                sleep_for,
                model,
            )
            time.sleep(sleep_for)
        try:
            request_key_index, request_client = self._get_client_snapshot()
            response = request_client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            in_tok, out_tok, total_tok = self._extract_response_usage_tokens(response)
            self._record_token_usage(
                model=model,
                mode="generate_content",
                input_tokens=in_tok,
                output_tokens=out_tok,
                total_tokens=total_tok,
                latency_ms=(time.perf_counter() - req_t0) * 1000.0,
            )
            self.health_monitor.mark_success(request_key_index)
            logger.info("[CHAT-PROVIDER] provider=gemini mode=generate_content model=%s", model)
            return response
        except Exception as e:
            error_msg = str(e).lower()
            classification = classify_api_error(e)
            reason_code = str(classification.get("reason_code") or "other")
            logger.warning(
                "Percobaan %d gagal (API key #%d, model %s): %s",
                attempt + 1,
                request_key_index + 1,
                model,
                e,
            )
            if "400" in error_msg or "invalid argument" in error_msg:
                logger.error(
                    "Request tidak valid untuk model %s (API key #%d), percobaan dihentikan.",
                    model,
                    request_key_index + 1,
                )
                raise

            if handle_api_error_retry(
                self,
                reason_code=reason_code,
                key_index=request_key_index,
                attempt=attempt,
                base_retry_delay=float(CHAT_BASE_RETRY_DELAY),
                rotate_sleep_seconds=min(1.0, float(CHAT_BASE_RETRY_DELAY)),
                quota_retry_delay=min(1.0, float(CHAT_BASE_RETRY_DELAY)),
                model_name=model,
                set_model_penalty_seconds=max(float(HIGH_DEMAND_MODEL_COOLDOWN), 60.0),
                high_demand_penalty_seconds=float(HIGH_DEMAND_MODEL_COOLDOWN),
                set_model_penalty_fn=self._set_model_penalty,
                set_model_high_demand_penalty_fn=self._set_model_high_demand_penalty,
                all_models_in_penalty_fn=lambda: all_chat_models_in_penalty(self),
                all_models_penalty_log="Semua model chat sedang cooldown karena quota. Coba ganti API key.",
                rotate_api_key_fn=self._rotate_api_key,
                high_demand_backoff_fn=self._high_demand_backoff,
            ):
                continue

            time.sleep(CHAT_BASE_RETRY_DELAY + attempt)
            if attempt == max_attempts - 1:
                raise


def generate_with_tools(
    self,
    system_prompt: str,
    history: List[types.Content],
    user_parts: List[types.Part],
) -> str:
    fallback_reason: Optional[str] = None

    def make_honest_fallback_system() -> str:
        reason_text = fallback_reason or "tool yang dibutuhkan sedang gagal diakses"
        called_tools = getattr(self._tool_call_local, "called_tools", None)
        called_list = sorted(list(called_tools)) if isinstance(called_tools, set) else []
        outbound_snapshot = _get_pending_outbound_snapshot(self)
        return _build_tool_failure_fallback_system(
            system_prompt=system_prompt,
            reason_text=reason_text,
            called_tool_names=called_list,
            outbound_snapshot=outbound_snapshot,
        )

    def make_config(tools_enabled: bool, system_instruction_text: str):
        return types.GenerateContentConfig(
            temperature=self.bot_config.temperature,
            top_p=self.bot_config.top_p,
            max_output_tokens=self.bot_config.max_output_tokens,
            system_instruction=system_instruction_text,
            tools=self._all_tools if tools_enabled else None,
            tool_config=(
                types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.AUTO
                    )
                )
                if tools_enabled
                else None
            ),
            automatic_function_calling=(
                types.AutomaticFunctionCallingConfig(
                    disable=True,
                )
                if tools_enabled
                else None
            ),
        )

    max_attempts = max(CHAT_MAX_RETRIES, len(self.api_keys) + 1)
    self._tool_call_local.web_cache = {}
    self._tool_call_local.called_tools = set()
    self._tool_call_local.inspect_image_inputs = []
    self._tool_call_local.web_image_inputs = []
    self._tool_call_local.last_terminal_cwd = ""
    gen_t0 = time.perf_counter()
    tools_enabled = True
    try:
        for attempt in range(max_attempts):
            att_t0 = time.perf_counter()
            request_key_index = int(self.current_key_index)
            active_model = select_chat_model_for_attempt(self)
            remaining = self._get_model_penalty_remaining(active_model)
            if remaining > 0:
                sleep_for = min(remaining, HIGH_DEMAND_MAX_DELAY)
                logger.warning(
                    "Menunggu %.1f detik sebelum generate dengan tools di model %s",
                    sleep_for,
                    active_model,
                )
                time.sleep(sleep_for)

            try:
                request_key_index, request_client = self._get_client_snapshot()
                attempt_system_prompt = system_prompt
                if (not tools_enabled) and fallback_reason:
                    attempt_system_prompt = make_honest_fallback_system()
                config = make_config(
                    tools_enabled=tools_enabled,
                    system_instruction_text=attempt_system_prompt,
                )
                chat = request_client.chats.create(
                    model=active_model,
                    history=history,
                    config=config,
                )
                tool_registry = build_tool_registry(self._tools or [])
                pending_parts = user_parts
                remote_calls = 0

                while True:
                    response = chat.send_message(pending_parts)
                    function_calls = _extract_function_calls(response)
                    planned_tool_names = _extract_function_call_names(response)
                    if function_calls and tools_enabled:
                        budgeted_calls = [
                            fc for fc in function_calls
                            if str(getattr(fc, "name", "") or "").strip() not in NON_BUDGETED_TOOL_NAMES
                        ]
                        remaining_budget = int(TOOL_MAX_REMOTE_CALLS) - remote_calls
                        if budgeted_calls and remaining_budget <= 0:
                            raise ValueError("Maximum remote tool calls reached before final response.")

                        called_tools = getattr(self._tool_call_local, "called_tools", None)
                        if isinstance(called_tools, set):
                            for fc in function_calls:
                                fn_name = str(getattr(fc, "name", "") or "").strip()
                                if fn_name:
                                    called_tools.add(fn_name)

                        tool_results = execute_tool_calls(
                            tool_registry,
                            function_calls,
                        )
                        budgeted_executed = sum(
                            1 for item in tool_results
                            if str(item.get("name") or "").strip() not in NON_BUDGETED_TOOL_NAMES
                        )
                        remote_calls += budgeted_executed
                        if not tool_results:
                            raise ValueError("Tool call detected but no valid executable function was found.")

                        pending_parts = [
                            types.Part.from_function_response(
                                name=item["name"],
                                response={"result": item["output"]},
                            )
                            for item in tool_results
                        ]
                        staged_inspect_parts = _consume_staged_inspect_image_parts(self)
                        if staged_inspect_parts:
                            pending_parts.extend(staged_inspect_parts)
                            logger.info(
                                "[TOOL-CALL] Added %d staged inspect image part(s) to model input.",
                                len(staged_inspect_parts),
                            )
                        staged_image_parts = _consume_staged_tool_image_parts(self)
                        if staged_image_parts:
                            pending_parts.extend(staged_image_parts)
                            logger.info("[TOOL-CALL] Added %d staged web image part(s) to model input.", len(staged_image_parts))
                        logger.info(
                            "[TOOL-CALL] Executed %d tool call(s), parallel=%s",
                            len(tool_results),
                            any(item.get("parallel") for item in tool_results),
                        )
                        continue

                    if response.text:
                        in_tok, out_tok, total_tok = self._extract_response_usage_tokens(response)
                        self._record_token_usage(
                            model=active_model,
                            mode="chat_with_tools" if tools_enabled else "chat_tools_disabled",
                            input_tokens=in_tok,
                            output_tokens=out_tok,
                            total_tokens=total_tok,
                            latency_ms=(time.perf_counter() - att_t0) * 1000.0,
                        )
                        self._update_visual_token_calibration(in_tok)
                        self.health_monitor.mark_success(request_key_index)
                        logger.info(
                            "[CHAT-PROVIDER] provider=gemini mode=%s model=%s",
                            "chat_with_tools" if tools_enabled else "chat_tools_disabled",
                            active_model,
                        )
                        logger.info(
                            "[LATENCY] generate_with_tools success in %.1fms",
                            (time.perf_counter() - gen_t0) * 1000.0,
                        )
                        return response.text.strip()
                    empty_summary = _summarize_empty_response(response)
                    logger.warning(
                        "Model belum memberi jawaban final (tools=%s, batch_tool=%d, pending=%d). Detail: %s",
                        tools_enabled,
                        remote_calls,
                        len(pending_parts or []),
                        empty_summary,
                    )
                    if _has_malformed_function_call_response(response):
                        called_tools = getattr(self._tool_call_local, "called_tools", None)
                        called_list = sorted(list(called_tools)) if isinstance(called_tools, set) else []
                        logger.error(
                            "[MALFORMED-FC] planned_tools=%s | called_tools=%s | remote_calls=%d | pending_parts=%d",
                            planned_tool_names,
                            called_list,
                            remote_calls,
                            len(pending_parts or []),
                        )
                        raise RuntimeError("malformed_function_call")
                    raise ValueError("Empty response received")
            except Exception as e:
                error_msg = str(e).lower()
                classification = classify_api_error(e)
                reason_code = str(classification.get("reason_code") or "other")
                logger.warning(
                    "Generate dengan tools gagal (percobaan %d/%d, API key #%d): %s",
                    attempt + 1,
                    max_attempts,
                    request_key_index + 1,
                    e,
                )

                if "malformed_function_call" in error_msg and tools_enabled:
                    logger.error(
                        "Model mengirim malformed function call. Mengulangi request tanpa Tools pada percobaan berikutnya."
                    )
                    fallback_reason = "model mengirim malformed function call pada request tools"
                    tools_enabled = False
                    time.sleep(min(0.5, float(CHAT_BASE_RETRY_DELAY)))
                    continue
                if "malformed_function_call" in error_msg and (not tools_enabled):
                    logger.error(
                        "Model tetap malformed meski tanpa Tools. Retry lokal tanpa cooldown/rotate."
                    )
                    fallback_reason = "model malformed meski tools nonaktif"
                    time.sleep(min(0.8, float(CHAT_BASE_RETRY_DELAY) + 0.2))
                    continue

                if ("400" in error_msg or "invalid argument" in error_msg or "not supported" in error_msg) and tools_enabled:
                    logger.error(
                        "Model menolak Tools. Mengulangi request yang sama tanpa Tools pada percobaan berikutnya."
                    )
                    fallback_reason = "model menolak pemanggilan tools pada request ini"
                    tools_enabled = False
                    time.sleep(min(0.5, float(CHAT_BASE_RETRY_DELAY)))
                    continue

                if _is_empty_response_error(error_msg) and (not tools_enabled):
                    logger.warning(
                        "Respons kosong berulang tanpa Tools. Retry lokal tanpa rotate model."
                    )
                    time.sleep(min(0.8, float(CHAT_BASE_RETRY_DELAY) + 0.2))
                    continue

                if handle_api_error_retry(
                    self,
                    reason_code=reason_code,
                    key_index=request_key_index,
                    attempt=attempt,
                    base_retry_delay=float(CHAT_BASE_RETRY_DELAY),
                    rotate_sleep_seconds=min(1.0, float(CHAT_BASE_RETRY_DELAY)),
                    quota_retry_delay=min(1.0, float(CHAT_BASE_RETRY_DELAY)),
                    model_name=active_model,
                    set_model_penalty_seconds=max(float(HIGH_DEMAND_MODEL_COOLDOWN), 60.0),
                    high_demand_penalty_seconds=float(HIGH_DEMAND_MODEL_COOLDOWN),
                    set_model_penalty_fn=self._set_model_penalty,
                    set_model_high_demand_penalty_fn=self._set_model_high_demand_penalty,
                    all_models_in_penalty_fn=lambda: all_chat_models_in_penalty(self),
                    all_models_penalty_log="Semua model chat sedang cooldown karena quota. Coba ganti API key.",
                    rotate_api_key_fn=self._rotate_api_key,
                    high_demand_backoff_fn=self._high_demand_backoff,
                ):
                    continue

                if attempt == max_attempts - 1:
                    raise
                time.sleep(CHAT_BASE_RETRY_DELAY + attempt)
            finally:
                logger.debug(
                    "[LATENCY] generate_with_tools took %.1fms",
                    (time.perf_counter() - att_t0) * 1000.0,
                )
    except Exception as e:
        logger.error("Generate dengan tools gagal setelah semua percobaan: %s", e)
    finally:
        self._tool_call_local.web_cache = None
        self._tool_call_local.called_tools = None
        self._tool_call_local.inspect_image_inputs = None
        self._tool_call_local.web_image_inputs = None
        self._tool_call_local.last_terminal_cwd = None

    logger.error("Semua percobaan tools gagal, lanjut ke mode tanpa tools")
    if fallback_reason is None:
        fallback_reason = "tool atau akses data live gagal setelah beberapa percobaan"
    return self._generate_no_tools(make_honest_fallback_system(), history, user_parts)


def generate_no_tools(
    self,
    system_prompt: str,
    history: List[types.Content],
    user_parts: List[types.Part],
) -> str:
    max_attempts = max(2, CHAT_MAX_RETRIES)
    gen_t0 = time.perf_counter()
    for attempt in range(max_attempts):
        request_key_index = int(self.current_key_index)
        active_model = select_chat_model_for_attempt(self)
        remaining = self._get_model_penalty_remaining(active_model)
        if remaining > 0:
            time.sleep(min(remaining, HIGH_DEMAND_MAX_DELAY))
        try:
            request_key_index, request_client = self._get_client_snapshot()
            config = types.GenerateContentConfig(
                temperature=self.bot_config.temperature,
                top_p=self.bot_config.top_p,
                max_output_tokens=self.bot_config.max_output_tokens,
                system_instruction=system_prompt,
            )
            chat = request_client.chats.create(
                model=active_model,
                history=history,
                config=config,
            )
            response = chat.send_message(user_parts)
            if response.text:
                in_tok, out_tok, total_tok = self._extract_response_usage_tokens(response)
                self._record_token_usage(
                    model=active_model,
                    mode="chat_plain",
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    total_tokens=total_tok,
                    latency_ms=(time.perf_counter() - gen_t0) * 1000.0,
                )
                self._update_visual_token_calibration(in_tok)
                self.health_monitor.mark_success(request_key_index)
                logger.info("[CHAT-PROVIDER] provider=gemini mode=chat_plain model=%s", active_model)
                return response.text.strip()
            empty_summary = _summarize_empty_response(response)
            logger.warning(
                "Model belum memberi jawaban final pada mode tanpa tools. Detail: %s",
                empty_summary,
            )
            if _has_malformed_function_call_response(response):
                logger.error(
                    "Model mengirim malformed function call pada mode tanpa tools. Retry lokal tanpa cooldown/rotate."
                )
                time.sleep(min(0.8, float(CHAT_BASE_RETRY_DELAY) + 0.2))
                continue
            logger.warning(
                "Respons kosong pada mode tanpa tools. Retry lokal tanpa rotate API atau cooldown model."
            )
            time.sleep(min(0.8, float(CHAT_BASE_RETRY_DELAY) + 0.2))
            continue
        except Exception as e:
            classification = classify_api_error(e)
            reason_code = str(classification.get("reason_code") or "other")

            if handle_api_error_retry(
                self,
                reason_code=reason_code,
                key_index=request_key_index,
                attempt=attempt,
                base_retry_delay=float(CHAT_BASE_RETRY_DELAY),
                rotate_sleep_seconds=min(1.0, float(CHAT_BASE_RETRY_DELAY)),
                quota_retry_delay=min(1.0, float(CHAT_BASE_RETRY_DELAY)),
                model_name=active_model,
                set_model_penalty_seconds=max(float(HIGH_DEMAND_MODEL_COOLDOWN), 60.0),
                high_demand_penalty_seconds=float(HIGH_DEMAND_MODEL_COOLDOWN),
                set_model_penalty_fn=self._set_model_penalty,
                set_model_high_demand_penalty_fn=self._set_model_high_demand_penalty,
                all_models_in_penalty_fn=lambda: all_chat_models_in_penalty(self),
                all_models_penalty_log="Semua model chat sedang cooldown karena quota. Coba ganti API key.",
                rotate_api_key_fn=self._rotate_api_key,
                high_demand_backoff_fn=self._high_demand_backoff,
            ):
                continue

            logger.error("Generate tanpa tools gagal: %s", e)
            time.sleep(CHAT_BASE_RETRY_DELAY + attempt)
    return ""
