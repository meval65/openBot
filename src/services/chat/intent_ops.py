import datetime
import logging
import os
import re
import time
import urllib.request
from urllib.error import URLError, HTTPError
from typing import List, Optional

from src.config import (
    MemoryType,
    TOOLS_ENABLE_AI_PC_INSPECT_IMAGES,
    TOOLS_ENABLE_AI_PC_SEND_FILES,
    TOOLS_ENABLE_AI_PERSONAL_COMPUTER,
    TOOLS_ENABLE_ANNOUNCE_ACTION,
    TOOLS_ENABLE_MEMORY,
    TOOLS_ENABLE_SCHEDULE,
    TOOLS_ENABLE_SEARCH_WEB,
)
from src.services.media.pipeline import ingest_web_image_to_cache, resolve_web_image
from src.services.chat.tool_policy import (
    INSPECT_IMAGE_MAX_BYTES,
    OUTBOUND_FILE_MAX_BYTES,
    OUTBOUND_FILE_MAX_ITEMS,
    OUTBOUND_MESSAGE_MAX_ITEMS,
    TERMINAL_OUTPUT_MAX_CHARS,
    TERMINAL_TIMEOUT_DEFAULT,
    TERMINAL_TIMEOUT_MAX,
    WEB_IMAGE_MAX_BYTES,
    WEB_IMAGE_MAX_COUNT,
)
from src.utils.time_utils import parse_local_dt

logger = logging.getLogger(__name__)

_LOCAL_DT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?$")
_UTC_SUFFIX_RE = re.compile(r"(?:Z|[+-]00:00)$", re.IGNORECASE)


def _sniff_image_mime(data: bytes, path_hint: str = "") -> str:
    blob = bytes(data or b"")
    if len(blob) >= 8 and blob.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(blob) >= 3 and blob[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(blob) >= 6 and (blob.startswith(b"GIF87a") or blob.startswith(b"GIF89a")):
        return "image/gif"
    if len(blob) >= 12 and blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return "image/webp"
    if len(blob) >= 2 and blob[:2] == b"BM":
        return "image/bmp"
    if len(blob) >= 4 and (blob[:4] in {b"II*\x00", b"MM\x00*"}):
        return "image/tiff"
    hint = str(path_hint or "").strip().lower()
    if hint.endswith(".png"):
        return "image/png"
    if hint.endswith(".jpg") or hint.endswith(".jpeg"):
        return "image/jpeg"
    if hint.endswith(".gif"):
        return "image/gif"
    if hint.endswith(".webp"):
        return "image/webp"
    if hint.endswith(".bmp"):
        return "image/bmp"
    if hint.endswith(".tif") or hint.endswith(".tiff"):
        return "image/tiff"
    return ""


def _download_web_image(url: str, timeout: float = 8.0):
    target = str(url or "").strip()
    if not target or not target.lower().startswith(("http://", "https://")):
        return None
    req = urllib.request.Request(
        target,
        headers={
            "User-Agent": "ViraBot/1.0",
            "Accept": "image/*,*/*;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = str(resp.headers.get("Content-Type", "")).split(";")[0].strip().lower()
            if content_type and not content_type.startswith("image/"):
                return None
            data = resp.read(WEB_IMAGE_MAX_BYTES + 1)
            if len(data) > WEB_IMAGE_MAX_BYTES:
                return None
            if not data:
                return None
            if not content_type:
                content_type = "image/jpeg"
            return {"data": data, "mime_type": content_type, "source_url": target}
    except (HTTPError, URLError, TimeoutError, ValueError):
        return None
    except Exception:
        return None


def _normalize_memory_type(raw_type: str) -> str:
    text = str(raw_type or "").strip().lower()
    allowed = {m.value for m in MemoryType}
    aliases = {
        "pref": MemoryType.PREFERENCE.value,
        "preferences": MemoryType.PREFERENCE.value,
        "facts": MemoryType.FACT.value,
        "decision": MemoryType.DECISION.value,
        "boundaries": MemoryType.BOUNDARY.value,
        "emotion": MemoryType.EMOTION.value,
        "mood": MemoryType.MOOD_STATE.value,
    }
    mapped = aliases.get(text, text)
    return mapped if mapped in allowed else MemoryType.GENERAL.value


def _parse_schedule_datetime(raw_value: str) -> Optional[datetime.datetime]:
    text = str(raw_value or "").strip()
    if not text:
        return None

    # Heuristic for tool-call robustness:
    # model often emits UTC suffix (Z/+00:00) even when user intent is local time.
    # Treat these as local wall-clock unless user explicitly sends timezone context.
    utc_stripped = _UTC_SUFFIX_RE.sub("", text).strip()
    if utc_stripped != text:
        candidate = utc_stripped.replace(" ", "T")
        if _LOCAL_DT_RE.match(candidate.replace("T", " ")):
            try:
                return datetime.datetime.fromisoformat(candidate)
            except ValueError:
                pass

    try:
        parsed = parse_local_dt(text)
        if parsed is not None:
            return parsed
    except Exception:
        pass

    if _LOCAL_DT_RE.match(text):
        normalized = text.replace(" ", "T")
        try:
            return datetime.datetime.fromisoformat(normalized)
        except ValueError:
            return None
    return None


def build_python_tools(self) -> list:
    def _bump_profile_score(amount: float, reason: str):
        try:
            current = float(self.session_manager.get_metadata("user_profile_update_score", 0.0) or 0.0)
        except Exception:
            current = 0.0
        self.session_manager.set_metadata(
            "user_profile_update_score",
            round(max(0.0, current + float(amount or 0.0)), 2),
            persist=True,
        )
        logger.info("[USER-PROFILE] score bump via tool | +%.2f | reason=%s", float(amount or 0.0), reason)

    def _resolve_outbound_file_item(raw_path: str):
        raw = str(raw_path or "").strip()
        if not raw:
            return None, "file_path kosong"
        default_cwd = str(getattr(self._tool_call_local, "last_terminal_cwd", "") or "").strip()
        resolved = self.terminal_service.resolve_file_for_telegram(
            raw,
            max_bytes=OUTBOUND_FILE_MAX_BYTES,
            default_cwd=default_cwd,
        )
        if not isinstance(resolved, dict) or not resolved.get("ok"):
            reason = resolved.get("error") if isinstance(resolved, dict) else "resolve gagal"
            return None, reason
        candidate = str(resolved.get("path") or "").strip()
        file_name = str(resolved.get("filename") or os.path.basename(candidate)).strip() or os.path.basename(candidate)
        size = int(resolved.get("size") or 0)
        item = {
            "path": candidate,
            "filename": file_name,
            "caption": "",
            "cleanup_after_send": bool(resolved.get("cleanup")),
            "size": size,
        }
        return item, ""

    def _resolve_inspectable_image(raw_path: str):
        raw = str(raw_path or "").strip()
        if not raw:
            return None, "file_path kosong"
        default_cwd = str(getattr(self._tool_call_local, "last_terminal_cwd", "") or "").strip()
        resolved = self.terminal_service.resolve_file_for_telegram(
            raw,
            max_bytes=INSPECT_IMAGE_MAX_BYTES,
            default_cwd=default_cwd,
        )
        if not isinstance(resolved, dict) or not resolved.get("ok"):
            reason = resolved.get("error") if isinstance(resolved, dict) else "resolve gagal"
            return None, reason

        local_copy = str(resolved.get("path") or "").strip()
        if not local_copy or not os.path.isfile(local_copy):
            return None, "file hasil copy tidak ditemukan"

        with open(local_copy, "rb") as f:
            image_bytes = f.read()
        if not image_bytes:
            return None, "file gambar kosong"
        mime_type = _sniff_image_mime(image_bytes, path_hint=local_copy)
        if not mime_type.startswith("image/"):
            return None, "file bukan gambar yang didukung berdasarkan isi file"

        item = {
            "data": image_bytes,
            "mime_type": mime_type,
            "source_url": "",
            "path": local_copy,
            "filename": os.path.basename(local_copy),
            "size": len(image_bytes),
        }
        return item, ""

    def search_web(
        query: str,
        topic: str = "general",
        search_level: int = 3,
        time_range: str = "none",
        include_image: bool = False,
    ) -> str:
        """
        Cari informasi live dari web.
        Pakai tool ini untuk berita, info terbaru, verifikasi fakta web, atau ketika user memang minta dicari di internet.
        Jangan pakai untuk pengetahuan umum yang sudah bisa dijawab tanpa web.
        topic: general | news | finance.
        search_level: 1..4, makin tinggi makin dalam tapi biasanya lebih berat.
        time_range: none | day | week | month | year.
        include_image: True hanya jika gambar hasil pencarian memang membantu jawaban.
        """
        called_tools = getattr(self._tool_call_local, "called_tools", None)
        if isinstance(called_tools, set):
            called_tools.add("search_web")
        norm_query = str(query or "").strip()
        norm_topic = str(topic or "general").strip().lower()
        if norm_topic not in {"general", "news", "finance"}:
            return "Gagal web search: topic harus salah satu dari general, news, atau finance."
        try:
            lvl = int(search_level)
        except Exception:
            lvl = 3
        lvl = max(1, min(4, lvl))
        # Lower level = faster, higher level = deeper.
        norm_depth = {
            1: "ultra-fast",
            2: "fast",
            3: "basic",
            4: "advanced",
        }[lvl]
        norm_time = str(time_range or "none").strip().lower()
        if norm_time not in {"none", "day", "week", "month", "year"}:
            norm_time = "none"
        include_img = bool(include_image)
        cache_key = (norm_query, norm_topic, lvl, norm_depth, norm_time, include_img)

        cache = getattr(self._tool_call_local, "web_cache", None)
        if (not include_img) and isinstance(cache, dict) and cache_key in cache:
            logger.info(
                f"[TOOL search_web] Cache hit | Query: {norm_query[:80]} | Topic: {norm_topic} | Level: {lvl} | Depth: {norm_depth} | Range: {norm_time} | Img: {include_img}"
            )
            return cache[cache_key]

        now_ts = time.time()
        with self._web_cache_lock:
            cached = self._web_search_cache.get(cache_key)
            if (not include_img) and cached and (now_ts - cached[0]) <= self._web_cache_ttl_sec:
                logger.info(
                    f"[TOOL search_web] Cache hit(global) | Query: {norm_query[:80]} | Topic: {norm_topic} | Level: {lvl} | Depth: {norm_depth} | Range: {norm_time} | Img: {include_img}"
                )
                if isinstance(cache, dict):
                    cache[cache_key] = cached[1]
                return cached[1]

        logger.info(
            f"[TOOL search_web] Query: {norm_query[:80]} | Topic: {norm_topic} | Level: {lvl} | Depth: {norm_depth} | Range: {norm_time} | Img: {include_img}"
        )
        result = self._tavily.search(
            norm_query,
            topic=norm_topic,
            max_results=5,
            search_depth=norm_depth,
            time_range=norm_time,
            include_image=include_img,
        )

        if isinstance(result, str) and result.startswith("[TOOL_ERROR search_web"):
            logger.warning(
                f"[TOOL search_web] Upstream returned error | Query: {norm_query[:80]} | Topic: {norm_topic} | Level: {lvl} | Depth: {norm_depth} | Range: {norm_time} | Img: {include_img} | Result: {result}"
            )
            return result

        payload = result if isinstance(result, dict) else {"text": str(result or ""), "images": []}
        text_result = str(payload.get("text") or "").strip()
        image_candidates = payload.get("images") if isinstance(payload.get("images"), list) else []

        downloaded_images = []
        if include_img and image_candidates:
            for img in image_candidates[:8]:
                if len(downloaded_images) >= WEB_IMAGE_MAX_COUNT:
                    continue
                if not isinstance(img, dict):
                    continue
                source_url = str(img.get("url") or "").strip()
                cached_item = resolve_web_image(self, source_url)
                if cached_item:
                    staged_ai = self.terminal_service.stage_local_file_to_workspace(
                        str(cached_item.get("path") or "").strip(),
                        media_kind="image",
                    )
                    downloaded_images.append(
                        {
                            "data": cached_item.get("data"),
                            "mime_type": str(cached_item.get("mime_type") or "image/jpeg"),
                            "source_url": source_url,
                            "path": str((staged_ai or {}).get("host_path") or cached_item.get("path") or "").strip(),
                            "ai_workspace_path": str((staged_ai or {}).get("container_path") or "").strip(),
                        }
                    )
                    continue
                downloaded = _download_web_image(source_url)
                if not downloaded:
                    continue
                persisted = ingest_web_image_to_cache(
                    self,
                    source_url=source_url,
                    raw_data=downloaded.get("data", b""),
                    mime_type=str(downloaded.get("mime_type") or "image/jpeg"),
                    image_description="",
                )
                if persisted:
                    staged_ai = self.terminal_service.stage_local_file_to_workspace(
                        str(persisted.get("path") or "").strip(),
                        media_kind="image",
                    )
                    downloaded_images.append(
                        {
                            "data": persisted.get("data"),
                            "mime_type": str(persisted.get("mime_type") or "image/jpeg"),
                            "source_url": source_url,
                            "path": str((staged_ai or {}).get("host_path") or persisted.get("path") or "").strip(),
                            "ai_workspace_path": str((staged_ai or {}).get("container_path") or "").strip(),
                        }
                    )

        if downloaded_images:
            # 1) Stage image bytes for multimodal follow-up in generation loop.
            staged = getattr(self._tool_call_local, "web_image_inputs", None)
            if not isinstance(staged, list):
                staged = []
            staged.extend(downloaded_images)
            self._tool_call_local.web_image_inputs = staged[:WEB_IMAGE_MAX_COUNT]

            # 2) Stage same images to be sent back to Telegram user after final response.
            try:
                self.stage_outbound_media(downloaded_images)
            except Exception:
                pass
        if not include_img:
            if isinstance(cache, dict):
                cache[cache_key] = text_result
            with self._web_cache_lock:
                self._web_search_cache[cache_key] = (now_ts, text_result)
        return text_result

    def create_schedule(datetime_iso: str, context: str, priority: int = 0) -> str:
        """
        Buat reminder baru.
        Gunakan jika user meminta diingatkan pada waktu tertentu dan kamu sudah punya waktu yang jelas.
        `context` harus singkat, jelas, dan menjelaskan isi reminder.
        """
        called_tools = getattr(self._tool_call_local, "called_tools", None)
        if isinstance(called_tools, set):
            called_tools.add("create_schedule")
        try:
            trigger_time = _parse_schedule_datetime(datetime_iso)
            clean_context = " ".join(str(context or "").split()).strip()
            if trigger_time is None:
                return "Gagal membuat reminder: format waktu tidak valid. Gunakan format seperti 2026-03-22T08:00:00."
            if not clean_context:
                return "Gagal membuat reminder: konteks reminder kosong."
            schedule_id = self.scheduler_service.add_schedule(
                trigger_time=trigger_time,
                context=clean_context,
                priority=int(max(0, min(10, int(priority or 0)))),
            )
            if not schedule_id:
                return "Reminder tidak dibuat. Mungkin duplikat, waktunya sudah lewat, atau input tidak valid."
            stored = self.scheduler_service.get_schedule_by_id(int(schedule_id))
            stored_context = clean_context
            pretty_time = trigger_time.strftime("%Y-%m-%d %H:%M")
            if isinstance(stored, dict):
                stored_context = " ".join(str(stored.get("context") or clean_context).split()).strip() or clean_context
                stored_dt = _parse_schedule_datetime(str(stored.get("scheduled_at") or ""))
                if stored_dt is not None:
                    pretty_time = stored_dt.strftime("%Y-%m-%d %H:%M")
            return f"Reminder berhasil dibuat. id={schedule_id}, waktu_local={pretty_time}, konteks={stored_context}"
        except Exception as e:
            logger.error(f"[TOOL create_schedule] Failed: {e}")
            return f"Gagal membuat reminder: {e}"

    def save_memory(summary: str, m_type: str = "general", priority: float = 0.7) -> str:
        """
        Simpan memory jangka panjang yang penting.
        Gunakan untuk preferensi, fakta personal, batasan, keputusan, atau hal relasional yang layak diingat nanti.
        Jangan simpan detail sementara atau hal yang sepele.
        """
        called_tools = getattr(self._tool_call_local, "called_tools", None)
        if isinstance(called_tools, set):
            called_tools.add("save_memory")
        try:
            clean_summary = " ".join(str(summary or "").split()).strip()
            if len(clean_summary) < 6:
                return "Gagal menyimpan memori: ringkasan terlalu pendek."

            mem_type = _normalize_memory_type(m_type)
            p = max(0.0, min(1.0, float(priority)))

            emb = self.analyzer.get_embedding(clean_summary)
            if emb is None:
                return "Gagal menyimpan memori: embedding tidak tersedia."
            emb_list = emb.tolist() if hasattr(emb, "tolist") else emb
            if isinstance(emb_list, (list, tuple)) and len(emb_list) == 0:
                return "Gagal menyimpan memori: embedding kosong."

            save_status = self.memory_manager.add_memory(
                summary=clean_summary,
                m_type=mem_type,
                priority=p,
                embedding=emb_list,
                embedding_namespace="memory",
            )
            if save_status == "created":
                _bump_profile_score(3.0, "save_memory")
                return f"Memori berhasil disimpan. type={mem_type}, priority={p:.2f}, summary={clean_summary}"
            if save_status == "duplicate":
                return f"Memori tidak ditambahkan karena sudah ada yang sangat mirip. summary={clean_summary}"
            if save_status == "invalid":
                return "Gagal menyimpan memori: ringkasan tidak valid."
            return f"Gagal menyimpan memori: status={save_status or 'unknown'}"
        except Exception as e:
            logger.error(f"[TOOL save_memory] Failed: {e}")
            return f"Gagal menyimpan memori: {e}"

    def list_memories(limit: int = 10, query: str = "", m_type: str = "") -> str:
        """
        Lihat atau cari memory yang sudah ada.
        Gunakan sebelum update/forget atau saat kamu perlu memastikan memory tertentu memang sudah tersimpan.
        """
        called_tools = getattr(self._tool_call_local, "called_tools", None)
        if isinstance(called_tools, set):
            called_tools.add("list_memories")
        try:
            max_items = max(1, min(20, int(limit or 10)))
            q = " ".join(str(query or "").split()).strip()
            mem_type_filter = str(m_type or "").strip()
            normalized_type = _normalize_memory_type(mem_type_filter) if mem_type_filter else ""
            lines = []

            if q:
                emb = self.analyzer.get_embedding(q)
                if emb is None:
                    return "Gagal membaca memori: embedding query tidak tersedia."
                emb_list = emb.tolist() if hasattr(emb, "tolist") else emb
                rows = self.memory_manager.get_relevant_memories(
                    query_embedding=emb_list,
                    memory_type=normalized_type or None,
                    max_results=max_items,
                    embedding_namespaces=["memory"],
                )
                if not rows:
                    return "Tidak ada memori relevan."
                for item in rows[:max_items]:
                    lines.append(
                        f"id={item.get('id')} | type={item.get('type')} | p={float(item.get('priority', 0.0)):.2f} | {item.get('summary')}"
                    )
                return "Daftar memori relevan berikut sudah menyertakan `id` untuk dipakai pada update/forget:\n" + "\n".join(lines)

            cursor = self.memory_manager.db.get_cursor()
            if normalized_type:
                cursor.execute(
                    """
                    SELECT id, summary, memory_type, priority
                    FROM memories
                    WHERE status='active' AND embedding_namespace='memory' AND memory_type=?
                    ORDER BY priority DESC, created_at DESC
                    LIMIT ?
                    """,
                    (normalized_type, max_items),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, summary, memory_type, priority
                    FROM memories
                    WHERE status='active' AND embedding_namespace='memory'
                    ORDER BY priority DESC, created_at DESC
                    LIMIT ?
                    """,
                    (max_items,),
                )
            rows = cursor.fetchall()
            if not rows:
                return "Tidak ada memori aktif."
            for idx, row in enumerate(rows, start=1):
                lines.append(
                    f"{idx}. id={row[0]} | type={row[2]} | p={float(row[3] or 0.0):.2f} | {row[1]}"
                )
            return "Daftar memori aktif berikut sudah menyertakan `id` untuk dipakai pada update/forget:\n" + "\n".join(lines)
        except Exception as e:
            logger.error(f"[TOOL list_memories] Failed: {e}")
            return f"Gagal membaca memori: {e}"

    def forget_memory(memory_id: str = "") -> str:
        """
        Arsipkan memory tertentu berdasarkan id.
        Gunakan hanya kalau kamu sudah tahu `memory_id` yang benar, biasanya setelah `list_memories`.
        """
        called_tools = getattr(self._tool_call_local, "called_tools", None)
        if isinstance(called_tools, set):
            called_tools.add("forget_memory")
        try:
            mem_id = str(memory_id or "").strip()
            if not mem_id:
                return "Gagal menghapus memori: `memory_id` wajib diisi. Gunakan `list_memories` dulu untuk melihat id yang tersedia."
            if not mem_id.isdigit() or int(mem_id) <= 0:
                return "Gagal menghapus memori: `memory_id` harus berupa angka positif."
            ok = self.memory_manager.archive_memory_by_id(mem_id)
            if ok:
                _bump_profile_score(3.0, "forget_memory")
                return f"Memori id={mem_id} berhasil diarsipkan."
            return f"Gagal mengarsipkan memori id={mem_id} (tidak ditemukan atau sudah nonaktif)."
        except Exception as e:
            logger.error(f"[TOOL forget_memory] Failed: {e}")
            return f"Gagal menghapus memori: {e}"

    def update_memory(memory_id: str, summary: str, priority: float = 0.7, m_type: str = "general") -> str:
        """
        Perbarui memory yang sudah ada.
        Gunakan untuk mengoreksi atau menyempurnakan memory lama, bukan untuk membuat memory baru.
        """
        called_tools = getattr(self._tool_call_local, "called_tools", None)
        if isinstance(called_tools, set):
            called_tools.add("update_memory")
        try:
            mem_id = str(memory_id or "").strip()
            clean_summary = " ".join(str(summary or "").split()).strip()
            if not mem_id:
                return "Gagal update memori: memory_id wajib diisi."
            if not mem_id.isdigit() or int(mem_id) <= 0:
                return "Gagal update memori: memory_id harus berupa angka positif."
            if len(clean_summary) < 3:
                return "Gagal update memori: summary terlalu pendek."

            p = max(0.0, min(1.0, float(priority)))
            normalized_type = _normalize_memory_type(m_type)
            emb = self.analyzer.get_embedding(clean_summary)
            if emb is None:
                return "Gagal update memori: embedding tidak tersedia."
            emb_list = emb.tolist() if hasattr(emb, "tolist") else emb
            if isinstance(emb_list, (list, tuple)) and len(emb_list) == 0:
                return "Gagal update memori: embedding kosong."
            ok = self.memory_manager.update_memory(
                memory_id=mem_id,
                new_summary=clean_summary,
                new_priority=p,
                new_m_type=normalized_type,
                new_embedding=emb_list,
            )
            if ok:
                _bump_profile_score(3.0, "update_memory")
                return f"Memori berhasil diupdate. id={mem_id}, type={normalized_type}, p={p:.2f}, summary={clean_summary}"
            return f"Gagal update memori id={mem_id} (tidak ditemukan atau sudah nonaktif)."
        except Exception as e:
            logger.error(f"[TOOL update_memory] Failed: {e}")
            return f"Gagal update memori: {e}"

    def list_schedules(limit: int = 10, priority: int = -1, datetime_iso: str = "") -> str:
        """
        Tampilkan reminder yang masih pending.
        Gunakan untuk mengecek jadwal, mencari id reminder, atau merangkum reminder user.
        """
        called_tools = getattr(self._tool_call_local, "called_tools", None)
        if isinstance(called_tools, set):
            called_tools.add("list_schedules")
        try:
            max_items = max(1, min(20, int(limit or 10)))
            min_priority = int(priority) if priority is not None else -1
            if min_priority < -1:
                min_priority = -1
            start_dt = _parse_schedule_datetime(datetime_iso) if str(datetime_iso or "").strip() else None
            if str(datetime_iso or "").strip() and start_dt is None:
                return "Gagal membaca daftar reminder: `datetime_iso` tidak valid."
            pending = self.scheduler_service.get_pending_schedules(
                lookahead_minutes=60 * 24 * 365,
                include_overdue=True,
            )
            filtered = []
            for item in pending:
                item_priority = int(item.get("priority") or 0)
                if min_priority >= 0 and item_priority < min_priority:
                    continue
                if start_dt is not None:
                    item_dt = _parse_schedule_datetime(str(item.get("scheduled_at") or ""))
                    if item_dt is None or item_dt < start_dt:
                        continue
                filtered.append(item)
            if not filtered:
                return "Tidak ada reminder pending."
            lines = []
            for item in filtered[:max_items]:
                lines.append(
                    f"id={item.get('id')} | waktu={item.get('scheduled_at')} | priority={item.get('priority')} | konteks={item.get('context')}"
                )
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[TOOL list_schedules] Failed: {e}")
            return f"Gagal membaca daftar reminder: {e}"

    def cancel_schedule(schedule_id: int = 0) -> str:
        """
        Batalkan reminder berdasarkan id.
        Gunakan hanya jika user memang ingin membatalkan dan kamu sudah punya `schedule_id` yang tepat.
        """
        called_tools = getattr(self._tool_call_local, "called_tools", None)
        if isinstance(called_tools, set):
            called_tools.add("cancel_schedule")
        try:
            sid = int(schedule_id or 0)
        except Exception:
            sid = 0
        try:
            if sid <= 0:
                return "Gagal membatalkan reminder: `schedule_id` wajib diisi. Gunakan `list_schedules` dulu untuk melihat id yang tersedia."
            success = self.scheduler_service.cancel_schedule(sid)
            return (
                f"Reminder id={sid} berhasil dibatalkan."
                if success
                else f"Reminder id={sid} tidak ditemukan atau sudah tidak pending."
            )
        except Exception as e:
            logger.error(f"[TOOL cancel_schedule] Failed: {e}")
            return f"Gagal membatalkan reminder: {e}"

    def ai_personal_computer(command: str, timeout_sec: int = TERMINAL_TIMEOUT_DEFAULT, cwd: str = "") -> str:
        """
        Jalankan command terminal di komputer pribadi AI.
        Gunakan untuk membaca file, menulis file, menjalankan script, mengatur workspace, atau tugas terminal lain yang benar-benar membantu kebutuhan user.
        Pilih command yang aman, rapi, dan tidak destruktif jika ada alternatif yang lebih aman.
        """
        called_tools = getattr(self._tool_call_local, "called_tools", None)
        if isinstance(called_tools, set):
            called_tools.add("ai_personal_computer")
        try:
            sandbox_ok, sandbox_reason = self.terminal_service.get_sandbox_status()
            if not sandbox_ok:
                return f"Tool terminal ditolak: {sandbox_reason}"
            cmd = str(command or "").strip()
            if not cmd:
                return "Gagal menjalankan command: command kosong."
            timeout = max(1, min(TERMINAL_TIMEOUT_MAX, int(timeout_sec or TERMINAL_TIMEOUT_DEFAULT)))
            result = self.terminal_service.execute(
                command=cmd,
                timeout_sec=timeout,
                cwd=str(cwd or "").strip(),
                source="ai",
                output_limit_chars=TERMINAL_OUTPUT_MAX_CHARS,
            )
            if not isinstance(result, dict):
                return "Gagal menjalankan command: hasil eksekusi tidak valid."
            self._tool_call_local.last_terminal_cwd = str(result.get("cwd") or "").strip()
            run_id = str(result.get("id") or "")
            exit_code = result.get("exit_code")
            timed_out = bool(result.get("timed_out"))
            output = str(result.get("output") or "").strip()
            err = str(result.get("error") or "").strip()
            if exit_code is None and err and not timed_out:
                return f"Gagal menjalankan command terminal: {err}"
            if timed_out:
                return (
                    f"Terminal timeout. id={run_id}, duration_ms={result.get('duration_ms')}, cwd={result.get('cwd')}\n"
                    + (output if output else (err or "Tidak ada output."))
                )
            if exit_code == 0:
                return (
                    f"Terminal success. id={run_id}, exit=0, duration_ms={result.get('duration_ms')}, cwd={result.get('cwd')}\n"
                    + (output if output else "(tanpa output)")
                )
            return (
                f"Terminal failed. id={run_id}, exit={exit_code}, duration_ms={result.get('duration_ms')}, cwd={result.get('cwd')}\n"
                + (output if output else (err or "Tidak ada output error."))
            )
        except Exception as e:
            logger.error(f"[TOOL ai_personal_computer] Failed: {e}")
            return f"Gagal menjalankan command terminal: {e}"

    def send_files_from_ai_personal_computer(file_paths: List[str]) -> str:
        """
        Siapkan beberapa file dari komputer pribadi AI untuk dikirim ke user Telegram.
        Gunakan setelah file benar-benar selesai dibuat atau memang ingin dibagikan ke user.
        """
        called_tools = getattr(self._tool_call_local, "called_tools", None)
        if isinstance(called_tools, set):
            called_tools.add("send_files_from_ai_personal_computer")
        try:
            sandbox_ok, sandbox_reason = self.terminal_service.get_sandbox_status()
            if not sandbox_ok:
                return f"Gagal kirim file: {sandbox_reason}"
            if not isinstance(file_paths, list) or not file_paths:
                return "Gagal kirim file: `file_paths` wajib berupa list dan tidak boleh kosong."

            staged_items = []
            failures = []
            for idx, raw_path in enumerate(file_paths, start=1):
                if idx > OUTBOUND_FILE_MAX_ITEMS:
                    failures.append(f"{idx}:dilewati karena batas maksimal {OUTBOUND_FILE_MAX_ITEMS} file per request")
                    continue
                item, reason = _resolve_outbound_file_item(raw_path)
                if item:
                    staged_items.append({k: v for k, v in item.items() if k != "size"})
                else:
                    failures.append(f"{idx}:{reason}")

            if not staged_items:
                detail = "; ".join(failures[:3]).strip()
                if detail:
                    detail = f" Detail: {detail}"
                return (
                    "Gagal kirim file: tidak ada file valid yang bisa di-stage."
                    + detail
                )

            staged_count = int(self.stage_outbound_files(staged_items[:OUTBOUND_FILE_MAX_ITEMS]) or 0)
            if staged_count < len(staged_items[:OUTBOUND_FILE_MAX_ITEMS]):
                for idx in range(staged_count + 1, len(staged_items[:OUTBOUND_FILE_MAX_ITEMS]) + 1):
                    failures.append(f"queue:{idx}:dilewati karena antrean outbound file sudah penuh")

            if staged_count <= 0:
                detail = "; ".join(failures[:3]).strip()
                if detail:
                    detail = f" Detail: {detail}"
                return "Gagal kirim file: antrean outbound file penuh atau semua file ditolak." + detail

            staged_effective = staged_items[:staged_count]
            names = ", ".join(str(item.get("filename") or "").strip() for item in staged_effective if item.get("filename"))
            response = f"{len(staged_effective)} file siap dikirim ke user"
            if names:
                response += f": {names}"
            if failures:
                response += f". Sebagian gagal di-resolve ({'; '.join(failures[:3])})"
            return response + "."
        except Exception as e:
            logger.error(f"[TOOL send_files_from_ai_personal_computer] Failed: {e}")
            return f"Gagal kirim file: {e}"

    def inspect_images_from_ai_personal_computer(file_paths: List[str]) -> str:
        """
        Ambil beberapa gambar dari komputer pribadi AI lalu masukkan lagi sebagai input visual.
        Gunakan jika kamu perlu melihat hasil gambar yang sudah ada di komputermu sendiri agar bisa menganalisis atau membandingkannya pada loop berikutnya.
        """
        called_tools = getattr(self._tool_call_local, "called_tools", None)
        if isinstance(called_tools, set):
            called_tools.add("inspect_images_from_ai_personal_computer")
        local_copies = []
        try:
            sandbox_ok, sandbox_reason = self.terminal_service.get_sandbox_status()
            if not sandbox_ok:
                return f"Gagal inspeksi gambar: {sandbox_reason}"
            if not isinstance(file_paths, list) or not file_paths:
                return "Gagal inspeksi gambar: `file_paths` wajib berupa list dan tidak boleh kosong."

            staged = getattr(self._tool_call_local, "inspect_image_inputs", None)
            if not isinstance(staged, list):
                staged = []
            failures = []
            loaded = []
            remaining_slots = max(0, WEB_IMAGE_MAX_COUNT - len(staged))
            for idx, raw_path in enumerate(file_paths[:OUTBOUND_FILE_MAX_ITEMS], start=1):
                if remaining_slots <= 0:
                    failures.append(f"{idx}:dilewati karena batas maksimal {WEB_IMAGE_MAX_COUNT} gambar per inspeksi")
                    break
                item, reason = _resolve_inspectable_image(raw_path)
                if not item:
                    failures.append(f"{idx}:{reason}")
                    continue
                local_copies.append(str(item.get("path") or "").strip())
                staged.append(
                    {
                        "data": item.get("data"),
                        "mime_type": item.get("mime_type"),
                        "source_url": "",
                        "path": item.get("path"),
                        "ai_workspace_path": "",
                    }
                )
                loaded.append(item)
                remaining_slots -= 1

            if not loaded:
                detail = "; ".join(failures[:3]).strip()
                if detail:
                    detail = f" Detail: {detail}"
                return "Gagal inspeksi gambar: tidak ada gambar valid yang bisa di-stage." + detail

            self._tool_call_local.inspect_image_inputs = staged[:WEB_IMAGE_MAX_COUNT]
            names = ", ".join(str(item.get("filename") or "").strip() for item in loaded if item.get("filename"))
            response = f"{len(loaded)} gambar siap diinspeksi model"
            if names:
                response += f": {names}"
            if failures:
                response += f". Sebagian gagal di-resolve ({'; '.join(failures[:3])})"
            response += ". Lanjutkan dengan analisis visual berdasarkan gambar-gambar ini."
            return response
        except Exception as e:
            logger.error(f"[TOOL inspect_images_from_ai_personal_computer] Failed: {e}")
            return f"Gagal inspeksi gambar: {e}"
        finally:
            for local_copy in local_copies:
                try:
                    if local_copy and os.path.isfile(local_copy):
                        os.remove(local_copy)
                except Exception:
                    pass

    def announce_action(message: str) -> str:
        """
        Kirim status singkat terpisah ke user sebelum aksi yang terasa lama.
        Gunakan untuk memberi tahu progres secara natural, bukan untuk setiap langkah kecil.
        """
        called_tools = getattr(self._tool_call_local, "called_tools", None)
        if isinstance(called_tools, set):
            called_tools.add("announce_action")
        clean = " ".join(str(message or "").split()).strip()
        if not clean:
            return "Gagal mengirim status: message kosong."
        try:
            self.stage_outbound_messages([clean][:OUTBOUND_MESSAGE_MAX_ITEMS])
            return "Status proses sudah dikirim ke user."
        except Exception as e:
            logger.error(f"[TOOL announce_action] Failed: {e}")
            return f"Gagal mengirim status proses: {e}"

    tools = []
    if TOOLS_ENABLE_SEARCH_WEB:
        tools.append(search_web)
    if TOOLS_ENABLE_SCHEDULE:
        tools.extend([create_schedule, list_schedules, cancel_schedule])
    if TOOLS_ENABLE_ANNOUNCE_ACTION:
        tools.append(announce_action)
    if TOOLS_ENABLE_AI_PERSONAL_COMPUTER:
        tools.append(ai_personal_computer)
    if TOOLS_ENABLE_AI_PC_INSPECT_IMAGES:
        tools.append(inspect_images_from_ai_personal_computer)
    if TOOLS_ENABLE_AI_PC_SEND_FILES:
        tools.append(send_files_from_ai_personal_computer)
    if TOOLS_ENABLE_MEMORY:
        tools.extend([save_memory, list_memories, forget_memory, update_memory])

    logger.info(
        "[TOOLS] Active=%s",
        ", ".join(getattr(fn, "__name__", "") for fn in tools) or "(none)",
    )
    return tools
