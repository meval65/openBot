import asyncio
import io
import logging
import os
import psutil
import threading
import time
import tracemalloc
from datetime import datetime

from telegram import InputFile, InputMediaPhoto
from telegram.constants import ParseMode
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from src.config import ADMIN_TELEGRAM_ID, PROACTIVE_MIN_SEND_GAP_SECONDS, TEMP_DIR
from src.services.analysis import ProactiveEngine
from src.services.chat.context import get_local_tz, now_local

logger = logging.getLogger(__name__)

# Singleton engine instance — persists across job ticks
_proactive_engine = ProactiveEngine()
_PROCESS_STARTED_AT = time.time()
_SCHEDULE_CHECK_LOCK = threading.Lock()
TELEGRAM_CAPTION_LIMIT = 1024


def _parse_local_datetime(value):
    if not value:
        return None
    try:
        if isinstance(value, str):
            dt = datetime.fromisoformat(value)
        else:
            dt = value
        if dt.tzinfo is None:
            dt = get_local_tz().localize(dt)
        return dt.astimezone(get_local_tz())
    except Exception:
        return None


def _in_proactive_cooldown(chat_handler) -> bool:
    last_sent = chat_handler.session_manager.get_metadata("last_proactive_sent_ts")
    last_sent_dt = _parse_local_datetime(last_sent)
    if not last_sent_dt:
        return False
    delta_seconds = (now_local() - last_sent_dt).total_seconds()
    if delta_seconds < PROACTIVE_MIN_SEND_GAP_SECONDS:
        logger.info(
            "[PROACTIVE] Cooldown active %.0fs remaining.",
            max(0.0, PROACTIVE_MIN_SEND_GAP_SECONDS - delta_seconds),
        )
        return True
    return False


async def _flush_proactive_outbound_messages(bot, chat_id: int, chat_handler) -> int:
    try:
        outbound_messages = chat_handler.pop_pending_outbound_messages()
    except Exception:
        outbound_messages = []

    sent_count = 0
    for text in outbound_messages:
        clean = " ".join(str(text or "").split()).strip()
        if not clean:
            continue
        await _safe_proactive_send_text(bot, chat_id, clean)
        sent_count += 1
    return sent_count


async def _safe_proactive_send_text(bot, chat_id: int, text: str):
    clean = " ".join(str(text or "").split()).strip()
    if not clean:
        return
    try:
        await bot.send_message(chat_id=chat_id, text=clean, parse_mode=ParseMode.HTML)
    except TelegramError:
        await bot.send_message(chat_id=chat_id, text=clean)


async def _send_proactive_outbound_media_with_caption(bot, chat_id: int, media_items: list, text: str):
    if not chat_id or not media_items:
        if text:
            await _safe_proactive_send_text(bot, chat_id, text)
        return False, []

    safe_text = str(text or "").strip()
    caption = safe_text[:TELEGRAM_CAPTION_LIMIT]
    remain = safe_text[TELEGRAM_CAPTION_LIMIT:].strip() if safe_text else ""

    prepared = []
    image_paths = []
    for idx, item in enumerate(media_items[:5]):
        if not isinstance(item, dict):
            continue
        data = item.get("data")
        if not isinstance(data, (bytes, bytearray)) or not data:
            continue
        mime = str(item.get("mime_type") or "image/jpeg").strip().lower()
        ext = ".png" if "png" in mime else ".jpg"
        bio = io.BytesIO(bytes(data))
        bio.name = f"proactive_{idx + 1}{ext}"
        prepared.append(bio)
        path = str(item.get("path") or "").strip()
        if path:
            image_paths.append(path)

    if not prepared:
        return False, []

    if len(prepared) == 1:
        try:
            await bot.send_photo(
                chat_id=chat_id,
                photo=prepared[0],
                caption=caption or None,
                parse_mode=ParseMode.HTML if caption else None,
            )
        except TelegramError:
            await bot.send_photo(
                chat_id=chat_id,
                photo=prepared[0],
                caption=caption or None,
            )
    else:
        media = [
            InputMediaPhoto(
                media=bio,
                caption=caption if idx == 0 and caption else None,
                parse_mode=ParseMode.HTML if idx == 0 and caption else None,
            )
            for idx, bio in enumerate(prepared)
        ]
        try:
            await bot.send_media_group(chat_id=chat_id, media=media)
        except TelegramError:
            plain_media = [
                InputMediaPhoto(media=bio, caption=caption if idx == 0 and caption else None)
                for idx, bio in enumerate(prepared)
            ]
            await bot.send_media_group(chat_id=chat_id, media=plain_media)

    if remain:
        await _safe_proactive_send_text(bot, chat_id, remain)
    return True, image_paths


async def _send_proactive_outbound_files_with_caption(bot, chat_id: int, file_items: list, text: str) -> bool:
    if not chat_id or not file_items:
        return False

    safe_text = str(text or "").strip()
    caption = safe_text[:TELEGRAM_CAPTION_LIMIT]
    remain = safe_text[TELEGRAM_CAPTION_LIMIT:].strip() if safe_text else ""

    prepared = []
    for item in file_items[:5]:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        if not path or not os.path.isfile(path):
            continue
        prepared.append(
            {
                "path": path,
                "filename": str(item.get("filename") or "").strip() or os.path.basename(path),
                "caption": " ".join(str(item.get("caption") or "").split()).strip(),
                "cleanup_after_send": bool(item.get("cleanup_after_send")),
            }
        )

    if not prepared:
        return False

    sent_count = 0
    for idx, item in enumerate(prepared):
        try:
            with open(item["path"], "rb") as fp:
                doc = InputFile(fp, filename=item["filename"])
                this_caption = (caption or item["caption"]) if idx == 0 else None
                try:
                    await bot.send_document(
                        chat_id=chat_id,
                        document=doc,
                        caption=this_caption,
                        parse_mode=ParseMode.HTML if this_caption else None,
                    )
                    sent_count += 1
                except TelegramError:
                    fp.seek(0)
                    doc_plain = InputFile(fp, filename=item["filename"])
                    await bot.send_document(
                        chat_id=chat_id,
                        document=doc_plain,
                        caption=this_caption,
                    )
                    sent_count += 1
        except Exception as e:
            logger.error(f"Failed sending proactive outbound file '{item.get('path')}': {e}")
        finally:
            if bool(item.get("cleanup_after_send")):
                try:
                    if os.path.isfile(item["path"]):
                        os.remove(item["path"])
                except Exception:
                    pass

    if sent_count <= 0:
        return False

    if remain:
        await _safe_proactive_send_text(bot, chat_id, remain)
    return True


async def _deliver_proactive_text_and_outbound(bot, chat_id: int, chat_handler, final_text: str) -> bool:
    await _flush_proactive_outbound_messages(bot, chat_id, chat_handler)

    try:
        outbound_media = chat_handler.pop_pending_outbound_media()
    except Exception:
        outbound_media = []
    try:
        outbound_files = chat_handler.pop_pending_outbound_files()
    except Exception:
        outbound_files = []

    delivered_any = False
    remaining_text = str(final_text or "").strip()
    if outbound_media:
        sent_ok, sent_paths = await _send_proactive_outbound_media_with_caption(
            bot,
            chat_id,
            outbound_media,
            remaining_text,
        )
        if sent_ok:
            delivered_any = True
            remaining_text = ""
            try:
                chat_handler.session_manager.attach_latest_model_image_paths(sent_paths)
            except Exception as meta_err:
                logger.warning(f"Failed to persist proactive image paths: {meta_err}")
    if outbound_files:
        sent_ok = await _send_proactive_outbound_files_with_caption(
            bot,
            chat_id,
            outbound_files,
            remaining_text,
        )
        if sent_ok:
            delivered_any = True
            remaining_text = ""
    if (not delivered_any) and remaining_text:
        await _safe_proactive_send_text(bot, chat_id, remaining_text)
        delivered_any = True
    return delivered_any


async def background_maintenance(context: ContextTypes.DEFAULT_TYPE):
    services = context.application.bot_data
    scheduler = services['scheduler']
    mem_mgr = services['mem_mgr']
    try:
        logger.info("Starting daily system maintenance...")
        db = services['db']
        deleted_count = await asyncio.to_thread(scheduler.cleanup_old_schedules, days_old=30)
        if deleted_count > 0:
            logger.info(f"Cleaned {deleted_count} old schedules.")
        await asyncio.to_thread(mem_mgr.optimize_memories)
        await asyncio.to_thread(db.maintenance)
        await asyncio.to_thread(_cleanup_orphan_media_groups, db)
        await asyncio.to_thread(_cleanup_stale_temp_files)
        logger.info("Daily maintenance completed.")
    except Exception as e:
        logger.error(f"Maintenance task failed: {e}")


def _cleanup_stale_temp_files(max_age_seconds: int = 3600):
    """Remove temp files older than max_age_seconds (default 1 hour)."""
    try:
        if not os.path.isdir(TEMP_DIR):
            return
        now = time.time()
        removed = 0
        for root, dirs, files in os.walk(TEMP_DIR):
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    if (now - os.path.getmtime(fpath)) > max_age_seconds:
                        os.remove(fpath)
                        removed += 1
                except OSError:
                    pass
        if removed:
            logger.info(f"[MAINTENANCE] Cleaned {removed} stale temp file(s) from {TEMP_DIR}.")
    except Exception as e:
        logger.warning(f"[MAINTENANCE] Temp cleanup failed: {e}")


def _cleanup_orphan_media_groups(db):
    try:
        cursor = db.get_cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_groups'")
        has_memory_groups = cursor.fetchone() is not None
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'")
        has_memories = cursor.fetchone() is not None
        if not has_memory_groups or not has_memories:
            logger.debug(
                "[MAINTENANCE] Skip orphan media cleanup (memory_groups=%s, memories=%s)",
                has_memory_groups,
                has_memories,
            )
            return
        cursor.execute("""
            SELECT mg.id, mg.file_path
            FROM memory_groups mg
            WHERE NOT EXISTS (
                SELECT 1 FROM memories m
                WHERE m.group_id = mg.id AND m.status = 'active'
            )
        """)
        orphans = cursor.fetchall()
        if not orphans:
            return

        for group_id, file_path in orphans:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"[MAINTENANCE] Deleted orphan image file: {file_path}")
                except OSError:
                    pass
            cursor.execute("DELETE FROM memory_groups WHERE id = ?", (group_id,))

        db.commit()
        logger.info(f"[MAINTENANCE] Cleaned {len(orphans)} orphan media group(s).")
    except Exception as e:
        logger.error(f"[MAINTENANCE] Orphan media cleanup failed: {e}")


async def background_schedule_checker(context: ContextTypes.DEFAULT_TYPE):
    if not _SCHEDULE_CHECK_LOCK.acquire(blocking=False):
        logger.info("[SCHEDULE] Previous checker run still active, skipping this tick.")
        return
    services = context.application.bot_data
    scheduler = services['scheduler']
    chat_handler = services['chat_handler']
    mem_mgr = services['mem_mgr']
    claim_note = ""
    claimed_ids = []
    try:
        # Claim due schedules first to avoid double-trigger with interaction flow.
        claim = await asyncio.to_thread(
            scheduler.claim_pending_schedules,
            0,
            True,
            50,
            "background",
        )
        due_items = claim.get("items", []) if isinstance(claim, dict) else []
        claim_note = str((claim or {}).get("claim_note") or "")
        claimed_ids = [int(item.get("id")) for item in due_items if item.get("id") is not None]
        if not due_items:
            return
        logger.info("[SCHEDULE] Found %d due item(s).", len(due_items))
            
        # Periksa apakah user sedang berinteraksi aktif dalam 2 menit terakhir
        last_interaction = chat_handler.session_manager.get_metadata("last_user_interaction")
        if last_interaction:
            try:
                last_dt = _parse_local_datetime(last_interaction)
                if not last_dt:
                    raise ValueError("invalid last_interaction")
                delta_seconds = (now_local() - last_dt).total_seconds()
                if delta_seconds < 30:
                    logger.info("User is currently active (last interaction < 30s ago). Deferring proactive trigger.")
                    return
            except Exception as e:
                logger.warning(f"Error checking last interaction time: {e}")

        unique_contexts = list(dict.fromkeys(item.get('context', '').strip() for item in due_items if item.get('context')))
        ctx_combined = " & ".join(unique_contexts)
        schedule_ids = [item['id'] for item in due_items]
        if not ctx_combined:
            logger.warning("[SCHEDULE] Due items have empty context. Keeping pending for retry.")
            if claim_note:
                await asyncio.to_thread(scheduler.release_claim, claim_note, claimed_ids)
            return

        allowed, reason = await asyncio.to_thread(
            _proactive_engine.should_allow_trigger,
            chat_handler,
            mem_mgr,
            scheduler,
            ctx_combined,
            "schedule",
        )
        if not allowed:
            logger.info("[SCHEDULE] Policy blocked trigger: %s", reason)
            if claim_note:
                await asyncio.to_thread(scheduler.release_claim, claim_note, claimed_ids)
            return
        
        try:
            ai_resp = await asyncio.to_thread(chat_handler.trigger_proactive_message, ctx_combined)
            if ai_resp:
                logger.info("[SCHEDULE] Sending proactive schedule message for IDs=%s", schedule_ids)
                if not str(ADMIN_TELEGRAM_ID or "").strip().isdigit():
                    logger.error("[SCHEDULE] ADMIN_TELEGRAM_ID invalid/empty; keeping pending for retry.")
                    if claim_note:
                        await asyncio.to_thread(scheduler.release_claim, claim_note, claimed_ids)
                    return
                chat_id = int(ADMIN_TELEGRAM_ID)
                delivered = await _deliver_proactive_text_and_outbound(
                    context.bot,
                    chat_id,
                    chat_handler,
                    ai_resp,
                )
                if not delivered:
                    logger.warning("[SCHEDULE] Final proactive delivery produced no outbound payload.")
                    if claim_note:
                        await asyncio.to_thread(scheduler.release_claim, claim_note, claimed_ids)
                    return
                await asyncio.to_thread(
                    chat_handler.finalize_proactive_delivery,
                    ctx_combined,
                    ai_resp,
                )
                if claim_note:
                    await asyncio.to_thread(
                        scheduler.complete_claimed_as_executed,
                        schedule_ids,
                        claim_note,
                        "Sent",
                    )
            else:
                reason = str(getattr(chat_handler, "_last_proactive_failure_reason", "") or "unknown")
                logger.warning(
                    "[SCHEDULE] Empty AI response for combined schedules. Keeping pending for retry: %s | reason=%s",
                    schedule_ids,
                    reason,
                )
                if claim_note:
                    await asyncio.to_thread(scheduler.release_claim, claim_note, claimed_ids)
        except Exception as e:
            logger.error(f"Schedule delivery failed; keeping pending for retry. IDs={schedule_ids} err={e}")
            if claim_note:
                await asyncio.to_thread(scheduler.release_claim, claim_note, claimed_ids)
    except Exception as e:
        logger.error(f"Scheduler checker loop failed: {e}")
        if claim_note:
            await asyncio.to_thread(scheduler.release_claim, claim_note, claimed_ids)
    finally:
        _SCHEDULE_CHECK_LOCK.release()


async def background_proactive_engine(context: ContextTypes.DEFAULT_TYPE):
    """Checks every 5 minutes whether the proactive engine's random timer has fired.
    When it fires, calls OpenRouter to decide if AI should initiate a conversation."""
    if not _proactive_engine.try_acquire_tick():
        return

    services = context.application.bot_data
    chat_handler = services['chat_handler']
    mem_mgr = services['mem_mgr']
    scheduler = services['scheduler']

    try:
        if _in_proactive_cooldown(chat_handler):
            return

        proactive_context = await asyncio.to_thread(
            _proactive_engine.run_tick,
            chat_handler,
            mem_mgr,
            scheduler,
        )

        if proactive_context:
            logger.info(f"[PROACTIVE-ENGINE] Firing spontaneous message. Context: {proactive_context[:100]}")
            ai_resp = await asyncio.to_thread(
                chat_handler.trigger_proactive_message,
                proactive_context,
            )
            if ai_resp:
                if not str(ADMIN_TELEGRAM_ID or "").strip().isdigit():
                    logger.error("[PROACTIVE-ENGINE] ADMIN_TELEGRAM_ID invalid/empty; skipping send.")
                    return
                chat_id = int(ADMIN_TELEGRAM_ID)
                delivered = await _deliver_proactive_text_and_outbound(
                    context.bot,
                    chat_id,
                    chat_handler,
                    ai_resp,
                )
                if not delivered:
                    logger.warning("[PROACTIVE-ENGINE] Final proactive delivery produced no outbound payload.")
                    return
                await asyncio.to_thread(
                    chat_handler.finalize_proactive_delivery,
                    proactive_context,
                    ai_resp,
                )
                logger.info("[PROACTIVE-ENGINE] Spontaneous message sent successfully.")
            else:
                logger.warning("[PROACTIVE-ENGINE] trigger_proactive_message returned empty.")
    except Exception as e:
        logger.error(f"[PROACTIVE-ENGINE] Background job failed: {e}")
    finally:
        _proactive_engine.finish_tick()


def _safe_process_metrics() -> dict:
    mem_mb = None
    peak_mb = None
    try:
        p = psutil.Process(os.getpid())
        mem_mb = p.memory_info().rss / (1024 * 1024)
    except Exception:
        pass

    if tracemalloc.is_tracing():
        try:
            current, peak = tracemalloc.get_traced_memory()
            if mem_mb is None:
                mem_mb = current / (1024 * 1024)
            peak_mb = peak / (1024 * 1024)
        except Exception:
            pass

    return {
        "pid": os.getpid(),
        "uptime_sec": max(0.0, time.time() - _PROCESS_STARTED_AT),
        "threads": threading.active_count(),
        "mem_mb": mem_mb if mem_mb is not None else -1.0,
        "mem_peak_mb": peak_mb if peak_mb is not None else -1.0,
    }


async def background_performance_logger(context: ContextTypes.DEFAULT_TYPE):
    services = context.application.bot_data
    chat_handler = services.get("chat_handler")
    scheduler = services.get("scheduler")
    mem_mgr = services.get("mem_mgr")

    if not chat_handler:
        return

    try:
        proc = _safe_process_metrics()
        perf = await asyncio.to_thread(chat_handler.get_performance_snapshot)

        pending_count = -1
        memory_count = -1
        try:
            pending = (
                await asyncio.to_thread(scheduler.get_pending_schedules, 60 * 24, True, 500)
                if scheduler else []
            )
            pending_count = len(pending or [])
        except Exception:
            pass

        try:
            if mem_mgr:
                stats = await asyncio.to_thread(mem_mgr.get_memory_stats)
                memory_count = int((stats or {}).get("active", 0))
        except Exception:
            pass

        logger.info(
            "[PERF] pid=%s uptime=%.0fs threads=%d mem=%.1fMB peak=%.1fMB "
            "req_total=%d success_rate=%.2f avg=%.1fms p50=%.1fms p95=%.1fms "
            "pending_sched=%d active_mem=%d last_err=%s",
            proc["pid"],
            proc["uptime_sec"],
            proc["threads"],
            proc["mem_mb"],
            proc["mem_peak_mb"],
            perf["requests_total"],
            perf["success_rate"],
            perf["latency_ms_avg"],
            perf["latency_ms_p50"],
            perf["latency_ms_p95"],
            pending_count,
            memory_count,
            perf["last_error"][:120] if perf.get("last_error") else "-",
        )
    except Exception as e:
        logger.error(f"[PERF] Performance logger failed: {e}")
