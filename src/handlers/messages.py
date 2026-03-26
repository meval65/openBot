import os
import asyncio
import uuid
import logging
import contextlib
from typing import Optional

from PIL import Image
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from src.utils import send_chunked_response, USER_LOCK
from src.handlers.media_group_cache import register_media_group_message
from src.handlers.outbound_delivery import (
    flush_outbound_messages,
    safe_send_text,
    send_outbound_files_with_caption,
    send_outbound_media_with_caption,
)
from src.handlers.sticker_command import cmd_sticker

from src.config import (
    MAX_TG_LEN,
    TEMP_DIR,
    IMAGE_STORE_DIR,
    VIDEO_STORE_DIR,
    VIDEO_MAX_DURATION_SECONDS,
    BOT_DISPLAY_NAME,
)

logger = logging.getLogger(__name__)


async def _get_user_profile_context(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[str]:
    user = update.effective_user
    if not user:
        return None

    cache = context.application.bot_data.setdefault("user_profile_cache", {})
    if not isinstance(cache, dict):
        cache = {}
        context.application.bot_data["user_profile_cache"] = cache

    key = str(user.id)
    now_ts = asyncio.get_running_loop().time()
    cached = cache.get(key)
    if isinstance(cached, dict) and (now_ts - float(cached.get("ts", 0.0))) <= 600:
        return cached.get("ctx")

    name = " ".join(
        p for p in [getattr(user, "first_name", None), getattr(user, "last_name", None)] if p
    ).strip()
    if not name:
        name = getattr(user, "username", "") or "User"

    bio = ""
    try:
        chat = await context.bot.get_chat(user.id)
        bio = str(getattr(chat, "bio", "") or "").strip()
    except Exception as e:
        logger.debug(f"User bio fetch skipped: {e}")

    parts = [f"name={name}"]
    if bio:
        if len(bio) > 280:
            bio = bio[:277].rstrip() + "..."
        parts.append(f"bio={bio}")
    profile_ctx = " | ".join(parts)

    cache[key] = {"ts": now_ts, "ctx": profile_ctx}
    return profile_ctx


async def _stream_words(
    update: Update,
    chat_handler,
    user_text: str,
    image_path: Optional[str],
    video_path: Optional[str] = None,
    user_profile_context: Optional[str] = None,
):
    final_text = await _run_with_typing(
        update,
        chat_handler,
        chat_handler.process_message,
        user_text,
        image_path,
        video_path,
        user_profile_context,
    )
    await _deliver_final_text_and_outbound(update, chat_handler, final_text)


async def _stream_video_sticker(
    update: Update,
    chat_handler,
    user_text: str,
    video_path: str,
    user_profile_context: Optional[str] = None,
):
    final_text = await _run_with_typing(
        update,
        chat_handler,
        chat_handler.process_video_sticker_message,
        video_path,
        user_text,
        user_profile_context,
    )
    await _deliver_final_text_and_outbound(update, chat_handler, final_text)


async def _run_with_typing(update: Update, chat_handler, fn, *args):
    stop_typing = asyncio.Event()

    async def _typing_pulse():
        chat = update.effective_chat
        if not chat:
            return
        while not stop_typing.is_set():
            try:
                await update.get_bot().send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)
            except Exception:
                pass
            try:
                await asyncio.wait_for(stop_typing.wait(), timeout=4.0)
            except asyncio.TimeoutError:
                continue

    typing_task = asyncio.create_task(_typing_pulse())
    final_text = "Terjadi error pada sistem."
    try:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, fn, *args)
        while not future.done():
            await flush_outbound_messages(chat_handler, lambda text: _safe_reply_markdown(update, text))
            try:
                await asyncio.wait_for(asyncio.shield(future), timeout=0.25)
            except asyncio.TimeoutError:
                continue
        await flush_outbound_messages(chat_handler, lambda text: _safe_reply_markdown(update, text))
        final_text = await future
        if not final_text or final_text == "ERROR":
            final_text = "Maaf, ada kesalahan sistem internal."
    except Exception as e:
        logger.error(f"Processing error: {e}")
    finally:
        stop_typing.set()
        with contextlib.suppress(Exception):
            await typing_task
    return final_text


async def _deliver_final_text_and_outbound(update: Update, chat_handler, final_text: str):
    delivered_any = False
    try:
        outbound_media = []
        outbound_files = []
        try:
            outbound_media = chat_handler.pop_pending_outbound_media()
        except Exception:
            outbound_media = []
        try:
            outbound_files = chat_handler.pop_pending_outbound_files()
        except Exception:
            outbound_files = []

        remaining_text = str(final_text or "").strip()
        if outbound_media:
            sent_ok, sent_paths = await send_outbound_media_with_caption(
                outbound_media,
                remaining_text,
                send_photo=lambda **kwargs: update.get_bot().send_photo(chat_id=update.effective_chat.id, **kwargs),
                send_media_group=lambda **kwargs: update.get_bot().send_media_group(chat_id=update.effective_chat.id, **kwargs),
                send_text=lambda text: send_chunked_response(update, text),
                file_prefix="web",
            )
            if sent_ok:
                delivered_any = True
                remaining_text = ""
                try:
                    chat_handler.session_manager.attach_latest_model_image_paths(sent_paths)
                except Exception as meta_err:
                    logger.warning(f"Failed to persist model image paths: {meta_err}")
        if outbound_files:
            sent_ok = await send_outbound_files_with_caption(
                outbound_files,
                remaining_text,
                send_document=lambda **kwargs: update.get_bot().send_document(chat_id=update.effective_chat.id, **kwargs),
                send_photo=lambda **kwargs: update.get_bot().send_photo(chat_id=update.effective_chat.id, **kwargs),
                send_text=lambda text: send_chunked_response(update, text),
                logger=logger,
                log_prefix="outbound",
            )
            if sent_ok:
                delivered_any = True
                remaining_text = ""
        if delivered_any:
            return

        if remaining_text:
            if len(remaining_text) <= MAX_TG_LEN:
                await _safe_reply_markdown(update, remaining_text)
            else:
                await send_chunked_response(update, remaining_text)
            delivered_any = True
    except Exception as deliver_err:
        logger.error(f"Delivery error after generation: {deliver_err}")
        try:
            await _safe_reply_markdown(update, final_text[:MAX_TG_LEN])
            delivered_any = True
        except Exception as fallback_err:
            logger.error(f"Fallback delivery failed: {fallback_err}")
    finally:
        try:
            chat_handler.finalize_pending_schedule_claim(
                delivered=delivered_any,
                note="Triggered by interaction",
            )
        except Exception as claim_err:
            logger.warning(f"Failed finalizing pending interaction schedules: {claim_err}")


async def _safe_reply_markdown(update: Update, text: str):
    await safe_send_text(
        text,
        send_html=lambda clean: update.message.reply_text(clean, parse_mode="HTML"),
        send_plain=lambda clean: update.message.reply_text(clean, parse_mode=None),
    )


async def handle_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    register_media_group_message(context, getattr(update, "message", None))
    chat_id = update.effective_chat.id

    if USER_LOCK.locked():
        await update.message.reply_text(f"Sebentar, {BOT_DISPLAY_NAME} masih memproses pesan sebelumnya.")
        return

    async with USER_LOCK:
        text_input = update.message.text or update.message.caption or ""
        # Route /sticker command in text/caption directly to utility path (no AI).
        normalized = (text_input or "").strip().lower()
        if normalized.startswith("/sticker"):
            await cmd_sticker(update, context)
            return

        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        user_profile_context = await _get_user_profile_context(update, context)

        img_path = None
        video_path = None
        video_sticker_path = None
        user_dir = os.path.join(TEMP_DIR, "chat_user")

        if update.message.reply_to_message:
            replied_text = update.message.reply_to_message.text or update.message.reply_to_message.caption or ""
            if replied_text:
                text_input = f"[Membalas pesan sebelumnya: '{replied_text}']\n{text_input}"

            if update.message.reply_to_message.photo:
                try:
                    os.makedirs(user_dir, exist_ok=True)
                    p_obj = await update.message.reply_to_message.photo[-1].get_file()
                    reply_img_path = os.path.join(user_dir, f"img_{uuid.uuid4().hex[:8]}.jpg")
                    await p_obj.download_to_drive(reply_img_path)
                    img_path = reply_img_path
                except Exception as e:
                    logger.error(f"Failed to download replied photo: {e}")

        if update.message.document:
            await update.message.reply_text("Fitur dokumen dinonaktifkan.")
            return

        if update.message.photo:
            img_path = await handle_photo(update, user_dir)
            if img_path is None:
                return

        if update.message.video:
            video_path = await handle_video(update, user_dir)
            if video_path is None:
                return

        if update.message.sticker:
            sticker_path = await handle_sticker(update, user_dir)
            if sticker_path is None:
                return
            if sticker_path.lower().endswith((".webm", ".mp4")):
                video_sticker_path = sticker_path
            else:
                img_path = sticker_path
                sticker_obj = update.message.sticker
                if getattr(sticker_obj, "is_animated", False):
                    sticker_ctx = "[Konteks: user mengirim sticker animasi (bukan foto biasa).]"
                else:
                    sticker_ctx = "[Konteks: user mengirim sticker statis (bukan foto biasa).]"
                if text_input and text_input.strip():
                    text_input = f"{text_input.strip()}\n\n{sticker_ctx}"
                else:
                    text_input = sticker_ctx

        if not text_input and not img_path and not video_path and not video_sticker_path:
            return

        try:
            services = context.application.bot_data
            chat_handler = services['chat_handler']
            if video_sticker_path:
                await _stream_video_sticker(
                    update,
                    chat_handler,
                    text_input,
                    video_sticker_path,
                    user_profile_context=user_profile_context,
                )
            else:
                await _stream_words(
                    update,
                    chat_handler,
                    text_input,
                    img_path,
                    video_path=video_path,
                    user_profile_context=user_profile_context,
                )
        except Exception as e:
            logger.error(f"Message process error: {e}", exc_info=True)
            await update.message.reply_text("Terjadi error pada sistem.")
        finally:
            _cleanup_temp_media_paths(img_path, video_sticker_path, video_path)


def _cleanup_temp_media_paths(img_path: Optional[str], video_sticker_path: Optional[str], video_path: Optional[str]):
    if img_path and os.path.exists(img_path) and IMAGE_STORE_DIR not in img_path:
        try:
            os.remove(img_path)
        except OSError:
            pass
    if video_sticker_path and os.path.exists(video_sticker_path):
        try:
            os.remove(video_sticker_path)
        except OSError:
            pass
    if video_path and os.path.exists(video_path) and VIDEO_STORE_DIR not in video_path:
        try:
            os.remove(video_path)
        except OSError:
            pass


async def handle_photo(update: Update, user_dir: str) -> Optional[str]:
    os.makedirs(user_dir, exist_ok=True)
    img_path = None
    try:
        p_obj = await update.message.photo[-1].get_file()
        img_path = os.path.join(user_dir, f"img_{uuid.uuid4().hex[:8]}.jpg")
        await p_obj.download_to_drive(img_path)

        file_size = os.path.getsize(img_path)
        if file_size < 100:
            raise ValueError(f"Downloaded image too small ({file_size} bytes)")

        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image: {e}")

        return img_path
    except Exception as e:
        logger.error(f"Photo handling failed: {e}")
        if img_path and os.path.exists(img_path):
            try:
                os.remove(img_path)
            except OSError:
                pass
        await update.message.reply_text("Gagal memproses gambar.")
        return None


async def handle_sticker(update: Update, user_dir: str) -> Optional[str]:
    os.makedirs(user_dir, exist_ok=True)
    sticker = update.message.sticker

    img_path = None
    try:
        f_obj = await sticker.get_file()
        if sticker.is_video:
            ext = ".webm"
        elif sticker.is_animated:
            await update.message.reply_text("Sticker animasi (.tgs) tidak didukung.")
            return None
        else:
            ext = ".webp"

        sticker_uid = str(getattr(sticker, "file_unique_id", "") or "").strip()
        if not sticker_uid:
            raise ValueError("Sticker tidak memiliki file_unique_id.")
        img_path = os.path.join(user_dir, f"sticker_{sticker_uid}{ext}")
        await f_obj.download_to_drive(img_path)

        file_size = os.path.getsize(img_path)
        if file_size < 50:
            raise ValueError(f"Downloaded sticker too small ({file_size} bytes)")

        if ext == ".webp":
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as e:
                raise ValueError(f"Invalid or corrupted sticker image: {e}")
        return img_path
    except Exception as e:
        logger.error(f"Sticker handling failed: {e}")
        if img_path and os.path.exists(img_path):
            try:
                os.remove(img_path)
            except OSError:
                pass
        await update.message.reply_text("Gagal memproses stiker.")
        return None


async def handle_video(update: Update, user_dir: str) -> Optional[str]:
    os.makedirs(user_dir, exist_ok=True)
    temp_path = None
    try:
        video = update.message.video
        if not video:
            return None

        duration = int(getattr(video, "duration", 0) or 0)
        if duration > VIDEO_MAX_DURATION_SECONDS:
            await update.message.reply_text(
                f"Video terlalu panjang ({duration}s). Maksimal {VIDEO_MAX_DURATION_SECONDS} detik."
            )
            return None

        f_obj = await video.get_file()
        ext = os.path.splitext(str(getattr(video, "file_name", "") or ""))[1].lower() or ".mp4"
        if ext not in {".mp4", ".webm", ".mov", ".m4v"}:
            ext = ".mp4"
        temp_path = os.path.join(user_dir, f"video_{uuid.uuid4().hex[:8]}{ext}")
        await f_obj.download_to_drive(temp_path)

        if not os.path.exists(temp_path) or os.path.getsize(temp_path) < 256:
            raise ValueError("Downloaded video invalid or too small")

        return temp_path
    except Exception as e:
        logger.error(f"Video handling failed: {e}")
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        await update.message.reply_text("Gagal memproses video.")
        return None
