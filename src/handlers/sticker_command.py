import mimetypes
import os
import tempfile
from typing import Optional, Tuple

from telegram import InputFile, Message, Update
from telegram.ext import ContextTypes

from src.config import TEMP_DIR
from src.handlers.media_group_cache import get_media_group_items, register_media_group_message
from src.services.media.sticker_tool import (
    convert_image_to_webp_sticker,
    convert_video_to_webm_sticker,
)


def _source_message(msg: Message) -> Message:
    return msg.reply_to_message or msg


def _guess_ext(file_name: str, mime: str, fallback: str) -> str:
    name_ext = os.path.splitext(str(file_name or ""))[1].lower()
    if name_ext:
        return name_ext
    mime_ext = mimetypes.guess_extension(str(mime or "").lower()) or ""
    if mime_ext:
        return mime_ext.lower()
    return fallback


def _extract_media_from_message(msg: Message) -> Tuple[Optional[object], str, str]:
    # Returns (file_obj, mode, ext_fallback)
    if msg.photo:
        return msg.photo[-1], "image", ".jpg"

    if msg.sticker:
        st = msg.sticker
        if getattr(st, "is_animated", False):
            return None, "unsupported_tgs", ""
        if getattr(st, "is_video", False):
            return st, "video", ".webm"
        return st, "image", ".webp"

    if msg.video:
        return msg.video, "video", ".mp4"

    if msg.animation:
        return msg.animation, "video", ".mp4"

    if msg.document:
        doc = msg.document
        mime = str(getattr(doc, "mime_type", "") or "").lower()
        fname = str(getattr(doc, "file_name", "") or "")
        ext = _guess_ext(fname, mime, ".bin")
        if mime.startswith("image/") or ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}:
            return doc, "image", ext or ".jpg"
        if mime.startswith("video/") or ext in {".mp4", ".webm", ".mov", ".m4v", ".avi", ".gif"}:
            return doc, "video", ext or ".mp4"

    return None, "none", ""


def _desc_from_message(msg: Message) -> Optional[dict]:
    media_obj, mode, ext_fallback = _extract_media_from_message(msg)
    if not media_obj:
        return None
    return {
        "kind": mode,
        "file_id": str(getattr(media_obj, "file_id", "") or ""),
        "file_name": str(getattr(media_obj, "file_name", "") or ""),
        "mime_type": str(getattr(media_obj, "mime_type", "") or ""),
        "ext_fallback": ext_fallback or ".bin",
    }


async def cmd_sticker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return

    src = _source_message(update.message)
    register_media_group_message(context, update.message)
    register_media_group_message(context, src)

    items = get_media_group_items(context, src)
    if not items:
        one = _desc_from_message(src)
        items = [one] if one else []

    items = [it for it in items if isinstance(it, dict) and str(it.get("file_id") or "").strip()]
    if not items:
        await update.message.reply_text(
            "Gunakan /sticker dengan reply ke gambar atau video/gif.\n"
            "Contoh: reply media lalu kirim /sticker"
        )
        return

    unsupported_tgs = [it for it in items if str(it.get("kind") or "") == "unsupported_tgs"]
    if unsupported_tgs and len(unsupported_tgs) == len(items):
        await update.message.reply_text("Sticker animasi (.tgs) tidak didukung untuk /sticker.")
        return

    os.makedirs(TEMP_DIR, exist_ok=True)
    converted = 0
    failed = 0
    skipped_tgs = 0

    for idx, item in enumerate(items, start=1):
        kind = str(item.get("kind") or "")
        if kind == "unsupported_tgs":
            skipped_tgs += 1
            continue
        if kind not in {"image", "video"}:
            failed += 1
            continue

        with tempfile.TemporaryDirectory(prefix="sticker_cmd_", dir=TEMP_DIR) as td:
            try:
                file_id = str(item.get("file_id") or "")
                f = await context.bot.get_file(file_id)
                file_name = str(item.get("file_name") or "")
                mime = str(item.get("mime_type") or "")
                ext_fallback = str(item.get("ext_fallback") or ".bin")
                ext = _guess_ext(file_name, mime, ext_fallback)
                src_path = os.path.join(td, f"in_{idx}{ext}")
                await f.download_to_drive(src_path)
            except Exception:
                failed += 1
                continue

            if kind == "image":
                out_path = os.path.join(td, f"sticker_{idx}.webp")
                ok = convert_image_to_webp_sticker(src_path, out_path)
            else:
                out_path = os.path.join(td, f"sticker_{idx}.webm")
                ok = convert_video_to_webm_sticker(src_path, out_path)

            if not ok:
                failed += 1
                continue

            try:
                with open(out_path, "rb") as fp:
                    await update.message.reply_sticker(sticker=InputFile(fp))
                converted += 1
            except Exception:
                failed += 1

    if converted <= 0:
        if skipped_tgs > 0 and failed == 0:
            await update.message.reply_text("Sticker animasi (.tgs) tidak didukung untuk /sticker.")
        else:
            await update.message.reply_text("Tidak ada media yang berhasil dikonversi menjadi sticker.")
        return

    if failed > 0 or skipped_tgs > 0:
        await update.message.reply_text(
            f"Selesai: {converted} berhasil, {failed} gagal, {skipped_tgs} dilewati (.tgs)."
        )
