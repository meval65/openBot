import io
import logging
import mimetypes
import os
from typing import Awaitable, Callable, Optional

from telegram import InputFile, InputMediaPhoto
from telegram.constants import ParseMode
from telegram.error import TelegramError


TELEGRAM_CAPTION_LIMIT = 1024


async def safe_send_text(
    text: str,
    send_html: Callable[[str], Awaitable[object]],
    send_plain: Callable[[str], Awaitable[object]],
):
    clean = " ".join(str(text or "").split()).strip()
    if not clean:
        return
    try:
        await send_html(clean)
    except TelegramError:
        await send_plain(clean)


async def flush_outbound_messages(chat_handler, send_text: Callable[[str], Awaitable[object]]) -> bool:
    try:
        outbound_messages = chat_handler.pop_pending_outbound_messages()
    except Exception:
        outbound_messages = []
    sent_any = False
    for text in outbound_messages:
        clean = " ".join(str(text or "").split()).strip()
        if not clean:
            continue
        await send_text(clean)
        sent_any = True
    return sent_any


async def send_outbound_media_with_caption(
    media_items: list,
    text: str,
    send_photo: Callable[..., Awaitable[object]],
    send_media_group: Callable[..., Awaitable[object]],
    send_text: Callable[[str], Awaitable[object]],
    file_prefix: str = "media",
):
    if not media_items:
        if text:
            await send_text(text)
        return False, []

    safe_text = str(text or "").strip()
    caption = safe_text[:TELEGRAM_CAPTION_LIMIT]
    remain = safe_text[TELEGRAM_CAPTION_LIMIT:].strip() if safe_text else ""

    prepared = []
    sent_media_refs = []
    for idx, item in enumerate(media_items[:5]):
        if not isinstance(item, dict):
            continue
        data = item.get("data")
        if not isinstance(data, (bytes, bytearray)) or not data:
            continue
        mime = str(item.get("mime_type") or "image/jpeg").strip().lower()
        ext = ".png" if "png" in mime else ".jpg"
        bio = io.BytesIO(bytes(data))
        bio.name = f"{file_prefix}_{idx + 1}{ext}"
        prepared.append(
            {
                "bio": bio,
                "path": str(item.get("path") or "").strip(),
                "ai_workspace_path": str(item.get("ai_workspace_path") or "").strip(),
            }
        )

    if not prepared:
        if safe_text:
            await send_text(safe_text)
        return False, []

    if len(prepared) == 1:
        try:
            await send_photo(
                photo=prepared[0]["bio"],
                caption=caption or None,
                parse_mode=ParseMode.HTML if caption else None,
            )
        except TelegramError:
            await send_photo(
                photo=prepared[0]["bio"],
                caption=caption or None,
            )
    else:
        media = [
            InputMediaPhoto(
                media=item["bio"],
                caption=caption if idx == 0 and caption else None,
                parse_mode=ParseMode.HTML if idx == 0 and caption else None,
            )
            for idx, item in enumerate(prepared)
        ]
        try:
            await send_media_group(media=media)
        except TelegramError:
            plain_media = [
                InputMediaPhoto(media=item["bio"], caption=caption if idx == 0 and caption else None)
                for idx, item in enumerate(prepared)
            ]
            await send_media_group(media=plain_media)

    for item in prepared:
        host_path = str(item.get("path") or "").strip()
        ai_workspace_path = str(item.get("ai_workspace_path") or "").strip()
        if host_path or ai_workspace_path:
            sent_media_refs.append(
                {
                    "kind": "image",
                    "host_path": host_path,
                    "ai_workspace_path": ai_workspace_path,
                    "role": "model",
                }
            )

    if remain:
        await send_text(remain)
    return True, sent_media_refs


async def send_outbound_files_with_caption(
    file_items: list,
    text: str,
    send_document: Callable[..., Awaitable[object]],
    send_photo: Optional[Callable[..., Awaitable[object]]],
    send_text: Callable[[str], Awaitable[object]],
    logger: Optional[logging.Logger] = None,
    log_prefix: str = "outbound",
) -> bool:
    if not file_items:
        if text:
            await send_text(text)
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
        if safe_text:
            await send_text(safe_text)
        return False

    sent_count = 0
    for idx, item in enumerate(prepared):
        try:
            mime_type, _ = mimetypes.guess_type(item["filename"])
            is_image = bool(mime_type and mime_type.startswith("image/"))
            with open(item["path"], "rb") as fp:
                this_caption = (caption or item["caption"]) if idx == 0 else None
                if is_image and callable(send_photo):
                    photo = InputFile(fp, filename=item["filename"])
                    try:
                        await send_photo(
                            photo=photo,
                            caption=this_caption,
                            parse_mode=ParseMode.HTML if this_caption else None,
                        )
                        sent_count += 1
                    except TelegramError:
                        fp.seek(0)
                        photo_plain = InputFile(fp, filename=item["filename"])
                        await send_photo(
                            photo=photo_plain,
                            caption=this_caption,
                        )
                        sent_count += 1
                else:
                    doc = InputFile(fp, filename=item["filename"])
                    try:
                        await send_document(
                            document=doc,
                            caption=this_caption,
                            parse_mode=ParseMode.HTML if this_caption else None,
                        )
                        sent_count += 1
                    except TelegramError:
                        fp.seek(0)
                        doc_plain = InputFile(fp, filename=item["filename"])
                        await send_document(
                            document=doc_plain,
                            caption=this_caption,
                        )
                        sent_count += 1
        except Exception as e:
            if logger:
                logger.error(f"Failed sending {log_prefix} file '{item.get('path')}': {e}")
        finally:
            if bool(item.get("cleanup_after_send")):
                try:
                    if os.path.isfile(item["path"]):
                        os.remove(item["path"])
                except Exception:
                    pass

    if sent_count <= 0:
        if safe_text:
            await send_text(safe_text)
        return False

    if remain:
        await send_text(remain)
    return True
