import os
import asyncio
import logging

from telegram import Update
from telegram.constants import ParseMode
from telegram.error import TelegramError

from src.config import MAX_FILE_SIZE, MAX_TG_LEN

logger = logging.getLogger(__name__)

USER_LOCK = asyncio.Lock()

async def read_file_content(file_path: str) -> str:
    try:
        def _read_safe():
            if os.path.getsize(file_path) > MAX_FILE_SIZE:
                return None
            with open(file_path, 'rb') as f:
                raw_data = f.read()
            try:
                return raw_data.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    return raw_data.decode('latin-1')
                except UnicodeDecodeError:
                    return raw_data.decode('ascii', errors='ignore')

        content = await asyncio.to_thread(_read_safe)
        if content is None:
            return "[... File too large to process ...]"
        return content
    except Exception as e:
        logger.error(f"File read error: {e}")
        return "[Error reading file]"


async def send_chunked_response(update: Update, text: str):
    if not text:
        return
    while text:
        if len(text) <= MAX_TG_LEN:
            chunk = text
            text = ""
        else:
            split_idx = text.rfind('\n', 0, MAX_TG_LEN)
            if split_idx == -1:
                split_idx = text.rfind(' ', 0, MAX_TG_LEN)
            if split_idx == -1:
                split_idx = MAX_TG_LEN
            chunk = text[:split_idx]
            text = text[split_idx:].lstrip()
        try:
            await update.message.reply_text(
                chunk,
                parse_mode=ParseMode.HTML
            )
        except TelegramError:
            await update.message.reply_text(chunk, parse_mode=None)
        if text:
            await asyncio.sleep(0.5)
