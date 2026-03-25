import asyncio
import shutil
import os
import logging

from telegram import Update
from telegram.ext import ContextTypes

from src.config import SESSION_DIR, TEMP_DIR

logger = logging.getLogger(__name__)


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    try:
        services = context.application.bot_data

        if data == "wipe_confirm":
            try:
                del_count = await asyncio.to_thread(services['mem_mgr'].wipe_all_memories)
                await asyncio.to_thread(services['chat_handler'].clear_session)
                if os.path.exists(SESSION_DIR):
                    await asyncio.to_thread(shutil.rmtree, SESSION_DIR, ignore_errors=True)
                if os.path.exists(TEMP_DIR):
                    await asyncio.to_thread(shutil.rmtree, TEMP_DIR, ignore_errors=True)
                await query.edit_message_text(f"Reset selesai. {del_count} memori dihapus.")
            except Exception as e:
                logger.error(f"Wipe confirm error: {e}")
                await query.edit_message_text("Reset gagal.")

        elif data == "wipe_cancel":
            await query.edit_message_text("Reset dibatalkan.")

        elif data == "new_session_confirm":
            try:
                await asyncio.to_thread(services['chat_handler'].clear_session)
                await query.edit_message_text("Sesi chat baru berhasil dimulai.")
            except Exception as e:
                logger.error(f"New session confirm error: {e}")
                await query.edit_message_text("Gagal memulai sesi baru.")

        elif data == "new_session_cancel":
            await query.edit_message_text("Sesi baru dibatalkan.")

    except Exception as e:
        logger.error(f"Callback error: {e}")
        await query.edit_message_text("Gagal memuat data.")
