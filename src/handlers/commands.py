import html

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from src.config import BOT_DISPLAY_NAME



async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_name = (
        " ".join(
            p for p in [getattr(user, "first_name", None), getattr(user, "last_name", None)] if p
        ).strip()
        if user
        else ""
    )
    if not user_name and user:
        user_name = getattr(user, "username", "") or "User"
    user_name_safe = html.escape(user_name or "User")

    txt = (
        f"<b>Hai, {user_name_safe}. Sistem {BOT_DISPLAY_NAME} Aktif.</b>\n\n"
        "<b>Daftar Perintah:</b>\n"
        "- /start\n"
        "- /sticker (reply ke gambar/video/gif)\n"
        "- /new_session\n"
        "- /wipe\n"
        "- /config\n"
        "- /settemp &lt;nilai&gt;\n"
        "- /settopp &lt;nilai&gt;\n"
        "- /setmaxtokens &lt;nilai&gt;\n"
        "- /setinstruction &lt;teks&gt;\n"
    )
    await update.message.reply_text(txt, parse_mode=ParseMode.HTML)


async def cmd_new_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("Ya", callback_data="new_session_confirm"),
         InlineKeyboardButton("Batal", callback_data="new_session_cancel")]
    ])
    await update.message.reply_text(
        "Mulai sesi baru dan reset konteks chat saat ini?",
        parse_mode=ParseMode.HTML,
        reply_markup=kb,
    )


async def cmd_wipe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("Ya, hapus semua", callback_data="wipe_confirm"),
         InlineKeyboardButton("Batal", callback_data="wipe_cancel")]
    ])
    await update.message.reply_text(
        "Peringatan: ini akan menghapus semua memori, jadwal, dan sesi. Lanjut?",
        parse_mode=ParseMode.HTML,
        reply_markup=kb,
    )


async def cmd_config(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cfg = context.application.bot_data.get('bot_config')
    if not cfg:
        await update.message.reply_text("Konfigurasi tidak ditemukan.")
        return

    instruction_preview_safe = html.escape(cfg.instruction)
    text = (
        f"<b>Konfigurasi Bot (runtime)</b>\n\n"
        f"Temperature: <code>{cfg.temperature}</code>\n"
        f"Top P: <code>{cfg.top_p}</code>\n"
        f"Max Tokens: <code>{cfg.max_output_tokens}</code>\n"
        f"Instruction:\n<code>{instruction_preview_safe or '(kosong)'}</code>"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def cmd_settemp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Gunakan: /settemp <nilai>. Contoh: /settemp 0.9")
        return

    try:
        val = float(context.args[0])
        if not (0.0 <= val <= 2.0):
            raise ValueError
    except ValueError:
        await update.message.reply_text("Nilai temperature harus antara 0.0 dan 2.0.")
        return

    cfg = context.application.bot_data.get('bot_config')
    if not cfg:
        await update.message.reply_text("Layanan tidak tersedia.")
        return

    cfg.set_temperature(val)
    await update.message.reply_text(f"Temperature diubah ke <code>{val}</code>.", parse_mode=ParseMode.HTML)


async def cmd_settopp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Gunakan: /settopp <nilai>. Contoh: /settopp 0.95")
        return

    try:
        val = float(context.args[0])
        if not (0.0 <= val <= 1.0):
            raise ValueError
    except ValueError:
        await update.message.reply_text("Nilai top_p harus antara 0.0 dan 1.0.")
        return

    cfg = context.application.bot_data.get('bot_config')
    if not cfg:
        await update.message.reply_text("Layanan tidak tersedia.")
        return

    cfg.set_top_p(val)
    await update.message.reply_text(f"Top P diubah ke <code>{val}</code>.", parse_mode=ParseMode.HTML)


async def cmd_setmaxtokens(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Gunakan: /setmaxtokens <nilai>. Contoh: /setmaxtokens 1024")
        return

    try:
        val = int(context.args[0])
        if not (64 <= val <= 8192):
            raise ValueError
    except ValueError:
        await update.message.reply_text("Nilai harus bilangan bulat antara 64 dan 8192.")
        return

    cfg = context.application.bot_data.get('bot_config')
    if not cfg:
        await update.message.reply_text("Layanan tidak tersedia.")
        return

    cfg.set_max_output_tokens(val)
    await update.message.reply_text(f"Max tokens diubah ke <code>{val}</code>.", parse_mode=ParseMode.HTML)


async def cmd_setinstruction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text_content = update.message.text or update.message.caption or ""

    if update.message.entities and update.message.entities[0].type == "bot_command":
        cmd_length = update.message.entities[0].length
        new_instruction = text_content[cmd_length:].strip()
    else:
        new_instruction = " ".join(context.args) if context.args else ""

    if not new_instruction:
        await update.message.reply_text("Gunakan: /setinstruction <teks instruksi>")
        return

    cfg = context.application.bot_data.get('bot_config')
    if not cfg:
        await update.message.reply_text("Layanan tidak tersedia.")
        return

    cfg.set_instruction(new_instruction)
    preview = (new_instruction[:80] + "...") if len(new_instruction) > 80 else new_instruction
    preview_safe = html.escape(preview)
    await update.message.reply_text(
        f"Instruction diperbarui:\n<code>{preview_safe}</code>",
        parse_mode=ParseMode.HTML,
    )

