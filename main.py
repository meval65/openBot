import atexit
import asyncio
import os
import logging
import signal
import sys
import tracemalloc
from pathlib import Path
import re

from telegram.constants import ParseMode
from telegram.error import TimedOut, NetworkError, RetryAfter
from telegram.ext import (
    ApplicationBuilder, ContextTypes, CommandHandler,
    MessageHandler, CallbackQueryHandler, filters, Application, Defaults
)

from src.database import DBConnection
from src.services.memory import MemoryManager
from src.services.embedding import MemoryAnalyzer
from src.services.scheduling import SchedulerService
from src.services.chat import ChatHandler
from src.config import CHAT_MODEL, BotConfig, STORAGE_DIR, LOG_DIR

from src.handlers.commands import (
    cmd_start, cmd_new_session, cmd_wipe,
    cmd_config, cmd_settemp, cmd_settopp, cmd_setmaxtokens, cmd_setinstruction
)
from src.handlers.sticker_command import cmd_sticker
from src.handlers.callbacks import callback_handler
from src.handlers.messages import handle_msg
from src.handlers.background import (
    background_maintenance, background_schedule_checker,
    background_proactive_engine,
    background_performance_logger,
)


def _resolve_bot_name() -> str:
    explicit = os.getenv("BOT_INSTANCE") or os.getenv("BOT_NAME")
    if explicit and explicit.strip():
        return explicit.strip()
    storage_name = Path(STORAGE_DIR).name
    return storage_name or "default"


BOT_NAME = _resolve_bot_name()
INSTANCE_LOCK_PATH = os.path.join(STORAGE_DIR, ".instance.lock")
_LOCK_ACQUIRED = False


def _setup_logging() -> int:
    log_level = logging.ERROR

    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.bot = BOT_NAME
        return record

    logging.setLogRecordFactory(record_factory)

    os.makedirs(STORAGE_DIR, exist_ok=True)
    logs_dir = LOG_DIR
    os.makedirs(logs_dir, exist_ok=True)
    log_file_path = os.path.join(logs_dir, f"{BOT_NAME}.log")

    logging.basicConfig(
        format="%(asctime)s | bot=%(bot)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        level=log_level,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file_path, encoding="utf-8"),
        ],
        force=True,
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
    return log_level


_setup_logging()
logger = logging.getLogger(__name__)


def _always_visible_log(message: str, level: int = logging.INFO):
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


def _read_lock_pid() -> int | None:
    try:
        with open(INSTANCE_LOCK_PATH, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            return None
        return int(text)
    except Exception:
        return None


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists but not signalable by current user.
        return True
    except OSError:
        return False
    return True


def _release_instance_lock():
    global _LOCK_ACQUIRED
    try:
        if not os.path.exists(INSTANCE_LOCK_PATH):
            return

        lock_pid = _read_lock_pid()
        current_pid = os.getpid()

        # Never remove another live process lock.
        if lock_pid is not None and lock_pid != current_pid and _is_pid_alive(lock_pid):
            return

        os.remove(INSTANCE_LOCK_PATH)
        _LOCK_ACQUIRED = False
    except OSError:
        pass


def _cleanup_stale_lock_if_any() -> bool:
    if not os.path.exists(INSTANCE_LOCK_PATH):
        return True

    lock_pid = _read_lock_pid()
    if lock_pid is None:
        try:
            os.remove(INSTANCE_LOCK_PATH)
            logger.warning("[LOCK] Removed invalid lock file (missing/invalid PID)")
            return True
        except OSError:
            return False

    if _is_pid_alive(lock_pid):
        return False

    try:
        os.remove(INSTANCE_LOCK_PATH)
        logger.warning(f"[LOCK] Removed stale lock from dead PID {lock_pid}")
        return True
    except OSError:
        return False


def _acquire_instance_lock() -> bool:
    global _LOCK_ACQUIRED
    os.makedirs(STORAGE_DIR, exist_ok=True)

    # Auto-heal stale lock on crash/restart scenarios.
    _cleanup_stale_lock_if_any()

    try:
        fd = os.open(INSTANCE_LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        _LOCK_ACQUIRED = True
        atexit.register(_release_instance_lock)
        return True
    except FileExistsError:
        lock_pid = _read_lock_pid()
        if lock_pid is not None:
            logger.error(f"[LOCK] Storage already used by PID {lock_pid}")
        return False


def _install_signal_cleanup():
    def _handle_stop_signal(signum, _frame):
        try:
            logger.info(f"[LOCK] Stop signal received ({signum}), releasing lock")
        except Exception:
            pass
        _release_instance_lock()
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_stop_signal)
        except Exception:
            pass


_install_signal_cleanup()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
_ENV_FILE_PATTERN = re.compile(r"^\.env\.[A-Za-z0-9_-]+$")


def _validate_managed_launch() -> tuple[bool, str]:
    launched_by_manager = str(os.getenv("LAUNCHED_BY_BOTS_PY", "") or "").strip() == "1"
    env_file = str(os.getenv("ENV_FILE", "") or "").strip()
    if not launched_by_manager:
        return False, "Bot harus dijalankan melalui bots.py."
    if not env_file or not _ENV_FILE_PATTERN.match(env_file):
        return False, "ENV_FILE harus menggunakan format .env.<nama>."
    if not os.path.exists(env_file):
        return False, f"ENV_FILE '{env_file}' tidak ditemukan."
    return True, ""


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    err = context.error
    if isinstance(err, RetryAfter):
        logger.warning("Update throttled by Telegram, retry_after=%ss", getattr(err, "retry_after", "?"))
        return
    if isinstance(err, (TimedOut, NetworkError)):
        logger.warning("Transient Telegram network timeout: %s", err)
        return
    logger.error(f"Update error: {err}")


async def post_init(app: Application):
    if not tracemalloc.is_tracing():
        tracemalloc.start(25)

    _always_visible_log("[STARTUP] Initializing core services...")
    db = DBConnection()
    _always_visible_log("[STARTUP] DBConnection ready.")
    mem_mgr = MemoryManager(db)
    _always_visible_log("[STARTUP] MemoryManager ready.")
    analyzer = MemoryAnalyzer()
    _always_visible_log("[STARTUP] MemoryAnalyzer ready.")
    scheduler = SchedulerService(db)
    _always_visible_log("[STARTUP] SchedulerService ready.")
    bot_config = BotConfig()
    _always_visible_log("[STARTUP] BotConfig ready.")
    chat_handler = ChatHandler(mem_mgr, analyzer, scheduler, bot_config)
    _always_visible_log("[STARTUP] ChatHandler ready.")

    app.bot_data.update({
        'db': db,
        'mem_mgr': mem_mgr,
        'analyzer': analyzer,
        'scheduler': scheduler,
        'chat_handler': chat_handler,
        'bot_config': bot_config,
        'startup_ready': False,
    })

    try:
        _always_visible_log("[STARTUP] Waiting for terminal warmup...")
        terminal_ok, terminal_reason = await asyncio.to_thread(chat_handler.wait_for_terminal_warmup, 60.0)
        if terminal_ok:
            _always_visible_log("[STARTUP] Terminal sandbox ready.")
        else:
            raise RuntimeError(f"Terminal sandbox not ready: {terminal_reason}")

        _always_visible_log("[STARTUP] Running selective preload...")
        await asyncio.to_thread(chat_handler.preload_selective)
        _always_visible_log("[STARTUP] Selective preload completed.")
    except Exception as e:
        _always_visible_log(f"[STARTUP] Service readiness failed: {e}", level=logging.CRITICAL)
        raise

    if app.job_queue:
        app.job_queue.run_repeating(
            background_maintenance,
            interval=86400,
            first=60,
            job_kwargs={"coalesce": True, "max_instances": 1, "misfire_grace_time": 300},
        )
        app.job_queue.run_repeating(
            background_schedule_checker,
            interval=60,
            first=10,
            job_kwargs={"coalesce": True, "max_instances": 1, "misfire_grace_time": 30},
        )
        app.job_queue.run_repeating(
            background_proactive_engine,
            interval=300,
            first=600,
            job_kwargs={"coalesce": True, "max_instances": 1, "misfire_grace_time": 60},
        )
        app.job_queue.run_repeating(
            background_performance_logger,
            interval=300,
            first=120,
            job_kwargs={"coalesce": True, "max_instances": 1, "misfire_grace_time": 60},
        )

    app.bot_data['startup_ready'] = True
    _always_visible_log(f"[STARTUP] System ready. Model: {CHAT_MODEL}")


async def post_shutdown(app: Application):
    if 'db' in app.bot_data:
        app.bot_data['db'].close()
    _release_instance_lock()
    logger.info("System shutdown")


if __name__ == '__main__':
    ok, reason = _validate_managed_launch()
    if not ok:
        raise SystemExit(reason)

    if not TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN missing in env file")

    if not _acquire_instance_lock():
        raise SystemExit(
            f"Bot instance '{BOT_NAME}' cannot start: STORAGE_DIR '{STORAGE_DIR}' is already in use."
        )

    _always_visible_log(f"[STARTUP] Starting bot '{BOT_NAME}' with storage '{STORAGE_DIR}'")

    defaults = Defaults(parse_mode=ParseMode.HTML)

    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .defaults(defaults)
        .concurrent_updates(True)
        .build()
    )

    app.add_handler(CommandHandler('start', cmd_start))
    app.add_handler(CommandHandler('sticker', cmd_sticker))
    app.add_handler(CommandHandler('new_session', cmd_new_session))
    app.add_handler(CommandHandler('wipe', cmd_wipe))
    app.add_handler(CommandHandler('config', cmd_config))
    app.add_handler(CommandHandler('settemp', cmd_settemp))
    app.add_handler(CommandHandler('settopp', cmd_settopp))
    app.add_handler(CommandHandler('setmaxtokens', cmd_setmaxtokens))
    app.add_handler(CommandHandler('setinstruction', cmd_setinstruction))
    app.add_handler(CallbackQueryHandler(callback_handler))

    app.add_handler(MessageHandler(
        (filters.TEXT | filters.PHOTO | filters.VIDEO | filters.Sticker.ALL) & (~filters.COMMAND),
        handle_msg
    ))

    app.add_error_handler(error_handler)

    _always_visible_log("[STARTUP] Bot starting polling loop")
    app.run_polling(drop_pending_updates=True)
