import os


def _bot_id_from_env_file() -> str:
    env_file = str(os.getenv("ENV_FILE", "") or "").strip()
    if not env_file:
        return ""
    base = os.path.basename(env_file)
    if base.startswith(".env.") and len(base) > len(".env."):
        return base[len(".env.") :]
    return ""


def _resolve_storage_dir() -> str:
    raw = str(os.getenv("STORAGE_DIR", "") or "").strip()
    norm = os.path.normpath(raw) if raw else ""
    placeholders = {"", ".", "storage", os.path.normpath("./storage"), os.path.normpath(".\\storage")}
    if norm in placeholders:
        bot_name = (
            str(os.getenv("BOT_INSTANCE", "") or "").strip()
            or str(os.getenv("BOT_NAME", "") or "").strip()
            or _bot_id_from_env_file()
        )
        if bot_name:
            return os.path.join("storage", bot_name)
        return "storage"
    return raw


STORAGE_DIR = _resolve_storage_dir()
DB_DIR = os.path.join(STORAGE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "memory.db")
RUNTIME_DIR = os.path.join(STORAGE_DIR, "runtime")
CACHE_DIR = os.path.join(RUNTIME_DIR, "cache")
HEALTH_DIR = os.path.join(RUNTIME_DIR, "health")
LOG_DIR = os.path.join(RUNTIME_DIR, "logs")
MEDIA_DIR = os.path.join(STORAGE_DIR, "media")
ANIMATED_COLLAGE_CACHE_DIR = os.path.join(CACHE_DIR, "animated_collages")

SESSION_DIR = os.path.join(STORAGE_DIR, "sessions")
TEMP_DIR = os.path.join(RUNTIME_DIR, "temp")
IMAGE_STORE_DIR = os.path.join(MEDIA_DIR, "images")
STICKER_STORE_DIR = os.path.join(MEDIA_DIR, "stickers")
STICKER_STATIC_STORE_DIR = os.path.join(STICKER_STORE_DIR, "static")
STICKER_VIDEO_STORE_DIR = os.path.join(STICKER_STORE_DIR, "video")
VIDEO_STORE_DIR = os.path.join(MEDIA_DIR, "videos")


def ensure_storage_layout():
    os.makedirs(STORAGE_DIR, exist_ok=True)
    os.makedirs(ANIMATED_COLLAGE_CACHE_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(RUNTIME_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(HEALTH_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MEDIA_DIR, exist_ok=True)
    os.makedirs(IMAGE_STORE_DIR, exist_ok=True)
    os.makedirs(STICKER_STORE_DIR, exist_ok=True)
    os.makedirs(STICKER_STATIC_STORE_DIR, exist_ok=True)
    os.makedirs(STICKER_VIDEO_STORE_DIR, exist_ok=True)
    os.makedirs(VIDEO_STORE_DIR, exist_ok=True)
    os.makedirs(SESSION_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
