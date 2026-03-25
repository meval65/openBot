import os

from . import env

_keys_str = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_API_KEYS = [k.strip() for k in _keys_str.split(",") if k.strip()]
GOOGLE_API_KEY = GOOGLE_API_KEYS[0] if GOOGLE_API_KEYS else None

ADMIN_TELEGRAM_ID = os.getenv("ADMIN_TELEGRAM_ID", "")
METEOSOURCE_API_KEY = os.getenv("METEOSOURCE_API_KEY")

_tavily_keys_str = os.getenv("TAVILY_API_KEY", "")
TAVILY_API_KEYS = [k.strip() for k in _tavily_keys_str.split(",") if k.strip()]

TIMEZONE = os.getenv("TIMEZONE", "Asia/Jakarta")
BOT_DISPLAY_NAME = (
    os.getenv("BOT_DISPLAY_NAME")
    or os.getenv("BOT_INSTANCE")
    or os.getenv("BOT_NAME")
    or "AI"
).strip()
