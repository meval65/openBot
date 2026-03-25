import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

ENV_FILE = str(os.getenv("ENV_FILE", ".env") or ".env").strip()


def _resolve_env_path(env_file: str) -> str:
    if os.path.isabs(env_file) and os.path.exists(env_file):
        return env_file
    candidates = [
        env_file,
        os.path.join(os.getcwd(), env_file),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), env_file),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return env_file


_ENV_PATH = _resolve_env_path(ENV_FILE)
# Use utf-8-sig so files with BOM still parse correct keys (e.g. GOOGLE_API_KEY).
load_dotenv(_ENV_PATH, override=True, encoding="utf-8-sig")
