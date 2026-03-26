import os

from . import env

AVAILABLE_CHAT_MODELS = [
    "models/gemini-3.1-flash-lite-preview",
    "models/gemini-2.5-flash-lite"
]

CHAT_MODEL = os.getenv("CHAT_MODEL") or AVAILABLE_CHAT_MODELS[0]
EMBEDDING_MODEL = "models/gemini-embedding-2-preview"
PROACTIVE_ANALYSIS_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
BACKGROUND_SUMMARY_MODEL = "models/gemma-3-12b-it"

def _env_bool(name: str, default: bool = True) -> bool:
    raw = str(os.getenv(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


TOOLS_ENABLE_ALL = _env_bool("TOOLS_ENABLE_ALL", True)
TOOLS_ENABLE_SEARCH_WEB = TOOLS_ENABLE_ALL and _env_bool("TOOLS_ENABLE_SEARCH_WEB", True)
TOOLS_ENABLE_SCHEDULE = TOOLS_ENABLE_ALL and _env_bool("TOOLS_ENABLE_SCHEDULE", True)
TOOLS_ENABLE_MEMORY = TOOLS_ENABLE_ALL and _env_bool("TOOLS_ENABLE_MEMORY", True)
TOOLS_ENABLE_ANNOUNCE_ACTION = TOOLS_ENABLE_ALL and _env_bool("TOOLS_ENABLE_ANNOUNCE_ACTION", True)
TOOLS_ENABLE_AI_PERSONAL_COMPUTER = TOOLS_ENABLE_ALL and _env_bool("TOOLS_ENABLE_AI_PERSONAL_COMPUTER", True)
TOOLS_ENABLE_AI_PC_INSPECT_IMAGES = TOOLS_ENABLE_AI_PERSONAL_COMPUTER and _env_bool("TOOLS_ENABLE_AI_PC_INSPECT_IMAGES", True)
TOOLS_ENABLE_AI_PC_SEND_FILES = TOOLS_ENABLE_AI_PERSONAL_COMPUTER and _env_bool("TOOLS_ENABLE_AI_PC_SEND_FILES", True)
