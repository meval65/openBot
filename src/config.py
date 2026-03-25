from .config_parts import env
from .config_parts.bot import BotConfig, INSTRUCTION, MemoryType, PERSONA_FILE
from .config_parts.credentials import *
from .config_parts.limits import *
from .config_parts.models import *
from .config_parts.paths import *
import os
import re

logger = env.logger

_ENV_FILE_PATTERN = re.compile(r"^\.env\.[A-Za-z0-9_-]+$")


def _should_auto_prepare_storage() -> bool:
    launched_by_manager = str(os.getenv("LAUNCHED_BY_BOTS_PY", "") or "").strip() == "1"
    env_file = str(os.getenv("ENV_FILE", "") or "").strip()
    return launched_by_manager and bool(_ENV_FILE_PATTERN.match(env_file))


if _should_auto_prepare_storage():
    ensure_storage_layout()
