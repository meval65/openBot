import json
import os
import time
from enum import Enum

from . import env
from . import limits
from . import paths


PERSONA_FILE = os.getenv("PERSONA_FILE")

if PERSONA_FILE and os.path.exists(PERSONA_FILE):
    with open(PERSONA_FILE, "r", encoding="utf-8") as f:
        INSTRUCTION = f.read()
else:
    INSTRUCTION = ""


class BotConfig:
    def __init__(self, storage_dir=None):
        if storage_dir is None:
            storage_dir = paths.STORAGE_DIR
        self.config_path = os.path.join(storage_dir, "bot_config.json")
        self.temperature = limits.TEMPERATURE
        self.top_p = limits.TOP_P
        self.max_output_tokens = limits.MAX_OUTPUT_TOKENS
        self.instruction = INSTRUCTION
        self._last_loaded_mtime = None
        self._last_check_ts = 0.0
        self._reload_check_interval_sec = 1.0
        self.load(force=True)

    def load(self, force: bool = False):
        now_ts = time.time()
        if not force and (now_ts - self._last_check_ts) < self._reload_check_interval_sec:
            return
        self._last_check_ts = now_ts

        if not os.path.exists(self.config_path):
            return
        try:
            mtime = os.path.getmtime(self.config_path)
        except Exception:
            mtime = None

        if not force and self._last_loaded_mtime is not None and mtime == self._last_loaded_mtime:
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.temperature = float(data.get("temperature", self.temperature))
                self.top_p = max(0.0, min(1.0, float(data.get("top_p", self.top_p))))
                self.max_output_tokens = int(data.get("max_output_tokens", self.max_output_tokens))
                self.instruction = str(data.get("instruction", self.instruction))
                self._last_loaded_mtime = mtime
        except Exception as e:
            env.logger.warning(f"[BOT-CONFIG] Failed to load config: {e}")

    def save(self):
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "max_output_tokens": self.max_output_tokens,
                        "instruction": self.instruction,
                    },
                    f,
                    indent=4,
                )
            try:
                self._last_loaded_mtime = os.path.getmtime(self.config_path)
            except Exception:
                self._last_loaded_mtime = None
            self._last_check_ts = time.time()
        except Exception as e:
            env.logger.warning(f"[BOT-CONFIG] Failed to save config: {e}")

    def set_temperature(self, val: float):
        self.temperature = float(val)
        self.save()

    def set_max_output_tokens(self, val: int):
        self.max_output_tokens = int(val)
        self.save()

    def set_top_p(self, val: float):
        self.top_p = max(0.0, min(1.0, float(val)))
        self.save()

    def set_instruction(self, val: str):
        self.instruction = val
        self.save()


class MemoryType(Enum):
    GENERAL = "general"
    EMOTION = "emotion"
    DECISION = "decision"
    PREFERENCE = "preference"
    BOUNDARY = "boundary"
    FACT = "fact"
    MOOD_STATE = "mood_state"
