import datetime
import pytz
from typing import Optional
from src.config import TIMEZONE

def get_local_tz() -> pytz.BaseTzInfo:
    """Get the configured active local timezone, falling back to Asia/Jakarta."""
    try:
        return pytz.timezone(TIMEZONE)
    except Exception:
        return pytz.timezone("Asia/Jakarta")

def now_local() -> datetime.datetime:
    """Return the current time timezone-aware based on local config."""
    return datetime.datetime.now(get_local_tz())

def to_naive(dt: datetime.datetime) -> datetime.datetime:
    """Convert an aware datetime into a naive datetime in the local timezone."""
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(get_local_tz()).replace(tzinfo=None)

def format_human_time(dt: datetime.datetime) -> str:
    """Format a datetime clearly like '2026-03-20 08:00 WIB'."""
    tz = get_local_tz()
    if dt.tzinfo is None:
        dt = tz.localize(dt)
    local_dt = dt.astimezone(tz)
    return local_dt.strftime("%Y-%m-%d %H:%M %Z")

def parse_local_dt(dt_str: str) -> Optional[datetime.datetime]:
    """Parse an ISO format string into a naive local datetime safely."""
    if not dt_str:
        return None
    try:
        text = str(dt_str).strip()
        # Python versions differ in handling trailing "Z"; normalize to +00:00.
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        dt = datetime.datetime.fromisoformat(text)
        return to_naive(dt)
    except ValueError:
        return None
