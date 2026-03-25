import os
import threading
import time
from typing import List, Tuple


_MAX_ENTRIES = 80
_MAX_DEPTH = 3
_SNAPSHOT_CACHE_TTL_SECONDS = 3.0
_SNAPSHOT_CACHE_MAX_KEYS = 16
_CACHE_LOCK = threading.RLock()
_SNAPSHOT_CACHE = {}


def _root_mtime(path: str) -> float:
    try:
        return float(os.path.getmtime(path))
    except Exception:
        return 0.0


def _build_cache_key(root: str, limit: int, depth_limit: int) -> tuple:
    return (root, int(limit), int(depth_limit))


def _get_cached_snapshot(key: tuple, root_mtime: float) -> str:
    now = time.monotonic()
    with _CACHE_LOCK:
        entry = _SNAPSHOT_CACHE.get(key)
        if not isinstance(entry, dict):
            return ""
        age = float(now - float(entry.get("ts", 0.0)))
        if age > _SNAPSHOT_CACHE_TTL_SECONDS:
            return ""
        if float(entry.get("root_mtime", 0.0)) != float(root_mtime):
            return ""
        text = str(entry.get("snapshot") or "")
        return text


def _set_cached_snapshot(key: tuple, root_mtime: float, snapshot: str):
    with _CACHE_LOCK:
        if len(_SNAPSHOT_CACHE) >= _SNAPSHOT_CACHE_MAX_KEYS and key not in _SNAPSHOT_CACHE:
            oldest_key = min(
                _SNAPSHOT_CACHE.keys(),
                key=lambda k: float((_SNAPSHOT_CACHE.get(k) or {}).get("ts", 0.0)),
                default=None,
            )
            if oldest_key is not None:
                _SNAPSHOT_CACHE.pop(oldest_key, None)
        _SNAPSHOT_CACHE[key] = {
            "ts": time.monotonic(),
            "root_mtime": float(root_mtime),
            "snapshot": str(snapshot or ""),
        }


def build_workspace_snapshot(workspace_dir: str, max_entries: int = _MAX_ENTRIES, max_depth: int = _MAX_DEPTH) -> str:
    root = os.path.abspath(str(workspace_dir or "").strip() or ".")
    if not os.path.isdir(root):
        return (
            "[Workspace Snapshot]\n"
            f"Path: {root}\n"
            "- workspace belum tersedia"
        )

    limit = max(20, int(max_entries or _MAX_ENTRIES))
    depth_limit = max(1, int(max_depth or _MAX_DEPTH))
    root_mtime = _root_mtime(root)
    cache_key = _build_cache_key(root, limit, depth_limit)
    cached_snapshot = _get_cached_snapshot(cache_key, root_mtime)
    if cached_snapshot:
        return cached_snapshot

    lines: List[str] = []
    stack: List[Tuple[str, int]] = [("", 0)]
    truncated = False

    while stack and len(lines) < limit:
        rel_dir, depth = stack.pop()
        abs_dir = os.path.join(root, rel_dir) if rel_dir else root
        try:
            with os.scandir(abs_dir) as it:
                entries = sorted(it, key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower()))
        except Exception:
            continue

        child_dirs: List[str] = []
        for entry in entries:
            rel_path = os.path.join(rel_dir, entry.name) if rel_dir else entry.name
            rel_posix = rel_path.replace("\\", "/")

            if entry.is_dir(follow_symlinks=False):
                lines.append(f"- d {rel_posix}/")
                if depth + 1 < depth_limit:
                    child_dirs.append(rel_path)
            else:
                size = 0
                try:
                    size = int(entry.stat(follow_symlinks=False).st_size or 0)
                except Exception:
                    size = 0
                lines.append(f"- f {rel_posix} ({size}B)")

            if len(lines) >= limit:
                truncated = True
                break

        if len(lines) >= limit:
            break

        for child in reversed(child_dirs):
            stack.append((child, depth + 1))

    if not lines:
        lines.append("- (kosong)")
    if truncated:
        lines.append(f"- ... dipotong (maks {limit} item)")

    snapshot = "\n".join(lines)
    snapshot_text = (
        "[Workspace Snapshot]\n"
        f"Path: {root}\n"
        "Daftar isi workspace terbaru:\n"
        f"{snapshot}"
    )
    _set_cached_snapshot(cache_key, root_mtime, snapshot_text)
    return snapshot_text
