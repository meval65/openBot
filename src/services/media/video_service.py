import datetime
import hashlib
import json
import logging
import mimetypes
import os
import shutil
import subprocess
import tempfile
from typing import Optional, Tuple

from PIL import Image, ImageDraw

from src.config import (
    ANIMATED_COLLAGE_CACHE_DIR,
    HISTORY_VISUAL_PART_TOKEN_BASE,
    STICKER_VIDEO_STORE_DIR,
    VIDEO_MAX_DURATION_SECONDS,
    VIDEO_VISUAL_BASE_TOKENS,
    VIDEO_VISUAL_TOKENS_PER_SECOND,
    VIDEO_TARGET_HEIGHT,
    VIDEO_SCALE_ALGO,
    VIDEO_STORE_DIR,
)
from . import catalog

logger = logging.getLogger(__name__)

os.makedirs(VIDEO_STORE_DIR, exist_ok=True)
os.makedirs(STICKER_VIDEO_STORE_DIR, exist_ok=True)
os.makedirs(ANIMATED_COLLAGE_CACHE_DIR, exist_ok=True)


def read_video_bytes(path: str) -> Tuple[bytes, str, str]:
    with open(path, "rb") as f:
        data = f.read()
    mime_type, _ = mimetypes.guess_type(path)
    if not mime_type:
        mime_type = "video/mp4"
    sha256_hex = hashlib.sha256(data).hexdigest()
    return data, mime_type, sha256_hex


def _ffprobe_json(path: str) -> dict:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {}
    try:
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if proc.returncode != 0 or not proc.stdout.strip():
            return {}
        return json.loads(proc.stdout)
    except Exception:
        return {}


def get_video_duration_seconds(path: str) -> float:
    meta = _ffprobe_json(path)
    if not meta:
        return 0.0
    try:
        duration = float((meta.get("format") or {}).get("duration") or 0.0)
        return max(0.0, duration)
    except Exception:
        return 0.0


def estimate_video_visual_units(path: str) -> float:
    """
    Duration-based visual estimate:
    est_tokens ~= (duration_sec * VIDEO_VISUAL_TOKENS_PER_SECOND) + VIDEO_VISUAL_BASE_TOKENS
    units = est_tokens / HISTORY_VISUAL_PART_TOKEN_BASE
    """
    try:
        duration = min(float(VIDEO_MAX_DURATION_SECONDS), get_video_duration_seconds(path))
        if duration <= 0:
            duration = 1.0
        est_tokens = (duration * float(VIDEO_VISUAL_TOKENS_PER_SECOND)) + float(VIDEO_VISUAL_BASE_TOKENS)
        denom = max(1.0, float(HISTORY_VISUAL_PART_TOKEN_BASE))
        return max(0.1, float(est_tokens / denom))
    except Exception:
        return 1.0


def _safe_scale_algo() -> str:
    # ffmpeg scale flags that are commonly available.
    if VIDEO_SCALE_ALGO in {"bicubic", "bilinear", "lanczos", "spline"}:
        return VIDEO_SCALE_ALGO
    return "bicubic"


def _video_profile_key() -> str:
    scale_algo = _safe_scale_algo()
    return f"h{int(VIDEO_TARGET_HEIGHT)}|d{int(VIDEO_MAX_DURATION_SECONDS)}|alg:{scale_algo}|v:1"


def _cache_lookup_key(original_hash: str) -> str:
    return f"{original_hash}|{_video_profile_key()}"


def _output_path_for_cache_key(cache_key: str, ext: str = ".mp4", target_dir: str = VIDEO_STORE_DIR) -> str:
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    safe_ext = str(ext or ".mp4").lower()
    if not safe_ext.startswith("."):
        safe_ext = f".{safe_ext}"
    return os.path.join(target_dir, f"{digest}_opt{safe_ext}")


def _is_sticker_video_source(path: str) -> bool:
    norm = str(path or "").replace("\\", "/").lower()
    base = os.path.basename(norm)
    sticker_dir = str(STICKER_VIDEO_STORE_DIR).replace("\\", "/").lower()
    return (
        base.startswith("sticker_")
        or "/stickers/" in norm
        or (sticker_dir and sticker_dir in norm)
    )


def _normalized_source_ext(path: str) -> str:
    ext = os.path.splitext(str(path or ""))[1].lower()
    if ext in {".mp4", ".webm", ".mov", ".m4v"}:
        return ext
    return ".mp4"


def _mime_for_video_path(path: str) -> str:
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type and mime_type.startswith("video/"):
        return mime_type
    return "video/mp4"


def _video_collage_cache_paths(video_hash: str) -> Tuple[str, str]:
    # v4: fixed 3x3 collage with interval-ms sampling + white padding slots.
    return (
        os.path.join(ANIMATED_COLLAGE_CACHE_DIR, f"vid_v4_{video_hash}.jpg"),
        os.path.join(ANIMATED_COLLAGE_CACHE_DIR, f"vid_v4_{video_hash}.json"),
    )


def get_video_collage_payload(path: str) -> Optional[Tuple[bytes, str, int]]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None

    _, _, video_hash = read_video_bytes(path)
    cache_path, meta_path = _video_collage_cache_paths(video_hash)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached_data = f.read()
            frame_count = 0
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                frame_count = int(meta.get("frame_count") or 0)
            return cached_data, "image/jpeg", frame_count
        except Exception:
            pass

    duration = min(3.0, max(0.1, get_video_duration_seconds(path)))
    target_frames = 9
    cols, rows = 3, 3

    try:
        with tempfile.TemporaryDirectory(prefix="vid_frames_", dir=ANIMATED_COLLAGE_CACHE_DIR) as td:
            pattern = os.path.join(td, "f_%03d.jpg")
            extract_cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                path,
                "-vf",
                # Keep original extracted frames (no pre-scaling) so collage uses raw source detail.
                "fps=24",
                "-vsync",
                "0",
                pattern,
            ]
            proc = subprocess.run(extract_cmd, capture_output=True, text=True, timeout=60)
            if proc.returncode != 0:
                return None

            candidates = sorted(
                os.path.join(td, n)
                for n in os.listdir(td)
                if n.lower().endswith(".jpg")
            )
            if not candidates:
                return None

            total = len(candidates)
            step_ms = max(1, int((duration * 1000.0) / float(target_frames)))
            timestamps_ms = [i * step_ms for i in range(target_frames)]

            selected_paths = []
            for ts_ms in timestamps_ms:
                t_sec = float(ts_ms) / 1000.0
                if t_sec >= duration:
                    selected_paths.append(None)
                    continue
                # Map time position to closest extracted frame index.
                idx = int(round((t_sec / max(duration, 1e-6)) * max(0, total - 1)))
                idx = max(0, min(total - 1, idx))
                selected_paths.append(candidates[idx] if idx < total else None)

            # Determine cell size from available frames; fallback sane default.
            cell_w = 256
            cell_h = 256
            for sp in selected_paths:
                if not sp:
                    continue
                try:
                    with Image.open(sp) as im:
                        cell_w, cell_h = im.convert("RGB").size
                    break
                except Exception:
                    continue

            # Tighter separators/border so grid feels compact.
            gap = 2
            pad = 2
            border = 1
            bg_color = (245, 245, 245)
            border_color = (180, 180, 180)
            collage_w = (pad * 2) + (cols * cell_w) + ((cols - 1) * gap)
            collage_h = (pad * 2) + (rows * cell_h) + ((rows - 1) * gap)
            collage = Image.new("RGB", (collage_w, collage_h), bg_color)
            draw = ImageDraw.Draw(collage)

            for i, sp in enumerate(selected_paths[: target_frames]):
                x = pad + (i % cols) * (cell_w + gap)
                y = pad + (i // cols) * (cell_h + gap)
                # White placeholder when timestamp is out of duration or frame unavailable.
                frame = Image.new("RGB", (cell_w, cell_h), (255, 255, 255))
                if sp:
                    try:
                        with Image.open(sp) as im:
                            loaded = im.convert("RGB")
                            if loaded.size != (cell_w, cell_h):
                                loaded = loaded.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
                            frame = loaded
                    except Exception:
                        pass
                collage.paste(frame, (x, y))
                # Border as clear separator between frames.
                draw.rectangle(
                    (x - border, y - border, x + cell_w + border - 1, y + cell_h + border - 1),
                    outline=border_color,
                    width=border,
                )

            # Final output budget: normalize collage into 768x768.
            if collage.size != (768, 768):
                collage = collage.resize((768, 768), Image.Resampling.LANCZOS)

            collage.save(cache_path, format="JPEG", quality=88, optimize=True)

        if not os.path.exists(cache_path) or os.path.getsize(cache_path) < 128:
            return None
        with open(cache_path, "rb") as f:
            collage_data = f.read()
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "source_hash": video_hash,
                        "frame_count": target_frames,
                        "duration_sec": round(float(duration), 3),
                        "grid": f"{cols}x{rows}",
                        "sample_mode": "fps24+time_ms+pad_white+border",
                        "step_ms": int(step_ms),
                        "timestamps_ms": timestamps_ms,
                    },
                    f,
                )
        except Exception:
            pass
        return collage_data, "image/jpeg", target_frames
    except Exception as e:
        logger.warning(f"[VIDEO-COLLAGE] Failed: {e}")
        return None


def _lookup_cached_video(db, cache_key: str) -> Optional[str]:
    try:
        row = catalog.get_video_cache(cache_key)
        if not row:
            return None
        cached_path = str(row.get("optimized_path") or "").strip()
        if not cached_path or not os.path.exists(cached_path):
            return None
        return cached_path
    except Exception as e:
        logger.warning(f"[VIDEO-CACHE] DB lookup failed: {e}")
        return None


def _upsert_video_cache(db, cache_key: str, optimized_path: str, mime_type: str):
    try:
        catalog.upsert_video_cache(cache_key, optimized_path, mime_type)
    except Exception as e:
        logger.warning(f"[VIDEO-CACHE] DB upsert failed: {e}")


def prepare_video_for_chat(db, src_path: str) -> str:
    _, _, original_hash = read_video_bytes(src_path)
    cache_key = _cache_lookup_key(original_hash)
    cached = _lookup_cached_video(db, cache_key)
    if cached:
        return cached

    source_ext = _normalized_source_ext(src_path)
    is_sticker_video = _is_sticker_video_source(src_path)
    target_dir = STICKER_VIDEO_STORE_DIR if is_sticker_video else VIDEO_STORE_DIR
    os.makedirs(target_dir, exist_ok=True)
    optimized_out_path = _output_path_for_cache_key(cache_key, ".mp4", target_dir=target_dir)
    fallback_out_path = _output_path_for_cache_key(cache_key, source_ext, target_dir=target_dir)

    # Keep sticker video in original format (typically .webm) for fidelity.
    if is_sticker_video:
        logger.info("[VIDEO] Sticker video kept ephemeral (not persisted to storage).")
        return src_path

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        if not os.path.exists(fallback_out_path):
            shutil.copyfile(src_path, fallback_out_path)
        _upsert_video_cache(db, cache_key, fallback_out_path, _mime_for_video_path(fallback_out_path))
        return fallback_out_path

    scale_algo = _safe_scale_algo()
    target_h = max(120, int(VIDEO_TARGET_HEIGHT))
    vf = f"scale=-2:{target_h}:flags={scale_algo}"
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        src_path,
        "-t",
        str(max(1, VIDEO_MAX_DURATION_SECONDS)),
        "-vf",
        vf,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        optimized_out_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if (
            proc.returncode != 0
            or not os.path.exists(optimized_out_path)
            or os.path.getsize(optimized_out_path) < 256
        ):
            logger.warning("[VIDEO] ffmpeg optimize failed, fallback to source copy.")
            if os.path.exists(optimized_out_path):
                try:
                    os.remove(optimized_out_path)
                except OSError:
                    pass
            if not os.path.exists(fallback_out_path):
                shutil.copyfile(src_path, fallback_out_path)
    except Exception as e:
        logger.warning(f"[VIDEO] ffmpeg optimize exception, fallback to source copy: {e}")
        if os.path.exists(optimized_out_path):
            try:
                os.remove(optimized_out_path)
            except OSError:
                pass
        if not os.path.exists(fallback_out_path):
            shutil.copyfile(src_path, fallback_out_path)

    final_path = optimized_out_path if os.path.exists(optimized_out_path) else fallback_out_path
    final_mime = _mime_for_video_path(final_path)
    meta = _ffprobe_json(final_path)
    if meta:
        try:
            duration = float((meta.get("format") or {}).get("duration") or 0.0)
            logger.info(
                "[VIDEO] Optimized hash=%s duration=%.2fs path=%s",
                original_hash[:12],
                duration,
                final_path,
            )
        except Exception:
            pass

    _upsert_video_cache(db, cache_key, final_path, final_mime)
    return final_path


def store_video_embedding(chat_handler, analyzer, memory_manager, video_path: str, caption: str):
    logger.info("[VIDEO-EMBED] Dinonaktifkan: multimodal embedding telah dihapus.")
    return


def generate_video_description(chat_handler, video_path: str, user_caption: str, extra_context: str = "") -> str:
    _, _, video_hash = read_video_bytes(video_path)
    cached_desc = catalog.get_video_description(video_hash)
    if cached_desc and not str(extra_context or "").strip():
        desc = str(cached_desc).strip()
        return str(desc or "").strip() or "User mengirim video."

    description = str(user_caption or "").strip()[:220] or "User mengirim video."
    extra = str(extra_context or "").strip()
    if extra:
        description = f"{description} | Konteks: {extra[:180]}"
    catalog.upsert_video_description(video_hash, description, video_path)
    return description
