import os
import uuid
import mimetypes
import datetime
import logging
import json
import threading
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict
import re

import PIL.Image
import PIL.ImageDraw
import PIL.ImageSequence
import numpy as np

from src.config import (
    ANIMATED_COLLAGE_CACHE_DIR,
    IMAGE_STORE_DIR,
    MAX_IMAGE_PIXELS,
    STICKER_STORE_DIR,
    STICKER_STATIC_STORE_DIR,
)
from . import catalog

logger = logging.getLogger(__name__)

os.makedirs(IMAGE_STORE_DIR, exist_ok=True)
os.makedirs(STICKER_STORE_DIR, exist_ok=True)
os.makedirs(STICKER_STATIC_STORE_DIR, exist_ok=True)
os.makedirs(ANIMATED_COLLAGE_CACHE_DIR, exist_ok=True)
TARGET_MEGAPIXELS = 0.85
TILE_SIZE = 1024
TILING_RATIO_THRESHOLD = 2.5
TILING_SHORT_SIDE_THRESHOLD = 1024
_MEDIA_CACHE_LOCK = threading.RLock()
_STICKER_ID_RE = re.compile(r"^sticker_([A-Za-z0-9_-]{8,})", re.IGNORECASE)


def _has_vector_payload(vector) -> bool:
    if vector is None:
        return False
    if hasattr(vector, "size"):
        try:
            return int(vector.size) > 0
        except Exception:
            pass
    if isinstance(vector, (list, tuple)):
        return len(vector) > 0
    return True


def _get_cache_value(chat_handler, key: str):
    try:
        row = catalog.get_image_description(key)
        if not row:
            return None
        return str(row.get("description") or "").strip() or None
    except Exception as e:
        logger.warning(f"[IMG-CACHE] Cache read failed: {e}")
        return None


def _set_cache_value(chat_handler, key: str, value: str, file_path: str = ""):
    try:
        catalog.upsert_image_description(key, str(value or "").strip(), str(file_path or "").strip())
    except Exception as e:
        logger.warning(f"[IMG-CACHE] Cache write failed: {e}")


def read_image_bytes(path: str):
    with open(path, "rb") as f:
        data = f.read()
    mime_type, _ = mimetypes.guess_type(path)
    if not mime_type:
        mime_type = "image/jpeg"
    media_hash = _compute_media_hash(path=path, data=data)
    return data, mime_type, media_hash


def _extract_sticker_unique_id_from_path(path: str) -> str:
    base = os.path.splitext(os.path.basename(str(path or "").strip()))[0]
    match = _STICKER_ID_RE.match(base)
    if not match:
        return ""
    return str(match.group(1) or "").strip().lower()


def _compute_phash_from_bytes(data: bytes, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    # FFT-based perceptual hash (pHash-like), robust to minor resizes/compression.
    size = int(hash_size * highfreq_factor)
    with PIL.Image.open(io.BytesIO(data)) as img:
        gray = img.convert("L").resize((size, size), PIL.Image.LANCZOS)
        pixels = np.asarray(gray, dtype=np.float32)

    freq = np.fft.fft2(pixels)
    low = np.abs(freq[:hash_size, :hash_size])
    flat = low.flatten()
    if flat.size <= 1:
        return "0" * (hash_size * hash_size // 4)
    med = np.median(flat[1:])  # skip DC component
    bits = (flat > med).astype(np.uint8)
    bit_str = "".join("1" if b else "0" for b in bits.tolist())
    hex_len = (len(bit_str) + 3) // 4
    return f"{int(bit_str, 2):0{hex_len}x}"


def _compute_media_hash(path: str, data: bytes) -> str:
    if _is_sticker_image_path(path):
        sid = _extract_sticker_unique_id_from_path(path)
        if not sid:
            raise ValueError("Sticker hash requires telegram file_unique_id.")
        return sid
    return _compute_phash_from_bytes(data)


def resize_image_if_needed(path: str, max_dim: int = MAX_IMAGE_PIXELS) -> None:
    try:
        with PIL.Image.open(path) as img:
            if getattr(img, "is_animated", False):
                logger.info(f"[IMG-RESIZE] Skip animated image: {path}")
                return
            if max(img.size) > max_dim:
                img_converted = img.convert("RGB")
                img_converted.thumbnail((max_dim, max_dim), PIL.Image.LANCZOS)
                ext = os.path.splitext(path)[1].lower()
                if ext in (".jpg", ".jpeg"):
                    img_converted.save(path, format="JPEG", quality=85, optimize=True)
                elif ext == ".webp":
                    img_converted.save(path, format="WEBP", quality=85, method=6)
                elif ext == ".png":
                    img_converted.save(path, format="PNG", optimize=True)
                else:
                    # Fallback for unknown extensions.
                    img_converted.save(path, format="JPEG", quality=85, optimize=True)
                logger.info(f"[IMG-RESIZE] Resized to max {max_dim}px (ext={ext or 'n/a'}): {path}")
    except Exception as e:
        logger.warning(f"[IMG-RESIZE] Failed: {e}")


def _is_sticker_image_path(path: str) -> bool:
    base_name = os.path.basename(str(path or "")).lower()
    norm = str(path or "").replace("\\", "/").lower()
    return (
        base_name.startswith("sticker_")
        or base_name.startswith("sticker_thumb_")
        or base_name.startswith("stickerthumb_")
        or "/stickers/" in norm
    )


def is_animated_image_file(path: str) -> bool:
    try:
        with PIL.Image.open(path) as img:
            return bool(getattr(img, "is_animated", False))
    except Exception:
        return False


def _sample_frame_indexes(frame_count: int, max_frames: int = 6) -> list[int]:
    if frame_count <= 0:
        return [0]
    if frame_count <= max_frames:
        return list(range(frame_count))
    if max_frames == 1:
        return [0]
    step = (frame_count - 1) / float(max_frames - 1)
    picks = []
    seen = set()
    for i in range(max_frames):
        idx = int(round(i * step))
        if idx not in seen:
            picks.append(idx)
            seen.add(idx)
    if len(picks) < max_frames:
        for idx in range(frame_count):
            if idx not in seen:
                picks.append(idx)
                seen.add(idx)
            if len(picks) >= max_frames:
                break
    return picks[:max_frames]


def _make_animated_collage_bytes(image_path: str) -> tuple[bytes, str, int]:
    with PIL.Image.open(image_path) as img:
        frame_count = int(getattr(img, "n_frames", 1) or 1)
        indexes = _sample_frame_indexes(frame_count, 6)
        frames = []
        for idx in indexes:
            try:
                img.seek(idx)
            except EOFError:
                break
            frame = img.convert("RGBA")
            frame.thumbnail((256, 256), PIL.Image.LANCZOS)
            canvas = PIL.Image.new("RGB", frame.size, (18, 20, 28))
            canvas.paste(frame, mask=frame.split()[-1] if frame.mode == "RGBA" else None)
            frames.append(canvas)

        if not frames:
            raise ValueError("No frames extracted from animated image")

        cols = 2 if len(frames) <= 4 else 3
        rows = (len(frames) + cols - 1) // cols
        cell_w = max(frame.width for frame in frames)
        cell_h = max(frame.height for frame in frames)
        gap = 8
        pad = 10
        collage = PIL.Image.new(
            "RGB",
            (
                pad * 2 + cols * cell_w + max(0, cols - 1) * gap,
                pad * 2 + rows * cell_h + max(0, rows - 1) * gap,
            ),
            (12, 14, 20),
        )
        draw = PIL.ImageDraw.Draw(collage)
        for i, frame in enumerate(frames):
            row = i // cols
            col = i % cols
            x = pad + col * (cell_w + gap)
            y = pad + row * (cell_h + gap)
            collage.paste(frame, (x, y))
            draw.rectangle((x, y, x + 24, y + 20), fill=(0, 0, 0))
            draw.text((x + 7, y + 4), str(i + 1), fill=(255, 255, 255))

        out = io.BytesIO()
        collage.save(out, format="JPEG", quality=88, optimize=True)
        return out.getvalue(), "image/jpeg", len(frames)


def get_cached_animated_collage_payload(image_path: str) -> tuple[bytes, str, int]:
    data, _, sha256_hex = read_image_bytes(image_path)
    cache_path = os.path.join(ANIMATED_COLLAGE_CACHE_DIR, f"{sha256_hex}.jpg")
    meta_path = os.path.join(ANIMATED_COLLAGE_CACHE_DIR, f"{sha256_hex}.json")
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

    collage_data, collage_mime, frame_count = _make_animated_collage_bytes(image_path)
    try:
        with open(cache_path, "wb") as f:
            f.write(collage_data)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"source_hash": sha256_hex, "frame_count": frame_count}, f)
    except Exception as e:
        logger.warning(f"[ANIM-COLLAGE] Cache write failed: {e}")
    return collage_data, collage_mime, frame_count


def get_image_analysis_payload(image_path: str) -> tuple[bytes, str, bool, int]:
    if is_animated_image_file(image_path):
        data, mime_type, frame_count = get_cached_animated_collage_payload(image_path)
        return data, mime_type, True, frame_count
    data, mime_type, _ = read_image_bytes(image_path)
    return data, mime_type, False, 0


def store_image_permanently(src_path: str) -> str:
    try:
        data, mime_type, sha256_hex = read_image_bytes(src_path)
        ext = os.path.splitext(src_path)[1].lower() or ".jpg"
        base_name = os.path.basename(src_path).lower()
        is_sticker = (
            base_name.startswith("sticker_")
            or base_name.startswith("sticker_thumb_")
            or base_name.startswith("stickerthumb_")
        )
        if is_sticker:
            logger.info(f"[IMG-STORE] Sticker kept ephemeral (not persisted to storage): {src_path}")
            return src_path
        target_dir = STICKER_STATIC_STORE_DIR if is_sticker else IMAGE_STORE_DIR
        os.makedirs(target_dir, exist_ok=True)
        dest_path = os.path.join(target_dir, f"{sha256_hex}{ext}")

        if os.path.exists(dest_path):
            try:
                os.remove(src_path)
            except OSError:
                pass
            logger.info(f"[IMG-STORE] Duplicate detected, reusing: {dest_path}")
            return dest_path

        with open(dest_path, "wb") as f:
            f.write(data)

        try:
            os.remove(src_path)
        except OSError:
            pass

        logger.info(f"[IMG-STORE] Stored permanently: {dest_path}")
        return dest_path
    except Exception as e:
        logger.error(f"[IMG-STORE] Failed: {e}")
        return src_path


def _upsert_web_image_source(source_url: str, media_hash: str, description: str = ""):
    source = str(source_url or "").strip()
    if not source:
        return
    catalog.upsert_web_image_source(source, media_hash, str(description or "").strip())


def _touch_media_description_usage(media_hash: str):
    row = catalog.get_image_description(media_hash)
    if row is None:
        return


def _upsert_web_image_raw_hash(raw_hash: str, media_hash: str):
    catalog.upsert_web_raw_hash(raw_hash, media_hash)


def _resolve_media_description_row_by_hash(media_hash: str) -> Optional[Dict]:
    h = str(media_hash or "").strip().lower()
    if not h:
        return None
    row = catalog.get_image_description(h)
    if row is None:
        return None
    return {
        "media_hash": h,
        "file_path": str(row.get("file_path") or "").strip(),
        "description": str(row.get("description") or "").strip(),
    }


def _reuse_existing_image(media_hash: str, source_url: str, description: str) -> Optional[Dict]:
    row = _resolve_media_description_row_by_hash(media_hash)
    if not row:
        return None
    file_path = row.get("file_path", "")
    if not file_path or not os.path.exists(file_path):
        return None
    data, stored_mime, actual_hash = read_image_bytes(file_path)
    _touch_media_description_usage(row["media_hash"])
    _upsert_web_image_source(source_url, row["media_hash"], description or row.get("description", ""))
    return {
        "data": data,
        "mime_type": stored_mime,
        "hash": actual_hash,
        "path": file_path,
        "description": description or row.get("description", ""),
        "reused_from_url": True,
    }


def _normalize_web_image_for_llm(raw_data: bytes, mime_type: str) -> Optional[Dict]:
    try:
        with PIL.Image.open(io.BytesIO(raw_data)) as img:
            frame = img.convert("RGB")
            orig_w, orig_h = frame.size
            if max(orig_w, orig_h) > int(MAX_IMAGE_PIXELS):
                frame.thumbnail((int(MAX_IMAGE_PIXELS), int(MAX_IMAGE_PIXELS)), PIL.Image.LANCZOS)
            out = io.BytesIO()
            frame.save(out, format="JPEG", quality=85, optimize=True)
            data = out.getvalue()
            file_hash = _compute_phash_from_bytes(data)
            return {
                "data": data,
                "mime_type": "image/jpeg",
                "hash": file_hash,
                "width": int(frame.width),
                "height": int(frame.height),
            }
    except Exception as e:
        logger.warning(f"[WEB-IMG] Normalize failed: {e}")
        return None


def ingest_web_image(
    chat_handler,
    source_url: str,
    raw_data: bytes,
    mime_type: str = "image/jpeg",
    image_description: str = "",
) -> Optional[Dict]:
    """
    Ingest image from web-search result.
    Flow:
    1) URL lookup in cache DB.
    2) If hit + file exists: reuse existing image.
    3) Else normalize (resize/downscale), hash, dedup by hash, store file.
    4) Persist media description + URL->hash mapping.
    """
    src = str(source_url or "").strip()
    if not raw_data:
        return None
    try:
        if src:
            reused_by_url = resolve_cached_web_image(chat_handler, src)
            if reused_by_url:
                if image_description:
                    _upsert_web_image_source(src, reused_by_url["hash"], image_description)
                    reused_by_url["description"] = image_description
                return reused_by_url

        raw_hash = _compute_phash_from_bytes(raw_data)
        mapped_hash = catalog.get_web_raw_hash(raw_hash)
        if mapped_hash:
            reused = _reuse_existing_image(mapped_hash, src, image_description)
            if reused:
                _upsert_web_image_raw_hash(raw_hash, mapped_hash)
                return reused

        direct_same_hash = _reuse_existing_image(raw_hash, src, image_description)
        if direct_same_hash:
            _upsert_web_image_raw_hash(raw_hash, raw_hash)
            return direct_same_hash

        normalized = _normalize_web_image_for_llm(raw_data, mime_type)
        if not normalized:
            return None

        media_hash = str(normalized["hash"]).strip().lower()
        dest_path = os.path.join(IMAGE_STORE_DIR, f"{media_hash}.jpg")
        os.makedirs(IMAGE_STORE_DIR, exist_ok=True)
        if not os.path.exists(dest_path):
            with open(dest_path, "wb") as f:
                f.write(normalized["data"])

        _set_cache_value(chat_handler, media_hash, image_description or "", dest_path)
        _upsert_web_image_source(src, media_hash, image_description)
        _upsert_web_image_raw_hash(raw_hash, media_hash)
        return {
            "data": normalized["data"],
            "mime_type": normalized["mime_type"],
            "hash": media_hash,
            "path": dest_path,
            "description": image_description or "",
            "reused_from_url": False,
        }
    except Exception as e:
        logger.warning(f"[WEB-IMG] Ingest failed: {e}")
        return None


def resolve_cached_web_image(chat_handler, source_url: str) -> Optional[Dict]:
    src = str(source_url or "").strip()
    if not src:
        return None
    try:
        source_row = catalog.get_web_image_source(src)
        if not source_row:
            return None
        media_hash = str(source_row.get("media_hash") or "").strip().lower()
        desc = str(source_row.get("description") or "").strip()
        media_row = _resolve_media_description_row_by_hash(media_hash)
        file_path = str((media_row or {}).get("file_path") or "").strip()
        if not desc:
            desc = str((media_row or {}).get("description") or "").strip()
        if not media_hash or not file_path or not os.path.exists(file_path):
            return None
        data, stored_mime, actual_hash = read_image_bytes(file_path)
        _touch_media_description_usage(media_hash)
        _upsert_web_image_source(src, media_hash, desc)
        return {
            "data": data,
            "mime_type": stored_mime,
            "hash": actual_hash,
            "path": file_path,
            "description": desc,
            "reused_from_url": True,
        }
    except Exception as e:
        logger.warning(f"[WEB-IMG] Resolve cached URL failed: {e}")
        return None


def _needs_downscale(w: int, h: int) -> bool:
    return (w * h) > (TARGET_MEGAPIXELS * 1.2 * 1_000_000)


def _needs_tiling(w: int, h: int) -> bool:
    long_side = max(w, h)
    short_side = min(w, h)
    return (long_side / short_side) > TILING_RATIO_THRESHOLD or short_side > TILING_SHORT_SIDE_THRESHOLD


def _downscale_image(img: PIL.Image.Image) -> PIL.Image.Image:
    w, h = img.size
    total_px = w * h
    target_px = int(TARGET_MEGAPIXELS * 1_000_000)
    if total_px <= target_px:
        return img
    scale = (target_px / total_px) ** 0.5
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), PIL.Image.LANCZOS)


def _slice_into_tiles(img: PIL.Image.Image) -> list:
    w, h = img.size
    tiles = []
    for top in range(0, h, TILE_SIZE):
        for left in range(0, w, TILE_SIZE):
            box = (left, top, min(left + TILE_SIZE, w), min(top + TILE_SIZE, h))
            tiles.append(img.crop(box))
    return tiles


def _save_tile(tile: PIL.Image.Image, base_dir: str, group_hash: str, idx: int) -> str:
    fname = f"{group_hash}_tile{idx}.jpg"
    path = os.path.join(base_dir, fname)
    tile_rgb = tile.convert("RGB")
    tile_rgb.save(path, format="JPEG", quality=85, optimize=True)
    return path


def _compose_visual_description(
    media_label: str,
    user_caption: str,
    extra_context: str = "",
    detail_hint: str = "",
) -> str:
    parts = []
    caption = str(user_caption or "").strip()
    if caption:
        parts.append(caption[:220])
    context = str(extra_context or "").strip()
    if context:
        parts.append(f"Konteks: {context[:180]}")
    if detail_hint:
        parts.append(detail_hint[:180])
    if parts:
        return " | ".join(parts)
    return f"User mengirim {media_label}."


def generate_image_description(chat_handler, image_path: str, user_caption: str, extra_context: str = "") -> str:
    _, _, sha256_hex = read_image_bytes(image_path)
    cached = _get_cache_value(chat_handler, sha256_hex)
    if cached and not str(extra_context or "").strip():
        logger.info(f"[IMG-CACHE] Found cached description for {sha256_hex}")
        return cached

    _, _, animated_mode, animated_frame_count = get_image_analysis_payload(image_path)
    detail_hint = ""
    if animated_mode:
        media_label = "sticker animasi" if _is_sticker_image_path(image_path) else "gambar animasi"
        if animated_frame_count > 0:
            detail_hint = f"Frame ringkasan: {int(animated_frame_count)}."
    else:
        media_label = "sebuah gambar"

    description = _compose_visual_description(
        media_label=media_label,
        user_caption=user_caption,
        extra_context=extra_context,
        detail_hint=detail_hint,
    )
    _set_cache_value(chat_handler, sha256_hex, description, image_path)
    return description


def _describe_tile(chat_handler, tile_path: str, caption: str, tile_idx: int, total: int) -> str:
    _, _, tile_hash = read_image_bytes(tile_path)
    cache_key = f"tile:{tile_hash}"
    cached = _get_cache_value(chat_handler, cache_key)
    if cached:
        return cached

    result = _compose_visual_description(
        media_label=f"bagian {tile_idx+1} dari gambar",
        user_caption=caption,
        detail_hint=f"Tile {tile_idx+1}/{total}.",
    )
    _set_cache_value(chat_handler, cache_key, result, tile_path)
    return result


def _register_group(db, group_id: str, file_hash: str, file_path: str,
                    media_type: str, description: str,
                    tile_count: int, w: int, h: int, orig_w: int, orig_h: int) -> str:
    try:
        cursor = db.get_cursor()
        cursor.execute("SELECT id FROM memory_groups WHERE file_hash = ?", (file_hash,))
        row = cursor.fetchone()
        if row:
            existing_id = row[0]
            logger.info(f"[IMG-GROUP] Reusing existing group {existing_id} for hash {file_hash}")
            return existing_id

        cursor.execute("""
            INSERT INTO memory_groups
                (id, file_hash, file_path, media_type, description, tile_count, width, height, original_width, original_height, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            group_id, file_hash, file_path, media_type, description,
            tile_count, w, h, orig_w, orig_h,
            datetime.datetime.now().isoformat()
        ))
        db.commit()
        return group_id
    except Exception as e:
        logger.error(f"[IMG-GROUP] Failed to register group: {e}")
        return group_id


def store_image_embedding(chat_handler, analyzer, memory_manager, image_path: str, caption: str, media_type: str = "image"):
    logger.info("[IMAGE-EMBED] Dinonaktifkan: multimodal embedding telah dihapus.")
    return


def _process_single(chat_handler, analyzer, memory_manager, image_path: str,
                    img: PIL.Image.Image, caption: str, media_type: str, orig_w: int, orig_h: int):
    needs_scale = _needs_downscale(orig_w, orig_h)
    if needs_scale:
        img_proc = _downscale_image(img.copy())
        img_proc.convert("RGB").save(image_path, format="JPEG", quality=85, optimize=True)
        img_proc.close()
    else:
        img_proc = img.copy()

    final_w, final_h = img_proc.size
    
    # Read the NEW file_hash potentially updated by downscaling
    _, _, file_hash = read_image_bytes(image_path)
    group_id = str(uuid.uuid4())

    desc_result = {}
    embed_result = {}

    def do_desc():
        desc_result["v"] = generate_image_description(chat_handler, image_path, caption)

    def do_embed():
        embed_result["v"] = analyzer.get_embedding(image_path, content_type="image")

    t1 = threading.Thread(target=do_desc)
    t2 = threading.Thread(target=do_embed)
    t1.start(); t2.start()
    t1.join(); t2.join()

    description = desc_result.get("v", caption or "User mengirim sebuah gambar.")
    embedding = embed_result.get("v", [])

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    summary = f"[{timestamp}] [IMG_PATH:{image_path}] {description}"

    actual_group_id = _register_group(
        chat_handler.cache_db, group_id, file_hash, image_path,
        media_type, description, 1, final_w, final_h, orig_w, orig_h
    )

    if _has_vector_payload(embedding):
        memory_manager.add_memory_with_group(
            summary=summary, m_type="general", priority=0.5,
            embedding=embedding, group_id=actual_group_id, embedding_namespace="image"
        )
        logger.info(f"[IMAGE-EMBED] Single stored: {summary[:80]}")


def _process_tiled(chat_handler, analyzer, memory_manager, image_path: str,
                   img: PIL.Image.Image, caption: str, media_type: str, orig_w: int, orig_h: int):
    _, _, file_hash = read_image_bytes(image_path)
    group_id = str(uuid.uuid4())

    tiles = _slice_into_tiles(img)
    tile_paths = []
    for i, tile in enumerate(tiles):
        tile_path = _save_tile(tile, IMAGE_STORE_DIR, file_hash, i)
        tile_paths.append(tile_path)

    tile_descs = [None] * len(tile_paths)
    with ThreadPoolExecutor(max_workers=min(4, len(tile_paths))) as pool:
        desc_futures = {
            pool.submit(_describe_tile, chat_handler, tp, caption, i, len(tile_paths)): i
            for i, tp in enumerate(tile_paths)
        }
        embed_future = pool.submit(analyzer.get_embeddings_batch, tile_paths)
        for f in as_completed(desc_futures):
            idx = desc_futures[f]
            try:
                tile_descs[idx] = f.result()
            except Exception:
                tile_descs[idx] = f"Bagian {idx+1} dari gambar."
        tile_vectors = embed_future.result()

    full_description = " | ".join(d for d in tile_descs if d)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")

    actual_group_id = _register_group(
        chat_handler.cache_db, group_id, file_hash, image_path,
        media_type, full_description, len(tile_paths),
        orig_w, orig_h, orig_w, orig_h
    )

    for i, (tile_path, tile_vec, tile_desc) in enumerate(zip(tile_paths, tile_vectors, tile_descs)):
        if not _has_vector_payload(tile_vec):
            continue
        summary = f"[{timestamp}] [IMG_PATH:{image_path}] [Tile {i+1}/{len(tile_paths)}] {tile_desc}"
        memory_manager.add_memory_with_group(
            summary=summary,
            m_type="general",
            priority=0.5,
            embedding=tile_vec,
            group_id=actual_group_id,
            embedding_namespace="image",
        )

    for tp in tile_paths:
        try:
            os.remove(tp)
        except OSError:
            pass

    logger.info(f"[IMAGE-EMBED] Tiled stored: {len(tile_paths)} tiles for group {group_id}")
