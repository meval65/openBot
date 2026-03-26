import os
from typing import Dict, Optional

from src.config import INPUT_IMAGE_MEDIA_RESOLUTION, STICKER_MEDIA_RESOLUTION
from src.services.media.image_service import (
    get_image_analysis_payload,
    ingest_web_image,
    read_image_bytes,
    resize_image_if_needed,
    resolve_cached_web_image,
    store_image_permanently,
)
from src.services.media.video_service import (
    estimate_video_visual_units,
    get_video_collage_payload,
    prepare_video_for_chat,
    read_video_bytes,
)


def is_sticker_path(path: str) -> bool:
    norm = str(path or "").replace("\\", "/").lower()
    base = os.path.basename(norm)
    return (
        base.startswith("sticker_")
        or base.startswith("stickerthumb_")
        or base.startswith("sticker_thumb_")
        or "/stickers/" in norm
    )


def ingest_local_image(path: str) -> Dict:
    resize_image_if_needed(path)
    analysis_data, analysis_mime, used_collage, frame_count = get_image_analysis_payload(path)
    stored_path = store_image_permanently(path)
    _, _, media_hash = read_image_bytes(stored_path)
    return {
        "kind": "image",
        "stored_path": stored_path,
        "hash": media_hash,
        "analysis_data": analysis_data,
        "analysis_mime": analysis_mime,
        "used_collage": bool(used_collage),
        "frame_count": int(frame_count or 0),
        "is_sticker": is_sticker_path(stored_path),
        "media_resolution": STICKER_MEDIA_RESOLUTION if is_sticker_path(stored_path) else INPUT_IMAGE_MEDIA_RESOLUTION,
    }


def load_image_analysis(path: str) -> Dict:
    analysis_data, analysis_mime, used_collage, frame_count = get_image_analysis_payload(path)
    return {
        "analysis_data": analysis_data,
        "analysis_mime": analysis_mime,
        "used_collage": bool(used_collage),
        "frame_count": int(frame_count or 0),
        "is_sticker": is_sticker_path(path),
        "media_resolution": STICKER_MEDIA_RESOLUTION if is_sticker_path(path) else INPUT_IMAGE_MEDIA_RESOLUTION,
    }


def ingest_local_video(db, path: str) -> Dict:
    stored_path = prepare_video_for_chat(db, path)
    media_data, media_mime, media_hash = read_video_bytes(stored_path)
    return {
        "kind": "video",
        "stored_path": stored_path,
        "hash": media_hash,
        "analysis_data": media_data,
        "analysis_mime": media_mime,
        "is_sticker": is_sticker_path(stored_path),
        "media_resolution": STICKER_MEDIA_RESOLUTION if is_sticker_path(stored_path) else INPUT_IMAGE_MEDIA_RESOLUTION,
        "visual_units": float(estimate_video_visual_units(stored_path)),
    }


def load_video_analysis(path: str) -> Dict:
    if is_sticker_path(path):
        collage = get_video_collage_payload(path)
        if collage:
            cdata, cmime, frame_count = collage
            return {
                "analysis_data": cdata,
                "analysis_mime": cmime,
                "used_collage": True,
                "frame_count": int(frame_count or 0),
                "visual_units": float(estimate_video_visual_units(path)),
                "is_sticker": True,
                "media_resolution": STICKER_MEDIA_RESOLUTION,
            }
    vdata, vmime, _ = read_video_bytes(path)
    return {
        "analysis_data": vdata,
        "analysis_mime": vmime,
        "used_collage": False,
        "frame_count": 0,
        "visual_units": float(estimate_video_visual_units(path)),
        "is_sticker": is_sticker_path(path),
        "media_resolution": STICKER_MEDIA_RESOLUTION if is_sticker_path(path) else INPUT_IMAGE_MEDIA_RESOLUTION,
    }


def build_video_sticker_payload(db, path: str) -> Dict:
    video = ingest_local_video(db, path)
    collage = get_video_collage_payload(video["stored_path"])
    if collage:
        cdata, cmime, frame_count = collage
        return {
            **video,
            "analysis_data": cdata,
            "analysis_mime": cmime,
            "used_collage": True,
            "frame_count": int(frame_count or 0),
        }
    return {
        **video,
        "used_collage": False,
        "frame_count": 0,
    }

def resolve_web_image(chat_handler, source_url: str) -> Optional[Dict]:
    return resolve_cached_web_image(chat_handler, source_url)


def ingest_web_image_to_cache(
    chat_handler,
    source_url: str,
    raw_data: bytes,
    mime_type: str = "image/jpeg",
    image_description: str = "",
) -> Optional[Dict]:
    return ingest_web_image(
        chat_handler,
        source_url=source_url,
        raw_data=raw_data,
        mime_type=mime_type,
        image_description=image_description,
    )
