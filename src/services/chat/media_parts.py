from google.genai import types


def to_media_resolution(level: str):
    key = f"MEDIA_RESOLUTION_{str(level or '').strip().upper()}"
    return getattr(types.MediaResolution, key, None)


def should_use_media_resolution(model_name: str) -> bool:
    return "gemini" in str(model_name or "").lower()


def part_from_bytes_with_resolution(data: bytes, mime_type: str, level: str, model_name: str):
    if not should_use_media_resolution(model_name):
        return types.Part.from_bytes(data=data, mime_type=mime_type)
    resolved = to_media_resolution(level)
    if resolved is None:
        return types.Part.from_bytes(data=data, mime_type=mime_type)
    try:
        return types.Part.from_bytes(
            data=data,
            mime_type=mime_type,
            media_resolution=resolved,
        )
    except Exception:
        return types.Part.from_bytes(data=data, mime_type=mime_type)
