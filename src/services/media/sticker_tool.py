import os
import shutil
import subprocess
from typing import Optional

from PIL import Image


MAX_STATIC_BYTES = 512 * 1024
MAX_VIDEO_BYTES = 256 * 1024
MAX_SIDE = 512
VIDEO_MAX_SECONDS = 3


def _ffmpeg_bin() -> Optional[str]:
    return shutil.which("ffmpeg")


def _fit_512(w: int, h: int) -> tuple[int, int]:
    if w <= 0 or h <= 0:
        return 512, 512
    if w >= h:
        nw = MAX_SIDE
        nh = max(1, int(round((h / w) * MAX_SIDE)))
    else:
        nh = MAX_SIDE
        nw = max(1, int(round((w / h) * MAX_SIDE)))
    return nw, nh


def convert_image_to_webp_sticker(src_path: str, out_path: str) -> bool:
    try:
        with Image.open(src_path) as im:
            im = im.convert("RGBA")
            nw, nh = _fit_512(im.width, im.height)
            im = im.resize((nw, nh), Image.Resampling.LANCZOS)

            # Try quality ladder first while keeping dimension.
            for q in (95, 90, 85, 80, 75, 70, 65, 60, 55, 50):
                im.save(
                    out_path,
                    format="WEBP",
                    quality=q,
                    method=6,
                    optimize=True,
                )
                if os.path.exists(out_path) and os.path.getsize(out_path) <= MAX_STATIC_BYTES:
                    return True

            # If still too large, downscale progressively.
            scale = 0.9
            while scale >= 0.55:
                rw = max(1, int(nw * scale))
                rh = max(1, int(nh * scale))
                rim = im.resize((rw, rh), Image.Resampling.LANCZOS)
                for q in (70, 60, 50, 40):
                    rim.save(
                        out_path,
                        format="WEBP",
                        quality=q,
                        method=6,
                        optimize=True,
                    )
                    if os.path.exists(out_path) and os.path.getsize(out_path) <= MAX_STATIC_BYTES:
                        return True
                scale -= 0.1
    except Exception:
        return False
    return False


def convert_video_to_webm_sticker(src_path: str, out_path: str) -> bool:
    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        return False

    # Bitrate ladder (kbps), then fallback with reduced fps.
    bitrates = [650, 560, 480, 420, 360, 300, 250, 210, 180, 150, 120, 100, 80]
    fps_ladder = [30, 24, 20, 15, 12]

    for fps in fps_ladder:
        for br in bitrates:
            vf = (
                "scale='if(gte(iw,ih),512,-2)':'if(gte(iw,ih),-2,512)':flags=lanczos,"
                f"fps={fps}"
            )
            cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                src_path,
                "-t",
                str(VIDEO_MAX_SECONDS),
                "-an",
                "-vf",
                vf,
                "-c:v",
                "libvpx-vp9",
                "-b:v",
                f"{br}k",
                "-minrate",
                f"{int(br * 0.6)}k",
                "-maxrate",
                f"{int(br * 1.1)}k",
                "-g",
                "30",
                "-row-mt",
                "1",
                "-deadline",
                "good",
                "-cpu-used",
                "4",
                "-pix_fmt",
                "yuv420p",
                out_path,
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            except Exception:
                continue

            if proc.returncode != 0:
                continue
            if os.path.exists(out_path) and os.path.getsize(out_path) <= MAX_VIDEO_BYTES:
                return True

    return False
