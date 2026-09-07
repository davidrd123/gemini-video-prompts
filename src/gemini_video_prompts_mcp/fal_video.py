"""H3 Max validation and fal queue transport. No automatic submission retries."""
from __future__ import annotations

import base64
import hashlib
import io
import mimetypes
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from dotenv import load_dotenv
from PIL import Image

from .seedance import probe_media_info

I2V = "minimax/h3-max/image-to-video"
R2V = "minimax/h3-max/reference-to-video"
T2V = "minimax/h3-max/text-to-video"
T2V_TURBO = "minimax/h3-max-turbo/text-to-video"
T2V_MODELS = (T2V, T2V_TURBO)
BILLING_NOTICE = (
    "fal H3 Max uses separately billed FAL_KEY API access, not a host subscription. "
    "Obtain user approval for the endpoint, clip count, duration, resolution, and references "
    "before setting allow_api_billing=true. A key is not spending authorization. "
    "Reference-to-video may charge for input references as well as output seconds. "
    "Check the endpoint pricing; local confirmation is not a spending cap."
)
MAX_INPUT_BYTES = 100 * 1024 * 1024  # Local inline-request limit, not a fal model limit.


def api_key() -> str:
    load_dotenv()
    key = os.getenv("FAL_KEY", "").strip()
    if not key:
        raise RuntimeError("FAL_KEY_MISSING: set FAL_KEY in .env or the process environment")
    return key


def build_request(*, prompt: str, model: str = I2V, image: str | None = None,
                  last_frame_image: str | None = None,
                  reference_images: list[str] | None = None,
                  reference_videos: list[str] | None = None,
                  reference_audios: list[str] | None = None, duration: int = 5,
                  resolution: str = "768p", aspect_ratio: str | None = None,
                  seed: int | None = None, prompt_expansion_mode: str = "balanced",
                  enable_safety_checker: bool = True) -> tuple[dict, list[dict]]:
    def invalid(message):
        raise RuntimeError("INVALID_INPUT: " + message)

    if model not in (I2V, R2V, *T2V_MODELS):
        invalid("choose a supported minimax/h3-max image, reference, or text-to-video endpoint")
    if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 50000:
        invalid("prompt must contain 1..50000 characters")
    if type(duration) is not int or not 5 <= duration <= 15:
        invalid("duration must be an integer in [5, 15]")
    if not isinstance(resolution, str) or resolution.upper() not in ("480P", "768P"):
        invalid("resolution must be 480p or 768p")
    if prompt_expansion_mode not in ("balanced", "quality"):
        invalid("documented prompt expansion modes are balanced and quality")
    if seed is not None and type(seed) is not int:
        invalid("seed must be an integer")
    if type(enable_safety_checker) is not bool:
        invalid("enable_safety_checker must be boolean")
    groups = [reference_images or [], reference_videos or [], reference_audios or []]
    if any(not isinstance(group, list) for group in groups):
        invalid("reference inputs must be lists")
    if model == I2V:
        if not image:
            invalid("image-to-video requires image; riff does not silently route to text-to-video")
        if any(groups) or aspect_ratio is not None:
            invalid("image-to-video uses image/last_frame_image and inherits the first frame ratio")
        fields = [("image_url", [image], "image", "First frame")]
        if last_frame_image:
            fields.append(("end_image_url", [last_frame_image], "image", "Last frame"))
    elif model == R2V:
        if image or last_frame_image:
            invalid("reference-to-video uses reference_* lists, not first/last frames")
        if not (groups[0] or groups[1]):
            invalid("reference-to-video requires at least one reference image or video")
        if any(len(group) > cap for group, cap in zip(groups, (9, 3, 3))) or sum(map(len, groups)) > 12:
            invalid("reference caps: 9 images, 3 videos, 3 audios, 12 files total")
        if aspect_ratio not in (None, "adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"):
            invalid("unsupported aspect_ratio")
        fields = list(zip(("reference_image_urls", "reference_video_urls", "reference_audio_urls"),
                          groups, ("image", "video", "audio"), ("Image", "Video", "Audio")))
    else:
        if image or last_frame_image or any(groups):
            invalid("text-to-video does not accept image, frame, or reference inputs")
        if aspect_ratio not in (None, "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"):
            invalid("unsupported aspect_ratio")
        fields = []
    params: dict[str, Any] = dict(prompt=prompt.strip(), duration=duration,
                                 resolution=resolution.upper(), prompt_expansion_mode=prompt_expansion_mode,
                                 enable_safety_checker=enable_safety_checker, sync_mode=False)
    if seed is not None:
        params["seed"] = seed
    if model == R2V:
        params["aspect_ratio"] = aspect_ratio or "adaptive"
    elif model in T2V_MODELS:
        params["aspect_ratio"] = aspect_ratio or "16:9"
    refs = []
    for field, paths, kind, label in fields:
        for i, path in enumerate(paths):
            if not isinstance(path, str) or not path.strip() or "://" in path or path.startswith("data:"):
                invalid("references must be local file paths; remote URLs are not accepted")
            refs.append(dict(field=field, index=i, kind=kind, path=str(Path(path).expanduser().resolve()),
                             token=f"{label} {i + 1}" if field.startswith("reference_") else label))
    return params, refs


def prepare_inputs(params: dict, references: list[dict]) -> tuple[dict, list[dict]]:
    """Encode precisely the bytes hashed; keep large data URIs out of durable records."""
    payload = dict(params)
    refs = []
    total_bytes = 0
    durations = {"video": 0.0, "audio": 0.0}
    for source in references:
        ref = dict(source)
        path = Path(ref["path"])
        if not path.is_file():
            raise RuntimeError(f"FILE_NOT_FOUND: {path}")
        if path.stat().st_size + total_bytes > MAX_INPUT_BYTES:
            raise RuntimeError("INVALID_INPUT: riff inline references exceed 100 MiB total")
        raw = path.read_bytes()
        total_bytes += len(raw)
        if not raw or total_bytes > MAX_INPUT_BYTES:
            raise RuntimeError("INVALID_INPUT: empty reference or total exceeds 100 MiB")
        if ref["kind"] == "image":
            try:
                with Image.open(io.BytesIO(raw)) as img:
                    mime = Image.MIME.get(img.format)
                    ref.update(width=img.width, height=img.height)
                    img.verify()
                if not mime:
                    raise ValueError("unknown image format")
            except Exception:
                raise RuntimeError(f"INVALID_INPUT: unreadable image {path}") from None
        else:
            mime = mimetypes.guess_type(path.name)[0]
            if path.suffix.lower() == ".m4a":
                mime = "audio/mp4"
            info = probe_media_info(str(path))
            duration = info.get("duration_s")
            if not mime or not mime.startswith(ref["kind"] + "/") or duration is None or not 2 <= duration <= 15:
                raise RuntimeError("INVALID_INPUT: video/audio references require ffprobe and 2..15 second clips")
            if ref["kind"] == "video" and not info.get("width"):
                raise RuntimeError("INVALID_INPUT: reference video has no video stream")
            if ref["kind"] == "audio" and not info.get("has_audio"):
                raise RuntimeError("INVALID_INPUT: reference audio has no audio stream")
            # Detect mutation during probing; the bytes submitted must match validation.
            if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(raw).digest():
                raise RuntimeError("INVALID_INPUT: reference changed during validation")
            durations[ref["kind"]] += duration
            if durations[ref["kind"]] > 15:
                raise RuntimeError("INVALID_INPUT: combined reference duration must be <=15s per modality")
            ref["media_info"] = info
        ref.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), hash_source="submitted_bytes")
        uri = f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")
        if ref["field"].startswith("reference_"):
            payload.setdefault(ref["field"], []).append(uri)
        else:
            payload[ref["field"]] = uri
        refs.append(ref)
    return payload, refs


def queue_request(method: str, url: str, *, payload: dict | None = None) -> dict:
    """Only send credentials to fal's queue origin; never follow redirects or retry."""
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != "queue.fal.run" or parsed.username:
        raise RuntimeError("FAL_INVALID_RESPONSE: invalid queue URL")
    try:
        with httpx.Client(timeout=60.0, follow_redirects=False) as client:
            response = client.request(method, url, json=payload,
                                      headers={"Authorization": "Key " + api_key()})
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError("expected object")
            return result
    except httpx.HTTPStatusError as exc:
        # Provider bodies may echo data URIs, URLs, or private prompt text.
        raise RuntimeError(f"FAL_HTTP_ERROR: HTTP {exc.response.status_code}") from None
    except (httpx.RequestError, ValueError):
        raise RuntimeError("FAL_REQUEST_UNCERTAIN: no usable response; do not automatically resubmit") from None


def download_video(url: str, path: Path) -> dict:
    """Download original CDN bytes without sending the fal API key."""
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not (parsed.hostname == "fal.media" or
                                         (parsed.hostname or "").endswith(".fal.media")) or parsed.username:
        raise RuntimeError("FAL_INVALID_RESPONSE: expected an HTTPS fal.media video URL")
    part = path.with_suffix(".part")
    digest = hashlib.sha256()
    size = 0
    try:
        with httpx.Client(timeout=120.0, follow_redirects=False) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with part.open("wb") as output:
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > 1024 * 1024 * 1024:
                            raise RuntimeError("FAL_DOWNLOAD_ERROR: output exceeds local 1 GiB limit")
                        output.write(chunk)
                        digest.update(chunk)
        if not size:
            raise RuntimeError("FAL_DOWNLOAD_ERROR: empty output")
        part.replace(path)
    except httpx.HTTPError:
        raise RuntimeError("FAL_DOWNLOAD_ERROR: download failed; poll the existing job to retry collection") from None
    finally:
        part.unlink(missing_ok=True)
    return dict(index=1, path=str(path), bytes=size, sha256=digest.hexdigest(),
                hash_source="downloaded_bytes", media_info=probe_media_info(str(path)))
