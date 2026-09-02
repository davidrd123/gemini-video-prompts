from __future__ import annotations

import base64
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from gemini_video_prompts.cli import build_resolved_image_job, generate_image_job


class _FakeTypes:
    class ImageConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs


def _png_b64() -> str:
    stream = io.BytesIO()
    Image.new("RGB", (12, 8), color=(10, 20, 30)).save(stream, "PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def test_interactions_image_job_sends_stateful_high_config_and_records_id(tmp_path: Path) -> None:
    captured: dict = {}

    class FakeInteractions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                id="interaction-test-123",
                output_image=SimpleNamespace(data=_png_b64()),
                output_text="planned the composition",
            )

    client = SimpleNamespace(interactions=FakeInteractions())
    job = build_resolved_image_job(
        prompt="Keep the front planes dark and the rim narrow.",
        model="gemini-3-pro-image",
        aspect_ratio="21:9",
        image_size="1K",
        temperature=1.0,
        num_outputs=1,
        out_root=str(tmp_path / "out"),
        config={
            "api": "interactions",
            "thinking_level": "high",
            "store": True,
            "previous_interaction_id": "interaction-parent-1",
        },
    )

    result = generate_image_job(
        client=client,
        gtypes=_FakeTypes,
        batch_path=tmp_path / "<inline>",
        job=job,
        run_day_dir=tmp_path / "day",
    )

    assert captured == {
        "model": "gemini-3-pro-image",
        "input": "Keep the front planes dark and the rim narrow.",
        "store": True,
        "response_format": {"type": "image", "aspect_ratio": "21:9", "image_size": "1K"},
        "generation_config": {"temperature": 1.0, "thinking_level": "high"},
        "previous_interaction_id": "interaction-parent-1",
    }
    assert result["interaction_ids"] == ["interaction-test-123"]
    assert result["resolved_params"]["api"] == "interactions"
    assert result["resolved_params"]["thinking_level"] == "high"
    assert result["resolved_params"]["store"] is True
    assert result["outputs"][0]["width"] == 12
    assert result["outputs"][0]["height"] == 8


def test_previous_interaction_requires_interactions_api(tmp_path: Path) -> None:
    job = build_resolved_image_job(
        prompt="continue",
        config={"previous_interaction_id": "interaction-parent-1"},
        out_root=str(tmp_path / "out"),
    )

    with pytest.raises(RuntimeError, match="previous_interaction_id requires"):
        generate_image_job(
            client=SimpleNamespace(),
            gtypes=_FakeTypes,
            batch_path=tmp_path / "<inline>",
            job=job,
            run_day_dir=tmp_path / "day",
        )


def test_interactions_image_job_sends_reference_as_inline_image(tmp_path: Path) -> None:
    captured: dict = {}

    class FakeInteractions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                id="interaction-image-edit-1",
                output_image=SimpleNamespace(data=_png_b64()),
                output_text=None,
            )

    reference = tmp_path / "reference.png"
    reference_bytes = base64.b64decode(_png_b64())
    reference.write_bytes(reference_bytes)
    client = SimpleNamespace(interactions=FakeInteractions())
    job = build_resolved_image_job(
        prompt="Relight this image without changing the camera or objects.",
        model="gemini-3.1-flash-image",
        image=str(reference),
        num_outputs=1,
        out_root=str(tmp_path / "out"),
        config={"api": "interactions", "thinking_level": "high", "store": True},
    )

    result = generate_image_job(
        client=client,
        gtypes=_FakeTypes,
        batch_path=tmp_path / "<inline>",
        job=job,
        run_day_dir=tmp_path / "day",
    )

    assert captured["input"][0] == {
        "type": "text",
        "text": "Relight this image without changing the camera or objects.",
    }
    assert captured["input"][1]["type"] == "image"
    assert captured["input"][1]["mime_type"] == "image/png"
    assert base64.b64decode(captured["input"][1]["data"]) == reference_bytes
    assert result["interaction_ids"] == ["interaction-image-edit-1"]
    assert result["input_count"] == 1
