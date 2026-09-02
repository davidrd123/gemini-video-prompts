from pathlib import Path

import pytest

from media_analysis_mcp import server


class _FakeClient:
    pass


class _FakeTypes:
    pass


@pytest.fixture
def image_paths(tmp_path: Path) -> list[Path]:
    pillow = server.gemini_media.require_pillow()
    paths = [tmp_path / "source.png", tmp_path / "variation.jpg"]
    pillow.new("RGB", (8, 8), color=(20, 30, 40)).save(paths[0])
    pillow.new("RGB", (8, 8), color=(80, 60, 40)).save(paths[1])
    return paths


@pytest.fixture(autouse=True)
def fake_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        server.gemini_media,
        "init_client",
        lambda: (_FakeClient(), _FakeTypes()),
    )


def test_analyze_images_keeps_sources_equal_labeled_and_ordered(
    monkeypatch: pytest.MonkeyPatch, image_paths: list[Path]
) -> None:
    captured: dict = {}

    def fake_call_unstructured(**kwargs):
        captured.update(kwargs)
        return "Both images preserve a dark foreground; only IMAGE 2 adds haze."

    monkeypatch.setattr(
        server.gemini_media, "call_unstructured", fake_call_unstructured
    )

    result = server.analyze_images(
        image_paths=[str(path) for path in image_paths],
        labels=["Source frame", "Lighting variation"],
        question="What persists, and what changes?",
        intent="Find transferable lighting relationships.",
        context="Do not rank the images.",
    )

    resolved = [str(path.resolve()) for path in image_paths]
    assert result["image_paths"] == resolved
    assert result["image_labels"] == ["Source frame", "Lighting variation"]
    assert result["thinking_level"] == "high"
    assert result["max_output_tokens"] == 65_536
    assert result["answer"].startswith("Both images")
    assert result["context_used"] == {
        "prompt": None,
        "intent": "Find transferable lighting relationships.",
        "context": "Do not rank the images.",
        "question": "What persists, and what changes?",
        "image_labels": ["Source frame", "Lighting variation"],
    }

    string_parts = [part for part in captured["contents"] if isinstance(part, str)]
    assert "IMAGE 1 — Source frame:" in string_parts
    assert "IMAGE 2 — Lighting variation:" in string_parts
    assert "equal-role source" in string_parts[-1]
    assert "Do not choose a winner" in string_parts[-1]
    assert captured["thinking_level"] == "high"
    assert captured["max_output_tokens"] == 65_536
    assert "equal evidentiary status" in captured["system_instruction"]


def test_analyze_images_defaults_labels_to_basenames(
    monkeypatch: pytest.MonkeyPatch, image_paths: list[Path]
) -> None:
    monkeypatch.setattr(
        server.gemini_media,
        "call_unstructured",
        lambda **kwargs: "A grounded collection read.",
    )

    result = server.analyze_images(
        image_paths=[str(path) for path in image_paths],
        question="Read the collection.",
    )

    assert result["image_labels"] == ["source.png", "variation.jpg"]


def test_analyze_images_allows_model_default_reasoning_controls(
    monkeypatch: pytest.MonkeyPatch, image_paths: list[Path]
) -> None:
    captured: dict = {}

    def fake_call_unstructured(**kwargs):
        captured.update(kwargs)
        return "Grounded answer."

    monkeypatch.setattr(
        server.gemini_media, "call_unstructured", fake_call_unstructured
    )
    result = server.analyze_images(
        image_paths=[str(path) for path in image_paths],
        question="What is shared?",
        thinking_level=None,
        max_output_tokens=None,
    )

    assert captured["thinking_level"] is None
    assert captured["max_output_tokens"] is None
    assert result["thinking_level"] is None
    assert result["max_output_tokens"] is None


@pytest.mark.parametrize("count", [0, 1, 11])
def test_analyze_images_requires_two_to_ten_paths(tmp_path: Path, count: int) -> None:
    paths = []
    for index in range(count):
        path = tmp_path / f"image-{index}.png"
        path.write_bytes(b"image")
        paths.append(str(path))

    with pytest.raises(RuntimeError, match="needs 2 to 10 image paths"):
        server.analyze_images(image_paths=paths, question="Compare them.")


def test_analyze_images_rejects_label_count_mismatch(
    image_paths: list[Path],
) -> None:
    with pytest.raises(RuntimeError, match="exactly one entry per image"):
        server.analyze_images(
            image_paths=[str(path) for path in image_paths],
            labels=["only one label"],
            question="Compare them.",
        )


def test_analyze_images_rejects_duplicate_paths(image_paths: list[Path]) -> None:
    duplicate = str(image_paths[0])
    with pytest.raises(RuntimeError, match="image paths must be distinct"):
        server.analyze_images(
            image_paths=[duplicate, duplicate], question="Compare them."
        )


def test_analyze_images_errors_on_missing_file(
    tmp_path: Path, image_paths: list[Path]
) -> None:
    missing = tmp_path / "missing.png"
    with pytest.raises(RuntimeError, match="^IMAGE_NOT_FOUND:"):
        server.analyze_images(
            image_paths=[str(image_paths[0]), str(missing)],
            question="Compare them.",
        )
