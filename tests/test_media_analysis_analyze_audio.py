from pathlib import Path
from types import SimpleNamespace

import pytest

from media_analysis_mcp import gemini_media, server


_UPLOAD_AND_POLL_AUDIO = gemini_media.upload_and_poll_audio
_UPLOAD_AND_POLL_VIDEO = gemini_media.upload_and_poll_video


class _FakeFileData:
    def __init__(self, *, file_uri, mime_type):
        self.file_uri = file_uri
        self.mime_type = mime_type


class _FakePart:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeTypes:
    FileData = _FakeFileData
    Part = _FakePart


class _FakeClient:
    pass


class _FakeUploaded:
    name = "files/audio-1"
    uri = "files/audio-1"
    mime_type = "audio/mp4a-latm"


@pytest.fixture
def audio_path(tmp_path: Path) -> Path:
    path = tmp_path / "notes.m4a"
    path.write_bytes(b"not-real-audio")
    return path


@pytest.fixture(autouse=True)
def fake_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        server.gemini_media,
        "init_client",
        lambda: (_FakeClient(), _FakeTypes()),
    )
    monkeypatch.setattr(
        server.gemini_media,
        "upload_and_poll_audio",
        lambda client, path, *, timeout_s=300: _FakeUploaded(),
    )
    monkeypatch.setattr(
        server.gemini_media,
        "cleanup_uploaded",
        lambda client, file_obj: None,
    )


def test_analyze_audio_preserves_detected_mime_and_defaults_model(
    monkeypatch: pytest.MonkeyPatch, audio_path: Path
) -> None:
    captured: dict = {}

    def fake_call_unstructured(**kwargs):
        captured.update(kwargs)
        return "[00:00] A faithful transcript."

    monkeypatch.setattr(
        server.gemini_media,
        "call_unstructured",
        fake_call_unstructured,
    )

    result = server.analyze_audio(
        audio_path=str(audio_path),
        question="Transcribe everything.",
    )

    assert result["model"] == "gemini-3.7-flash"
    assert result["detected_mime_type"] == "audio/mp4a-latm"
    assert result["audio_path"] == str(audio_path.resolve())
    media_part = captured["contents"][2]
    assert media_part.kwargs["file_data"].mime_type == "audio/mp4a-latm"
    assert "VideoMetadata" not in media_part.kwargs


def test_analyze_audio_cleans_up_when_model_call_fails(
    monkeypatch: pytest.MonkeyPatch, audio_path: Path
) -> None:
    cleaned: list[str] = []
    monkeypatch.setattr(
        server.gemini_media,
        "cleanup_uploaded",
        lambda client, file_obj: cleaned.append(file_obj.name),
    )
    monkeypatch.setattr(
        server.gemini_media,
        "call_unstructured",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )

    with pytest.raises(RuntimeError, match="synthetic failure"):
        server.analyze_audio(str(audio_path), "Transcribe it.")

    assert cleaned == ["files/audio-1"]


def test_analyze_audio_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="^AUDIO_NOT_FOUND:"):
        server.analyze_audio(str(tmp_path / "missing.m4a"), "Transcribe it.")


def test_analyze_audio_rejects_blank_question(audio_path: Path) -> None:
    with pytest.raises(RuntimeError, match="^INVALID_INPUT:"):
        server.analyze_audio(str(audio_path), "  \n")


def test_analyze_audio_rejects_non_audio_mime(
    monkeypatch: pytest.MonkeyPatch, audio_path: Path
) -> None:
    class WrongMime(_FakeUploaded):
        mime_type = "video/mp4"

    monkeypatch.setattr(
        server.gemini_media,
        "upload_and_poll_audio",
        lambda client, path, *, timeout_s=300: WrongMime(),
    )

    with pytest.raises(RuntimeError, match="^AUDIO_MIME_MISMATCH:"):
        server.analyze_audio(str(audio_path), "Transcribe it.")


def test_upload_and_poll_audio_failed_state_uses_audio_code_and_cleans_up(
    audio_path: Path,
) -> None:
    deleted: list[str] = []

    class Files:
        def upload(self, *, file: str):
            return SimpleNamespace(name="files/audio-failed")

        def get(self, *, name: str):
            return SimpleNamespace(
                name=name,
                state=SimpleNamespace(name="FAILED"),
            )

        def delete(self, *, name: str) -> None:
            deleted.append(name)

    client = SimpleNamespace(files=Files())

    with pytest.raises(RuntimeError, match="^AUDIO_PROCESSING_FAILED:"):
        _UPLOAD_AND_POLL_AUDIO(client, str(audio_path))

    assert deleted == ["files/audio-failed"]


def test_upload_and_poll_video_keeps_existing_missing_file_error(tmp_path: Path) -> None:
    client = SimpleNamespace(files=SimpleNamespace())

    with pytest.raises(RuntimeError, match="^VIDEO_NOT_FOUND:"):
        _UPLOAD_AND_POLL_VIDEO(client, str(tmp_path / "missing.mp4"))
