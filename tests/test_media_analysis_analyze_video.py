import inspect
from pathlib import Path

import pytest

from media_analysis_mcp import server


class _FakeClient:
    pass


@pytest.mark.parametrize(
    "tool",
    [
        server.analyze_video,
        server.analyze_videos,
        server.describe_video,
        server.score_video,
    ],
)
def test_all_video_analysis_tools_default_to_gemini_3_6_flash(tool) -> None:
    default = inspect.signature(tool).parameters["model"].default
    assert server.DEFAULT_VIDEO_ANALYSIS_MODEL == "gemini-3.6-flash"
    assert default == server.DEFAULT_VIDEO_ANALYSIS_MODEL


class _FakeFileData:
    def __init__(self, *, file_uri, mime_type):
        self.file_uri = file_uri
        self.mime_type = mime_type


class _FakePart:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeVideoMetadata:
    def __init__(self, *, fps, start_offset):
        self.fps = fps
        self.start_offset = start_offset


class _FakeTypes:
    FileData = _FakeFileData
    Part = _FakePart
    VideoMetadata = _FakeVideoMetadata


class _FakeUploaded:
    name = "files/fake-upload"
    uri = "files/fake-upload"
    mime_type = "video/mp4"


@pytest.fixture
def video_path(tmp_path: Path) -> Path:
    """A file that just needs to exist — Files API upload is mocked."""
    p = tmp_path / "fake.mp4"
    p.write_bytes(b"not-a-real-mp4")
    return p


@pytest.fixture
def video_paths(tmp_path: Path) -> list[Path]:
    paths = [tmp_path / "master.mp4", tmp_path / "candidate.mov"]
    for path in paths:
        path.write_bytes(b"not-a-real-video")
    return paths


@pytest.fixture(autouse=True)
def fake_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        server.gemini_media,
        "init_client",
        lambda: (_FakeClient(), _FakeTypes()),
    )
    monkeypatch.setattr(
        server.gemini_media,
        "upload_and_poll_video",
        lambda client, path, *, timeout_s=300: _FakeUploaded(),
    )
    monkeypatch.setattr(
        server.gemini_media,
        "cleanup_uploaded",
        lambda client, file_obj: None,
    )
    # require_pillow is called even when no ref images are passed; return a stub.
    monkeypatch.setattr(server.gemini_media, "require_pillow", lambda: object())


def test_analyze_video_returns_question_and_answer(
    monkeypatch: pytest.MonkeyPatch, video_path: Path
) -> None:
    captured: dict = {}

    def fake_call_unstructured(**kwargs):
        captured["model"] = kwargs["model"]
        captured["contents"] = kwargs["contents"]
        return "The camera is locked off; the glow ignites near midpoint."

    monkeypatch.setattr(
        server.gemini_media, "call_unstructured", fake_call_unstructured
    )

    result = server.analyze_video(
        video_path=str(video_path),
        question="describe the camera move",
        intent="ad-hoc motion read",
        model="test-model",
    )

    assert result["model"] == "test-model"
    assert result["video_path"] == str(video_path.resolve())
    assert result["question"] == "describe the camera move"
    assert result["answer"] == "The camera is locked off; the glow ignites near midpoint."
    assert result["context_used"]["question"] == "describe the camera move"
    assert result["context_used"]["prompt"] is None
    # Question is anchored in the first content block (context_block).
    assert "Question: describe the camera move" in captured["contents"][0]


def test_analyze_video_defaults_to_gemini_3_6_flash(
    monkeypatch: pytest.MonkeyPatch, video_path: Path
) -> None:
    captured: dict = {}

    def fake_call_unstructured(**kwargs):
        captured["model"] = kwargs["model"]
        return "A detailed answer."

    monkeypatch.setattr(
        server.gemini_media, "call_unstructured", fake_call_unstructured
    )

    result = server.analyze_video(
        video_path=str(video_path),
        question="catalog the audiovisual construction",
    )

    assert captured["model"] == server.DEFAULT_VIDEO_ANALYSIS_MODEL
    assert result["model"] == server.DEFAULT_VIDEO_ANALYSIS_MODEL


def test_analyze_video_errors_on_missing_file(tmp_path: Path) -> None:
    bogus = tmp_path / "not-here.mp4"
    with pytest.raises(RuntimeError, match="^VIDEO_NOT_FOUND:"):
        server.analyze_video(video_path=str(bogus), question="anything")


def test_analyze_video_rejects_blank_question(video_path: Path) -> None:
    with pytest.raises(RuntimeError, match="^INVALID_INPUT: question is required"):
        server.analyze_video(video_path=str(video_path), question="\t\n")


def test_analyze_video_cleanup_runs_on_call_failure(
    monkeypatch: pytest.MonkeyPatch, video_path: Path
) -> None:
    """The Files API upload must be cleaned up even when the structured call
    raises — the same try/finally invariant that protects describe_video."""
    cleanup_calls: list = []

    monkeypatch.setattr(
        server.gemini_media,
        "cleanup_uploaded",
        lambda client, file_obj: cleanup_calls.append(file_obj.name),
    )

    def boom(**kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(server.gemini_media, "call_unstructured", boom)

    with pytest.raises(RuntimeError, match="synthetic failure"):
        server.analyze_video(video_path=str(video_path), question="anything")

    assert cleanup_calls == ["files/fake-upload"]


def test_analyze_videos_labels_uploads_and_cleans_up_in_order(
    monkeypatch: pytest.MonkeyPatch, video_paths: list[Path]
) -> None:
    captured: dict = {}
    upload_calls: list[str] = []
    cleanup_calls: list[str] = []

    class Uploaded:
        def __init__(self, index: int) -> None:
            self.name = f"files/video-{index}"
            self.uri = self.name

    def fake_upload(client, path, *, timeout_s=300):
        upload_calls.append(path)
        return Uploaded(len(upload_calls))

    def fake_call_unstructured(**kwargs):
        captured["contents"] = kwargs["contents"]
        return "Candidate B has the cleanest causal action."

    monkeypatch.setattr(server.gemini_media, "upload_and_poll_video", fake_upload)
    monkeypatch.setattr(
        server.gemini_media, "call_unstructured", fake_call_unstructured
    )
    monkeypatch.setattr(
        server.gemini_media,
        "cleanup_uploaded",
        lambda client, file_obj: cleanup_calls.append(file_obj.name),
    )

    result = server.analyze_videos(
        video_paths=[str(path) for path in video_paths],
        labels=["Current master", "Doorway candidate"],
        question="Which clip better establishes impossible geography?",
        fps=6,
    )

    resolved = [str(path.resolve()) for path in video_paths]
    assert upload_calls == resolved
    assert cleanup_calls == ["files/video-1", "files/video-2"]
    assert result["video_paths"] == resolved
    assert result["video_labels"] == ["Current master", "Doorway candidate"]
    assert result["answer"] == "Candidate B has the cleanest causal action."
    assert result["context_used"]["video_labels"] == result["video_labels"]

    labels = [part for part in captured["contents"] if isinstance(part, str)]
    assert "VIDEO 1 — Current master:" in labels
    assert "VIDEO 2 — Doorway candidate:" in labels
    video_parts = [
        part for part in captured["contents"] if isinstance(part, _FakePart)
    ]
    assert len(video_parts) == 2
    assert [part.kwargs["file_data"].file_uri for part in video_parts] == [
        "files/video-1",
        "files/video-2",
    ]
    assert [part.kwargs["video_metadata"].fps for part in video_parts] == [6, 6]


def test_analyze_videos_defaults_labels_to_basenames(
    monkeypatch: pytest.MonkeyPatch, video_paths: list[Path]
) -> None:
    monkeypatch.setattr(
        server.gemini_media,
        "call_unstructured",
        lambda **kwargs: "A grounded comparison.",
    )

    result = server.analyze_videos(
        video_paths=[str(path) for path in video_paths],
        question="Compare them.",
    )

    assert result["video_labels"] == ["master.mp4", "candidate.mov"]


@pytest.mark.parametrize("count", [0, 1, 11])
def test_analyze_videos_requires_two_to_ten_paths(
    tmp_path: Path, count: int
) -> None:
    paths = []
    for index in range(count):
        path = tmp_path / f"video-{index}.mp4"
        path.write_bytes(b"video")
        paths.append(str(path))

    with pytest.raises(RuntimeError, match="needs 2 to 10 video paths"):
        server.analyze_videos(video_paths=paths, question="Compare them.")


def test_analyze_videos_rejects_label_count_mismatch(
    video_paths: list[Path],
) -> None:
    with pytest.raises(RuntimeError, match="exactly one entry per video"):
        server.analyze_videos(
            video_paths=[str(path) for path in video_paths],
            labels=["only one label"],
            question="Compare them.",
        )


def test_analyze_videos_cleanup_runs_when_model_call_fails(
    monkeypatch: pytest.MonkeyPatch, video_paths: list[Path]
) -> None:
    cleanup_calls: list[str] = []

    class Uploaded:
        def __init__(self, index: int) -> None:
            self.name = f"files/video-{index}"
            self.uri = self.name

    upload_count = 0

    def fake_upload(client, path, *, timeout_s=300):
        nonlocal upload_count
        upload_count += 1
        return Uploaded(upload_count)

    monkeypatch.setattr(server.gemini_media, "upload_and_poll_video", fake_upload)
    monkeypatch.setattr(
        server.gemini_media,
        "cleanup_uploaded",
        lambda client, file_obj: cleanup_calls.append(file_obj.name),
    )
    monkeypatch.setattr(
        server.gemini_media,
        "call_unstructured",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )

    with pytest.raises(RuntimeError, match="synthetic failure"):
        server.analyze_videos(
            video_paths=[str(path) for path in video_paths],
            question="Compare them.",
        )

    assert cleanup_calls == ["files/video-1", "files/video-2"]


def test_analyze_videos_cleans_prior_upload_when_later_upload_fails(
    monkeypatch: pytest.MonkeyPatch, video_paths: list[Path]
) -> None:
    cleanup_calls: list[str] = []

    class Uploaded:
        name = "files/video-1"
        uri = name

    upload_count = 0

    def fake_upload(client, path, *, timeout_s=300):
        nonlocal upload_count
        upload_count += 1
        if upload_count == 2:
            raise RuntimeError("VIDEO_UPLOAD_FAILED: synthetic failure")
        return Uploaded()

    monkeypatch.setattr(server.gemini_media, "upload_and_poll_video", fake_upload)
    monkeypatch.setattr(
        server.gemini_media,
        "cleanup_uploaded",
        lambda client, file_obj: cleanup_calls.append(file_obj.name),
    )

    with pytest.raises(RuntimeError, match="VIDEO_UPLOAD_FAILED"):
        server.analyze_videos(
            video_paths=[str(path) for path in video_paths],
            question="Compare them.",
        )

    assert cleanup_calls == ["files/video-1"]
