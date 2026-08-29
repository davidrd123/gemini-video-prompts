import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from gemini_video_prompts_mcp import seedance
from gemini_video_prompts_mcp import server as gen_server


@pytest.fixture
def image_path(tmp_path: Path) -> Path:
    image = tmp_path / "first.png"
    image.write_bytes(b"not a real png; existence is enough for this contract")
    return image


def test_generate_video_dry_run_modes() -> None:
    text_only = gen_server.generate_video(prompt="A quiet establishing shot", dry_run=True)
    video_ref = gen_server.generate_video(
        prompt="Use [Video1] as motion reference",
        reference_videos=["/tmp/motion.mp4"],
        dry_run=True,
    )
    first_frame = gen_server.generate_video(
        prompt="Animate [Image1]",
        image="/tmp/first.png",
        dry_run=True,
    )
    # 2.0 lets reference_videos layer on top of a first frame; 2.5 does not.
    image_and_video_20 = gen_server.generate_video(
        prompt="Use [Image1] and [Video1]",
        image="/tmp/first.png",
        reference_videos=["/tmp/motion.mp4"],
        model="bytedance/seedance-2.0",
        dry_run=True,
    )
    explicit_seed = gen_server.generate_video(
        prompt="A repeatable camera test",
        seed=1234,
        dry_run=True,
    )

    assert text_only["mode"] == "text_to_video"
    assert text_only["model"] == "bytedance/seedance-2.5"
    assert text_only["resolved_params"]["aspect_ratio"] == "16:9"
    assert video_ref["mode"] == "omni_reference"
    assert first_frame["mode"] == "first_last_frames"
    assert first_frame["resolved_params"]["aspect_ratio"] == "adaptive"
    assert image_and_video_20["mode"] == "first_last_frames"
    assert image_and_video_20["resolved_params"]["aspect_ratio"] == "adaptive"
    with pytest.raises(RuntimeError, match="^INVALID_INPUT:.*exclusive with all"):
        gen_server.generate_video(
            prompt="Use [Image1] and [Video1]",
            image="/tmp/first.png",
            reference_videos=["/tmp/motion.mp4"],
            dry_run=True,
        )
    assert explicit_seed["seed_provenance"] == {
        "requested_seed": 1234,
        "effective_seed": 1234,
        "source": "request",
        "matches_requested": True,
    }


V20 = "bytedance/seedance-2.0"
V25 = "bytedance/seedance-2.5"


@pytest.mark.parametrize(
    "kwargs",
    [
        # Model-independent rules
        {"prompt": "x", "image": "a.png", "reference_images": ["b.png"]},
        {"prompt": "x", "last_frame_image": "b.png"},
        {"prompt": "x", "duration": 0},
        {"prompt": "x", "duration": 3},
        {"prompt": "x", "reference_audios": ["a.wav"]},
        {"prompt": "x", "model": "someone/other-video-model"},
        # 2.5 (default) limits
        {"prompt": "x", "reference_videos": [f"{i}.mp4" for i in range(11)]},
        {"prompt": "x", "reference_audios": [f"{i}.wav" for i in range(11)],
         "reference_images": ["a.png"]},
        {"prompt": "x", "reference_images": [f"{i}.png" for i in range(31)]},
        {"prompt": "x", "duration": 31},
        {"prompt": "x", "resolution": "1080p"},
        {"prompt": "x", "resolution": "4k"},
        {"prompt": "x", "aspect_ratio": "9:21"},
        {"prompt": "x", "image": "a.png", "reference_videos": ["m.mp4"]},
        {"prompt": "x", "image": "a.png", "reference_audios": ["m.wav"]},
        # 2.0 limits (tighter caps, shorter max duration)
        {"prompt": "x", "model": V20,
         "reference_videos": ["1.mp4", "2.mp4", "3.mp4", "4.mp4"]},
        {"prompt": "x", "model": V20,
         "reference_images": [f"{i}.png" for i in range(10)]},
        {"prompt": "x", "model": V20, "duration": 16},
    ],
)
def test_seedance_validation_errors_are_coded(kwargs: dict) -> None:
    with pytest.raises(RuntimeError, match="^INVALID_INPUT:"):
        seedance.build_seedance_video_params(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        # 2.5 (default): larger sets, 30s, first-frame + adaptive
        {"prompt": "x", "reference_videos": [f"{i}.mp4" for i in range(10)]},
        {"prompt": "x", "reference_images": [f"{i}.png" for i in range(30)]},
        {"prompt": "x", "duration": 30},
        {"prompt": "x", "duration": -1},
        {"prompt": "x", "image": "a.png", "aspect_ratio": "adaptive"},
        # Live-probed 2026-08-29: 2.5 accepts an explicit ratio with a first
        # frame despite the schema text saying it "requires 'adaptive'".
        {"prompt": "x", "image": "a.png", "aspect_ratio": "16:9"},
        {"prompt": "x", "image": "a.png", "last_frame_image": "b.png",
         "aspect_ratio": "1:1"},
        {"prompt": "x", "image": "a.png", "last_frame_image": "b.png"},
        {"prompt": "x", "model": f"{V25}:ca38262bae0952bf80a7f10eda58af86"},
        # 2.0: 4k + 9:21 still valid, image + video layering still valid
        {"prompt": "x", "model": V20, "resolution": "4k"},
        {"prompt": "x", "model": V20, "resolution": "1080p", "aspect_ratio": "9:21"},
        {"prompt": "x", "model": V20, "image": "a.png", "reference_videos": ["m.mp4"]},
        {"prompt": "x", "model": V20, "image": "a.png", "aspect_ratio": "16:9"},
        {"prompt": "x", "model": V20, "duration": 15},
    ],
)
def test_seedance_validation_accepts_per_model_limits(kwargs: dict) -> None:
    params = seedance.build_seedance_video_params(**kwargs)
    assert params["prompt"] == "x"
    assert "model" not in params  # model_ref is routing, never a Replicate input


def test_seedance_default_aspect_ratio_follows_first_frame() -> None:
    text = seedance.build_seedance_video_params(prompt="x")
    framed = seedance.build_seedance_video_params(prompt="x", image="a.png")
    explicit = seedance.build_seedance_video_params(
        prompt="x", image="a.png", aspect_ratio="16:9"
    )

    assert text["aspect_ratio"] == "16:9"
    assert framed["aspect_ratio"] == "adaptive"
    assert explicit["aspect_ratio"] == "16:9"  # explicit values are never rewritten


def test_seedance_spec_table_matches_replicate_schema() -> None:
    v25 = seedance.resolve_spec(V25)
    v20 = seedance.resolve_spec("bytedance/seedance-2.0:a6dcbae88b153e75")

    assert seedance.SEEDANCE_MODEL_DEFAULT == V25
    assert (v25.max_duration, v20.max_duration) == (30, 15)
    assert v25.resolutions == {"480p", "720p"}
    assert v20.resolutions == {"480p", "720p", "1080p", "4k"}
    assert "9:21" in v20.aspect_ratios and "9:21" not in v25.aspect_ratios
    assert (v25.max_reference_images, v25.max_reference_videos, v25.max_reference_audios) == (30, 10, 10)
    assert (v20.max_reference_images, v20.max_reference_videos, v20.max_reference_audios) == (9, 3, 3)
    assert v25.frames_exclusive_with_all_references and not v20.frames_exclusive_with_all_references


def test_seed_provenance_prefers_provider_log_over_echoed_input() -> None:
    provenance = seedance.resolve_seed_provenance(
        requested_seed=41,
        prediction={"input": {"seed": 42}, "logs": "Using seed: 43"},
    )

    assert provenance == {
        "requested_seed": 41,
        "effective_seed": 43,
        "source": "provider_logs",
        "matches_requested": False,
    }


def test_seed_provenance_recovers_provider_generated_seed_from_logs() -> None:
    provenance = seedance.resolve_seed_provenance(
        requested_seed=None,
        prediction={"logs": "Preparing model\nUsing seed: 2091770884\nGenerating"},
    )

    assert provenance == {
        "requested_seed": None,
        "effective_seed": 2091770884,
        "source": "provider_logs",
        "matches_requested": None,
    }


def test_seedance_hashes_the_same_open_stream_passed_to_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image_module = pytest.importorskip("PIL.Image")
    image = tmp_path / "reference.png"
    image_module.new("RGB", (16, 9), color=(12, 34, 56)).save(image)
    captured: dict = {}

    def fake_create_prediction(**kwargs):
        handle = kwargs["params"]["image"]
        captured["handle"] = handle
        captured["bytes"] = handle.read()
        return {"id": "pred-stream-hash", "status": "starting"}

    monkeypatch.setattr(
        seedance.replicate_min,
        "create_prediction",
        fake_create_prediction,
    )
    params = seedance.build_seedance_video_params(
        prompt="Animate [Image1]",
        image=str(image),
        aspect_ratio="adaptive",
    )

    prediction = seedance.create_seedance_prediction(api_params=params)

    identity = prediction["_reference_file_identities"][str(image.resolve())]
    assert captured["bytes"] == image.read_bytes()
    assert identity == {
        "bytes": len(captured["bytes"]),
        "sha256": hashlib.sha256(captured["bytes"]).hexdigest(),
        "hash_source": "uploaded_file_stream",
    }
    assert captured["handle"].closed is True


def test_seedance_reference_image_aspect_ratio_preflight(tmp_path: Path) -> None:
    image_module = pytest.importorskip("PIL.Image")
    matching = tmp_path / "matching.png"
    mismatched = tmp_path / "mismatched.png"
    image_module.new("RGB", (16, 9)).save(matching)
    image_module.new("RGB", (9, 16)).save(mismatched)

    seedance.assert_reference_aspect_ratios(
        {
            "aspect_ratio": "16:9",
            "reference_images": [str(matching)],
        }
    )

    with pytest.raises(RuntimeError) as exc_info:
        seedance.assert_reference_aspect_ratios(
            {
                "aspect_ratio": "16:9",
                "reference_images": [str(mismatched)],
            }
        )
    assert str(exc_info.value).startswith("INVALID_INPUT:")
    assert "mismatched.png" in str(exc_info.value)


def test_generate_video_preserves_coded_error_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, image_path: Path
) -> None:
    def fake_run_seedance_job(**kwargs):
        return {
            "success": False,
            "error": {
                "message": "REPLICATE_TIMEOUT: prediction did not complete within 1s",
                "type": "RuntimeError",
            },
            "outputs": [],
            "metrics": {},
            "cold_start": False,
        }

    monkeypatch.setattr(gen_server.seedance, "run_seedance_job", fake_run_seedance_job)

    with pytest.raises(RuntimeError, match="^REPLICATE_TIMEOUT:"):
        gen_server.generate_video(
            prompt="Use [Image1]",
            image=str(image_path),
            out_root=str(tmp_path / "out"),
        )


def test_generate_video_writes_job_json_and_cold_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, image_path: Path
) -> None:
    submission_time = "2026-05-09T10:00:00Z"
    collection_time = "2026-05-09T10:00:30Z"
    timestamps = iter([submission_time, collection_time])

    def fake_run_seedance_job(**kwargs):
        output_path = kwargs["out_dir"] / "fake_00.mp4"
        output_path.write_bytes(b"fake video")
        return {
            "success": True,
            "model": {"version": "@latest"},
            "outputs": [
                {
                    "path": str(output_path),
                    "url": "https://example.com/fake.mp4",
                    "bytes": output_path.stat().st_size,
                }
            ],
            "metrics": {
                "predict_time_s": 1.0,
                "download_time_s": 0.1,
                "elapsed_s": 1.1,
            },
            "cold_start": True,
        }

    monkeypatch.setattr(gen_server.seedance, "run_seedance_job", fake_run_seedance_job)
    monkeypatch.setattr(gen_server, "now_iso", lambda: next(timestamps))
    monkeypatch.setattr(
        gen_server.seedance,
        "probe_media_info",
        lambda path: {
            "duration_s": None,
            "fps": None,
            "width": None,
            "height": None,
            "has_audio": None,
        },
    )

    result = gen_server.generate_video(
        prompt="Use [Image1]",
        image=str(image_path),
        out_root=str(tmp_path / "out"),
    )

    job_json = Path(result["job_dir"]) / "job.json"
    assert job_json.is_file()
    assert result["metrics"]["cold_start"] is True
    assert result["created_at"] == submission_time
    assert result["started_at"] == submission_time
    assert result["collected_at"] == collection_time
    saved = json.loads(job_json.read_text(encoding="utf-8"))
    assert saved["status"] == "ok"
    assert saved["created_at"] == submission_time
    assert saved["collected_at"] == collection_time
    assert saved["metrics"]["cold_start"] is True
    assert saved["seed_provenance"]["source"] == "unavailable"
    assert saved["references"][0]["sha256"] == hashlib.sha256(
        image_path.read_bytes()
    ).hexdigest()
    assert saved["outputs"][0]["sha256"] == hashlib.sha256(b"fake video").hexdigest()


def test_start_video_job_writes_local_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, image_path: Path
) -> None:
    def fake_create_seedance_prediction(**kwargs):
        assert kwargs["webhook_url"] is None
        reference_bytes = image_path.read_bytes()
        return {
            "id": "pred-starting",
            "status": "starting",
            "version": "@latest",
            "created_at": "2026-05-09T10:00:00Z",
            "started_at": None,
            "completed_at": None,
            "output": None,
            "metrics": None,
            "error": None,
            "_reference_file_identities": {
                str(image_path.resolve()): {
                    "bytes": len(reference_bytes),
                    "sha256": hashlib.sha256(reference_bytes).hexdigest(),
                    "hash_source": "uploaded_file_stream",
                }
            },
        }

    monkeypatch.setattr(
        gen_server.seedance,
        "create_seedance_prediction",
        fake_create_seedance_prediction,
    )

    result = gen_server.start_video_job(
        prompt="Use [Image1]",
        image=str(image_path),
        out_root=str(tmp_path / "out"),
    )

    assert result["status"] == "starting"
    assert result["prediction_id"] == "pred-starting"
    assert result["outputs_downloaded"] is False
    assert Path(result["status_path"]).is_file()
    assert Path(result["request_path"]).is_file()

    saved = json.loads(Path(result["status_path"]).read_text(encoding="utf-8"))
    assert saved["job_id"] == result["job_id"]
    assert saved["prediction_id"] == "pred-starting"
    assert saved["generation_id"] == result["job_id"]
    assert saved["seed_provenance"]["effective_seed"] is None
    assert saved["references"][0]["bytes"] == image_path.stat().st_size
    assert saved["references"][0]["sha256"] == hashlib.sha256(
        image_path.read_bytes()
    ).hexdigest()
    assert saved["references"][0]["hash_source"] == "uploaded_file_stream"
    saved_request = json.loads(
        Path(result["request_path"]).read_text(encoding="utf-8")
    )
    assert (
        saved_request["references"][0]["hash_source"]
        == "uploaded_file_stream"
    )

    index_path = tmp_path / "out" / "jobs" / "index.ndjson"
    entries = [json.loads(line) for line in index_path.read_text().splitlines()]
    assert entries[0]["event"] == "created"
    assert entries[0]["generation_id"] == result["job_id"]
    assert (
        entries[0]["reference_files"][0]["hash_source"]
        == "uploaded_file_stream"
    )


def test_start_video_job_uses_unique_job_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, image_path: Path
) -> None:
    prediction_ids = iter(["pred-one", "pred-two"])

    monkeypatch.setattr(
        gen_server.seedance,
        "create_seedance_prediction",
        lambda **kwargs: {
            "id": next(prediction_ids),
            "status": "starting",
            "version": "@latest",
            "created_at": "2026-05-09T10:00:00Z",
            "started_at": None,
            "completed_at": None,
            "output": None,
            "metrics": None,
            "error": None,
        },
    )

    first = gen_server.start_video_job(
        prompt="Use [Image1]",
        image=str(image_path),
        out_root=str(tmp_path / "out"),
    )
    second = gen_server.start_video_job(
        prompt="Use [Image1]",
        image=str(image_path),
        out_root=str(tmp_path / "out"),
    )

    assert first["job_id"] != second["job_id"]
    assert first["job_dir"] != second["job_dir"]
    assert first["job_dir"].endswith(first["job_id"])
    assert second["job_dir"].endswith(second["job_id"])


def test_start_video_job_survives_index_write_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, image_path: Path
) -> None:
    monkeypatch.setattr(
        gen_server.seedance,
        "create_seedance_prediction",
        lambda **kwargs: {
            "id": "pred-index-warning",
            "status": "starting",
            "input": {},
        },
    )
    monkeypatch.setattr(
        gen_server,
        "_append_generation_index",
        lambda *args, **kwargs: "synthetic index failure",
    )

    result = gen_server.start_video_job(
        prompt="Use [Image1]",
        image=str(image_path),
        out_root=str(tmp_path / "out"),
    )

    assert result["job_id"]
    assert result["provenance_warnings"] == ["synthetic index failure"]
    saved = json.loads(Path(result["status_path"]).read_text(encoding="utf-8"))
    assert saved["prediction_id"] == "pred-index-warning"
    assert saved["provenance_warnings"] == ["synthetic index failure"]


def test_start_video_job_create_failure_leaves_no_local_job_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, image_path: Path
) -> None:
    def fake_create_seedance_prediction(**kwargs):
        raise RuntimeError("REPLICATE_API_TOKEN_MISSING: missing token")

    monkeypatch.setattr(
        gen_server.seedance,
        "create_seedance_prediction",
        fake_create_seedance_prediction,
    )

    out_root = tmp_path / "out"
    with pytest.raises(RuntimeError, match="^REPLICATE_API_TOKEN_MISSING:"):
        gen_server.start_video_job(
            prompt="Use [Image1]",
            image=str(image_path),
            out_root=str(out_root),
        )

    assert not (out_root / "jobs").exists()


def test_get_video_job_poll_false_returns_local_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, image_path: Path
) -> None:
    monkeypatch.setattr(
        gen_server.seedance,
        "create_seedance_prediction",
        lambda **kwargs: {
            "id": "pred-local",
            "status": "processing",
            "version": "@latest",
            "created_at": "2026-05-09T10:00:00Z",
            "started_at": "2026-05-09T10:00:01Z",
            "completed_at": None,
            "output": None,
            "metrics": None,
            "error": None,
        },
    )
    monkeypatch.setattr(
        gen_server.seedance,
        "get_seedance_prediction",
        lambda prediction_id: pytest.fail("poll=False should not call provider"),
    )

    started = gen_server.start_video_job(
        prompt="Use [Image1]",
        image=str(image_path),
        out_root=str(tmp_path / "out"),
    )
    result = gen_server.get_video_job(
        started["job_id"],
        out_root=str(tmp_path / "out"),
        poll=False,
    )

    assert result["status"] == "processing"
    assert result["prediction_id"] == "pred-local"


def test_get_video_job_unknown_job_id_is_coded(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="^JOB_NOT_FOUND: missing-job"):
        gen_server.get_video_job("missing-job", out_root=str(tmp_path / "out"))


def test_get_video_job_finalizes_succeeded_prediction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, image_path: Path
) -> None:
    collection_time = "2026-05-09T10:00:31Z"
    download_complete = False

    monkeypatch.setattr(
        gen_server.seedance,
        "create_seedance_prediction",
        lambda **kwargs: {
            "id": "pred-done",
            "status": "processing",
            "version": "@latest",
            "created_at": "2026-05-09T10:00:00Z",
            "started_at": "2026-05-09T10:00:01Z",
            "completed_at": None,
            "output": None,
            "metrics": None,
            "error": None,
        },
    )
    monkeypatch.setattr(
        gen_server.seedance,
        "get_seedance_prediction",
        lambda prediction_id: {
            "id": prediction_id,
            "status": "succeeded",
            "version": "@latest",
            "created_at": "2026-05-09T10:00:00Z",
            "started_at": "2026-05-09T10:00:01Z",
            "completed_at": "2026-05-09T10:00:30Z",
            "output": ["https://example.com/fake.mp4"],
            "metrics": {"predict_time": 1.5},
            "logs": "Using seed: 2091770884\nGenerating video...",
            "error": None,
        },
    )

    def fake_download_prediction_outputs(**kwargs):
        nonlocal download_complete
        output_path = kwargs["out_dir"] / "fake_00.mp4"
        output_path.write_bytes(b"fake video")
        download_complete = True
        return [
            {
                "path": str(output_path),
                "url": kwargs["outputs"][0],
                "bytes": output_path.stat().st_size,
                "_metrics": {"download_time_s": 0.2},
            }
        ]

    monkeypatch.setattr(
        gen_server.seedance,
        "download_prediction_outputs",
        fake_download_prediction_outputs,
    )
    monkeypatch.setattr(
        gen_server.seedance,
        "probe_media_info",
        lambda path: {
            "duration_s": None,
            "fps": None,
            "width": None,
            "height": None,
            "has_audio": None,
        },
    )

    started = gen_server.start_video_job(
        prompt="Use [Image1]",
        image=str(image_path),
        out_root=str(tmp_path / "out"),
    )

    def collection_clock() -> str:
        assert download_complete, "collected_at sampled before download completed"
        return collection_time

    monkeypatch.setattr(gen_server, "now_iso", collection_clock)
    result = gen_server.get_video_job(
        started["job_id"],
        out_root=str(tmp_path / "out"),
    )

    assert result["status"] == "succeeded"
    assert result["outputs_downloaded"] is True
    assert result["result"]["status"] == "ok"
    assert result["result"]["created_at"] == started["created_at"]
    assert result["result"]["collected_at"] == collection_time
    assert result["result"]["outputs"][0]["path"].endswith(".mp4")
    assert result["result"]["metrics"]["predict_time_s"] == 1.5
    assert result["seed_provenance"] == {
        "requested_seed": None,
        "effective_seed": 2091770884,
        "source": "provider_logs",
        "matches_requested": None,
    }
    assert result["result"]["seed_provenance"] == result["seed_provenance"]

    job_json = Path(result["job_dir"]) / "job.json"
    assert job_json.is_file()
    saved_job = json.loads(job_json.read_text(encoding="utf-8"))
    assert saved_job["prediction_id"] == "pred-done"
    assert saved_job["generation_id"] == started["job_id"]
    assert saved_job["created_at"] == started["created_at"]
    assert saved_job["collected_at"] == result["result"]["collected_at"]
    assert saved_job["seed_provenance"]["effective_seed"] == 2091770884
    assert saved_job["outputs"][0]["sha256"] == hashlib.sha256(
        b"fake video"
    ).hexdigest()

    provenance = gen_server.get_generation(
        started["job_id"],
        out_root=str(tmp_path / "out"),
    )
    assert provenance["state"] == "succeeded"
    assert provenance["request"]["prompt"] == "Use [Image1]"
    assert provenance["request"]["created_at"] == started["created_at"]
    assert provenance["status_record"]["created_at"] == started["created_at"]
    assert provenance["result"]["created_at"] == started["created_at"]
    assert provenance["result"]["prediction_id"] == "pred-done"
    assert provenance["seed_provenance"]["effective_seed"] == 2091770884

    listed = gen_server.list_generations(
        out_root=str(tmp_path / "out"),
        query="Use [Image1]",
        status_filter="succeeded",
    )
    assert listed["total"] == 1
    assert listed["generations"][0]["job_id"] == started["job_id"]
    assert listed["generations"][0]["seed_provenance"]["effective_seed"] == 2091770884
    assert (
        listed["generations"][0]["collected_at"]
        == result["result"]["collected_at"]
    )
    assert listed["generations"][0]["reference_files"][0]["sha256"]
    assert listed["generations"][0]["output_files"][0]["sha256"]

    second_read = gen_server.get_video_job(
        started["job_id"],
        out_root=str(tmp_path / "out"),
    )
    assert second_read["status"] == "succeeded"
    index_path = tmp_path / "out" / "jobs" / "index.ndjson"
    index_events = [
        json.loads(line)["event"] for line in index_path.read_text().splitlines()
    ]
    assert index_events == ["created", "succeeded"]


def test_list_generations_recovers_seed_from_pre_index_status(tmp_path: Path) -> None:
    out_root = tmp_path / "out"
    status_dir = out_root / "jobs" / "legacy-job"
    status_dir.mkdir(parents=True)
    status = {
        "job_id": "legacy-job",
        "prediction_id": "pred-legacy",
        "provider": "replicate",
        "status": "succeeded",
        "created_at": "2026-01-01T00:00:00Z",
        "title": "Legacy doorway shot",
        "model": "bytedance/seedance-2.0",
        "mode": "text_to_video",
        "prompt": "A doorway opens",
        "resolved_params": {"seed": None},
        "references": [],
        "job_dir": str(tmp_path / "legacy-output"),
        "provider_prediction": {"logs": "Using seed: 7654321\n"},
    }
    (status_dir / "status.json").write_text(
        json.dumps(status, indent=2) + "\n",
        encoding="utf-8",
    )

    listed = gen_server.list_generations(out_root=str(out_root))

    assert listed["index_path"].endswith("jobs/index.ndjson")
    assert listed["total"] == 1
    assert listed["generations"][0]["job_id"] == "legacy-job"
    assert listed["generations"][0]["seed_provenance"] == {
        "requested_seed": None,
        "effective_seed": 7654321,
        "source": "provider_logs",
        "matches_requested": None,
    }


def test_index_recovers_when_marker_exists_without_event(tmp_path: Path) -> None:
    out_root = tmp_path / "out"
    job_id = "marker-only-job"
    marker_dir = out_root / "jobs" / job_id
    marker_dir.mkdir(parents=True)
    (marker_dir / ".index-event-created").write_text("stale marker\n")
    status = {
        "job_id": job_id,
        "status": "starting",
        "references": [],
    }

    warning = gen_server._append_generation_index(
        out_root,
        status,
        event="created",
    )
    second_warning = gen_server._append_generation_index(
        out_root,
        status,
        event="created",
    )

    assert warning is None
    assert second_warning is None
    index_path = out_root / "jobs" / "index.ndjson"
    entries = [json.loads(line) for line in index_path.read_text().splitlines()]
    assert [(entry["job_id"], entry["event"]) for entry in entries] == [
        (job_id, "created")
    ]


def test_index_separates_new_event_after_partial_crash_tail(
    tmp_path: Path,
) -> None:
    out_root = tmp_path / "out"
    index_path = out_root / "jobs" / "index.ndjson"
    index_path.parent.mkdir(parents=True)
    index_path.write_bytes(b'{"job_id":"crashed"')
    status = {
        "job_id": "recovered-job",
        "status": "starting",
        "references": [],
    }

    warning = gen_server._append_generation_index(
        out_root,
        status,
        event="created",
    )

    assert warning is None
    entries, warnings = gen_server._read_generation_index(out_root)
    assert [entry["job_id"] for entry in entries] == ["recovered-job"]
    assert len(warnings) == 1
    assert index_path.read_bytes().startswith(b'{"job_id":"crashed"\n')


def test_index_recovers_after_partial_multibyte_crash_tail(
    tmp_path: Path,
) -> None:
    out_root = tmp_path / "out"
    index_path = out_root / "jobs" / "index.ndjson"
    index_path.parent.mkdir(parents=True)
    index_path.write_bytes(b'{"job_id":"crashed","prompt":"\xe2')
    status = {
        "job_id": "recovered-unicode-job",
        "status": "starting",
        "references": [],
    }

    warning = gen_server._append_generation_index(
        out_root,
        status,
        event="created",
    )

    assert warning is None
    entries, warnings = gen_server._read_generation_index(out_root)
    assert [entry["job_id"] for entry in entries] == [
        "recovered-unicode-job"
    ]
    assert len(warnings) == 1
    assert "invalid JSON" in warnings[0]


def test_shared_index_serializes_events_from_different_jobs(tmp_path: Path) -> None:
    out_root = tmp_path / "out"
    statuses = [
        {
            "job_id": f"job-{index}",
            "status": "starting",
            "references": [],
        }
        for index in range(8)
    ]

    with ThreadPoolExecutor(max_workers=8) as executor:
        warnings = list(
            executor.map(
                lambda status: gen_server._append_generation_index(
                    out_root,
                    status,
                    event="created",
                ),
                statuses,
            )
        )

    assert warnings == [None] * len(statuses)
    index_path = out_root / "jobs" / "index.ndjson"
    entries = [json.loads(line) for line in index_path.read_text().splitlines()]
    assert {entry["job_id"] for entry in entries} == {
        status["job_id"] for status in statuses
    }


def test_get_video_job_finalizes_terminal_status_without_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, image_path: Path
) -> None:
    monkeypatch.setattr(
        gen_server.seedance,
        "create_seedance_prediction",
        lambda **kwargs: {
            "id": "pred-terminal",
            "status": "succeeded",
            "version": "@latest",
            "created_at": "2026-05-09T10:00:00Z",
            "started_at": "2026-05-09T10:00:01Z",
            "completed_at": "2026-05-09T10:00:30Z",
            "output": ["https://example.com/fake.mp4"],
            "metrics": {"predict_time": 1.0},
            "error": None,
        },
    )

    monkeypatch.setattr(
        gen_server.seedance,
        "download_prediction_outputs",
        lambda **kwargs: [
            {
                "path": str(kwargs["out_dir"] / "fake_00.mp4"),
                "url": kwargs["outputs"][0],
                "bytes": 10,
                "_metrics": {"download_time_s": 0.1},
            }
        ],
    )
    monkeypatch.setattr(
        gen_server.seedance,
        "probe_media_info",
        lambda path: {
            "duration_s": None,
            "fps": None,
            "width": None,
            "height": None,
            "has_audio": None,
        },
    )

    started = gen_server.start_video_job(
        prompt="Use [Image1]",
        image=str(image_path),
        out_root=str(tmp_path / "out"),
    )
    saved_status = json.loads(Path(started["status_path"]).read_text(encoding="utf-8"))
    saved_status.pop("result", None)
    saved_status["outputs_downloaded"] = False
    Path(started["status_path"]).write_text(
        json.dumps(saved_status, indent=2) + "\n",
        encoding="utf-8",
    )

    result = gen_server.get_video_job(
        started["job_id"],
        out_root=str(tmp_path / "out"),
    )

    assert result["status"] == "succeeded"
    assert result["outputs_downloaded"] is True
    assert result["result"]["prediction_id"] == "pred-terminal"


def test_concurrent_terminal_merges_download_and_index_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, image_path: Path
) -> None:
    completed_prediction = {
        "id": "pred-concurrent",
        "status": "succeeded",
        "version": "@latest",
        "completed_at": "2026-05-09T10:00:30Z",
        "output": ["https://example.com/fake.mp4"],
        "metrics": {},
        "logs": "Using seed: 123456\n",
        "error": None,
    }
    monkeypatch.setattr(
        gen_server.seedance,
        "create_seedance_prediction",
        lambda **kwargs: {
            "id": "pred-concurrent",
            "status": "processing",
            "input": {},
        },
    )

    download_calls: list[int] = []

    def fake_download_prediction_outputs(**kwargs):
        download_calls.append(1)
        time.sleep(0.03)
        output_path = kwargs["out_dir"] / "fake_00.mp4"
        output_path.write_bytes(b"fake video")
        return [
            {
                "path": str(output_path),
                "url": kwargs["outputs"][0],
                "bytes": output_path.stat().st_size,
            }
        ]

    monkeypatch.setattr(
        gen_server.seedance,
        "download_prediction_outputs",
        fake_download_prediction_outputs,
    )
    monkeypatch.setattr(
        gen_server.seedance,
        "probe_media_info",
        lambda path: {},
    )

    out_root = tmp_path / "out"
    started = gen_server.start_video_job(
        prompt="Use [Image1]",
        image=str(image_path),
        out_root=str(out_root),
    )

    def finalize():
        return gen_server._merge_prediction_status(
            status=dict(started),
            prediction=completed_prediction,
            out_root=out_root,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: finalize(), range(2)))

    assert [result["status"] for result in results] == ["succeeded", "succeeded"]
    assert len(download_calls) == 1
    index_path = out_root / "jobs" / "index.ndjson"
    events = [json.loads(line)["event"] for line in index_path.read_text().splitlines()]
    assert events == ["created", "succeeded"]


def test_cancel_video_job_updates_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, image_path: Path
) -> None:
    monkeypatch.setattr(
        gen_server.seedance,
        "create_seedance_prediction",
        lambda **kwargs: {
            "id": "pred-cancel",
            "status": "processing",
            "version": "@latest",
            "created_at": "2026-05-09T10:00:00Z",
            "started_at": "2026-05-09T10:00:01Z",
            "completed_at": None,
            "output": None,
            "metrics": None,
            "error": None,
        },
    )
    monkeypatch.setattr(
        gen_server.seedance,
        "cancel_seedance_prediction",
        lambda prediction_id: {
            "id": prediction_id,
            "status": "canceled",
            "version": "@latest",
            "created_at": "2026-05-09T10:00:00Z",
            "started_at": "2026-05-09T10:00:01Z",
            "completed_at": "2026-05-09T10:00:03Z",
            "output": None,
            "metrics": None,
            "error": None,
        },
    )

    started = gen_server.start_video_job(
        prompt="Use [Image1]",
        image=str(image_path),
        out_root=str(tmp_path / "out"),
    )
    result = gen_server.cancel_video_job(
        started["job_id"],
        out_root=str(tmp_path / "out"),
    )

    assert result["status"] == "canceled"
    assert result["outputs_downloaded"] is False
    saved = json.loads(Path(result["status_path"]).read_text(encoding="utf-8"))
    assert saved["status"] == "canceled"
