from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from gemini_video_prompts_mcp import fal_video as fal, server


def png(path):
    Image.new("RGB", (32, 24), "blue").save(path)
    return str(path)


def build(**kwargs):
    return fal.build_request(**{ "prompt": "A cup", "image": "cup.png", **kwargs})


@pytest.mark.parametrize("resolution", ["480p", "768p", "480P", "768P"])
def test_resolution_and_preview_no_io(tmp_path, monkeypatch, resolution):
    monkeypatch.setattr(fal, "api_key", lambda: pytest.fail("credentials read"))
    result = server.start_fal_video_job(prompt="A cup", image="missing.png", resolution=resolution,
                                       dry_run=True, out_root=str(tmp_path / "out"))
    assert result["resolved_params"]["resolution"] == resolution.upper()
    assert result["billing_confirmation_required"] is True
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kwargs", [
    {"resolution": "720p"}, {"duration": 4}, {"duration": 16}, {"duration": True},
    {"duration": 5.5}, {"model": "minimax/h3/image-to-video"}, {"image": None},
    {"reference_images": ["r.png"]}, {"aspect_ratio": "16:9"},
    {"prompt_expansion_mode": "disabled"}, {"seed": True}, {"prompt": " "},
    {"image": "https://example.com/p.png"}, {"enable_safety_checker": "false"},
    {"model": fal.R2V, "image": None, "reference_audios": ["a.wav"]},
    {"model": fal.R2V, "image": None, "reference_images": ["a.png"] * 10},
    {"model": fal.R2V, "image": None, "reference_images": ["a.png"] * 9,
     "reference_videos": ["v.mp4"] * 3, "reference_audios": ["a.wav"]},
])
def test_invalid_before_submission(kwargs):
    with pytest.raises(RuntimeError, match="INVALID_INPUT"):
        build(**kwargs)


def test_billing_gate_precedes_key_files_and_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(fal, "api_key", lambda: pytest.fail("credentials read"))
    with pytest.raises(RuntimeError, match="API_BILLING_CONFIRMATION_REQUIRED"):
        server.start_fal_video_job(prompt="A cup", image="missing.png", out_root=str(tmp_path))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("model", [fal.T2V, fal.T2V_TURBO])
def test_text_to_video_preview_shape_and_reference_rejection(tmp_path, monkeypatch, model):
    monkeypatch.setattr(fal, "api_key", lambda: pytest.fail("credentials read"))
    plan = server.start_fal_video_job(
        prompt="A cup", model=model, resolution="480p", dry_run=True,
        out_root=str(tmp_path / "out"),
    )
    assert plan["resolved_params"] == {
        "prompt": "A cup", "duration": 5, "resolution": "480P",
        "prompt_expansion_mode": "balanced", "enable_safety_checker": True,
        "sync_mode": False, "aspect_ratio": "16:9",
    }
    assert plan["references"] == []
    with pytest.raises(RuntimeError, match="text-to-video does not accept"):
        fal.build_request(prompt="A cup", model=model, image="frame.png")
    with pytest.raises(RuntimeError, match="text-to-video does not accept"):
        fal.build_request(prompt="A cup", model=model, reference_images=["ref.png"])


@pytest.mark.parametrize("model", [fal.T2V, fal.T2V_TURBO])
def test_text_to_video_submits_exact_route_and_payload(tmp_path, transport, model):
    state, calls = transport
    state["root"] = str(tmp_path / "out")
    server.start_fal_video_job(
        prompt="A cup", model=model, resolution="480p", aspect_ratio="9:16",
        seed=9, out_root=state["root"], allow_api_billing=True,
    )
    assert calls[0].url.path == "/" + model
    assert json.loads(calls[0].content) == {
        "prompt": "A cup", "duration": 5, "resolution": "480P",
        "prompt_expansion_mode": "balanced", "enable_safety_checker": True,
        "sync_mode": False, "aspect_ratio": "9:16", "seed": 9,
    }


def test_ordered_references_and_exact_bytes(tmp_path):
    a = png(tmp_path / "a.png")
    b = png(tmp_path / "b.png")
    params, refs = build(model=fal.R2V, image=None, reference_images=[b, a],
                         prompt_expansion_mode="quality")
    payload, refs = fal.prepare_inputs(params, refs)
    assert [r["token"] for r in refs] == ["Image 1", "Image 2"]
    assert [r["path"] for r in refs] == [b, a]
    assert payload["prompt_expansion_mode"] == "quality"
    assert payload["aspect_ratio"] == "adaptive"
    for uri, ref in zip(payload["reference_image_urls"], refs):
        raw = base64.b64decode(uri.split(",", 1)[1])
        assert raw == Path(ref["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == ref["sha256"]


@pytest.mark.parametrize("durations", [[1], [16], [8, 8]])
def test_reference_duration_limits(tmp_path, monkeypatch, durations):
    paths = []
    for i, _ in enumerate(durations):
        path = tmp_path / f"{i}.mp4"
        path.write_bytes(b"fixture")
        paths.append(str(path))
    monkeypatch.setattr(fal, "probe_media_info", lambda path: dict(duration_s=durations[int(Path(path).stem)], width=32))
    params, refs = build(model=fal.R2V, image=None, reference_videos=paths)
    with pytest.raises(RuntimeError, match="INVALID_INPUT"):
        fal.prepare_inputs(params, refs)


@pytest.fixture
def transport(monkeypatch):
    real_client = httpx.Client
    calls = []
    state = {"queue_status": "COMPLETED", "submit_error": False, "download_error": False,
             "generation_error": False}
    def handle(req):
        calls.append(req)
        if req.method == "POST":
            if state["submit_error"]:
                raise httpx.ReadTimeout("private error contents", request=req)
            # The durable record must already exist before the billable POST.
            assert list(Path(state["root"]).glob("jobs/*/request.json"))
            return httpx.Response(200, json=dict(request_id="req-123",
                status_url="https://queue.fal.run/minimax/h3-max/requests/req-123/status",
                response_url="https://queue.fal.run/minimax/h3-max/requests/req-123",
                cancel_url="https://queue.fal.run/minimax/h3-max/requests/req-123/cancel"))
        if req.method == "PUT":
            return httpx.Response(202, json={"status": "CANCELLATION_REQUESTED"})
        if req.url.path.endswith("/status"):
            data = {"status": state["queue_status"]}
            if state["generation_error"]:
                data["error"] = "provider failure"
            return httpx.Response(200, json=data)
        if req.url.host == "queue.fal.run":
            return httpx.Response(200, json={"video": {"url": "https://v3.fal.media/test.mp4"},
                                "expanded_prompt": "A richer cup shot", "seed": 42,
                                "timings": {"inference": 2.5}})
        assert "authorization" not in req.headers
        return httpx.Response(503 if state["download_error"] else 200, content=b"original video bytes")
    monkeypatch.setattr(fal.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw))
    monkeypatch.setattr(fal, "api_key", lambda: "test-key")
    monkeypatch.setattr(fal, "probe_media_info", lambda _: {"duration_s": 5, "width": 32, "height": 24})
    return state, calls


def submit(tmp_path, transport):
    state, _ = transport
    state["root"] = str(tmp_path / "out")
    return server.start_fal_video_job(prompt="A cup", image=png(tmp_path / "a.png"),
                                      out_root=state["root"], allow_api_billing=True)


def test_full_queue_lifecycle_and_no_duplicate_collection(tmp_path, transport):
    state, calls = transport
    job = submit(tmp_path, transport)
    assert calls[0].url.path == "/minimax/h3-max/image-to-video"
    assert json.loads(calls[0].content)["image_url"].startswith("data:image/png;base64,")
    root, job_id = state["root"], job["job_id"]
    assert server.get_video_job(job_id, root, poll=False)["status"] == "queued"
    assert len(calls) == 1
    state["queue_status"] = "IN_PROGRESS"
    assert server.get_video_job(job_id, root)["status"] == "processing"
    state["queue_status"] = "COMPLETED"
    result = server.get_video_job(job_id, root)
    assert result["status"] == "succeeded"
    assert result["result"]["expanded_prompt"] == "A richer cup shot"
    assert result["result"]["prompt"] == "A cup"
    assert result["seed_provenance"]["returned_seed"] == 42
    output = result["result"]["outputs"][0]
    assert Path(output["path"]).read_bytes() == b"original video bytes"
    assert output["sha256"] == hashlib.sha256(b"original video bytes").hexdigest()
    count = len(calls)
    assert server.get_video_job(job_id, root) == result
    assert server.get_generation(job_id, root)["result"] == result["result"]
    assert server.list_generations(root)["generations"][0]["provider"] == "fal"
    assert len(calls) == count
    assert "data:image" not in "".join(p.read_text() for p in Path(root).rglob("*.json"))


def test_uncertain_submit_is_durable_and_not_retried(tmp_path, transport):
    state, calls = transport
    state["submit_error"] = True
    with pytest.raises(RuntimeError, match="FAL_SUBMISSION_UNCERTAIN") as exc:
        submit(tmp_path, transport)
    assert "private error" not in str(exc.value)
    assert len(calls) == 1
    status = json.loads(next(Path(state["root"]).glob("jobs/*/status.json")).read_text())
    assert status["status"] == "submission_unknown"
    with pytest.raises(RuntimeError, match="FAL_SUBMISSION_UNCERTAIN"):
        server.get_video_job(status["job_id"], state["root"])
    assert len(calls) == 1


def test_collection_retry_does_not_resubmit(tmp_path, transport):
    state, calls = transport
    job = submit(tmp_path, transport)
    state["download_error"] = True
    with pytest.raises(RuntimeError, match="FAL_DOWNLOAD_ERROR"):
        server.get_video_job(job["job_id"], state["root"])
    state["download_error"] = False
    assert server.get_video_job(job["job_id"], state["root"])["status"] == "succeeded"
    assert sum(r.method == "POST" for r in calls) == 1


def test_cancel_acceptance_does_not_claim_stopped(tmp_path, transport):
    state, _ = transport
    job = submit(tmp_path, transport)
    result = server.cancel_video_job(job["job_id"], state["root"])
    assert result["status"] == "queued"
    assert result["cancellation_response"]["status"] == "CANCELLATION_REQUESTED"
    assert server.get_video_job(job["job_id"], state["root"])["status"] == "succeeded"


def test_generation_error_is_terminal_without_download(tmp_path, transport):
    state, calls = transport
    job = submit(tmp_path, transport)
    state["generation_error"] = True
    assert server.get_video_job(job["job_id"], state["root"])["status"] == "failed"
    assert len(calls) == 2


def test_credentials_not_sent_to_receipt_redirect_or_other_host(monkeypatch):
    monkeypatch.setattr(fal, "api_key", lambda: pytest.fail("key accessed"))
    with pytest.raises(RuntimeError, match="FAL_INVALID_RESPONSE"):
        fal.queue_request("GET", "https://example.com/status")


def test_reference_endpoint_and_last_frame_payload(tmp_path, transport):
    state, calls = transport
    state['root'] = str(tmp_path / 'out')
    image = png(tmp_path / 'one.png')
    server.start_fal_video_job(prompt='Image 1 walks.', model=fal.R2V,
                              reference_images=[image], prompt_expansion_mode='quality',
                              resolution='480p', out_root=state['root'], allow_api_billing=True)
    assert calls[0].url.path == '/minimax/h3-max/reference-to-video'
    payload = json.loads(calls[0].content)
    assert payload['reference_image_urls'] and 'image_url' not in payload
    assert payload['prompt_expansion_mode'] == 'quality' and payload['resolution'] == '480P'
    params, refs = build(image=image, last_frame_image=image)
    payload, _ = fal.prepare_inputs(params, refs)
    assert payload['image_url'] == payload['end_image_url']


def test_fal_doctor_independent_of_other_credentials(monkeypatch):
    from riff_mcp_doctor import doctor
    monkeypatch.setattr(doctor, 'load_dotenv_if_available', lambda: False)
    monkeypatch.setenv('FAL_KEY', 'test-key')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.delenv('REPLICATE_API_TOKEN', raising=False)
    checks = doctor.run_fal_checks(network=True)
    assert not any(check.status == 'fail' for check in checks)
    assert [check.name for check in checks if check.category == 'env'] == ['FAL_KEY']
    assert checks[-1].status == 'skipped'
