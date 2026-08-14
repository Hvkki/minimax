"""HTTP API tests: auth, validation, routing.

Runs against a real FastAPI TestClient with the Modal-backed pieces stubbed, so
it needs no GPU, no Modal account and no weights. What it covers is exactly the
part that must not be wrong on a public URL: that requests without a valid key
are refused, and that an illegal request is rejected *before* anything is
spawned onto a paid GPU.

Run: python -m pytest tests/test_api.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

KEY = "test-key-do-not-use-in-production"


class FakeCall:
    """Stand-in for modal.FunctionCall."""

    def __init__(self, object_id: str = "fc-stub"):
        self.object_id = object_id


class FakeMethod:
    def __init__(self, recorder: list):
        self.recorder = recorder

    def spawn(self, job_id, spec):
        self.recorder.append((job_id, spec))
        return FakeCall()


class FakeRenderer:
    """Records how the renderer was configured without launching anything."""

    calls: list = []
    options: list = []

    def __init__(self):
        self.render = FakeMethod(FakeRenderer.calls)

    @classmethod
    def with_options(cls, **kwargs):
        cls.options.append(kwargs)
        return cls


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    import serve_api

    monkeypatch.setenv("API_KEY", KEY)
    monkeypatch.setattr(serve_api, "jobs", {})
    FakeRenderer.calls.clear()
    FakeRenderer.options.clear()
    monkeypatch.setattr(serve_api, "Renderer", FakeRenderer)

    return TestClient(serve_api.build_api())


def auth() -> dict:
    return {"X-API-Key": KEY}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_health_needs_no_key(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.parametrize("path", ["/", "/jobs", "/jobs/abc"])
def test_reads_require_a_key(client, path):
    assert client.get(path).status_code == 401


def test_render_requires_a_key(client):
    assert client.post("/render", json={"prompt": "x"}).status_code == 401


def test_wrong_key_is_refused(client):
    response = client.post("/render", json={"prompt": "x"},
                           headers={"X-API-Key": "wrong"})
    assert response.status_code == 401
    assert not FakeRenderer.calls, "a bad key must never reach the GPU"


def test_missing_server_secret_is_a_server_error_not_an_open_door(client, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    response = client.get("/", headers={"X-API-Key": ""})
    assert response.status_code == 500


# ---------------------------------------------------------------------------
# Validation happens before any GPU is spawned
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "duration,ok",
    [(5.0, True), (14.375, True), (15.0, True), (2.0, False), (20.0, False)],
)
def test_duration_is_bounded_by_the_model(client, duration, ok):
    response = client.post("/render", json={"prompt": "x", "duration_s": duration},
                           headers=auth())
    assert (response.status_code == 202) is ok
    if not ok:
        assert not FakeRenderer.calls


def test_thirteen_references_are_rejected(client):
    references = [{"type": "image", "uri": f"https://e/{n}.png"} for n in range(13)]
    response = client.post("/render", json={"prompt": "x", "references": references},
                           headers=auth())
    assert response.status_code == 422
    assert not FakeRenderer.calls, "an illegal set must not cost a GPU second"


def test_audio_only_references_are_rejected(client):
    response = client.post(
        "/render",
        json={"prompt": "x", "references": [{"type": "audio", "uri": "https://e/a.wav"}]},
        headers=auth(),
    )
    assert response.status_code == 422
    assert "only input" in response.json()["detail"]
    assert not FakeRenderer.calls


def test_ten_images_are_rejected(client):
    references = [{"type": "image", "uri": f"https://e/{n}.png"} for n in range(10)]
    response = client.post("/render", json={"prompt": "x", "references": references},
                           headers=auth())
    assert response.status_code == 422
    assert not FakeRenderer.calls


def test_empty_prompt_is_rejected(client):
    assert client.post("/render", json={"prompt": ""}, headers=auth()).status_code == 422


def test_unknown_resolution_is_rejected(client):
    response = client.post("/render", json={"prompt": "x", "resolution": "8k"},
                           headers=auth())
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Happy path and routing decisions
# ---------------------------------------------------------------------------


def test_text_only_render_is_accepted_and_uses_the_cheap_partition(client):
    response = client.post("/render", json={"prompt": "a lighthouse at dusk"},
                           headers=auth())
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued" and len(body["job_id"]) == 32

    assert len(FakeRenderer.calls) == 1
    options = FakeRenderer.options[-1]
    assert options["env"]["GIGGSDANCE_VARIANT"] == "fl2va"
    assert options["gpu"] == "B300"          # single GPU: no ":n" suffix


def test_twelve_references_route_to_ref2va(client):
    references = (
        [{"type": "image", "uri": f"https://e/{n}.png"} for n in range(9)]
        + [{"type": "video", "uri": "https://e/a.mp4"},
           {"type": "video", "uri": "https://e/b.mp4"},
           {"type": "audio", "uri": "https://e/v.wav"}]
    )
    response = client.post("/render", json={"prompt": "x", "references": references},
                           headers=auth())
    assert response.status_code == 202
    assert FakeRenderer.options[-1]["env"]["GIGGSDANCE_VARIANT"] == "ref2va"

    _, spec = FakeRenderer.calls[-1]
    assert len(spec["reference_order"]) == 12
    # Order must survive: it drives the <Picture N>/<Video N>/<Audio N> tags.
    assert [kind for kind, _ in spec["reference_order"]] == \
        ["image"] * 9 + ["video"] * 2 + ["audio"]


def test_two_keyframes_stay_on_fl2va(client):
    references = [{"type": "image", "uri": "https://e/a.png"},
                  {"type": "image", "uri": "https://e/b.png"}]
    client.post("/render", json={"prompt": "x", "references": references}, headers=auth())
    assert FakeRenderer.options[-1]["env"]["GIGGSDANCE_VARIANT"] == "fl2va"


def test_multi_gpu_requests_the_right_shape(client):
    client.post("/render", json={"prompt": "x", "gpus": 4}, headers=auth())
    options = FakeRenderer.options[-1]
    assert options["gpu"] == "B300:4"
    assert options["env"]["GIGGSDANCE_GPUS"] == "4"


def test_lora_and_fp8_are_passed_through_only_when_asked(client):
    client.post("/render", json={"prompt": "x"}, headers=auth())
    plain = FakeRenderer.options[-1]["env"]
    assert "GIGGSDANCE_LORA" not in plain and "GIGGSDANCE_FP8" not in plain

    client.post("/render", json={"prompt": "x", "lora": "lightx2v",
                                 "quantize_fp8": True}, headers=auth())
    opted = FakeRenderer.options[-1]["env"]
    assert opted["GIGGSDANCE_LORA"] == "lightx2v"
    assert opted["GIGGSDANCE_FP8"] == "1"


def test_gpu_count_is_capped(client):
    assert client.post("/render", json={"prompt": "x", "gpus": 99},
                       headers=auth()).status_code == 422


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------


def test_unknown_job_is_404(client):
    assert client.get("/jobs/nope", headers=auth()).status_code == 404
    assert client.get("/jobs/nope/file", headers=auth()).status_code == 404


def test_submitted_job_appears_in_the_listing(client):
    job_id = client.post("/render", json={"prompt": "x"}, headers=auth()).json()["job_id"]
    listing = client.get("/jobs", headers=auth()).json()["jobs"]
    assert any(entry["job_id"] == job_id for entry in listing)


def test_capabilities_reports_the_real_limits(client):
    body = client.get("/", headers=auth()).json()
    assert body["references"] == {
        "images": 9, "videos": 3, "audios": 3, "total": 12,
        "audio_requires_visual": True, "order_is_semantic": True,
    }
    assert body["generation"]["frame_rule"] == "17n+5"
    assert body["generation"]["native_fps"] == 24
    # The B300 rate is a third-party figure and must be reported as unverified.
    assert body["gpu"]["rate_verified"] is False
