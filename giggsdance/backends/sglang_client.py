"""SGLang Diffusion client for MiniMax H3.

Why SGLang rather than diffusers or vLLM
----------------------------------------
Measured on 8x B300 SXM6, 1344x768, 124 frames, 50 steps (SGLang's own sweep):

    FL2VA BF16   19.04 s   83.6 GB/GPU     load 118 s
    FL2VA FP8    18.03 s   51.9 GB/GPU     load 116 s
    Ref2VA BF16  29.12 s   84.0 GB/GPU     load 114 s
    Ref2VA FP8   27.12 s   52.8 GB/GPU     load 124 s

Two things decide the backend:

1. **vLLM-Omni cannot do 12 references.** Its documented limitation is one image
   plus one audio reference, or videos with no separate audio. SGLang's ref2va
   accepts combined image + video + audio conditions, which is the whole point of
   the omni-reference mode.
2. The reference diffusers implementation is BF16 written for clarity, not speed.
   SGLang measured 1.41x over it, and is where the Turbo LoRA recipes are
   documented.

References cost about 1.53x a text-only request (29.12 vs 19.04 s), because they
lengthen the packed sequence and attention is quadratic in it.

On quantization and quality
---------------------------
FP8 is **not** lossless, and on a B300 it is not really a speed lever either --
it bought 5% (19.04 -> 18.03 s) while cutting memory 38%. It is a *capacity*
tool, and with 288 GB you do not need capacity. What is genuinely lossless is
listed in :class:`QualityMode` below.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..references import ReferenceSet

DEFAULT_PORT = 30010
DEFAULT_STEPS = 50


class QualityMode(str, Enum):
    """What each mode costs you, measured rather than asserted.

    LOSSLESS  Bit-exact against the reference implementation. Speed comes only
              from parallelism, resident placement, VAE patch parallelism and
              warmup matching -- none of which touch the denoising math.

    HIGH      SGLang's audited Cache-DiT path: 1.40x faster (75.10 -> 53.70 s on
              4x H200) at SSIM 0.931 / PSNR 28.16 dB against lossless. Same seed,
              slightly different realisation. Note it is currently validated only
              for one exact workload and deployment.

    TURBO     A distilled LoRA, 4-8 denoiser evaluations instead of 50. By far
              the largest speedup available, and explicitly *not* the same model
              -- output differs, it is not "the same clip faster".

    FP8 is deliberately not a mode here. It is a memory option
    (``quantize_fp8``) because that is what it actually buys.
    """

    LOSSLESS = "lossless"
    HIGH = "high"
    TURBO = "turbo"


# Turbo LoRA recipes, exactly as SGLang documents them. num_inference_steps
# counts sigma grid points *including* the terminal zero, so the denoiser runs
# one fewer evaluation -- which is why a "4-step" adapter is configured as 5.
TURBO_LORAS: dict[str, dict[str, Any]] = {
    "lightx2v": {
        "repo": "lightx2v/Minimax-h3-Turbo",
        "file": "minimax_h3_fl2v_turbo_4step_v0.1.safetensors",
        "num_inference_steps": 5,      # 4 denoiser evaluations
        "lora_scale": 1.0,
        "lora_alpha": 8,               # required: absent from the checkpoint metadata
        "note": "most aggressive; alpha must be set explicitly",
    },
    "larryvrh": {
        "repo": "larryvrh/MiniMax-H3-Turbo-Lora",
        "file": "minimax_h3_turbo_v4_step600_ema.safetensors",
        "num_inference_steps": 9,      # 8 denoiser evaluations
        "lora_scale": 1.0,
        "lora_alpha": None,
        "note": "start here when fine detail matters more than latency",
    },
}


@dataclass
class Target:
    """Output geometry. SGLang derives the canvas and frame count from this."""

    duration_seconds: float = 5.0
    aspect_ratio: str = "16:9"
    short_edge: int = 768

    def to_payload(self) -> dict[str, Any]:
        return {
            "short_edge": int(self.short_edge),
            "aspect_ratio": self.aspect_ratio,
            "duration_seconds": float(self.duration_seconds),
        }


@dataclass
class RenderRequest:
    prompt: str
    task: str = "t2va"                       # t2va | fl2va | ref2va
    target: Target = field(default_factory=Target)
    seed: int = 0
    num_inference_steps: int = DEFAULT_STEPS
    flow_shift: float = 12.0
    audio_flow_shift: float = 3.0
    quality: QualityMode = QualityMode.LOSSLESS
    conditions: list[dict[str, Any]] = field(default_factory=list)
    lora_scale: float | None = None
    num_outputs: int = 1

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task": self.task,
            "prompt": self.prompt,
            "conditions": self.conditions,
            "target": self.target.to_payload(),
            "seed": int(self.seed),
            "num_inference_steps": int(self.num_inference_steps),
            "flow_shift": self.flow_shift,
            "audio_flow_shift": self.audio_flow_shift,
        }
        # Only send `quality` for the audited accelerated path. TURBO is a LoRA,
        # not a quality level, and sending both is explicitly not validated.
        if self.quality is QualityMode.HIGH:
            payload["quality"] = "high"
        elif self.quality is QualityMode.LOSSLESS:
            payload["quality"] = "lossless"
        if self.lora_scale is not None:
            payload["lora_scale"] = self.lora_scale
        if self.num_outputs > 1:
            payload["num_outputs_per_prompt"] = int(self.num_outputs)
        return payload


def build_conditions(references: ReferenceSet, task: str) -> list[dict[str, Any]]:
    """Map a validated reference set onto SGLang's ``conditions`` list.

    ``fl2va`` treats images as *keyframes* -- the literal first and/or last frame
    of the clip, with supported index sets [0], [-1] and [0, -1]. ``ref2va``
    treats them as semantic references that may be recomposed or cropped.
    Confusing the two is the most common mistake here: a screenshot you want
    animated from that exact composition is fl2va, a character you want to
    reappear is ref2va.

    Order is preserved, because it must match the one-based material tags
    (``<Picture 1>``, ``<Video 1>``, ``<Audio 1>``) in the prompt.
    """
    def uri(path: str) -> str:
        return path if "://" in path else f"file://{Path(path).resolve()}"

    if task == "fl2va":
        images = references.images
        if len(images) > 2:
            raise ValueError(
                f"fl2va takes at most 2 keyframes (first and last), got {len(images)}. "
                "Use ref2va for more images -- they are references, not keyframes."
            )
        indices = {1: [0], 2: [0, -1]}.get(len(images), [])
        return [
            {"role": "keyframe", "type": "image", "uri": uri(path), "frame_index": index}
            for path, index in zip(images, indices)
        ]

    if task != "ref2va":
        return []

    type_for = {"image": "image", "video": "video", "audio": "audio"}
    return [
        {"role": "reference", "type": type_for[kind], "uri": uri(path)}
        for kind, path in references.order
    ]


class SGLangError(RuntimeError):
    pass


class SGLangClient:
    """Minimal client for SGLang's asynchronous OpenAI-compatible video API."""

    def __init__(self, base_url: str = f"http://127.0.0.1:{DEFAULT_PORT}", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # -- plumbing ---------------------------------------------------------

    def _request(self, method: str, path: str, body: dict | None = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if data else {}
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                if response.headers.get("Content-Type", "").startswith("application/json"):
                    return json.loads(raw)
                return raw
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise SGLangError(f"HTTP {exc.code} on {method} {path}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise SGLangError(f"cannot reach SGLang at {self.base_url}: {exc.reason}") from exc

    def healthy(self) -> bool:
        try:
            self._request("GET", "/health")
            return True
        except SGLangError:
            return False

    def wait_until_healthy(self, deadline_s: float = 1800.0, interval: float = 3.0) -> float:
        """Block until the server answers /health. Returns seconds waited.

        The first launch has to load ~144 GB, which SGLang measured at 114-124 s
        on B300, so the default deadline is deliberately generous.
        """
        started = time.time()
        while time.time() - started < deadline_s:
            if self.healthy():
                return time.time() - started
            time.sleep(interval)
        raise SGLangError(f"SGLang did not become healthy within {deadline_s}s")

    # -- generation -------------------------------------------------------

    def submit(self, request: RenderRequest) -> str:
        response = self._request("POST", "/v1/videos", request.to_payload())
        for key in ("id", "video_id", "task_id"):
            value = (response or {}).get(key)
            if isinstance(value, str) and value:
                return value
        raise SGLangError(f"no job id in response: {json.dumps(response)[:400]}")

    def status(self, video_id: str) -> dict:
        return self._request("GET", f"/v1/videos/{video_id}")

    def download(self, video_id: str, destination: Path, variant: int | None = None) -> Path:
        path = f"/v1/videos/{video_id}/content"
        if variant is not None:
            path += f"?variant={variant}"
        payload = self._request("GET", path)
        if not isinstance(payload, (bytes, bytearray)):
            raise SGLangError("expected MP4 bytes from the content endpoint")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return destination

    def render(
        self,
        request: RenderRequest,
        destination: Path,
        poll_interval: float = 2.0,
        max_wait_s: float = 3600.0,
    ) -> tuple[Path, float]:
        """Submit, poll, download. Returns (path, seconds_elapsed)."""
        started = time.time()
        video_id = self.submit(request)

        while True:
            if time.time() - started > max_wait_s:
                raise SGLangError(f"generation exceeded {max_wait_s}s")
            state = str((self.status(video_id) or {}).get("status", "")).lower()
            if state in ("completed", "succeeded", "success"):
                break
            if state in ("failed", "error", "cancelled"):
                raise SGLangError(
                    f"job {video_id} ended as {state}: "
                    f"{json.dumps(self.status(video_id))[:400]}"
                )
            time.sleep(poll_interval)

        return self.download(video_id, destination), time.time() - started


def build_serve_command(
    model_path: str,
    variant: str = "fl2va",
    port: int = DEFAULT_PORT,
    num_gpus: int = 1,
    quantize_fp8: bool = False,
    lora: str | None = None,
    warmup_resolution: str | None = "1344x768",
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build the ``sglang serve`` command.

    Lossless choices baked in:

    * ``--performance-mode speed`` keeps components resident. For H3 this
      deliberately leaves the DiT eager, because SGLang's torch.compile path
      changes numerical output and must not be used for ground truth.
    * ``--ulysses-degree == num_gpus`` (pure Ulysses). SGLang measured this as
      both the capacity default and the faster topology on H200; a U2xRing2
      hybrid currently fails with an attention-mask mismatch.
    * ``--warmup-resolutions`` matched to the served shape removes about 10 s of
      first-request cost, for free.

    ``quantize_fp8`` is exposed but off by default: on B300 it measured 5% faster
    while cutting memory 38%, so it is a capacity tool, and it is not bit-exact.
    """
    command = [
        "sglang", "serve",
        "--model-path", model_path,
        "--model-variant", variant,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--performance-mode", "speed",
        "--num-gpus", str(num_gpus),
        "--ulysses-degree", str(num_gpus),
    ]
    if warmup_resolution:
        command += ["--warmup-resolutions", warmup_resolution]
    if quantize_fp8:
        command += ["--quantization", "fp8"]
    if lora:
        recipe = TURBO_LORAS.get(lora)
        if recipe is None:
            raise ValueError(f"unknown LoRA {lora!r}; known: {sorted(TURBO_LORAS)}")
        # SGLang pins adapters by repo plus filename. The exact flag spelling is
        # version-dependent and was not verifiable from the docs at the time of
        # writing, so it is passed through here and can be overridden with
        # extra_args if a future release renames it.
        command += ["--lora-path", f"{recipe['repo']}:{recipe['file']}"]
        if recipe.get("lora_alpha") is not None:
            command += ["--lora-alpha", str(recipe["lora_alpha"])]
    command += list(extra_args or [])
    return command


def steps_for(mode: QualityMode, lora: str | None, default: int = DEFAULT_STEPS) -> int:
    """The step count that goes with a mode/LoRA pair."""
    if lora:
        return int(TURBO_LORAS[lora]["num_inference_steps"])
    return default
