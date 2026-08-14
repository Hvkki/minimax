#!/usr/bin/env python3
"""
Giggsdance HTTP API — a permanent HTTPS endpoint on Modal. No ngrok.

    modal secret create giggsdance-api API_KEY=$(openssl rand -hex 32)
    modal deploy serve_api.py

You get a stable URL:

    https://<workspace>--giggsdance-api-fastapi-app.modal.run

--------------------------------------------------------------------------------
WHY NO NGROK
--------------------------------------------------------------------------------
ngrok exists to expose a server on *your own machine*. Nothing here runs on your
machine. Modal serves this container over public HTTPS with a real certificate,
so routing through ngrok would only add a hop, a dependency, and a laptop that
must stay awake. If you ever do want a raw TCP tunnel -- to talk to SGLang's own
port directly, say -- Modal has `modal.forward(port)` built in at no extra charge.

--------------------------------------------------------------------------------
ARCHITECTURE
--------------------------------------------------------------------------------
    CPU container   FastAPI, no torch, no SGLang    scales to zero, pennies
          | .spawn()
    GPU container   B300 + SGLang, ~144 GB          wakes only to render

The split matters. Serving JSON from the GPU container would bill B300 rates
($7.10/hr) to answer HTTP. The API image deliberately omits torch entirely so it
cold-starts in seconds rather than minutes.

Everything is asynchronous because renders take 20 s to 13 minutes -- a 15 s
Ref2VA clip measured 784 s on 4x B300. No HTTP request survives that, so the
contract is submit -> poll -> download.

--------------------------------------------------------------------------------
ENDPOINTS
--------------------------------------------------------------------------------
    GET  /health              no auth, readiness probe
    GET  /                    capabilities and limits
    POST /uploads             multipart, returns a URI usable as a reference
    POST /render              returns { job_id }
    GET  /jobs/{id}           status, stage, elapsed, cost
    GET  /jobs/{id}/file      the mp4
    GET  /jobs                recent jobs
    DELETE /jobs/{id}         cancel or delete

All except /health need `X-API-Key`.

Powered by MiniMax H3. See NOTICE.md -- the licence excludes the EU, UK, South
Korea and USA and covers outputs as well as weights. Mark published output as
AI-generated.
"""

from __future__ import annotations

import os
import secrets as pysecrets
import time
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

import modal
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from giggsdance.constraints import MAX_DURATION_S, MIN_DURATION_S
from giggsdance.references import MAX_TOTAL, ReferenceError, ReferenceSet, validate

# --------------------------------------------------------------------------
# Shared config
# --------------------------------------------------------------------------

MODEL_ID = "MiniMaxAI/MiniMax-H3"
WEIGHTS_DIR = "/weights"
MODEL_PATH = f"{WEIGHTS_DIR}/MiniMax-H3"
OUTPUT_DIR = "/outputs"
UPLOAD_DIR = "/uploads"
PORT = 30010

GPU = os.environ.get("GIGGSDANCE_GPU", "B300")
GPU_RATES = {"B200": 6.25 / 3600, "B200+": 6.25 / 3600, "B300": 7.10 / 3600}
UNVERIFIED_RATES = {"B300"}
RATE = GPU_RATES.get(GPU, GPU_RATES["B200"])

ONE_PARTITION_GB = 144.06
RESULT_TTL_HOURS = 48

class Reference(BaseModel):
    type: Literal["image", "video", "audio"]
    uri: str


class RenderSpec(BaseModel):
    """One render request. Every limit here mirrors H3's real constraints."""

    prompt: str = Field(min_length=1, max_length=7000)
    duration_s: float = 5.0
    aspect_ratio: str = "16:9"
    short_edge: Literal[768, 1440] = 768
    resolution: Literal["native", "720p", "1080p", "1440p", "2160p"] = "native"
    fps: Literal[24, 30, 60] = 60
    seed: int = 0
    steps: Optional[int] = None
    quality: Literal["lossless", "high"] = "lossless"
    lora: Literal["none", "lightx2v", "larryvrh"] = "none"
    gpus: int = Field(default=1, ge=1, le=8)
    quantize_fp8: bool = False
    crf: int = Field(default=18, ge=0, le=51)
    preset: str = "veryfast"
    references: list[Reference] = Field(default_factory=list)

    @field_validator("duration_s")
    @classmethod
    def _duration(cls, value: float) -> float:
        if not (MIN_DURATION_S - 1e-6 <= value <= MAX_DURATION_S + 1e-6):
            raise ValueError(
                f"duration_s must be {MIN_DURATION_S}-{MAX_DURATION_S}; H3 cannot "
                "generate shorter or longer. Trim in post for under 5 s."
            )
        return value

    @field_validator("references")
    @classmethod
    def _references(cls, value: list) -> list:
        if len(value) > MAX_TOTAL:
            raise ValueError(f"at most {MAX_TOTAL} references, got {len(value)}")
        return value


app = modal.App("giggsdance-api")

weights = modal.Volume.from_name("h3-weights", create_if_missing=True)
outputs = modal.Volume.from_name("giggsdance-outputs", create_if_missing=True)
uploads = modal.Volume.from_name("giggsdance-uploads", create_if_missing=True)
jobs = modal.Dict.from_name("giggsdance-jobs", create_if_missing=True)

# Required. A public URL with no auth is an open invitation to spend your credits.
api_secret = modal.Secret.from_name("giggsdance-api", required_keys=["API_KEY"])

# The API image has no torch and no SGLang on purpose: it exists to answer HTTP.
api_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]==0.115.*", "pydantic>=2.7", "numpy")
    .add_local_python_source("giggsdance")
)

# CUDA 13.1+ so the image can run on either B200 or B300.
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "wget")
    .pip_install("torch", "torchvision",
                 extra_index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("sglang[diffusion]", "huggingface_hub[hf_transfer]",
                 "numpy", "pillow", "av",
                 # this module is imported in the GPU container as well
                 "fastapi[standard]==0.115.*", "pydantic>=2.7")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": f"{WEIGHTS_DIR}/hf"})
    .add_local_python_source("giggsdance")
)


# --------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------

@app.function(image=gpu_image, volumes={WEIGHTS_DIR: weights},
              timeout=6 * 60 * 60, cpu=8.0, memory=16384)
def fetch_weights(variant: str = "fl2va", force: bool = False) -> dict:
    """Download the partition SGLang needs, once. Idempotent."""
    from huggingface_hub import snapshot_download

    wanted = "Ref2VA" if variant == "ref2va" else "FL2VA"
    marker = Path(MODEL_PATH) / wanted

    def gb(path: Path) -> float:
        if not path.is_dir():
            return 0.0
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9

    present = gb(marker)
    if not force and present > 100:
        return {"downloaded": False, "gb": present, "variant": variant}

    started = time.time()
    snapshot_download(MODEL_ID, local_dir=MODEL_PATH, max_workers=16,
                      allow_patterns=[f"{wanted}/*", "*.json", "*.txt"])
    weights.commit()
    return {"downloaded": True, "gb": gb(marker),
            "seconds": time.time() - started, "variant": variant}


# --------------------------------------------------------------------------
# GPU renderer
# --------------------------------------------------------------------------

@app.cls(
    image=gpu_image, gpu=GPU,
    volumes={WEIGHTS_DIR: weights, OUTPUT_DIR: outputs, UPLOAD_DIR: uploads},
    timeout=2 * 60 * 60, memory=200 * 1024,
    min_containers=0,
    scaledown_window=600,   # 10 min warm reuse: the load is ~115 s, so batching pays
    max_containers=2,       # hard ceiling on simultaneous GPU spend
)
class Renderer:
    @modal.enter()
    def start(self):
        import subprocess

        from giggsdance.backends.sglang_client import SGLangClient, build_serve_command

        variant = os.environ.get("GIGGSDANCE_VARIANT", "fl2va")
        gpus = int(os.environ.get("GIGGSDANCE_GPUS", "1"))
        lora = os.environ.get("GIGGSDANCE_LORA") or None
        fp8 = os.environ.get("GIGGSDANCE_FP8") == "1"

        command = build_serve_command(
            MODEL_PATH if Path(MODEL_PATH).exists() else MODEL_ID,
            variant=variant, port=PORT, num_gpus=gpus, quantize_fp8=fp8, lora=lora,
        )
        print("launching:", " ".join(command))

        self.logfile = open("/tmp/sglang.log", "wb")
        self.proc = subprocess.Popen(command, stdout=self.logfile,
                                     stderr=subprocess.STDOUT)
        self.client = SGLangClient(f"http://127.0.0.1:{PORT}")

        began = time.perf_counter()
        try:
            self.client.wait_until_healthy(deadline_s=2400)
        except Exception:
            tail = Path("/tmp/sglang.log").read_text(errors="replace")[-4000:]
            raise RuntimeError(f"SGLang failed to start:\n{tail}")
        self.load_seconds = time.perf_counter() - began
        self.variant, self.gpus, self.lora = variant, gpus, lora
        print(f"SGLang healthy in {self.load_seconds:.1f}s")

    @modal.exit()
    def stop(self):
        try:
            self.proc.terminate()
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()

    @modal.method()
    def render(self, job_id: str, spec: dict) -> dict:
        """Render one clip, write it to the outputs Volume, return metadata.

        The MP4 is deliberately *not* returned. Passing 150 MB through a function
        result is slow and memory-hungry; the API streams it from the Volume.
        """
        import subprocess

        from giggsdance.backends.sglang_client import (
            TURBO_LORAS, QualityMode, RenderRequest, Target, build_conditions,
            steps_for,
        )
        from giggsdance.references import ReferenceSet

        began = time.perf_counter()
        timings: dict[str, float] = {"model_load": self.load_seconds}

        order = [(k, v) for k, v in spec.get("reference_order", [])]
        references = ReferenceSet(order=order)
        for kind, path in order:
            {"image": references.images, "video": references.videos,
             "audio": references.audios}[kind].append(path)
        task = references.workflow if order else "t2va"

        quality = spec.get("quality", "lossless")
        mode = QualityMode(quality) if quality in ("lossless", "high") else QualityMode.LOSSLESS
        request = RenderRequest(
            prompt=spec["prompt"],
            task=task,
            target=Target(
                duration_seconds=spec.get("duration_s", 5.0),
                aspect_ratio=spec.get("aspect_ratio", "16:9"),
                short_edge=spec.get("short_edge", 768),
            ),
            seed=spec.get("seed", 0),
            num_inference_steps=spec.get("steps") or steps_for(mode, self.lora),
            quality=mode,
            conditions=build_conditions(references, task),
            lora_scale=TURBO_LORAS[self.lora]["lora_scale"] if self.lora else None,
        )

        raw = Path(f"/tmp/{job_id}_raw.mp4")
        mark = time.perf_counter()
        self.client.render(request, raw, max_wait_s=5400)
        timings["generate"] = time.perf_counter() - mark

        mark = time.perf_counter()
        final = self._postprocess(
            raw, Path(OUTPUT_DIR) / f"{job_id}.mp4",
            spec.get("resolution", "native"), int(spec.get("fps", 60)),
            int(spec.get("crf", 18)), spec.get("preset", "veryfast"),
        )
        timings["postprocess"] = time.perf_counter() - mark
        outputs.commit()

        import json as _json
        probe = _json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(final)],
            check=True, capture_output=True, text=True).stdout)
        stream = next(s for s in probe["streams"] if s["codec_type"] == "video")

        billable = (time.perf_counter() - began) + self.load_seconds
        return {
            "timings": timings,
            "billable_s": billable,
            "gpu_seconds": billable * self.gpus,
            "cost_usd": billable * self.gpus * RATE,
            "task": task,
            "steps": request.num_inference_steps,
            "quality": mode.value,
            "lora": self.lora,
            "gpus": self.gpus,
            "references": len(order),
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "fps": stream.get("r_frame_rate"),
            "pix_fmt": stream.get("pix_fmt"),
            "duration_s": float(probe["format"].get("duration", 0.0)),
            "file_bytes": final.stat().st_size,
        }

    def _postprocess(self, source: Path, target: Path, resolution: str,
                     fps: int, crf: int, preset: str) -> Path:
        import json as _json
        import subprocess

        from giggsdance.stages.upscale import resolve_target

        info = _json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
             str(source)], check=True, capture_output=True, text=True).stdout)
        stream = next(s for s in info["streams"] if s["codec_type"] == "video")
        width, height = int(stream["width"]), int(stream["height"])
        out_w, out_h = resolve_target(resolution, width, height)

        filters = []
        if fps != 24:
            filters.append(
                f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:vsbmc=1")
        if (out_w, out_h) != (width, height):
            filters.append(f"scale={out_w}:{out_h}:flags=lanczos")
        filters += [
            "deband=1thr=0.008:2thr=0.008:3thr=0.008:4thr=0.008:range=16:blur=true",
            "noise=alls=2:allf=t",
            "format=yuv420p10le",
            "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv",
        ]
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-i", str(source),
            "-vf", ",".join(filters),
            "-c:v", "libx265", "-crf", str(crf), "-preset", preset,
            "-profile:v", "main10", "-tag:v", "hvc1",
            "-x265-params", "log-level=error:aq-mode=3",
            "-c:a", "aac", "-b:a", "384k", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart", str(target),
        ], check=True)
        return target


# --------------------------------------------------------------------------
# Housekeeping
# --------------------------------------------------------------------------

@app.function(image=api_image, volumes={OUTPUT_DIR: outputs, UPLOAD_DIR: uploads},
              schedule=modal.Period(hours=6), timeout=900)
def prune() -> dict:
    """Delete results older than the TTL.

    A 15 s 1440p clip is well over 100 MB. Without this, Volume storage grows
    forever and quietly becomes the biggest line on the bill.
    """
    cutoff = time.time() - RESULT_TTL_HOURS * 3600
    removed, freed = 0, 0
    for directory in (Path(OUTPUT_DIR), Path(UPLOAD_DIR)):
        if not directory.is_dir():
            continue
        for entry in directory.iterdir():
            if entry.is_file() and entry.stat().st_mtime < cutoff:
                freed += entry.stat().st_size
                entry.unlink()
                removed += 1
    outputs.commit()
    uploads.commit()

    for key in list(jobs.keys()):
        record = jobs.get(key) or {}
        if record.get("created_at", time.time()) < cutoff:
            jobs.pop(key, None)

    print(f"pruned {removed} files, freed {freed / 1e6:.1f} MB")
    return {"removed": removed, "freed_mb": freed / 1e6}


# --------------------------------------------------------------------------
# HTTP API
# --------------------------------------------------------------------------

def build_api():
    """Construct the FastAPI app.

    Kept separate from the Modal decorators so it can be imported and exercised
    with TestClient locally -- auth, validation and routing are all testable
    without a GPU, a Modal account, or 144 GB of weights.
    """
    api = FastAPI(
        title="Giggsdance",
        version="0.2.0",
        description="MiniMax H3 video generation with 60 fps conversion. "
                    "Async: POST /render, poll GET /jobs/{id}, then /jobs/{id}/file.",
    )

    def require_key(x_api_key: str = Header(default="")) -> None:
        expected = os.environ.get("API_KEY", "")
        if not expected:
            raise HTTPException(500, "server misconfigured: API_KEY secret missing")
        # compare_digest so a wrong key cannot be found by timing the response.
        if not pysecrets.compare_digest(x_api_key, expected):
            raise HTTPException(401, "bad or missing X-API-Key")

    @api.get("/health")
    def health() -> dict:
        return {"status": "ok", "gpu": GPU, "model": MODEL_ID}

    @api.get("/", dependencies=[Depends(require_key)])
    def capabilities() -> dict:
        return {
            "model": MODEL_ID,
            "generation": {
                "duration_s": [MIN_DURATION_S, MAX_DURATION_S],
                "native_fps": 24,
                "frame_rule": "17n+5",
                "audio": "32 kHz stereo, generated jointly",
            },
            "references": {
                "images": 9, "videos": 3, "audios": 3, "total": MAX_TOTAL,
                "audio_requires_visual": True,
                "order_is_semantic": True,
            },
            "output": {
                "resolutions": ["native", "720p", "1080p", "1440p", "2160p"],
                "fps": [24, 30, 60],
                "codec": "libx265 main10 (10-bit), AAC 384k",
            },
            "quality": {
                "lossless": "bit-exact reference path (default)",
                "high": "audited Cache-DiT, 1.40x, SSIM 0.931",
                "lora": "lightx2v (4 evals) or larryvrh (8) -- a different model",
                "fp8": "NOT bit-exact and only ~5% on B300; use gpus instead",
            },
            "gpu": {"type": GPU, "usd_per_hour": round(RATE * 3600, 2),
                    "rate_verified": GPU not in UNVERIFIED_RATES},
            "notice": "Licence excludes EU/UK/South Korea/USA and covers outputs. "
                      "Mark published output as AI-generated.",
        }

    @api.post("/uploads", dependencies=[Depends(require_key)])
    async def upload(file: UploadFile = File(...)) -> dict:
        suffix = Path(file.filename or "").suffix.lower() or ".bin"
        name = f"{uuid.uuid4().hex}{suffix}"
        destination = Path(UPLOAD_DIR) / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = await file.read()
        destination.write_bytes(payload)
        uploads.commit()
        return {
            "uri": f"file://{destination}",
            "bytes": len(payload),
            "expires_in_hours": RESULT_TTL_HOURS,
        }

    @api.post("/render", dependencies=[Depends(require_key)], status_code=202)
    def render(spec: RenderSpec) -> dict:
        # Validate the reference set here, on a CPU container, so an illegal
        # combination never costs a GPU second.
        references = ReferenceSet(
            images=[r.uri for r in spec.references if r.type == "image"],
            videos=[r.uri for r in spec.references if r.type == "video"],
            audios=[r.uri for r in spec.references if r.type == "audio"],
            order=[(r.type, r.uri) for r in spec.references],
        )
        try:
            validate(references)
        except ReferenceError as exc:
            raise HTTPException(422, str(exc))

        variant = "ref2va" if references.workflow == "ref2va" else "fl2va"
        job_id = uuid.uuid4().hex

        env = {
            "GIGGSDANCE_VARIANT": variant,
            "GIGGSDANCE_GPUS": str(spec.gpus),
        }
        if spec.lora != "none":
            env["GIGGSDANCE_LORA"] = spec.lora
        if spec.quantize_fp8:
            env["GIGGSDANCE_FP8"] = "1"

        # A distinct env/gpu combination gets its own container pool, which is how
        # one deployment can serve fl2va and ref2va without reloading weights.
        renderer = Renderer.with_options(
            gpu=f"{GPU}:{spec.gpus}" if spec.gpus > 1 else GPU, env=env,
        )()
        call = renderer.render.spawn(job_id, spec.model_dump() | {
            "reference_order": [(r.type, r.uri) for r in spec.references],
        })

        jobs[job_id] = {
            "job_id": job_id,
            "call_id": call.object_id,
            "status": "queued",
            "created_at": time.time(),
            "variant": variant,
            "gpus": spec.gpus,
            "lora": spec.lora,
            "quality": spec.quality,
            "references": len(spec.references),
            "spec": spec.model_dump(),
        }
        return {
            "job_id": job_id,
            "status": "queued",
            "poll": f"/jobs/{job_id}",
            "note": "First job on a cold container waits ~115 s for the "
                    "~144 GB load, then ~30 s warmup.",
        }

    def _refresh(record: dict) -> dict:
        """Ask Modal whether the spawned call has finished."""
        if record.get("status") in ("completed", "failed", "cancelled"):
            return record
        try:
            call = modal.FunctionCall.from_id(record["call_id"])
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"lost track of the call: {exc}"
            jobs[record["job_id"]] = record
            return record

        try:
            result = call.get(timeout=0)
        except TimeoutError:
            record["status"] = "running"
            jobs[record["job_id"]] = record
            return record
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            jobs[record["job_id"]] = record
            return record

        record["status"] = "completed"
        record["result"] = result
        record["finished_at"] = time.time()
        jobs[record["job_id"]] = record
        return record

    @api.get("/jobs/{job_id}", dependencies=[Depends(require_key)])
    def job_status(job_id: str) -> dict:
        record = jobs.get(job_id)
        if record is None:
            raise HTTPException(404, "unknown job_id")
        record = _refresh(dict(record))
        elapsed = (record.get("finished_at") or time.time()) - record["created_at"]
        body = {
            "job_id": job_id,
            "status": record["status"],
            "elapsed_s": round(elapsed, 1),
            "variant": record.get("variant"),
            "gpus": record.get("gpus"),
            "quality": record.get("quality"),
            "lora": record.get("lora"),
            "references": record.get("references"),
        }
        if record["status"] == "completed":
            result = record["result"]
            body |= {
                "file": f"/jobs/{job_id}/file",
                "output": f"{result['width']}x{result['height']}",
                "fps": result["fps"],
                "pix_fmt": result["pix_fmt"],
                "duration_s": round(result["duration_s"], 3),
                "file_bytes": result["file_bytes"],
                "steps": result["steps"],
                "timings": {k: round(v, 1) for k, v in result["timings"].items()},
                "gpu_seconds": round(result["gpu_seconds"], 1),
                "cost_usd": round(result["cost_usd"], 4),
                "cost_rate_verified": GPU not in UNVERIFIED_RATES,
                "expires_in_hours": RESULT_TTL_HOURS,
            }
        elif record["status"] == "failed":
            body["error"] = record.get("error", "unknown")
        return body

    @api.get("/jobs/{job_id}/file", dependencies=[Depends(require_key)])
    def job_file(job_id: str):
        record = jobs.get(job_id)
        if record is None:
            raise HTTPException(404, "unknown job_id")
        record = _refresh(dict(record))
        if record["status"] != "completed":
            raise HTTPException(409, f"job is {record['status']}, not completed")

        outputs.reload()  # see what the GPU container committed
        path = Path(OUTPUT_DIR) / f"{job_id}.mp4"
        if not path.exists():
            raise HTTPException(410, f"result expired (TTL {RESULT_TTL_HOURS} h)")
        return FileResponse(path, media_type="video/mp4",
                            filename=f"giggsdance_{job_id}.mp4")

    @api.get("/jobs", dependencies=[Depends(require_key)])
    def list_jobs(limit: int = 20) -> dict:
        records = sorted(
            (dict(jobs.get(k) or {}) for k in jobs.keys()),
            key=lambda r: r.get("created_at", 0), reverse=True,
        )[:max(1, min(limit, 100))]
        return {"jobs": [
            {"job_id": r.get("job_id"), "status": r.get("status"),
             "created_at": r.get("created_at"), "variant": r.get("variant")}
            for r in records
        ]}

    @api.delete("/jobs/{job_id}", dependencies=[Depends(require_key)])
    def delete_job(job_id: str) -> dict:
        record = jobs.get(job_id)
        if record is None:
            raise HTTPException(404, "unknown job_id")
        if record.get("status") in ("queued", "running"):
            try:
                modal.FunctionCall.from_id(record["call_id"]).cancel()
            except Exception:
                pass
        outputs.reload()
        path = Path(OUTPUT_DIR) / f"{job_id}.mp4"
        if path.exists():
            path.unlink()
            outputs.commit()
        jobs.pop(job_id, None)
        return {"job_id": job_id, "deleted": True}

    @api.exception_handler(ValueError)
    def value_error(_request, exc: ValueError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    return api


@app.function(
    image=api_image,
    volumes={OUTPUT_DIR: outputs, UPLOAD_DIR: uploads},
    secrets=[api_secret],
    min_containers=0,
    timeout=600,
)
@modal.asgi_app()
def fastapi_app():
    return build_api()


# --------------------------------------------------------------------------
# Local helper: prefetch weights before the first API call
# --------------------------------------------------------------------------

@app.local_entrypoint()
def main(variant: str = "fl2va", force: bool = False):
    """Warm the Volume so the first HTTP render is not also a 144 GB download."""
    print(f"fetching {variant} weights (~{ONE_PARTITION_GB:.0f} GB, one time)...")
    info = fetch_weights.remote(variant=variant, force=force)
    if info["downloaded"]:
        print(f"downloaded {info['gb']:.1f} GB in {info['seconds'] / 60:.1f} min")
    else:
        print(f"already present: {info['gb']:.1f} GB")
    print("\nNow deploy the API:")
    print("    modal secret create giggsdance-api API_KEY=$(openssl rand -hex 32)")
    print("    modal deploy serve_api.py")
