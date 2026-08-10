#!/usr/bin/env python3
"""
Giggsdance -- one command, does everything.

    modal run run.py

That single command will, in order:

  1. check whether the H3 weights are already in your Modal Volume
  2. download them if not (~90 GB, on cheap CPU, skipped on every later run)
  3. fetch the super-resolution checkpoint, but only if you asked for upscaling
  4. render one clip: generate -> 60 fps -> (optional upscale) -> 10-bit encode
  5. save the .mp4 next to this file and print exactly what it cost

Everything is idempotent, so re-running is safe and skips whatever is done.

    # defaults: 5s, native resolution (no upscaling), 60fps, 8 steps, <$1
    modal run run.py

    # your own prompt
    modal run run.py --prompt "a paper boat drifting down a rain-filled gutter"

    # turn upscaling on (costs roughly 4x more post-processing time)
    modal run run.py --resolution 1440p --steps 24 --budget-usd 3

    # measure the cold start on its own before committing to a render
    modal run run.py --probe-only

    # dump the pipeline signature, if generation misbehaves
    modal run run.py --describe

    # validate the whole pipeline locally, free, no GPU and no weights
    python run.py --dry-run

Powered by MiniMax H3. Read NOTICE.md before use -- the licence excludes the EU,
UK, South Korea and USA, and the restriction covers the model's *outputs*, not
just its weights. If you publish the result, mark it as AI-generated.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import modal

from giggsdance.constraints import SRC_FPS, frames_for_duration, resolve_canvas
from giggsdance.stages.encode import EncodeSettings, build_encode_command, write_wav
from giggsdance.stages.interpolate import plan_interpolation
from giggsdance.stages.upscale import (
    blend_tiles,
    pick_scale,
    plan_geometry,
    plan_tiles,
    resolve_target,
)

MODEL_ID = "MiniMaxAI/MiniMax-H3"
WEIGHTS_DIR = "/weights"
UPSCALER_DIR = "/upscalers"
MODEL_PATH = f"{WEIGHTS_DIR}/MiniMax-H3"

# Modal's B200 rate. gpu="B200+" may land on a B300 but is billed as a B200.
USD_PER_SECOND = 6.25 / 3600.0
DEFAULT_BUDGET_USD = 1.00

UPSCALERS = {
    2: ("RealESRGAN_x2plus.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"),
    4: ("RealESRGAN_x4plus.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"),
}

# What a complete t2va/fl2va checkpoint must contain. Used to decide whether the
# Volume already holds usable weights, so a re-run does not re-download 90 GB.
REQUIRED_PARTS = ("transformer", "text_encoder", "vae", "audio_vae", "tokenizer")

DEFAULT_PROMPT = (
    "integrated_multimodal_description: [Shot 1] Cinematic wide shot, slow push in. "
    "A lone lighthouse on black volcanic rock at dusk, its beam sweeping through "
    "drifting sea mist. Heavy swells break against the rock and throw spray up into "
    "the light. The sky is deep indigo fading to burnt orange at the horizon, a "
    "smooth clean gradient.\n"
    "overall_soundscape: Deep rhythmic booms of waves against stone, hissing spray, "
    "a low wind rising and falling, and the distant two-tone moan of a foghorn.\n"
    "non_diegetic_music: Sparse ambient score, sustained low strings under a single "
    "distant piano note repeating slowly."
)

# CUDA 13.1+ is required for B300, and gpu="B200+" can land on either chip, so
# the image has to be able to run on both.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "wget")
    .pip_install("torch", "torchvision",
                 extra_index_url="https://download.pytorch.org/whl/cu130")
    .pip_install(
        "git+https://github.com/huggingface/diffusers.git",
        "transformers>=4.57.0", "accelerate", "safetensors",
        "huggingface_hub[hf_transfer]", "spandrel==0.4.2",
        "pillow", "numpy", "sentencepiece", "protobuf", "av",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": f"{WEIGHTS_DIR}/hf"})
    .add_local_python_source("giggsdance")
)

app = modal.App("giggsdance")
weights_volume = modal.Volume.from_name("h3-weights", create_if_missing=True)
upscaler_volume = modal.Volume.from_name("giggsdance-upscalers", create_if_missing=True)


# ==========================================================================
# Step 1-2: weights, idempotent
# ==========================================================================

@app.function(
    image=image, volumes={WEIGHTS_DIR: weights_volume},
    timeout=6 * 60 * 60, cpu=8.0, memory=16384,
)
def ensure_weights(force: bool = False, workflow: str = "t2va"):
    """Download H3 only if the Volume does not already have it.

    Only the components t2va/fl2va need. ``transformer_ref/`` is another 61.7 GB
    and is required solely for ref2va (omni-reference) generation.
    """
    from huggingface_hub import snapshot_download

    root = Path(MODEL_PATH)

    def survey():
        found = {}
        for part in REQUIRED_PARTS:
            directory = root / part
            if directory.is_dir():
                size = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
                if size > 1_000_000:
                    found[part] = size
        return found

    present = survey()
    if not force and len(present) == len(REQUIRED_PARTS):
        total = sum(present.values())
        print(f"weights already present: {total / 1e9:.1f} GB -- skipping download")
        return {"downloaded": False, "gb": total / 1e9, "seconds": 0.0}

    if present:
        missing = [p for p in REQUIRED_PARTS if p not in present]
        print(f"incomplete checkpoint (missing {missing}); resuming download")
    else:
        print("no weights found; downloading ~90 GB (one time, on CPU)")

    patterns = [
        "modular_model_index.json", "model_index.json", "*.json", "*.txt",
        "text_encoder/*", "tokenizer/*", "processor/*",
        "vae/*", "audio_vae/*", "scheduler/*", "audio_scheduler/*",
    ]
    patterns.append("transformer_ref/*" if workflow == "ref2va" else "transformer/*")

    started = time.time()
    snapshot_download(MODEL_ID, local_dir=MODEL_PATH,
                      allow_patterns=patterns, max_workers=16)
    weights_volume.commit()
    elapsed = time.time() - started

    final = survey()
    total = sum(final.values())
    print(f"\ndownloaded in {elapsed / 60:.1f} min, {total / 1e9:.1f} GB total")
    for part, size in sorted(final.items()):
        print(f"  {part:16} {size / 1e9:7.2f} GB")

    still_missing = [p for p in REQUIRED_PARTS if p not in final]
    if still_missing:
        raise RuntimeError(f"download finished but {still_missing} are still missing")
    return {"downloaded": True, "gb": total / 1e9, "seconds": elapsed}


@app.function(image=image, volumes={UPSCALER_DIR: upscaler_volume}, timeout=1800)
def ensure_upscaler(scale: int):
    """Fetch a Real-ESRGAN checkpoint (~67 MB) if it is not already there."""
    import urllib.request

    if scale not in UPSCALERS:
        return {"needed": False}
    name, url = UPSCALERS[scale]
    target = Path(UPSCALER_DIR) / name
    if target.exists() and target.stat().st_size > 1_000_000:
        print(f"{name} already present ({target.stat().st_size / 1e6:.1f} MB)")
        return {"needed": True, "downloaded": False, "name": name}

    print(f"downloading {name} ...")
    urllib.request.urlretrieve(url, target)
    upscaler_volume.commit()
    print(f"  -> {target.stat().st_size / 1e6:.1f} MB")
    return {"needed": True, "downloaded": True, "name": name}


# ==========================================================================
# Step 3: render
# ==========================================================================

@app.cls(
    image=image,
    gpu="B200+",
    volumes={WEIGHTS_DIR: weights_volume, UPSCALER_DIR: upscaler_volume},
    timeout=2 * 60 * 60,
    memory=180 * 1024,
    min_containers=0,      # never pay for idle: a warm B200 is ~$150/day
    scaledown_window=300,  # but reuse the container for 5 min of back-to-back runs
)
class Renderer:
    @modal.enter()
    def load(self):
        """Load H3 once per container. ~124 GB, and the least predictable cost."""
        import torch
        from diffusers import ComponentsManager, ModularPipeline

        self.torch = torch
        started = time.perf_counter()

        self.gpu_name = torch.cuda.get_device_name(0)
        self.gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {self.gpu_name} ({self.gpu_vram:.0f} GB), "
              f"torch {torch.__version__}, CUDA {torch.version.cuda}")

        source = MODEL_PATH if Path(MODEL_PATH).exists() else MODEL_ID
        print(f"loading H3 from {source} "
              f"(~124 GB: 61.7 transformer + 62.1 Qwen3-VL conditioner)")

        self.manager = ComponentsManager()
        self.pipe = ModularPipeline.from_pretrained(
            source, components_manager=self.manager
        )
        self.pipe.load_components(workflow="t2va", dtype=torch.bfloat16)

        # 124 GB fits resident on a B200 (180 GB) or B300 (288 GB), so keep a
        # large activation margin rather than staging through host RAM.
        margin = "40GB" if self.gpu_vram >= 150 else "12GB"
        self.manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin=margin)
        print(f"offload margin: {margin}")

        try:
            self.pipe.transformer.set_attention_backend("_flash_3_hub")
            print("attention backend: _flash_3_hub")
        except Exception as exc:
            print(f"attention backend left at default ({exc})")

        self.load_seconds = time.perf_counter() - started
        print(f"model load: {self.load_seconds:.1f}s "
              f"(${self.load_seconds * USD_PER_SECOND:.4f})")

    @modal.method()
    def report_load(self) -> float:
        return self.load_seconds

    @modal.method()
    def describe(self) -> str:
        """The pipeline's real signature -- ground truth if generation misbehaves."""
        return str(getattr(self.pipe, "doc", "<no doc available>"))

    @modal.method()
    def render(
        self,
        prompt: str,
        duration_s: float = 5.0,
        aspect_ratio: str = "16:9",
        resolution: str = "native",
        fps: int = 60,
        steps: int = 8,
        seed: int = 0,
        crf: int = 18,
        preset: str = "veryfast",
        tile: int = 512,
        overlap: int = 48,
    ) -> dict:
        import numpy as np

        stages: dict[str, float] = {"model_load": self.load_seconds}

        def tick(name, start):
            stages[name] = stages.get(name, 0.0) + (time.perf_counter() - start)

        wall_start = time.perf_counter()

        num_frames = frames_for_duration(duration_s)
        canvas = resolve_canvas(aspect_ratio)
        out_w, out_h = resolve_target(resolution, canvas.width, canvas.height)
        print(f"\n{num_frames} frames @24fps = {num_frames / 24:.4f}s | "
              f"canvas {canvas} -> {out_w}x{out_h} @{fps}fps")

        # -- generate ---------------------------------------------------
        mark = time.perf_counter()
        frames, audio, sample_rate = self._generate(
            prompt, num_frames, canvas.width, canvas.height, steps, seed
        )
        tick("generate", mark)
        src_n, src_h, src_w = frames.shape[0], frames.shape[1], frames.shape[2]
        print(f"generated {src_n} frames at {src_w}x{src_h}, "
              f"audio {tuple(audio.shape)} @ {sample_rate} Hz "
              f"({stages['generate']:.1f}s)")

        # -- plan -------------------------------------------------------
        plan = plan_interpolation(src_n, SRC_FPS, float(fps))
        crop_h = plan_geometry(src_w, src_h, out_w, out_h, 2).crop_height
        scale = pick_scale(crop_h, out_h)
        geo = plan_geometry(src_w, src_h, out_w, out_h, model_scale=max(1, scale))
        label = "crop only, no model" if scale == 1 else f"{scale}x super-resolution"
        print(f"crop {geo.crop_width}x{geo.crop_height} -> {label} -> "
              f"{geo.out_width}x{geo.out_height}, {plan.num_dst_frames} output frames")

        model = self._load_upscaler(scale)

        work = Path("/tmp/giggsdance")
        work.mkdir(parents=True, exist_ok=True)
        mark = time.perf_counter()
        wav = write_wav(work / "audio.wav", audio, sample_rate)
        tick("audio", mark)

        out_path = work / "output.mp4"
        settings = EncodeSettings(codec="h265", crf=crf, preset=preset, fps=float(fps))
        command = build_encode_command(
            geo.out_width, geo.out_height, out_path, settings, wav
        )

        # -- stream interpolate -> upscale -> encode --------------------
        # Never materialise the full output: 863 frames of 1440p float32 is 38 GB.
        source = self.torch.from_numpy(frames)
        proc = subprocess.Popen(command, stdin=subprocess.PIPE)
        written = 0
        try:
            for timing in plan.timings:
                mark = time.perf_counter()
                frame = self._interpolate(source, timing.left, timing.t)
                tick("interpolate", mark)

                mark = time.perf_counter()
                final = self._postprocess(frame, geo, model, tile, overlap)
                tick("upscale", mark)

                mark = time.perf_counter()
                proc.stdin.write(np.ascontiguousarray(
                    (np.clip(final, 0, 1) * 65535.0 + 0.5).astype("<u2")).tobytes())
                tick("encode_pipe", mark)

                written += 1
                if written % 120 == 0:
                    print(f"  {written}/{plan.num_dst_frames}")
        finally:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
            mark = time.perf_counter()
            code = proc.wait()
            tick("encode_finalize", mark)
        if code != 0:
            raise RuntimeError(f"ffmpeg exited with {code}")

        probe = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(out_path)],
            check=True, capture_output=True, text=True).stdout)

        return {
            "stages": stages,
            "billable_s": (time.perf_counter() - wall_start) + self.load_seconds,
            "render_s": time.perf_counter() - wall_start,
            "src_frames": src_n,
            "out_frames": written,
            "canvas": f"{src_w}x{src_h}",
            "output": f"{geo.out_width}x{geo.out_height}",
            "fps": fps,
            "resolution": resolution,
            "steps": steps,
            "scale": scale,
            "video_s": src_n / SRC_FPS,
            "file_bytes": out_path.stat().st_size,
            "gpu": self.gpu_name,
            "gpu_vram_gb": round(self.gpu_vram, 1),
            "probe": probe,
            "video": out_path.read_bytes(),
        }

    # -- internals -----------------------------------------------------

    def _generate(self, prompt, num_frames, width, height, steps, seed):
        """NOTE: this is the one part of the repo never executed by the author."""
        import numpy as np

        results = self.pipe(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=steps,
            generator=self.torch.Generator().manual_seed(int(seed)),
            output=["videos", "audio", "sampling_rate"],
            output_type="pt",
        )
        videos = results["videos"][0]
        audio = results["audio"]
        audio = audio[0] if isinstance(audio, (list, tuple)) else audio

        if hasattr(videos, "detach"):
            videos = videos.detach().float().cpu().numpy()
        videos = np.asarray(videos)
        if videos.ndim == 5:
            videos = videos[0]
        if videos.shape[1] == 3 and videos.shape[-1] != 3:
            videos = np.transpose(videos, (0, 2, 3, 1))
        videos = videos.astype(np.float32)
        if videos.max() > 1.5:
            videos /= 255.0

        if hasattr(audio, "detach"):
            audio = audio.detach().float().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32)
        while audio.ndim > 2:
            audio = audio[0]

        return np.clip(videos, 0, 1), audio, int(results["sampling_rate"])

    def _interpolate(self, source, left, t):
        """Blend at the exact timestamp. Cadence is correct; RIFE would be sharper."""
        if t <= 1e-9:
            return source[left]
        right = source[min(left + 1, source.shape[0] - 1)]
        return source[left] * (1.0 - t) + right * t

    def _load_upscaler(self, scale):
        if scale <= 1:
            print("upscaler: none (target is at or below source height)")
            return None
        from spandrel import ModelLoader

        name, _ = UPSCALERS[scale]
        path = Path(UPSCALER_DIR) / name
        if not path.exists():
            raise FileNotFoundError(f"{path} missing -- ensure_upscaler should have run")
        descriptor = ModelLoader().load_from_file(str(path))
        model = descriptor.model.eval().to("cuda").half()
        print(f"upscaler: {name} ({getattr(descriptor, 'scale', scale)}x)")
        return model

    def _postprocess(self, frame, geo, model, tile, overlap):
        import numpy as np
        import torch

        left, upper, right, lower = geo.crop_box
        patch = frame[upper:lower, left:right]
        if hasattr(patch, "cpu"):
            patch = patch.cpu().numpy()
        patch = np.ascontiguousarray(np.asarray(patch, dtype=np.float32))
        height, width = patch.shape[:2]

        if model is None:
            if (height, width) == (geo.out_height, geo.out_width):
                return patch                      # native: genuinely free
            tensor = torch.from_numpy(patch).to("cuda").permute(2, 0, 1).unsqueeze(0)
            resized = torch.nn.functional.interpolate(
                tensor, size=(geo.out_height, geo.out_width),
                mode="bicubic", align_corners=False, antialias=True)
            return resized.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()

        grid = plan_tiles(width, height, tile, overlap)
        outputs = []
        with torch.no_grad():
            full = torch.from_numpy(patch).to("cuda")
            for piece in grid:
                tx, ty = piece.x, piece.y
                chunk = full[ty:ty + piece.height, tx:tx + piece.width]
                inp = chunk.permute(2, 0, 1).unsqueeze(0).half()
                res = model(inp).float().clamp(0, 1).squeeze(0).permute(1, 2, 0)
                outputs.append((piece, res.cpu().numpy()))

        blended = blend_tiles(outputs, width, height, geo.scale, grid)
        if blended.shape[:2] != (geo.out_height, geo.out_width):
            import torch as t2
            tensor = t2.from_numpy(blended).permute(2, 0, 1).unsqueeze(0)
            blended = t2.nn.functional.interpolate(
                tensor, size=(geo.out_height, geo.out_width),
                mode="bicubic", align_corners=False, antialias=True,
            ).squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
        return blended


# ==========================================================================
# Entrypoints
# ==========================================================================

@app.local_entrypoint()
def main(
    prompt: str = DEFAULT_PROMPT,
    duration_s: float = 5.0,
    resolution: str = "native",
    fps: int = 60,
    steps: int = 8,
    seed: int = 0,
    aspect_ratio: str = "16:9",
    crf: int = 18,
    preset: str = "veryfast",
    budget_usd: float = DEFAULT_BUDGET_USD,
    out: str = "output.mp4",
    force_download: bool = False,
    probe_only: bool = False,
    describe: bool = False,
):
    """Do everything: fetch what is missing, render one clip, report the cost.

    This is deliberately the *only* local entrypoint in the file. Modal auto-runs
    a single entrypoint but treats several as ambiguous, so `modal run run.py`
    with no arguments would stop and ask which one -- which defeats the purpose.
    The two diagnostic modes are flags instead of separate entrypoints.
    """
    banner = "=" * 74
    print(f"{banner}\nGIGGSDANCE -- powered by MiniMax H3\n{banner}")
    print("Licence: not valid in the EU, UK, South Korea or USA, and the")
    print("restriction covers outputs too. See NOTICE.md. Mark published")
    print("results as AI-generated.\n")

    if describe:
        _describe()
        return
    if probe_only:
        _probe_only()
        return

    canvas = resolve_canvas(aspect_ratio)
    num_frames = frames_for_duration(duration_s)
    out_w, out_h = resolve_target(resolution, canvas.width, canvas.height)
    crop_h = plan_geometry(canvas.width, canvas.height, out_w, out_h, 2).crop_height
    scale = pick_scale(crop_h, out_h)
    plan_frames = plan_interpolation(num_frames, SRC_FPS, float(fps)).num_dst_frames

    max_seconds = int(budget_usd / USD_PER_SECOND)
    print(f"plan:    {num_frames} frames @24fps ({num_frames / 24:.3f}s) at {canvas}")
    print(f"         -> {plan_frames} frames @{fps}fps at {out_w}x{out_h}")
    print(f"         -> {'no super-resolution' if scale <= 1 else f'{scale}x super-resolution'}"
          f", {steps} steps")
    print(f"budget:  ${budget_usd:.2f} = {max_seconds}s of B200 time "
          f"({max_seconds / 60:.1f} min), enforced as a hard timeout\n")

    print("[1/3] weights")
    weights_info = ensure_weights.remote(force=force_download)
    if weights_info["downloaded"]:
        print(f"      fetched {weights_info['gb']:.1f} GB in "
              f"{weights_info['seconds'] / 60:.1f} min")

    print("[2/3] upscaler")
    if scale <= 1:
        print("      not needed at this resolution")
    else:
        info = ensure_upscaler.remote(scale)
        print(f"      {info.get('name')} ready")

    print("[3/3] render")
    renderer = Renderer.with_options(timeout=max_seconds)()
    started = time.time()
    try:
        result = renderer.render.remote(
            prompt=prompt, duration_s=duration_s, aspect_ratio=aspect_ratio,
            resolution=resolution, fps=fps, steps=steps, seed=seed,
            crf=crf, preset=preset,
        )
    except Exception as exc:
        spent = time.time() - started
        print(f"\nfailed after {spent:.0f}s (at most ${spent * USD_PER_SECOND:.2f}): {exc}")
        print("If that was the timeout: raise --budget-usd, or lower --steps.")
        print("If generation itself failed: modal run run.py --describe")
        raise

    Path(out).write_bytes(result.pop("video"))
    _report(result, result.pop("probe"), time.time() - started, out, budget_usd)


def _probe_only():
    """Measure only the cold start and model load, then stop.

    The load precedes any generation and is the least predictable number in the
    system. Learn it once and every later budget becomes reliable.
    """
    print("measuring cold start + model load, no rendering...")
    seconds = Renderer().report_load.remote()
    print(f"\nmodel load: {seconds:.1f}s = ${seconds * USD_PER_SECOND:.4f}")
    print(f"a $1.00 budget leaves {int(1.0 / USD_PER_SECOND - seconds)}s to render in")
    print("a warm container (within 300s of the last call) skips this entirely")


def _describe():
    """Print the diffusers pipeline signature -- use this if generation fails."""
    print(Renderer().describe.remote())


def _report(result, probe, round_trip, out_path, budget_usd):
    stages = result["stages"]
    load = stages.get("model_load", 0.0)
    render = {k: v for k, v in stages.items() if k != "model_load"}
    render_total = sum(render.values()) or 1e-9
    billable = result["billable_s"]
    actual = billable * USD_PER_SECOND
    frames = result["out_frames"]
    video_s = result["video_s"]

    line = "=" * 74
    print(f"\n{line}\nMEASURED\n{line}")
    print(f"GPU        {result['gpu']} ({result['gpu_vram_gb']} GB)")
    print(f"generated  {result['src_frames']} frames at {result['canvas']}, "
          f"{result['steps']} steps")
    print(f"output     {result['output']} @{result['fps']}fps, {frames} frames, "
          f"{video_s:.3f}s")
    print(f"upscaling  {'none' if result['scale'] <= 1 else str(result['scale']) + 'x'}")
    print(f"file       {result['file_bytes'] / 1e6:.1f} MB "
          f"({result['file_bytes'] * 8 / video_s / 1e6:.1f} Mbps)")

    print(f"\n{'stage':<18}{'seconds':>10}{'%':>7}{'USD':>9}   per frame")
    print("-" * 74)
    for name, seconds in sorted(render.items(), key=lambda kv: -kv[1]):
        print(f"{name:<18}{seconds:>10.2f}{seconds / render_total * 100:>6.1f}%"
              f"{seconds * USD_PER_SECOND:>9.4f}   {seconds / frames * 1000:>7.1f} ms")
    print("-" * 74)
    print(f"{'render':<18}{render_total:>10.2f}{100.0:>6.1f}%"
          f"{render_total * USD_PER_SECOND:>9.4f}")
    print(f"{'model load':<18}{load:>10.2f}{'':>7}{load * USD_PER_SECOND:>9.4f}")
    print(f"{'TOTAL':<18}{billable:>10.2f}{'':>7}{actual:>9.4f}")

    verdict = "UNDER" if actual <= budget_usd else "OVER"
    print(f"\nspent ${actual:.4f} of ${budget_usd:.2f} budget "
          f"({actual / budget_usd * 100:.0f}%, {verdict})")
    print(f"  per second of video      ${actual / video_s:.4f}")
    print(f"  again on a warm container ${render_total * USD_PER_SECOND:.4f}")
    print(f"  wall clock               {round_trip:.0f}s")
    if load > render_total:
        print(f"\n  Model load ({load:.0f}s) cost more than the render ({render_total:.0f}s).")
        print(f"  Batching clips is your biggest saving -- each extra clip is "
              f"~${render_total * USD_PER_SECOND:.4f}.")

    per_frame = render_total / frames
    print(f"\n{line}\nPROJECTED (linear, verify before trusting)\n{line}")
    for label, seconds in [("14.375s max clip", 14.375),
                           ("1 minute (5 clips)", 60.0),
                           ("10 minutes (42 clips)", 600.0)]:
        count = int(seconds * result["fps"])
        secs = count * per_frame
        print(f"  {label:<24}{count:>7} frames  {secs / 60:>6.1f} min  "
              f"${secs * USD_PER_SECOND:>7.2f} warm")
    print("\nChained clips each need their own generation pass, and H3 writes every")
    print("clip's soundtrack independently -- joins have audible ambience seams.")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    if "--dry-run" in sys.argv:
        sys.path.insert(0, str(Path(__file__).parent))
        from bench_one_video import dry_run
        raise SystemExit(dry_run())
    print(__doc__)
    print("Run:  modal run run.py")
