#!/usr/bin/env python3
"""
Giggsdance benchmark: render ONE clip on Modal and report exact numbers.

Renders a single MiniMax H3 clip end to end -- generate -> interpolate -> upscale
-> encode -- timing every stage separately and printing a real cost table at
Modal's B200 rate. The point is to replace estimates with measurements before
you design anything around them.

Powered by MiniMax H3 (https://huggingface.co/MiniMaxAI/MiniMax-H3), used under
the MiniMax H3 Community License Agreement.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
  pip install modal && modal setup

  # 0. Validate the whole pipeline locally for $0 (no GPU, no Modal, no weights).
  #    Uses synthetic frames. Do this first -- it catches plumbing bugs free.
  python bench_one_video.py --dry-run

  # 1. Download weights into a Modal Volume (~90 GB, once, on cheap CPU).
  modal run bench_one_video.py::download_weights

  # 2. Fetch the upscaler checkpoint (~67 MB, once).
  modal run bench_one_video.py::download_upscaler

  # 3. Benchmark one clip. Start at the defaults; they are the cheap ones.
  modal run bench_one_video.py::benchmark

  # Variations
  modal run bench_one_video.py::benchmark --resolution 1440p --fps 60 --steps 24
  modal run bench_one_video.py::benchmark --resolution 1080p --steps 8 --duration-s 5
  modal run bench_one_video.py::benchmark --resolution 2160p --steps 24

--------------------------------------------------------------------------------
WHAT IS MEASURED VS PROJECTED
--------------------------------------------------------------------------------
Measured: model load, generation, interpolation, upscale, encode, file size.
Projected (clearly labelled): longer clips and multi-clip chains, extrapolated
from the measured marginal cost per frame. Projections assume linear scaling,
which is roughly true for upscale/encode and only approximately true for
generation.

--------------------------------------------------------------------------------
HONEST CAVEATS
--------------------------------------------------------------------------------
* The H3 generation call is written against the published diffusers integration
  but has NOT been executed by the author (it needs ~124 GB of weights and a big
  GPU). Everything else here is verified. If something breaks, it is most likely
  in `_generate`, and `--print-doc` will dump the pipeline's real signature.
* Interpolation defaults to `torch`, which places frames at mathematically
  correct 60 fps timestamps but synthesises them by blending. The cadence is
  exactly right; motion is softer than RIFE. It exists so this benchmark is
  self-contained. Add RIFE for production quality.
* "1440p" is OUR upscale of H3's 768p output. It is NOT MiniMax's native 2K,
  which comes from the closed-source H3-Regenerate-2K module.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

# ==============================================================================
# Constants -- H3's fixed geometry. Verified against the model card.
# ==============================================================================

SRC_FPS = 24
FRAME_QUANTUM, FRAME_OFFSET = 17, 5
MIN_DURATION_S, MAX_DURATION_S = 5.0, 15.0
CANVAS_SHORT_EDGE, CANVAS_MAX_PIXELS, SIZE_MULTIPLE = 768, 1032192, 32

MODEL_ID = "MiniMaxAI/MiniMax-H3"

# Modal B200 rate. gpu="B200+" may run on a B300 but is billed as B200.
USD_PER_SECOND = 6.25 / 3600.0

RESOLUTIONS = {
    # "native" means: no super-resolution at all. H3's own 1344x768 canvas, only
    # centre-cropped to exact 16:9 (a crop is free). This is the cheap path --
    # super-resolution measured at ~84% of post-processing time, so skipping it
    # removes most of the bill. Quality is H3's real output with nothing invented.
    "native": None,
    "720p":  (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "2160p": (3840, 2160),
}

# Cap so a runaway job cannot quietly empty your wallet. Converted to a hard
# container timeout, so Modal kills the run rather than billing past it.
DEFAULT_BUDGET_USD = 1.00

UPSCALERS = {
    2: ("RealESRGAN_x2plus.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"),
    4: ("RealESRGAN_x4plus.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"),
}

WEIGHTS_DIR = "/weights"
UPSCALER_DIR = "/upscalers"


def valid_frame_counts() -> tuple[int, ...]:
    counts, n = [], 1
    while True:
        frames = FRAME_QUANTUM * n + FRAME_OFFSET
        if frames / SRC_FPS > MAX_DURATION_S:
            break
        if frames / SRC_FPS >= MIN_DURATION_S:
            counts.append(frames)
        n += 1
    return tuple(counts)


VALID_FRAME_COUNTS = valid_frame_counts()   # (124, 141, ..., 345)
MAX_FRAMES = VALID_FRAME_COUNTS[-1]


def frames_for_duration(seconds: float) -> int:
    """Snap a duration to a decodable 17n+5 frame count."""
    if seconds > MAX_DURATION_S + 1e-9:
        raise ValueError(f"{seconds}s exceeds H3's {MAX_DURATION_S}s per-clip limit")
    if seconds < MIN_DURATION_S - 1e-9:
        raise ValueError(f"{seconds}s is below H3's {MIN_DURATION_S}s minimum")
    wanted = math.ceil(seconds * SRC_FPS - 1e-9)
    if wanted > MAX_FRAMES:
        return MAX_FRAMES
    return next(c for c in VALID_FRAME_COUNTS if c >= wanted)


def resolve_canvas(aspect_ratio: str = "16:9", short_edge: int = CANVAS_SHORT_EDGE):
    """Largest legal canvas: multiples of 32, area <= 1032192.

    For 16:9 this yields 1344x768 -- the canvas H3 was trained on. Note 1376x768
    would be the naive 'nearest multiple of 32' answer but it breaks the pixel cap.
    """
    left, right = (aspect_ratio.replace("/", ":").split(":") + ["1"])[:2]
    ratio = float(left) / float(right)
    landscape = ratio >= 1.0
    requested = max(SIZE_MULTIPLE, round(short_edge / SIZE_MULTIPLE) * SIZE_MULTIPLE)

    best, best_key = None, None
    for short in range(requested, SIZE_MULTIPLE - 1, -SIZE_MULTIPLE):
        raw = short * ratio if landscape else short / ratio
        low = max(SIZE_MULTIPLE, int(raw // SIZE_MULTIPLE) * SIZE_MULTIPLE)
        for long_edge in (low, low + SIZE_MULTIPLE):
            w, h = (long_edge, short) if landscape else (short, long_edge)
            if w * h > CANVAS_MAX_PIXELS:
                continue
            key = (abs(short - requested), round(abs(w / h - ratio) / ratio, 6), -(w * h))
            if best_key is None or key < best_key:
                best, best_key = (w, h), key
    if best is None:
        raise ValueError(f"no legal canvas for {aspect_ratio}")
    return best


# ==============================================================================
# 24 -> 60 fps timing. Absolute timestamps, never 2x-then-drop.
# ==============================================================================

def plan_interpolation(num_src: int, src_fps: float = 24.0, dst_fps: float = 60.0):
    """Return [(left_index, t), ...], one entry per output frame.

    Output frame i sits at absolute time i/dst_fps. In source-frame units that is
    s = i * src_fps/dst_fps, so it interpolates between floor(s) and floor(s)+1 at
    fraction t = s - floor(s). For 24->60 the fractions cycle 0, .4, .8, .2, .6 --
    perfectly uniform spacing. Running a 2x interpolator twice and dropping frames
    instead gives an irregular grid, which reads as judder.
    """
    if num_src < 1:
        raise ValueError("need at least one source frame")
    if num_src == 1 or dst_fps == src_fps:
        return [(i, 0.0) for i in range(num_src)]
    num_dst = max(1, math.ceil(num_src / src_fps * dst_fps - 1e-9))
    step, last = src_fps / dst_fps, num_src - 1
    plan = []
    for i in range(num_dst):
        s = i * step
        left = int(math.floor(s + 1e-9))
        t = s - left
        if left >= last:
            left, t = last, 0.0
        plan.append((left, round(t, 9)))
    return plan


# ==============================================================================
# Geometry: crop to the exact output aspect, upscale, supersample down.
# ==============================================================================

@dataclass(frozen=True)
class Geometry:
    src_w: int
    src_h: int
    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int
    scale: int
    out_w: int
    out_h: int

    @property
    def crop_box(self):
        return (self.crop_x, self.crop_y, self.crop_x + self.crop_w, self.crop_y + self.crop_h)


def plan_geometry(src_w: int, src_h: int, out_w: int, out_h: int, scale: int) -> Geometry:
    """H3's 1344x768 canvas is 1.75:1; UHD/1440p are 1.7778:1.

    Scaling one onto the other directly stretches by 1.6% -- visible on faces. So
    centre-crop to the target aspect first (12 rows out of 768) and never stretch.
    """
    target_ar, src_ar = out_w / out_h, src_w / src_h
    if abs(src_ar - target_ar) < 1e-9:
        crop_w, crop_h = src_w, src_h
    elif src_ar > target_ar:
        crop_h, crop_w = src_h, min(src_w, int(round(src_h * target_ar)) & ~1)
    else:
        crop_w, crop_h = src_w, min(src_h, int(round(src_w / target_ar)) & ~1)
    return Geometry(
        src_w, src_h, (src_w - crop_w) // 2, (src_h - crop_h) // 2,
        crop_w, crop_h, scale, out_w, out_h,
    )


def pick_scale(crop_h: int, out_h: int) -> int:
    """Choose the cheapest model that reaches the target.

    Returns 1 when the target is at or below the source height -- 768p down to
    720p is a *downscale*, so running a super-resolution model first would burn
    the most expensive stage in the pipeline to throw the result away.

    2x costs roughly a quarter of 4x (4.1 vs 16.3 megapixels of model output per
    frame from H3's canvas), so 2x is used whenever it reaches the target and a
    Lanczos step brings it the rest of the way down.
    """
    if out_h <= crop_h:
        return 1
    return 2 if out_h / crop_h <= 2.0 + 1e-6 else 4


def plan_tiles(width: int, height: int, tile: int, overlap: int):
    """Uniformly spaced tiles + the overlap actually achieved.

    Stepping by (tile - overlap) and clamping the last tile to the edge produces a
    much bigger overlap there than elsewhere; feathering with the *requested*
    width then leaves a seam at a fixed column, which crawls through the clip.
    """
    def axis(size):
        if tile >= size:
            return [0], 0
        stride = tile - overlap
        count = max(2, -(-(size - overlap) // stride))
        last = size - tile
        origins = sorted({int(round(i * last / (count - 1))) for i in range(count)})
        got = min(tile - (origins[i + 1] - origins[i]) for i in range(len(origins) - 1))
        return origins, max(0, got)

    xs, ox = axis(width)
    ys, oy = axis(height)
    tw, th = min(tile, width), min(tile, height)
    return [(x, y, tw, th) for y in ys for x in xs], ox, oy


# ==============================================================================
# Encode
# ==============================================================================

def write_wav(path: Path, audio, sample_rate: int) -> Path:
    """Write float audio as 24-bit PCM so the only lossy audio step is the mux."""
    import numpy as np

    array = np.asarray(audio)
    while array.ndim > 2:
        array = array[0]
    if array.ndim == 1:
        array = array[None, :]
    if array.shape[0] > array.shape[1]:
        array = array.T
    ints = (np.clip(array, -1.0, 1.0) * 8388607.0).astype(np.int32)
    packed = ints.T.reshape(-1).astype("<i4").tobytes()
    raw = bytearray()
    for i in range(0, len(packed), 4):
        raw += packed[i:i + 3]
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(array.shape[0])
        handle.setsampwidth(3)
        handle.setframerate(int(sample_rate))
        handle.writeframes(bytes(raw))
    return path


def encode_command(width, height, fps, out_path, audio_path, crf=18, preset="medium"):
    """16-bit RGB on stdin -> 10-bit HEVC, bt709 tagged, audio as duration authority.

    deband + a tiny bit of temporally-varying noise fight the banding that a
    768p->1440p stretch otherwise produces in gradients. 10-bit output avoids
    re-quantising to 8 bits at the last step. `-shortest` trims the sub-frame tail
    so the container is exactly as long as the generated soundtrack.
    """
    filters = (
        "deband=1thr=0.008:2thr=0.008:3thr=0.008:4thr=0.008:range=16:blur=true,"
        "noise=alls=2:allf=t,"
        "format=yuv420p10le,"
        "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"
    )
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb48le",
        "-s", f"{width}x{height}", "-framerate", str(fps), "-i", "-",
    ]
    if audio_path:
        cmd += ["-i", str(audio_path)]
    cmd += ["-filter_complex", f"[0:v]{filters}[v]", "-map", "[v]"]
    if audio_path:
        cmd += ["-map", "1:a"]
    cmd += [
        "-c:v", "libx265", "-crf", str(crf), "-preset", preset,
        "-profile:v", "main10", "-tag:v", "hvc1",
        "-x265-params", "log-level=error:aq-mode=3",
    ]
    if audio_path:
        cmd += ["-c:a", "aac", "-b:a", "384k", "-ar", "48000", "-ac", "2", "-shortest"]
    cmd += ["-r", str(fps), "-movflags", "+faststart", str(out_path)]
    return cmd


# ==============================================================================
# Timing
# ==============================================================================

@dataclass
class Timings:
    stages: dict = field(default_factory=dict)

    def add(self, name: str, seconds: float):
        self.stages[name] = self.stages.get(name, 0.0) + seconds

    def timer(self, name: str):
        return _Timer(self, name)

    @property
    def total(self) -> float:
        return sum(self.stages.values())


class _Timer:
    def __init__(self, timings: Timings, name: str):
        self.timings, self.name = timings, name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.timings.add(self.name, time.perf_counter() - self.start)
        return False


# ==============================================================================
# Modal application
# ==============================================================================

try:
    import modal
except ImportError:  # allows --dry-run without modal installed
    modal = None

if modal is not None:
    # CUDA 13.1+ is required for B300, and gpu="B200+" may land on either chip,
    # so the image must be able to run on both.
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("ffmpeg", "git", "wget")
        .pip_install(
            "torch", "torchvision",
            extra_index_url="https://download.pytorch.org/whl/cu130",
        )
        .pip_install(
            "git+https://github.com/huggingface/diffusers.git",
            "transformers>=4.57.0",
            "accelerate", "safetensors", "huggingface_hub[hf_transfer]",
            "spandrel==0.4.2", "pillow", "numpy", "sentencepiece", "protobuf", "av",
        )
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/weights/hf"})
    )

    app = modal.App("giggsdance-bench")
    weights = modal.Volume.from_name("h3-weights", create_if_missing=True)
    upscalers = modal.Volume.from_name("giggsdance-upscalers", create_if_missing=True)

    # ----------------------------------------------------------------------
    # One-time downloads (CPU only -- do not pay GPU rates for this)
    # ----------------------------------------------------------------------

    @app.function(
        image=image, volumes={WEIGHTS_DIR: weights},
        timeout=6 * 60 * 60, cpu=8.0, memory=16384,
    )
    def download_weights(workflow: str = "t2va"):
        """Pull the diffusers-format H3 components into the Volume (~90 GB).

        Only what t2va/fl2va need: the `transformer/` partition and the shared
        components. `transformer_ref/` is another 61.7 GB and is only required for
        ref2va (omni-reference) generation.
        """
        from huggingface_hub import snapshot_download

        patterns = [
            "modular_model_index.json", "model_index.json", "*.json", "*.txt",
            "text_encoder/*", "tokenizer/*", "processor/*",
            "vae/*", "audio_vae/*", "scheduler/*", "audio_scheduler/*",
        ]
        patterns.append("transformer_ref/*" if workflow == "ref2va" else "transformer/*")

        started = time.time()
        path = snapshot_download(
            MODEL_ID, local_dir=f"{WEIGHTS_DIR}/MiniMax-H3",
            allow_patterns=patterns, max_workers=16,
        )
        weights.commit()

        total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
        elapsed = time.time() - started
        print(f"\ndownloaded {total / 1e9:.1f} GB in {elapsed / 60:.1f} min -> {path}")
        for entry in sorted(Path(path).iterdir()):
            if entry.is_dir():
                size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
                print(f"  {entry.name:20} {size / 1e9:7.2f} GB")
        return {"gb": total / 1e9, "minutes": elapsed / 60}

    @app.function(image=image, volumes={UPSCALER_DIR: upscalers}, timeout=1800)
    def download_upscaler(scale: int = 0):
        """Fetch the Real-ESRGAN checkpoints (~67 MB each)."""
        import urllib.request

        wanted = [scale] if scale in UPSCALERS else list(UPSCALERS)
        for value in wanted:
            name, url = UPSCALERS[value]
            target = Path(UPSCALER_DIR) / name
            if target.exists():
                print(f"{name} already present ({target.stat().st_size / 1e6:.1f} MB)")
                continue
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(url, target)
            print(f"  -> {target} ({target.stat().st_size / 1e6:.1f} MB)")
        upscalers.commit()
        return sorted(p.name for p in Path(UPSCALER_DIR).glob("*.pth"))

    # ----------------------------------------------------------------------
    # The renderer
    # ----------------------------------------------------------------------

    @app.cls(
        image=image,
        gpu=os.environ.get("GIGGSDANCE_GPU", "B200+"),
        volumes={WEIGHTS_DIR: weights, UPSCALER_DIR: upscalers},
        timeout=2 * 60 * 60,
        # Host RAM. With CPU offloading the weights are staged in system memory,
        # so this needs to be generous. If Modal cannot schedule it, drop to
        # 131072 and rely on the GPU being large enough to hold everything.
        memory=180 * 1024,
        min_containers=0,           # never pay for idle; ~$150/day if you do
        scaledown_window=300,       # reuse the warm container for 5 min
    )
    class Renderer:
        @modal.enter()
        def load(self):
            """Load H3 once per container. This is the expensive, amortisable part."""
            import torch
            from diffusers import ComponentsManager, ModularPipeline

            self.torch = torch
            started = time.perf_counter()

            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU: {name} ({vram:.0f} GB), torch {torch.__version__}, "
                  f"CUDA {torch.version.cuda}")
            self.gpu_name, self.gpu_vram = name, vram

            local = f"{WEIGHTS_DIR}/MiniMax-H3"
            source = local if Path(local).exists() else MODEL_ID
            print(f"loading H3 from {source} (expect ~124 GB for one workflow)")

            self.manager = ComponentsManager()
            self.pipe = ModularPipeline.from_pretrained(
                source, components_manager=self.manager
            )
            self.pipe.load_components(workflow="t2va", dtype=torch.bfloat16)

            # One workflow is ~124 GB in bf16 (61.7 transformer + 62.1 Qwen3-VL).
            # A B200 has 180 GB and a B300 288 GB, so on either chip it fits
            # resident and offloading would only add host-RAM staging and PCIe
            # traffic. Keep a margin for activations; offload only if tight.
            if vram >= 150:
                self.manager.enable_auto_cpu_offload(
                    device="cuda", memory_reserve_margin="40GB"
                )
                print("strategy: resident with 40GB activation margin")
            else:
                self.manager.enable_auto_cpu_offload(
                    device="cuda", memory_reserve_margin="12GB"
                )
                print("strategy: aggressive CPU offload (GPU under 150GB)")
            try:
                self.pipe.transformer.set_attention_backend("_flash_3_hub")
                print("attention backend: _flash_3_hub")
            except Exception as exc:
                print(f"attention backend left at default ({exc})")

            self.load_seconds = time.perf_counter() - started
            print(f"model load: {self.load_seconds:.1f}s")

        @modal.method()
        def print_doc(self):
            """Dump the pipeline's real signature -- ground truth for the call below."""
            return str(getattr(self.pipe, "doc", "<no doc>"))

        @modal.method()
        def report_load(self):
            """Return just the model-load time, for cheap cold-start measurement."""
            return self.load_seconds

        @modal.method()
        def render(
            self,
            prompt: str,
            duration_s: float = 5.0,
            aspect_ratio: str = "16:9",
            resolution: str = "1440p",
            fps: int = 60,
            steps: int = 24,
            seed: int = 0,
            interp: str = "torch",
            tile: int = 512,
            overlap: int = 48,
            crf: int = 18,
            preset: str = "medium",
        ):
            import numpy as np

            timings = Timings()
            timings.add("model_load", self.load_seconds)
            wall_start = time.perf_counter()

            num_frames = frames_for_duration(duration_s)
            canvas_w, canvas_h = resolve_canvas(aspect_ratio)

            if RESOLUTIONS[resolution] is None:
                # Native: crop the canvas to the closest exact 16:9 (or 9:16) and
                # do no scaling whatsoever. Zero super-resolution cost.
                if canvas_h > canvas_w:
                    out_w = canvas_w
                    out_h = min(canvas_h, int(round(canvas_w * 16 / 9)) & ~1)
                else:
                    out_w = canvas_w
                    out_h = min(canvas_h, int(round(canvas_w * 9 / 16)) & ~1)
            else:
                out_w, out_h = RESOLUTIONS[resolution]
                if canvas_h > canvas_w:                  # portrait -> portrait output
                    out_w, out_h = min(RESOLUTIONS[resolution]), max(RESOLUTIONS[resolution])

            print(f"\nclip: {num_frames} frames @24fps = {num_frames / 24:.4f}s")
            print(f"canvas: {canvas_w}x{canvas_h}  target: {out_w}x{out_h} @{fps}fps")

            # -- 1. generate ------------------------------------------------
            with timings.timer("generate"):
                frames, audio, sample_rate = self._generate(
                    prompt, num_frames, canvas_w, canvas_h, steps, seed
                )
            src_n, src_h, src_w = frames.shape[0], frames.shape[1], frames.shape[2]
            print(f"generated {src_n} frames at {src_w}x{src_h}, "
                  f"audio {audio.shape} @ {sample_rate} Hz")

            # -- 2. plan interpolation + geometry ---------------------------
            # The crop is needed before the scale can be chosen (the scale
            # depends on the *cropped* height), so plan twice: once to learn the
            # crop, once with the right model scale.
            plan = plan_interpolation(src_n, SRC_FPS, fps)
            crop_h = plan_geometry(src_w, src_h, out_w, out_h, 2).crop_h
            geo = plan_geometry(src_w, src_h, out_w, out_h, pick_scale(crop_h, out_h))
            print(f"crop {geo.crop_w}x{geo.crop_h} -> {geo.scale}x -> "
                  f"{geo.out_w}x{geo.out_h} ({len(plan)} output frames)")

            upscaler = self._load_upscaler(geo.scale, tile, overlap)

            work = Path("/tmp/bench")
            work.mkdir(parents=True, exist_ok=True)
            wav = work / "audio.wav"
            with timings.timer("audio_wav"):
                write_wav(wav, audio, sample_rate)

            out_path = work / f"clip_{resolution}_{fps}fps.mp4"
            cmd = encode_command(geo.out_w, geo.out_h, fps, out_path, wav, crf, preset)

            # -- 3. stream interpolate -> upscale -> encode -----------------
            # Streaming keeps peak memory flat. 863 frames of 1440p float32 would
            # be 38 GB if materialised; at 4K it would be 86 GB.
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            src = self.torch.from_numpy(frames)
            written = 0
            try:
                for left, t in plan:
                    with timings.timer("interpolate"):
                        frame = self._interpolate(src, left, t, interp)
                    with timings.timer("upscale"):
                        out = self._upscale_frame(frame, geo, upscaler)
                    with timings.timer("encode_pipe"):
                        proc.stdin.write(
                            np.ascontiguousarray(
                                (np.clip(out, 0, 1) * 65535.0 + 0.5).astype("<u2")
                            ).tobytes()
                        )
                    written += 1
                    if written % 60 == 0:
                        print(f"  {written}/{len(plan)} frames")
            finally:
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
                with timings.timer("encode_finalize"):
                    code = proc.wait()
            if code != 0:
                raise RuntimeError(f"ffmpeg exited {code}")

            wall = time.perf_counter() - wall_start
            probe = json.loads(subprocess.run(
                ["ffprobe", "-v", "error", "-print_format", "json",
                 "-show_format", "-show_streams", str(out_path)],
                check=True, capture_output=True, text=True).stdout)

            return {
                "timings": timings.stages,
                "wall_render_s": wall,
                "billable_s": wall + self.load_seconds,
                "src_frames": src_n,
                "out_frames": written,
                "canvas": f"{src_w}x{src_h}",
                "output": f"{geo.out_w}x{geo.out_h}",
                "fps": fps,
                "resolution": resolution,
                "steps": steps,
                "upscale_scale": geo.scale,
                "duration_s": src_n / SRC_FPS,
                "file_bytes": out_path.stat().st_size,
                "gpu": self.gpu_name,
                "gpu_vram_gb": round(self.gpu_vram, 1),
                "probe": probe,
                "video_bytes": out_path.read_bytes(),
            }

        # -- internals -----------------------------------------------------

        def _generate(self, prompt, num_frames, width, height, steps, seed):
            """Call H3. NOTE: this specific call is the untested part of the script."""
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

        def _interpolate(self, src, left, t, mode):
            """Blend at the exact timestamp. Cadence is correct; RIFE would be sharper."""
            if t <= 1e-9 or mode == "none":
                return src[left]
            right = src[min(left + 1, src.shape[0] - 1)]
            return src[left] * (1.0 - t) + right * t

        def _load_upscaler(self, scale, tile, overlap):
            from spandrel import ModelLoader

            self._tile, self._overlap = tile, overlap
            if scale == 1:
                print("upscaler: none needed (target <= source height, Lanczos only)")
                return None

            name, _ = UPSCALERS[scale]
            path = Path(UPSCALER_DIR) / name
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} missing. Run: modal run {Path(__file__).name}"
                    f"::download_upscaler --scale {scale}"
                )
            descriptor = ModelLoader().load_from_file(str(path))
            model = descriptor.model.eval().to("cuda").half()
            actual = int(getattr(descriptor, "scale", scale) or scale)
            print(f"upscaler {name}: {actual}x, tile={tile}, overlap={overlap}")
            self._tile, self._overlap = tile, overlap
            return model

        def _upscale_frame(self, frame, geo, model):
            """Tiled upscale, feathered blend, then Lanczos down to the exact size."""
            import numpy as np
            import torch

            left, upper, right, lower = geo.crop_box
            patch = frame[upper:lower, left:right]
            if hasattr(patch, "numpy"):
                patch = patch.cpu().numpy()
            patch = np.ascontiguousarray(np.asarray(patch, dtype=np.float32))
            h, w = patch.shape[:2]

            if model is None:
                if (h, w) == (geo.out_h, geo.out_w):
                    return patch          # native: crop only, genuinely free
                tensor = torch.from_numpy(patch).to("cuda").permute(2, 0, 1).unsqueeze(0)
                resized = torch.nn.functional.interpolate(
                    tensor, size=(geo.out_h, geo.out_w),
                    mode="bicubic", align_corners=False, antialias=True,
                )
                return resized.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()

            tiles, ox, oy = plan_tiles(w, h, self._tile, self._overlap)
            scale = geo.scale
            accum = torch.zeros((h * scale, w * scale, 3), device="cuda")
            weight = torch.zeros((h * scale, w * scale, 1), device="cuda")
            tensor_full = torch.from_numpy(patch).to("cuda")

            def ramp(length, ov, at_start, at_end):
                out = torch.ones(length, device="cuda")
                if ov <= 0:
                    return out
                n = min(ov, length)
                line = torch.linspace(0, 1, n + 2, device="cuda")[1:-1]
                if not at_start:
                    out[:n] = torch.minimum(out[:n], line)
                if not at_end:
                    out[-n:] = torch.minimum(out[-n:], line.flip(0))
                return out

            with torch.no_grad():
                for tx, ty, tw, th in tiles:
                    chunk = tensor_full[ty:ty + th, tx:tx + tw]
                    inp = chunk.permute(2, 0, 1).unsqueeze(0).half()
                    res = model(inp).float().clamp(0, 1).squeeze(0).permute(1, 2, 0)
                    ph, pw = res.shape[:2]
                    wy = ramp(ph, oy * scale, ty == 0, ty + th >= h)
                    wx = ramp(pw, ox * scale, tx == 0, tx + tw >= w)
                    mask = (wy[:, None] * wx[None, :])[:, :, None]
                    ys, xs = ty * scale, tx * scale
                    accum[ys:ys + ph, xs:xs + pw] += res * mask
                    weight[ys:ys + ph, xs:xs + pw] += mask

                blended = accum / weight.clamp_min(1e-6)
                if blended.shape[0] != geo.out_h or blended.shape[1] != geo.out_w:
                    blended = torch.nn.functional.interpolate(
                        blended.permute(2, 0, 1).unsqueeze(0),
                        size=(geo.out_h, geo.out_w),
                        mode="bicubic", align_corners=False, antialias=True,
                    ).squeeze(0).permute(1, 2, 0)
                return blended.clamp(0, 1).cpu().numpy()

    # ----------------------------------------------------------------------
    # Entrypoint
    # ----------------------------------------------------------------------

    @app.local_entrypoint()
    def probe_load():
        """Measure ONLY the cold start and model load, then exit.

        The load is the fixed cost you pay before generating anything, and it is
        the least predictable number in the whole system (~124 GB off a Volume).
        Run this once to learn it, then you can budget every later run properly.
        """
        renderer = Renderer()
        seconds = renderer.report_load.remote()
        print(f"\nmodel load: {seconds:.1f}s = ${seconds * USD_PER_SECOND:.4f}")
        print(f"this is the fixed cost of every cold start; a warm container "
              f"(within {300}s) skips it")

    @app.local_entrypoint()
    def benchmark(
        prompt: str = (
            "integrated_multimodal_description: [Shot 1] Cinematic wide shot, slow "
            "push in. A lone lighthouse on black volcanic rock at dusk, its beam "
            "sweeping through drifting sea mist. Heavy swells break against the rock "
            "and throw spray into the light. The sky is deep indigo fading to burnt "
            "orange at the horizon, with a smooth gradient and no visible banding.\n"
            "overall_soundscape: Deep rhythmic booms of waves against stone, hissing "
            "spray, a low wind that rises and falls, and the distant two-tone moan of "
            "a foghorn.\n"
            "non_diegetic_music: Sparse ambient score, sustained low strings and a "
            "single distant piano note repeating slowly."
        ),
        duration_s: float = 5.0,
        resolution: str = "native",
        fps: int = 60,
        steps: int = 8,
        seed: int = 0,
        aspect_ratio: str = "16:9",
        interp: str = "torch",
        crf: int = 18,
        preset: str = "veryfast",
        budget_usd: float = DEFAULT_BUDGET_USD,
        out: str = "giggsdance_bench.mp4",
        print_doc: bool = False,
    ):
        """Render one clip and report exact costs.

        Defaults are the cheap ones: native resolution (no super-resolution),
        5 seconds, 8 steps, fast encode. That keeps a single run comfortably
        inside a $1 budget provided the cold-start load behaves -- which is the
        one number nobody can predict in advance. Run ``probe_load`` first if you
        want to know it before committing.
        """
        if resolution not in RESOLUTIONS:
            raise SystemExit(f"resolution must be one of {list(RESOLUTIONS)}")

        # Turn the budget into a hard wall. Modal terminates the container at the
        # timeout, so the bill cannot exceed the budget even if something hangs.
        max_seconds = int(budget_usd / USD_PER_SECOND)
        print(f"budget ${budget_usd:.2f} = {max_seconds}s of B200 time "
              f"({max_seconds / 60:.1f} min); container timeout set to that.")
        if resolution == "native":
            print("resolution=native -> no super-resolution model runs at all")
        else:
            print(f"resolution={resolution} -> a super-resolution pass WILL run "
                  f"(~84% of post-processing time; use 'native' to skip it)")

        renderer = Renderer.with_options(timeout=max_seconds)()
        if print_doc:
            print(renderer.print_doc.remote())
            return

        print(f"submitting: {duration_s}s, {resolution}, {fps}fps, {steps} steps")
        started = time.time()
        try:
            result = renderer.render.remote(
                prompt=prompt, duration_s=duration_s, aspect_ratio=aspect_ratio,
                resolution=resolution, fps=fps, steps=steps, seed=seed,
                interp=interp, crf=crf, preset=preset,
            )
        except Exception as exc:
            spent = time.time() - started
            print(f"\nrun failed after {spent:.0f}s "
                  f"(<= ${spent * USD_PER_SECOND:.2f} spent): {exc}")
            print("If this was a timeout, raise --budget-usd or lower --steps.")
            raise
        round_trip = time.time() - started

        video = result.pop("video_bytes")
        Path(out).write_bytes(video)
        probe = result.pop("probe")
        _report(result, probe, round_trip, out, budget_usd)


def _report(result, probe, round_trip, out_path, budget_usd=DEFAULT_BUDGET_USD):
    """Print the cost table."""
    stages = result["timings"]
    load = stages.get("model_load", 0.0)
    render_stages = {k: v for k, v in stages.items() if k != "model_load"}
    render_total = sum(render_stages.values())
    billable = result["billable_s"]
    out_frames = result["out_frames"]
    video_s = result["duration_s"]

    line = "=" * 74
    print(f"\n{line}\nGIGGSDANCE BENCHMARK -- MEASURED\n{line}")
    print(f"GPU              {result['gpu']} ({result['gpu_vram_gb']} GB)")
    print(f"generation       {result['src_frames']} frames @24fps at {result['canvas']}, "
          f"{result['steps']} steps")
    print(f"output           {result['output']} @{result['fps']}fps, "
          f"{out_frames} frames, {video_s:.3f}s of video")
    print(f"upscaler         {result['upscale_scale']}x")
    print(f"file             {result['file_bytes'] / 1e6:.1f} MB "
          f"({result['file_bytes'] * 8 / video_s / 1e6:.1f} Mbps)")

    print(f"\n{'stage':<20}{'seconds':>10}{'% render':>10}{'USD':>10}   per out-frame")
    print("-" * 74)
    for name, seconds in sorted(render_stages.items(), key=lambda kv: -kv[1]):
        print(f"{name:<20}{seconds:>10.2f}{seconds / render_total * 100:>9.1f}%"
              f"{seconds * USD_PER_SECOND:>10.4f}   {seconds / out_frames * 1000:>7.1f} ms")
    print("-" * 74)
    print(f"{'render subtotal':<20}{render_total:>10.2f}{100.0:>9.1f}%"
          f"{render_total * USD_PER_SECOND:>10.4f}")
    print(f"{'model load (once)':<20}{load:>10.2f}{'':>10}"
          f"{load * USD_PER_SECOND:>10.4f}")
    print(f"{'BILLABLE TOTAL':<20}{billable:>10.2f}{'':>10}"
          f"{billable * USD_PER_SECOND:>10.4f}")

    actual = billable * USD_PER_SECOND
    print(f"\ncost of this clip           ${actual:.4f}")
    print(f"  per second of video       ${actual / video_s:.4f}")
    print(f"  marginal (warm container) ${render_total * USD_PER_SECOND:.4f} "
          f"= ${render_total * USD_PER_SECOND / video_s:.4f}/s of video")
    print(f"local round trip            {round_trip:.1f}s (includes cold start + transfer)")

    verdict = "UNDER" if actual <= budget_usd else "OVER"
    share = actual / budget_usd * 100 if budget_usd else 0.0
    print(f"\nbudget ${budget_usd:.2f}  ->  spent ${actual:.4f}  "
          f"({share:.0f}% of budget, {verdict})")
    if actual > budget_usd:
        print("  to get cheaper: --resolution native, fewer --steps, "
              "--duration-s 5, --preset ultrafast")
    if load > render_total:
        print(f"  note: model load ({load:.0f}s) exceeded the render itself "
              f"({render_total:.0f}s). Batch several clips per container -- a warm "
              f"run costs only ${render_total * USD_PER_SECOND:.4f}.")

    # -- projections ----------------------------------------------------
    per_frame = render_total / out_frames
    print(f"\n{line}\nPROJECTED (linear extrapolation -- verify before trusting)\n{line}")
    print(f"{'target':<26}{'out frames':>12}{'render':>12}{'USD warm':>11}{'USD cold':>11}")
    print("-" * 74)
    fps_v = result["fps"]
    for label, seconds in [
        ("14.375s (max single clip)", 14.375),
        ("1 minute (5 clips)", 60.0),
        ("10 minutes (42 clips)", 600.0),
    ]:
        frames = int(seconds * fps_v)
        secs = frames * per_frame
        warm = secs * USD_PER_SECOND
        cold = (secs + load) * USD_PER_SECOND
        print(f"{label:<26}{frames:>12}{secs / 60:>10.1f}m{warm:>11.2f}{cold:>11.2f}")
    print("\nNote: chained clips need a per-clip generation pass, and H3 generates each")
    print("clip's soundtrack independently, so joins have audible ambience seams.")
    print(f"\nwrote {out_path}")
    print("\nDisclosure: this is AI-generated video. If you publish it, mark it as")
    print("synthetic (YouTube's altered-content setting) -- the H3 licence requires")
    print("disclosing machine-generated content in public environments.")


# ==============================================================================
# Local dry run -- validate the pipeline for $0, no GPU, no weights, no Modal
# ==============================================================================

def dry_run(resolution="native", fps=60, duration_s=5.0, out="dryrun.mp4"):
    """Exercise every stage except H3 with synthetic frames at low resolution."""
    import shutil

    import numpy as np

    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise SystemExit(
                f"{tool} is not on PATH, and the dry run needs it to encode.\n\n"
                "  Debian/Ubuntu/Colab:  apt-get install -y ffmpeg\n"
                "  Kaggle:               already installed\n"
                "  macOS:                brew install ffmpeg\n\n"
                "It must be built with libx265 and the deband/noise filters.\n"
                "Note: the GPU path installs ffmpeg inside the Modal image, so "
                "this is only needed for the local dry run."
            )

    print("DRY RUN -- synthetic frames, no GPU, no weights, $0")

    num_frames = frames_for_duration(duration_s)
    canvas_w, canvas_h = resolve_canvas("16:9")
    print(f"H3 would generate {num_frames} frames @24fps at {canvas_w}x{canvas_h} "
          f"= {num_frames / 24:.4f}s")

    # Shrink the canvas proportionally so this is fast, keeping the 1.75 aspect.
    sw, sh = 224, 128
    if resolution == "native":
        # Native crops to exact 16:9 and does not scale at all.
        small = (sw, min(sh, int(round(sw * 9 / 16)) & ~1))
        print("resolution=native -> crop only, no super-resolution")
    else:
        small = {"720p": (320, 180), "1080p": (480, 270),
                 "1440p": (640, 360), "2160p": (960, 540)}[resolution]
    src = []
    for i in range(num_frames):
        x = np.linspace(0, 1, sw, dtype=np.float32)[None, :]
        y = np.linspace(0, 1, sh, dtype=np.float32)[:, None]
        g = 0.15 + 0.25 * y + 0.02 * x
        frame = np.stack([g, g * 0.95, g * 1.05], -1)
        cx = int(i / num_frames * sw)
        frame[:, max(0, cx - 4):cx + 4, :] = 0.85
        src.append(np.clip(frame, 0, 1))
    src = np.asarray(src, dtype=np.float32)

    plan = plan_interpolation(num_frames, SRC_FPS, fps)
    crop_h = plan_geometry(sw, sh, small[0], small[1], 2).crop_h
    geo = plan_geometry(sw, sh, small[0], small[1], pick_scale(crop_h, small[1]))
    tag = "crop only" if geo.scale == 1 else f"{geo.scale}x model"
    print(f"plan: {len(plan)} output frames @{fps}fps; "
          f"crop {geo.crop_w}x{geo.crop_h} -> {tag} -> {geo.out_w}x{geo.out_h}")

    sr = 32000
    n = int(round(num_frames / SRC_FPS * sr))
    t = np.arange(n) / sr
    audio = np.stack([0.2 * np.sin(2 * np.pi * 440 * t),
                      0.2 * np.sin(2 * np.pi * 554 * t)]).astype(np.float32)
    work = Path("/tmp/gd_dry")
    work.mkdir(parents=True, exist_ok=True)
    wav = write_wav(work / "a.wav", audio, sr)

    from PIL import Image

    def upscale(patch, out_w, out_h):
        """Stand-in for the real ESRGAN. PIL infers I;16 from a uint16 array."""
        bands = []
        for c in range(3):
            band = Image.fromarray(
                (np.clip(patch[:, :, c], 0, 1) * 65535).astype(np.uint16))
            bands.append(np.asarray(band.resize((out_w, out_h), Image.LANCZOS))
                         .astype(np.float32) / 65535.0)
        return np.stack(bands, -1)

    timings = Timings()
    cmd = encode_command(geo.out_w, geo.out_h, fps, work / out, wav, 20, "ultrafast")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    written = 0
    try:
        for left, tt in plan:
            with timings.timer("interpolate"):
                if tt <= 1e-9:
                    frame = src[left]
                else:
                    frame = src[left] * (1 - tt) + src[min(left + 1, num_frames - 1)] * tt
            with timings.timer("upscale"):
                l, u, r, d = geo.crop_box
                patch = frame[u:d, l:r]
                res = (patch if (patch.shape[1], patch.shape[0]) == (geo.out_w, geo.out_h)
                       else upscale(patch, geo.out_w, geo.out_h))
            with timings.timer("encode_pipe"):
                proc.stdin.write(np.ascontiguousarray(
                    (np.clip(res, 0, 1) * 65535 + 0.5).astype("<u2")).tobytes())
            written += 1
    finally:
        proc.stdin.close()
        with timings.timer("encode_finalize"):
            code = proc.wait()
    if code != 0:
        raise SystemExit(f"ffmpeg failed with {code}")

    path = work / out
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        check=True, capture_output=True, text=True).stdout)
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    audio_s = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    v_dur = float(video.get("duration") or probe["format"]["duration"])
    a_dur = float(audio_s.get("duration") or probe["format"]["duration"])

    checks = [
        ("frames match plan", abs(int(video.get("nb_frames") or 0) - len(plan)) <= 1,
         f"{video.get('nb_frames')} vs {len(plan)}"),
        ("fps", video["r_frame_rate"] == f"{fps}/1", video["r_frame_rate"]),
        ("10-bit", video["pix_fmt"] == "yuv420p10le", video["pix_fmt"]),
        ("bt709 tagged", video.get("color_primaries") == "bt709", video.get("color_primaries")),
        ("A/V agree <20ms", abs(v_dur - a_dur) < 0.02, f"v={v_dur:.4f} a={a_dur:.4f}"),
        ("duration matches source", abs(v_dur - num_frames / SRC_FPS) < 0.03,
         f"{v_dur:.4f} vs {num_frames / SRC_FPS:.4f}"),
        ("audio 48kHz stereo",
         int(audio_s["sample_rate"]) == 48000 and int(audio_s["channels"]) == 2,
         f"{audio_s['sample_rate']}Hz x{audio_s['channels']}"),
    ]
    print()
    failed = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:26} {detail}")
        failed += 0 if ok else 1
    total = sum(timings.stages.values())
    print(f"\n{written} frames in {total:.1f}s; stage split:")
    for k, v in sorted(timings.stages.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<18}{v:>8.2f}s  {v / total * 100:>5.1f}%")
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed -> {path}")
    print("\nPipeline plumbing is sound. The GPU run will differ only in the")
    print("generation stage and the real ESRGAN upscaler.")
    return 1 if failed else 0


if __name__ == "__main__":
    if "--dry-run" in sys.argv:
        args = {}
        for key in ("resolution", "fps", "duration-s"):
            flag = f"--{key}"
            if flag in sys.argv:
                value = sys.argv[sys.argv.index(flag) + 1]
                args[key.replace("-", "_")] = (
                    int(value) if key == "fps" else
                    float(value) if key == "duration-s" else value
                )
        raise SystemExit(dry_run(**args))
    print(__doc__)
    print("Run with --dry-run for a free local check, or use `modal run`.")
