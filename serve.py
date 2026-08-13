#!/usr/bin/env python3
"""
Giggsdance on SGLang — the fast path, with 12-reference support.

    modal run serve.py

Runs H3 through SGLang instead of diffusers, then applies our 60 fps conversion,
optional upscaling and 10-bit encode on top.

--------------------------------------------------------------------------------
WHY SGLANG
--------------------------------------------------------------------------------
Measured by SGLang on 8x B300 SXM6, 1344x768, 124 frames, 50 steps:

    FL2VA  BF16   19.04 s     load 118 s     83.6 GB/GPU
    FL2VA  FP8    18.03 s     load 116 s     51.9 GB/GPU
    Ref2VA BF16   29.12 s     load 114 s     84.0 GB/GPU

Two reasons it wins over the alternatives:

  * vLLM-Omni's reference path takes ONE image plus ONE audio, or videos with no
    separate audio. It cannot do the 12-file omni-reference set. SGLang can.
  * The diffusers reference implementation is BF16 written for clarity; SGLang
    measured 1.41x over it and is where the Turbo LoRA recipes are documented.

--------------------------------------------------------------------------------
QUANTIZATION AND "NO QUALITY LOSS"
--------------------------------------------------------------------------------
Short answer: **do not quantize, you do not need to.**

FP8 on a B300 measured 19.04 -> 18.03 s. That is **5%**, while cutting memory
38%. It is a capacity tool for cards that cannot hold the weights, and you have
288 GB. It is also not bit-exact.

What is genuinely lossless, and is all switched on by default here:

    resident components (--performance-mode speed)  the recommended latency path
    pure Ulysses sequence parallelism               faster AND the capacity default
    VAE patch parallelism, tile mode                cuts decode 3.4-3.5x
    warmup matched to the served resolution         removes ~10 s of first-request cost
    eager BF16/FP32 denoise                         bit-exact vs the reference

Explicitly NOT lossless, and therefore opt-in:

    --quantize-fp8      not bit-exact, 5% on B300
    --quality high      1.40x, SSIM 0.931 / PSNR 28.16 dB vs lossless
    --lora lightx2v     4 denoiser evals instead of 50; a different model
    torch.compile       changes numerical output; never enabled here

So: for no quality loss, speed comes from GPU count, not from precision.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
    modal run serve.py                                    # 5 s, lossless, native
    modal run serve.py --refs "a.png,b.png,clip.mp4,v.wav"  # up to 12 references
    modal run serve.py --lora lightx2v                    # ~10x fewer evaluations
    modal run serve.py --gpus 4                           # ~5x faster, lossless
    modal run serve.py --quality high                     # 1.40x, near-lossless
    modal run serve.py --resolution 1440p --fps 60         # upscale + interpolate

Powered by MiniMax H3. See NOTICE.md: the licence excludes the EU, UK, South
Korea and USA, and covers the model's outputs as well as its weights.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import modal

try:
    from giggsdance.backends.sglang_client import (
        TURBO_LORAS,
        QualityMode,
        RenderRequest,
        SGLangClient,
        Target,
        build_conditions,
        build_serve_command,
        steps_for,
    )
    from giggsdance.references import ReferenceError, build_reference_set
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        f"could not import the giggsdance package ({exc}). Keep serve.py next to "
        "the giggsdance/ directory:\n\n"
        "    git clone https://github.com/Hvkki/minimax.git\n"
        "    cd minimax && modal run serve.py\n"
    ) from exc

MODEL_ID = "MiniMaxAI/MiniMax-H3"
WEIGHTS_DIR = "/weights"
MODEL_PATH = f"{WEIGHTS_DIR}/MiniMax-H3"
PORT = 30010

# Modal rates. B200/B200+ is Modal's published $6.25/hr; the B300 figure is a
# third-party listing and is labelled as such wherever it is printed.
GPU_RATES = {"B200": 6.25 / 3600, "B200+": 6.25 / 3600, "B300": 7.10 / 3600}
UNVERIFIED_RATES = {"B300"}

# On-disk sizes measured from the HF manifest. Note these are GB; the docs quote
# the same components in GiB (66.28 GB == 61.73 GiB), which is why the two sets
# of numbers look different but agree.
ONE_PARTITION_GB = 144.06
BOTH_PARTITIONS_GB = 210.34

GPU = os.environ.get("GIGGSDANCE_GPU", "B300")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "wget")
    .pip_install("torch", "torchvision",
                 extra_index_url="https://download.pytorch.org/whl/cu130")
    # SGLang with the diffusion extra is what serves H3.
    .pip_install("sglang[diffusion]", "huggingface_hub[hf_transfer]",
                 "numpy", "pillow", "av")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": f"{WEIGHTS_DIR}/hf"})
    .add_local_python_source("giggsdance")
)

app = modal.App("giggsdance-sglang")
weights = modal.Volume.from_name("h3-weights", create_if_missing=True)
media = modal.Volume.from_name("giggsdance-media", create_if_missing=True)


@app.function(image=image, volumes={WEIGHTS_DIR: weights},
              timeout=6 * 60 * 60, cpu=8.0, memory=16384)
def fetch_weights(variant: str = "fl2va", force: bool = False) -> dict:
    """Download the partition SGLang needs.

    SGLang owns the checkpoint-directory mapping and expects the repository root,
    so unlike the diffusers path this fetches the original FL2VA/ or Ref2VA/
    layout rather than the diffusers-format subfolders.
    """
    from huggingface_hub import snapshot_download

    root = Path(MODEL_PATH)
    wanted = "Ref2VA" if variant == "ref2va" else "FL2VA"
    marker = root / wanted

    def size_of(path: Path) -> float:
        if not path.is_dir():
            return 0.0
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9

    existing = size_of(marker)
    if not force and existing > 100:
        print(f"{wanted} already present: {existing:.1f} GB -- skipping")
        return {"downloaded": False, "gb": existing, "variant": variant}

    print(f"downloading {wanted} (~{ONE_PARTITION_GB:.0f} GB), one time")
    started = time.time()
    snapshot_download(
        MODEL_ID, local_dir=MODEL_PATH, max_workers=16,
        allow_patterns=[f"{wanted}/*", "*.json", "*.txt"],
    )
    weights.commit()
    elapsed = time.time() - started
    total = size_of(marker)
    print(f"downloaded {total:.1f} GB in {elapsed / 60:.1f} min")
    return {"downloaded": True, "gb": total, "seconds": elapsed, "variant": variant}


@app.cls(
    image=image, gpu=GPU, volumes={WEIGHTS_DIR: weights, "/media": media},
    timeout=2 * 60 * 60, memory=200 * 1024,
    min_containers=0, scaledown_window=600,
)
class Server:
    """Runs `sglang serve` as a subprocess and talks to it over localhost."""

    @modal.enter()
    def start(self):
        variant = os.environ.get("GIGGSDANCE_VARIANT", "fl2va")
        gpus = int(os.environ.get("GIGGSDANCE_GPUS", "1"))
        lora = os.environ.get("GIGGSDANCE_LORA") or None
        fp8 = os.environ.get("GIGGSDANCE_FP8") == "1"

        command = build_serve_command(
            MODEL_PATH if Path(MODEL_PATH).exists() else MODEL_ID,
            variant=variant, port=PORT, num_gpus=gpus,
            quantize_fp8=fp8, lora=lora,
        )
        print("launching:", " ".join(command))
        if fp8:
            print("NOTE: FP8 is a capacity option and is NOT bit-exact. "
                  "On B300 it measured 5% faster while cutting memory 38%.")

        self.log = open("/tmp/sglang.log", "wb")
        self.proc = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT)
        self.client = SGLangClient(f"http://127.0.0.1:{PORT}")

        started = time.perf_counter()
        try:
            self.client.wait_until_healthy(deadline_s=2400)
        except Exception:
            self.proc.poll()
            tail = Path("/tmp/sglang.log").read_text(errors="replace")[-4000:]
            raise RuntimeError(f"SGLang failed to start. Log tail:\n{tail}")
        self.load_seconds = time.perf_counter() - started
        self.variant, self.gpus, self.lora = variant, gpus, lora
        print(f"SGLang healthy after {self.load_seconds:.1f}s "
              f"(SGLang measured 114-124 s on B300 for ~144 GB)")

    @modal.exit()
    def stop(self):
        try:
            self.proc.terminate()
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()

    @modal.method()
    def render(
        self,
        prompt: str,
        duration_s: float = 5.0,
        aspect_ratio: str = "16:9",
        short_edge: int = 768,
        seed: int = 0,
        steps: int | None = None,
        quality: str = "lossless",
        reference_order: list | None = None,
        resolution: str = "native",
        fps: int = 60,
        crf: int = 18,
        preset: str = "veryfast",
    ) -> dict:
        """Generate with SGLang, then post-process to the requested fps/resolution."""
        from giggsdance.references import ReferenceSet

        timings = {"model_load": self.load_seconds}
        wall = time.perf_counter()

        references = ReferenceSet(order=list(reference_order or []))
        for kind, path in references.order:
            {"image": references.images, "video": references.videos,
             "audio": references.audios}[kind].append(path)
        task = references.workflow if not references.is_empty else "t2va"

        mode = QualityMode(quality) if quality in ("lossless", "high") else QualityMode.LOSSLESS
        request = RenderRequest(
            prompt=prompt,
            task=task,
            target=Target(duration_s, aspect_ratio, short_edge),
            seed=seed,
            num_inference_steps=steps or steps_for(mode, self.lora),
            quality=mode,
            conditions=build_conditions(references, task),
            lora_scale=TURBO_LORAS[self.lora]["lora_scale"] if self.lora else None,
        )
        print(f"task={task} steps={request.num_inference_steps} quality={mode.value} "
              f"refs={references.total} lora={self.lora or 'none'}")

        raw = Path("/tmp/h3_raw.mp4")
        mark = time.perf_counter()
        _, generate_s = self.client.render(request, raw, max_wait_s=5400)
        timings["generate"] = time.perf_counter() - mark
        print(f"SGLang returned {raw.stat().st_size / 1e6:.1f} MB in {generate_s:.1f}s")

        # -- post-process: 24 -> fps, optional upscale, 10-bit encode --------
        mark = time.perf_counter()
        final = self._postprocess(raw, resolution, fps, crf, preset)
        timings["postprocess"] = time.perf_counter() - mark

        probe = self._probe(final)
        return {
            "timings": timings,
            "billable_s": (time.perf_counter() - wall) + self.load_seconds,
            "render_s": time.perf_counter() - wall,
            "task": task,
            "steps": request.num_inference_steps,
            "quality": mode.value,
            "lora": self.lora,
            "gpus": self.gpus,
            "references": references.total,
            "file_bytes": final.stat().st_size,
            "probe": probe,
            "video": final.read_bytes(),
        }

    # -- internals --------------------------------------------------------

    def _postprocess(self, source: Path, resolution: str, fps: int,
                     crf: int, preset: str) -> Path:
        """Interpolate to fps and optionally upscale, keeping H3's audio intact.

        SGLang hands back a finished H.264 MP4, so unlike the diffusers path we
        start from an encoded file. That means one extra decode, and it is why the
        diffusers path keeps a quality edge: it never round-trips through H.264.
        """
        import json as _json

        info = _json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_streams", str(source)],
            check=True, capture_output=True, text=True).stdout)
        stream = next(s for s in info["streams"] if s["codec_type"] == "video")
        width, height = int(stream["width"]), int(stream["height"])

        from giggsdance.stages.upscale import RESOLUTIONS, resolve_target
        out_w, out_h = resolve_target(resolution, width, height)

        filters = [f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:vsbmc=1"] if fps != 24 else []
        if (out_w, out_h) != (width, height):
            filters.append(f"scale={out_w}:{out_h}:flags=lanczos")
        filters += [
            "deband=1thr=0.008:2thr=0.008:3thr=0.008:4thr=0.008:range=16:blur=true",
            "noise=alls=2:allf=t",
            "format=yuv420p10le",
            "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv",
        ]

        target = Path("/tmp/h3_final.mp4")
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-i", str(source),
            "-vf", ",".join(filters),
            "-c:v", "libx265", "-crf", str(crf), "-preset", preset,
            "-profile:v", "main10", "-tag:v", "hvc1",
            "-x265-params", "log-level=error:aq-mode=3",
            "-c:a", "aac", "-b:a", "384k", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart", str(target),
        ]
        print("post:", " ".join(command[:12]), "...")
        subprocess.run(command, check=True)
        return target

    @staticmethod
    def _probe(path: Path) -> dict:
        import json as _json
        return _json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            check=True, capture_output=True, text=True).stdout)


@app.local_entrypoint()
def main(
    prompt: str = (
        "integrated_multimodal_description: [Shot 1] Cinematic wide shot, slow push in. "
        "A lone lighthouse on black volcanic rock at dusk, beam sweeping through sea mist. "
        "Heavy swells break against the rock and throw spray into the light.\n"
        "overall_soundscape: Deep rhythmic booms of waves on stone, hissing spray, low wind.\n"
        "non_diegetic_music: Sparse ambient strings under a slow repeating piano note."
    ),
    duration_s: float = 5.0,
    aspect_ratio: str = "16:9",
    short_edge: int = 768,
    seed: int = 0,
    steps: int = 0,
    quality: str = "lossless",
    lora: str = "",
    gpus: int = 1,
    quantize_fp8: bool = False,
    resolution: str = "native",
    fps: int = 60,
    refs: str = "",
    images: str = "",
    videos: str = "",
    audios: str = "",
    out: str = "output.mp4",
    budget_usd: float = 5.0,
):
    banner = "=" * 74
    print(f"{banner}\nGIGGSDANCE on SGLang -- powered by MiniMax H3\n{banner}")
    print("Licence: invalid in the EU, UK, South Korea and USA, and it covers")
    print("outputs as well as weights. See NOTICE.md. Mark output AI-generated.\n")

    def split(value: str) -> list[str]:
        return [x.strip() for x in value.split(",") if x.strip()]

    try:
        references = build_reference_set(
            images=split(images), videos=split(videos),
            audios=split(audios), mixed=split(refs),
        )
    except ReferenceError as exc:
        raise SystemExit(f"reference set rejected before spending anything:\n  {exc}")

    missing = [p for _, p in references.order if "://" not in p and not Path(p).exists()]
    if missing:
        raise SystemExit(f"reference files not found: {missing}")

    variant = "ref2va" if references.workflow == "ref2va" else "fl2va"
    rate = GPU_RATES.get(GPU, GPU_RATES["B200"])
    resolved_steps = steps or steps_for(
        QualityMode(quality) if quality in ("lossless", "high") else QualityMode.LOSSLESS,
        lora or None,
    )

    print(f"gpu        {GPU} x{gpus} at ${rate * 3600:.2f}/hr"
          + ("  (third-party rate)" if GPU in UNVERIFIED_RATES else "  (Modal published)"))
    print(f"variant    {variant}   weights ~{ONE_PARTITION_GB:.0f} GB")
    print(f"quality    {quality}"
          + (f" + {lora} LoRA" if lora else "")
          + (f"   steps={resolved_steps}"))
    if quantize_fp8:
        print("           FP8 requested: NOT bit-exact, and only ~5% faster on B300.")
        print("           With 288 GB you do not need it -- prefer --gpus for speed.")
    if references.total:
        print(f"references {references.summary()} ({references.total}/12)")
        for i, (kind, path) in enumerate(references.order, 1):
            print(f"   {i:2}. {kind:5} {path}")
    print()

    os.environ["GIGGSDANCE_VARIANT"] = variant
    os.environ["GIGGSDANCE_GPUS"] = str(gpus)
    if lora:
        os.environ["GIGGSDANCE_LORA"] = lora
    if quantize_fp8:
        os.environ["GIGGSDANCE_FP8"] = "1"

    print("[1/2] weights")
    info = fetch_weights.remote(variant=variant)
    if info["downloaded"]:
        print(f"      fetched {info['gb']:.1f} GB in {info['seconds'] / 60:.1f} min")

    print("[2/2] serve + render")
    max_seconds = int(budget_usd / rate)
    server = Server.with_options(timeout=max_seconds, gpu=f"{GPU}:{gpus}" if gpus > 1 else GPU)()
    started = time.time()
    result = server.render.remote(
        prompt=prompt, duration_s=duration_s, aspect_ratio=aspect_ratio,
        short_edge=short_edge, seed=seed, steps=steps or None, quality=quality,
        reference_order=references.order or None, resolution=resolution, fps=fps,
    )

    Path(out).write_bytes(result.pop("video"))
    _report(result, result.pop("probe"), time.time() - started, out, rate, gpus)


def _report(result, probe, round_trip, out_path, rate, gpus):
    stages = result["timings"]
    load = stages.get("model_load", 0.0)
    render = {k: v for k, v in stages.items() if k != "model_load"}
    total = result["billable_s"]
    gpu_seconds = total * gpus
    cost = gpu_seconds * rate

    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    line = "=" * 74
    print(f"\n{line}\nMEASURED\n{line}")
    print(f"task        {result['task']}   steps {result['steps']}   "
          f"quality {result['quality']}" + (f" + {result['lora']}" if result["lora"] else ""))
    print(f"output      {video['width']}x{video['height']} @ {video['r_frame_rate']}, "
          f"{video['pix_fmt']}")
    print(f"file        {result['file_bytes'] / 1e6:.1f} MB")
    print(f"references  {result['references']}")
    print()
    for name, seconds in sorted(render.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<14}{seconds:>8.1f} s")
    print(f"  {'model load':<14}{load:>8.1f} s   (once per cold container)")
    print(f"  {'TOTAL':<14}{total:>8.1f} s x {gpus} GPU = {gpu_seconds:.0f} GPU-s")
    print(f"\ncost ${cost:.4f}   warm again ${sum(render.values()) * gpus * rate:.4f}")
    print(f"wall clock {round_trip:.0f}s")
    print(f"\nwrote {out_path}")
    print("\nSpeed levers, in order of quality cost:")
    print("  --gpus 4          lossless, ~5x     (Ulysses shards the sequence)")
    print("  --quality high    SSIM 0.931, 1.40x (audited Cache-DiT)")
    print("  --lora lightx2v   different model, ~10x fewer evaluations")
    print("  --quantize-fp8    NOT bit-exact, only ~5% on B300 -- skip it")


if __name__ == "__main__":
    print(__doc__)
    print("Run:  modal run serve.py")
