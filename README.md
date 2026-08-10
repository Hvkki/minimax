# Giggsdance

A 60 fps, high-resolution rendering pipeline on top of [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), built to run on Modal.

H3 generates **768p, 24 fps, 5–15 s clips with a jointly generated stereo soundtrack**. This repo turns that into something you can upload: correct 24→60 fps conversion, optional super-resolution to 1080p/1440p/2160p, banding control, 10-bit output, and exact A/V sync.

> **Before anything else, read [NOTICE.md](NOTICE.md).** The H3 licence does not grant rights in the EU, UK, South Korea or USA, and the restriction covers the model's **outputs**, not just its weights.

---

## Status: honest version

| Component | State |
|---|---|
| `constraints.py` — frame/canvas rules | **verified**, 36 unit tests |
| `stages/interpolate.py` — 24→60 fps timing | **verified** |
| `stages/upscale.py` — crop, tile, supersample | **verified** |
| `stages/encode.py` — 10-bit mux | **verified**, 7 end-to-end checks |
| `frameio.py` — 16-bit round-trip | **verified** |
| `prompt.py` — Context-IR-style prompts | **verified** |
| `stages/generate.py` — the H3 call | **written, never executed** |

That last row matters. The H3 generation call is written against the published diffusers integration and model card, but running it needs ~124 GB of weights and a large GPU. If something breaks, it is almost certainly there — run `modal run bench_one_video.py::benchmark --print-doc` to dump the pipeline's real signature.

---

## Quick start

```bash
pip install -r requirements.txt
modal setup

# 0. FREE. No GPU, no weights, no Modal. Validates the whole pipeline.
python bench_one_video.py --dry-run

# 1. Weights into a Modal Volume (~90 GB, once, on CPU — not GPU rates)
modal run bench_one_video.py::download_weights

# 2. Measure the cold-start cost once, so you can budget properly
modal run bench_one_video.py::probe_load

# 3. Render one clip and get exact numbers (defaults stay under $1)
modal run bench_one_video.py::benchmark
```

Do step 0 first. It is free and it catches plumbing bugs.

Super-resolution is **off by default** (`--resolution native`) because it measured at ~84% of post-processing time. Turn it on only when you want it:

```bash
modal run bench_one_video.py::download_upscaler          # 67 MB, once
modal run bench_one_video.py::benchmark --resolution 1440p --steps 24
```

---

## Cost

Modal bills `gpu="B200+"` at the **B200 rate of $6.25/hr** ($0.001736/s) even when it lands on a B300. So:

| Budget | GPU time |
|---|---|
| $0.50 | 4.8 min |
| **$1.00** | **9.6 min** |
| $2.00 | 19.2 min |

`--budget-usd` converts your budget into a hard container timeout, so an overrun gets killed rather than billed.

**The wildcard is the cold start.** Loading ~124 GB (61.7 GB transformer + 62.1 GB Qwen3-VL conditioner) happens before a single frame is generated, and nobody can predict it without measuring — that is what `probe_load` is for. A warm container within `scaledown_window` (300 s) skips it entirely, so **batching clips is the single biggest saving**.

`min_containers=0` is deliberate. Keeping one B200 warm around the clock is roughly **$150/day ≈ $4,500/month**.

---

## What it gets right

**60 fps that actually looks like 60 fps.** 24→60 is a 2.5x ratio, not an integer. The common shortcut — run a 2x interpolator twice, then drop frames — leaves the retained frames on an irregular time grid, which reads as judder. Instead every output frame is placed at its own absolute timestamp (`i/60` s) and synthesised there. Verified: gaps are uniformly `1/60` s, and the interpolation fractions form an exact 5-step cycle (0, .2, .4, .6, .8).

**H3's canvas is not 16:9.** For 16:9 the model's real canvas is **1344x768** = 1.75:1, because the naive multiple-of-32 answer (1376x768) breaks H3's 1,032,192-pixel cap. UHD is 1.7778:1. Scaling one onto the other stretches by 1.6%, which is visible on faces — so the pipeline centre-crops 12 rows instead and never stretches.

**Frame counts are constrained.** `num_frames` must be `17n + 5` and land in 5–15 s, giving exactly 14 legal values (124 … 345). Asking for 15 s clamps to 345 (14.375 s) rather than failing.

**Banding control.** A 768p source stretched to 1440p spreads every 8-bit step across ~2 output pixels, turning invisible gradients into visible contours. Mitigated with 16-bit intermediates, 10-bit output, `deband`, and a small amount of temporally-varying noise for dithering.

**Audio is the duration authority.** Frames are placed by absolute time so they cannot drift; `-shortest` trims the sub-frame tail. Measured A/V agreement: **0.7 ms**.

**Seam-free tiling.** Stepping by `tile - overlap` and clamping the last tile produces a larger overlap there than elsewhere; feathering with the requested width then leaves a seam at a fixed column that crawls through the clip. Tiles are distributed uniformly and the real overlap is used for the blend. Verified: a flat field stays flat to within 1e-5.

---

## Known limitations

**No native 2K.** H3's own 2K comes from `H3-Regenerate-2K`, which is **not open source** — API only. `--resolution 1440p` here is *our* upscale of 768p. It is not the same thing.

**No Context-IR.** Also closed-source and API-only, and the model card says it is critical to output quality. `prompt.py` reconstructs its documented output structure locally (`integrated_multimodal_description` / `overall_soundscape` / `non_diegetic_music`, `[Shot N]` blocks, timecodes, `<d>` dialogue tags), which is a large improvement over a bare one-liner but is not the real thing. Set `MINIMAX_API_KEY` to use the official endpoint.

**Interpolation quality.** The default `torch` interpolator places frames at mathematically correct timestamps but synthesises them by blending, so motion is softer than optical flow. Cadence is right; sharpness is not. RIFE v4.x (arbitrary-timestep) is the upgrade — v4.x specifically, because earlier versions only do the midpoint and cannot hit 0.2/0.4/0.6/0.8. Note that ffmpeg's `minterpolate` is **8-bit only** — every 10/16-bit format silently converts — so it undoes the banding work.

**Chained clips have audible audio seams.** H3 generates each clip's soundtrack independently. Chaining gives visual continuity, but ambience, level and musical key jump at every join. A crossfade softens it; it does not create a continuous score. This is the hardest unsolved problem here and it is not oversold anywhere in the code.

**Long-form drift.** Anchored continuation accumulates colour, contrast and identity drift. Claims of "zero drift" over dozens of chained clips are not credible.

---

## Hardware

One workflow in bf16 is ~124 GB.

| Setup | Works? | Notes |
|---|---|---|
| 1x B200 (180 GB) / B300 (288 GB) | yes | resident, no offload; `B200+` is billed as B200 |
| 1x 80 GB (A100/H100) | yes | CPU offload, slower |
| 24–32 GB consumer | yes | int8 + block streaming, needs **~75 GB system RAM** |
| < 24 GB or CPU only | no | there is no usable CPU path for a 33B video transformer |

`gpu="B200+"` can land on a B300, which **requires CUDA 13.1+** — the Modal image uses the cu130 torch index. Set `GIGGSDANCE_GPU=B200` if scheduling fails.

`ref2va` chaining needs the second transformer partition (+61.7 GB → ~186 GB) and so needs a B300; `fl2va` chaining uses one partition and fits a B200.

---

## Tests

```bash
python -m pytest tests/test_pipeline_math.py -v   # 36 tests, ~1s, no GPU
python tests/smoke_render.py                     # end-to-end, needs ffmpeg, ~8s
```

`smoke_render.py` renders a real file at 640x360 and asserts frame count, fps, bit depth, colour tags, A/V agreement and audio format. Keep it small — a full 4K smoke test takes ~13 minutes on CPU and proves nothing extra.

---

## Licence

Code in this repository: MIT (see [LICENSE](LICENSE)).
MiniMax H3 and its outputs: MiniMax H3 Community License Agreement (see [NOTICE.md](NOTICE.md)).

Powered by MiniMax H3.
