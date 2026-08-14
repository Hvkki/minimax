# Giggsdance

A 60 fps, high-resolution rendering pipeline on top of [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), built to run on Modal.

H3 generates **768p, 24 fps, 5–15 s clips with a jointly generated stereo soundtrack**. This repo turns that into something you can upload: correct 24→60 fps conversion, optional super-resolution to 1080p/1440p/2160p, banding control, 10-bit output, and exact A/V sync.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hvkki/minimax/blob/main/Giggsdance.ipynb)

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
| `run.py` — one-command orchestrator | wiring verified, generation untested |

The H3 generation call is written against the published diffusers integration and model card, but running it needs ~144 GB of weights and a large GPU, so it has never been executed. If something breaks, it is almost certainly there — run `modal run run.py --describe` to dump the pipeline's real signature and compare it against what we send.

---

## Budget warning

Modal's free tier is **$30/month, but only about $1 is available until a payment method is added**. Self-hosting H3 does not fit in $1: ~90 GB of weights must be downloaded and held, and ~144 GB loaded into GPU memory on every cold start *before a single frame exists*. The model load alone can eat most of a dollar.

With the full $30 unlocked, everything here is comfortable. On $1, use [MiniMax's hosted API](https://platform.minimax.io/) instead — no weights, no load, no storage, no territory restriction, and you get `H3-Context-IR` and native 2K, neither of which is open source. This repo's prompt builder, 60 fps conversion, upscaling and encoding all still work on top of API output.

## Quick start

```bash
git clone https://github.com/Hvkki/minimax.git && cd minimax
pip install -r requirements.txt
modal setup

modal run run.py
```

> **Modal runs local files.** It does not clone from GitHub and needs no GitHub integration. Pasting this repo's URL into the Modal dashboard returns `Bad Request: Unsupported URL`, and giving one to the CLI returns `Invalid object reference`. Clone first, then point `modal run` at the file — Modal uploads your local code itself. (You can run it from any directory; Modal puts the script's own folder on `sys.path`. Just keep `run.py` next to `giggsdance/`.)

That one command does everything: checks the Modal Volume for weights, downloads them if missing (~90 GB, on cheap CPU, skipped forever after), fetches the upscaler only if you asked for upscaling, renders a clip, writes `output.mp4` next to the file, and prints exactly what it cost. Every step is idempotent, so re-running is safe and skips whatever is already done.

Free sanity check first, if you like — no GPU, no weights, no Modal account:

```bash
python run.py --dry-run
```

Common variations:

```bash
modal run run.py --prompt "a paper boat drifting down a rain-filled gutter"
modal run run.py --duration-s 10 --steps 16
modal run run.py --resolution 1440p --steps 24 --budget-usd 3   # upscaling on
modal run run.py --probe-only                                   # measure cold start only
modal run run.py --describe                                     # dump pipeline signature
```

`run.py` deliberately exposes a **single** Modal entrypoint. Modal auto-runs one entrypoint but treats several as ambiguous, so extra entrypoints would make bare `modal run run.py` stop and ask which — the diagnostics are flags instead.

`bench_one_video.py` is the same pipeline with granular, separately invokable steps if you want to drive each stage yourself.

### Super-resolution is off by default

`--resolution native` runs **no** super-resolution model. It keeps H3's own pixels and only crops 1344x768 to 1344x756 for exact 16:9 (a crop is free, a scale is not). Measured effect:

| | with upscaler | native |
|---|---|---|
| pipeline time | 6.3s | **1.0s** |
| upscale share | 84.3% | **1.2%** |

Turn it on with `--resolution 1440p` when you want it; expect roughly 4x the post-processing time.

---


## Fastest path: Modal Notebooks (no token needed)

At [modal.com/notebooks](https://modal.com/notebooks) use **Import notebook → From URL** and paste:

```
https://github.com/Hvkki/minimax/blob/main/Giggsdance_Modal.ipynb
```

That dialog imports a **notebook**, so it needs a link to a `.ipynb` file. Pasting the repository root (`https://github.com/Hvkki/minimax`) is what produces `Bad Request: Unsupported URL`. If the `blob` link is rejected, use the raw form:

```
https://raw.githubusercontent.com/Hvkki/minimax/main/Giggsdance_Modal.ipynb
```

Inside a Modal notebook you are already authenticated, so **there is no token to paste anywhere**. Set the notebook's own hardware to **CPU** — the GPU work happens in separate Modal functions, so a GPU attached to the notebook would idle while still billing.

Then run the cells: a free dry run, a no-GPU check, the full GPU check, and the render.

Prefer a single cell in a blank notebook? This is equivalent:

```python
!git clone -q https://github.com/Hvkki/minimax.git /root/minimax && cd /root/minimax && pip install -q pytest && modal run doctor.py --skip-gpu
```

### One command, anywhere

```bash
modal run doctor.py
```

Five steps, then it stops — nothing rendered, nothing left running:

| Step | What it does |
|---|---|
| 1 | local environment: modal version, package, credentials |
| 2 | downloads ~90 GB of weights into a Volume, **skipped if already there** |
| 3 | runs the unit tests **inside the container**, not just on your machine |
| 4 | boots a B200, loads H3 (~144 GB), checks our call against the pipeline's real signature |
| 5 | prints a PASS/FAIL table, the spend, and the next command |

Step 4 is the valuable one: it catches a diffusers signature drift for the price of a model load instead of halfway through a paid render.

`modal run doctor.py --skip-gpu` does steps 1–3 only, so it costs almost nothing.

### Tokens

**None are needed for the weights.** `MiniMaxAI/MiniMax-H3` is a public, ungated repo (`gated: false`), so downloads are anonymous. The only credential is your Modal token, and inside Modal Notebooks even that is automatic.

If you want a Hugging Face token attached anyway, use a Modal Secret — never a literal in a cell:

```bash
modal secret create hf-token HF_TOKEN=hf_xxx
GIGGSDANCE_SECRETS=hf-token modal run doctor.py
```

Anything named in `GIGGSDANCE_SECRETS` is attached to every function and arrives as environment variables.

## In another notebook (Kaggle, Colab, Jupyter)

Ready-made notebooks — pick one:

| Platform | Notebook |
|---|---|
| **Kaggle** | [`Giggsdance_Kaggle.ipynb`](Giggsdance_Kaggle.ipynb) — File → Import Notebook, or upload it |
| **Colab** | [`Giggsdance.ipynb`](https://colab.research.google.com/github/Hvkki/minimax/blob/main/Giggsdance.ipynb) |

**On Kaggle, two settings matter before you run anything:** *Settings → Internet* must be **ON** (off by default, and nothing can reach Modal or PyPI without it), and *Settings → Accelerator* should be **None** — Kaggle's own GPU is not used, so attaching one only burns your quota.

Kaggle's 16 GB T4/P100, ~30 GB RAM and disk quota cannot hold H3's ~144 GB of weights; even the int8 path needs ~75 GB of host RAM. So Kaggle is the client and **Modal supplies the GPU**. Credentials come from *Add-ons → Secrets* (`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`) rather than being typed into a cell, and output is written to `/kaggle/working` so it shows up in the Output tab.

### By hand

[Open the notebook in Colab](https://colab.research.google.com/github/Hvkki/minimax/blob/main/Giggsdance.ipynb) and run the cells, or do it by hand:

```python
# Cell 1 -- the ! prefix matters. Without it the cell is parsed as
# Python and you get "SyntaxError: invalid syntax".
!git clone -q https://github.com/Hvkki/minimax.git
%cd minimax
!pip install -q modal numpy pillow

# Cell 2 -- credentials from modal.com/settings/tokens
import os
os.environ["MODAL_TOKEN_ID"] = "ak-..."
os.environ["MODAL_TOKEN_SECRET"] = "as-..."

# Cell 3 -- free, no GPU, no account
import notebook
notebook.dry_run()

# Cell 4 -- render, and preview it inline
path, report = notebook.render()
```

`modal run` is a shell command and `@app.local_entrypoint()` is CLI-only, so `notebook.py` drives Modal through its Python API inside an `app.run()` block instead. `notebook.check_setup()` reports exactly which of the three usual things is missing (package, modal, credentials).

Nothing runs on the notebook's own machine, so a free Colab runtime is enough — the GPU is Modal's.




## HTTP API — permanent HTTPS, no ngrok

```bash
modal secret create giggsdance-api API_KEY=$(openssl rand -hex 32)
modal run serve_api.py        # prefetch weights so the first call isn't a 144 GB download
modal deploy serve_api.py     # permanent URL
```

You get `https://<workspace>--giggsdance-api-fastapi-app.modal.run`.

**ngrok is not needed and would not help.** ngrok exposes a server on *your own machine*; nothing here runs there. Modal serves this over public HTTPS with a real certificate. If you ever want a raw TCP tunnel — to reach SGLang's own port, say — Modal has `modal.forward(port)` built in at no extra charge.

| Endpoint | |
|---|---|
| `GET /health` | no auth, readiness |
| `GET /` | capabilities and real limits |
| `POST /uploads` | multipart → a URI usable as a reference |
| `POST /render` | → `{ job_id }` (202) |
| `GET /jobs/{id}` | status, stage timings, measured cost |
| `GET /jobs/{id}/file` | the mp4 |
| `GET /jobs` · `DELETE /jobs/{id}` | list · cancel/delete |

All but `/health` need `X-API-Key`, compared with `compare_digest`.

```bash
URL=https://<workspace>--giggsdance-api-fastapi-app.modal.run
KEY=<your key>

JOB=$(curl -sS -X POST $URL/render -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' -d '{
    "prompt": "a lighthouse at dusk, beam sweeping through sea mist",
    "duration_s": 5, "resolution": "native", "fps": 60
  }' | jq -r .job_id)

curl -sS $URL/jobs/$JOB -H "X-API-Key: $KEY" | jq
curl -sS $URL/jobs/$JOB/file -H "X-API-Key: $KEY" -o out.mp4
```

With references (order is semantic — it sets the `<Picture N>`/`<Video N>`/`<Audio N>` tags):

```jsonc
{
  "prompt": "...",
  "gpus": 4,                     // lossless ~5x
  "quality": "lossless",         // or "high": 1.40x, SSIM 0.931
  "lora": "none",                // or "lightx2v": 4 evals, a different model
  "references": [
    {"type": "image", "uri": "https://.../hero.png"},
    {"type": "video", "uri": "https://.../motion.mp4"},
    {"type": "audio", "uri": "https://.../voice.wav"}
  ]
}
```

### Design notes

**Async by necessity.** A 15 s Ref2VA clip measured 784 s on 4× B300. No HTTP request survives that, so it's submit → poll → download.

**The web container has no torch.** FastAPI runs on CPU and `spawn()`s the GPU. Serving JSON from the B300 would bill $7.10/hr to answer HTTP; the API image omits torch entirely so it cold-starts in seconds.

**Validation happens on CPU.** An illegal reference set — 13 files, 10 images, audio alone — is rejected with 422 *before* a GPU is spawned, so a bad request costs nothing.

**`max_containers=2`** caps simultaneous GPU spend. H3 does one generation per diffusion batch, so requests queue rather than fanning out.

**Results expire after 48 h.** A scheduled `prune` deletes old files — a 15 s 1440p clip is >100 MB, and Volume storage otherwise becomes the biggest line on the bill.

## Backends: SGLang is the fast path

| | diffusers (`run.py`) | **SGLang (`serve.py`)** | vLLM-Omni |
|---|---|---|---|
| Speed | baseline | **1.41x** | similar |
| 12 references | via package API | **yes** | **no** — 1 image + 1 audio only |
| Turbo LoRAs | manual | **documented recipes** | — |
| Quality edge | no H.264 round-trip | one extra decode | — |

```bash
modal run serve.py                                   # 5 s, lossless
modal run serve.py --refs "a.png,b.png,clip.mp4,v.wav"   # up to 12 references
modal run serve.py --gpus 4                          # lossless, ~5x faster
modal run serve.py --lora lightx2v                   # ~10x fewer evaluations
```

vLLM-Omni's documented reference limit is **one image plus one audio**, or videos with no separate audio. If you need the 12-file omni-reference set, SGLang is the only option.

### Measured latency — SGLang's own 8x B300 sweep

1344x768, 124 frames (5.17 s), 50 steps:

| Weights | Precision | Latency | Load | Peak/GPU |
|---|---|---|---|---|
| FL2VA | BF16 | **19.04 s** | 118 s | 83.6 GB |
| FL2VA | FP8 | 18.03 s | 116 s | 51.9 GB |
| Ref2VA | BF16 | **29.12 s** | 114 s | 84.0 GB |

**Cold start is ~114–124 s.** References cost **1.53x**. Scaling 8→1 GPU is ~5.2–5.8x (from AMD's published GPU-count sweep), so a single B300 is roughly **100–110 s at 50 steps, ~17–20 s at 8**.

### Quantization: don't

FP8 on a B300 measured **19.04 → 18.03 s — 5%** — while cutting memory 38%. It's a *capacity* tool for cards that can't hold the weights, and it is **not bit-exact**. With 288 GB you don't need it.

**Lossless, and on by default:**

| | Effect |
|---|---|
| `--performance-mode speed` | resident components, eager BF16/FP32 |
| pure Ulysses (`--gpus N`) | faster *and* the capacity default |
| VAE patch parallelism, tile | cuts decode **3.4–3.5x** |
| warmup matched to resolution | removes ~10 s of first-request cost |

**Not lossless, opt-in:**

| | Cost |
|---|---|
| `--quality high` | 1.40x, SSIM 0.931 / PSNR 28.16 dB |
| `--lora lightx2v` | 4 evals vs 50 — a *different model*, not the same clip faster |
| `--quantize-fp8` | not bit-exact, ~5% |
| torch.compile | changes numerical output — never enabled here |

**For no quality loss, speed comes from GPU count, not precision.** `--gpus 4` is lossless and roughly 5x.

## Multimodal references (12 files)

`ref2va` conditions on an **ordered** mix of media:

```bash
modal run run.py --refs "hero.png,pose.png,style.png,motion.mp4,voice.wav"
modal run run.py --images "a.png,b.png" --videos "clip.mp4" --audios "voice.wav"
```

| Input | Limit |
|---|---|
| Images | ≤ 9 |
| Videos | ≤ 3, each 2–15 s, total ≤ 15 s |
| Audio | ≤ 3, each 2–15 s, **never alone** |
| Total | ≤ 12 files |

Every limit is checked **locally before a GPU boots**, so a bad set costs nothing.

**Order is semantic.** It sets the `<Picture N>` / `<Video N>` / `<Audio N>` labels and each reference's place on the shared rotary clock — reordering the same files is a different request, not a cosmetic change.

One or two bare images stay on the cheaper `fl2va` path (they're keyframes, not references). Three or more, or any video/audio, promotes to `ref2va` and its separate 66.3 GB partition.

### Real weight sizes

Measured from the Hugging Face manifest, not from documentation — the docs' "61.7 / 62.1 GB" figures are low:

| Component | GB |
|---|---|
| `transformer/` (t2va, fl2va) | 66.28 |
| `transformer_ref/` (ref2va) | 66.28 |
| `text_encoder/` (Qwen3-VL) | 66.73 |
| `vae/` | 10.42 |
| `audio_vae/` + tokenizer | 0.63 |
| **One workflow** | **144.06** |
| **Both partitions** | **210.34** |

That's why B300 is the default: 210 GB does not fit a B200's 180 GB, and even one workflow leaves a B200 only ~36 GB for activations.

## Cost

Modal bills `gpu="B200+"` at the **B200 rate of $6.25/hr** ($0.001736/s) even when it lands on a B300. So:

| Budget | GPU time |
|---|---|
| $0.50 | 4.8 min |
| **$1.00** | **9.6 min** |
| $2.00 | 19.2 min |

`--budget-usd` converts your budget into a hard container timeout, so an overrun gets killed rather than billed. Default is $1.00.

**The wildcard is the cold start.** Loading ~144 GB happens before a single frame is generated, and nobody can predict it without measuring — that is what `probe_load` is for. A warm container within `scaledown_window` (300 s) skips it entirely, so **batching clips is the single biggest saving**.

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

One workflow in bf16 is ~144 GB.

| Setup | Works? | Notes |
|---|---|---|
| 1x B300 (288 GB) | yes | **recommended** — 144 GB spare, and the only card that fits both partitions |
| 1x B200 (180 GB) | tight | 144 GB of weights leaves ~36 GB for activations |
| 1x 80 GB (A100/H100) | yes | CPU offload, slower |
| 24–32 GB consumer | yes | int8 + block streaming, needs **~75 GB system RAM** |
| < 24 GB or CPU only | no | there is no usable CPU path for a 33B video transformer |

`gpu="B200+"` can land on a B300, which **requires CUDA 13.1+** — the Modal image uses the cu130 torch index. Set `GIGGSDANCE_GPU=B200` if scheduling fails.

`ref2va` chaining needs the second transformer partition (+66.3 GB → ~210 GB) and so needs a B300; `fl2va` chaining uses one partition and fits a B200.

---

## Tests

```bash
python -m pytest tests/test_pipeline_math.py -v   # 42 tests, ~1s, no GPU
python tests/smoke_render.py                     # end-to-end, needs ffmpeg, ~8s
python run.py --dry-run                          # full pipeline, synthetic frames
```

`smoke_render.py` renders a real file at 640x360 and asserts frame count, fps, bit depth, colour tags, A/V agreement and audio format. Keep it small — a full 4K smoke test takes ~13 minutes on CPU and proves nothing extra.

---

## Licence

Code in this repository: MIT (see [LICENSE](LICENSE)).
MiniMax H3 and its outputs: MiniMax H3 Community License Agreement (see [NOTICE.md](NOTICE.md)).

Powered by MiniMax H3.
