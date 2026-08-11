#!/usr/bin/env python3
"""
One command that sets everything up, tests it, and tells you where you stand.

    modal run doctor.py

It does five things and then stops:

  1. reports the local environment (modal version, package, credentials)
  2. downloads the H3 weights into a Modal Volume, if they are not there yet
  3. runs the unit tests inside the Modal container, not just on your laptop
  4. boots a GPU, loads H3, and checks our generation call against the
     pipeline's real signature
  5. prints what worked, what did not, what it cost, and the next command

Nothing is rendered. Nothing is left running.

--------------------------------------------------------------------------------
TOKENS
--------------------------------------------------------------------------------
You need none for the weights: MiniMaxAI/MiniMax-H3 is a public, ungated
repository, so downloads are anonymous.

The only credential in play is your Modal token, and there are two ways to avoid
handling it at all:

  * Run this from a notebook at modal.com/notebooks. You are already inside
    Modal there, so authentication is automatic -- nothing to paste.
  * Or set MODAL_TOKEN_ID / MODAL_TOKEN_SECRET from modal.com/settings/tokens.

If you do want a Hugging Face token attached anyway (higher rate limits, or a
gated mirror), put it in a Modal Secret and name it:

    modal secret create hf-token HF_TOKEN=hf_xxx
    GIGGSDANCE_SECRETS=hf-token modal run doctor.py

Any secret named in GIGGSDANCE_SECRETS is attached to every function here, and
its keys arrive as environment variables. Nothing is read from your shell
history or written to disk.

--------------------------------------------------------------------------------
NOTE ON REPOSITORY URLS
--------------------------------------------------------------------------------
Modal has no "run this GitHub repo" feature. Pasting a repository URL into the
dashboard returns "Bad Request: Unsupported URL", and handing one to the CLI
returns "Invalid object reference". Modal executes code from your local
filesystem, or from a Modal Notebook. See the README for the one-cell version.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import modal

MODEL_ID = "MiniMaxAI/MiniMax-H3"
WEIGHTS_DIR = "/weights"
MODEL_PATH = f"{WEIGHTS_DIR}/MiniMax-H3"

USD_PER_SECOND = 6.25 / 3600.0  # Modal B200; gpu="B200+" is billed at this rate

REQUIRED_PARTS = ("transformer", "text_encoder", "vae", "audio_vae", "tokenizer")

# The keyword arguments we actually send when generating. Step 4 compares these
# against the pipeline's declared inputs so a signature drift is caught here,
# cheaply, instead of halfway through a paid render.
GENERATION_KWARGS = (
    "prompt", "num_frames", "height", "width",
    "num_inference_steps", "generator", "output", "output_type",
)

# Optional: any Modal Secret named in GIGGSDANCE_SECRETS gets attached.
_SECRET_NAMES = [s.strip() for s in os.environ.get("GIGGSDANCE_SECRETS", "").split(",") if s.strip()]
SECRETS = [modal.Secret.from_name(name) for name in _SECRET_NAMES]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git")
    .pip_install("torch", "torchvision",
                 extra_index_url="https://download.pytorch.org/whl/cu130")
    .pip_install(
        "git+https://github.com/huggingface/diffusers.git",
        "transformers>=4.57.0", "accelerate", "safetensors",
        "huggingface_hub[hf_transfer]", "pillow", "numpy",
        "sentencepiece", "protobuf", "av", "pytest",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": f"{WEIGHTS_DIR}/hf"})
    .add_local_python_source("giggsdance")
    .add_local_dir("tests", remote_path="/root/tests")
)

app = modal.App("giggsdance-doctor")
weights_volume = modal.Volume.from_name("h3-weights", create_if_missing=True)


# ==========================================================================
# Step 2: weights
# ==========================================================================

@app.function(
    image=image, volumes={WEIGHTS_DIR: weights_volume}, secrets=SECRETS,
    timeout=6 * 60 * 60, cpu=8.0, memory=16384,
)
def fetch_weights(force: bool = False) -> dict:
    """Download H3's t2va/fl2va components if the Volume lacks them."""
    from huggingface_hub import snapshot_download

    root = Path(MODEL_PATH)

    def survey() -> dict[str, int]:
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
        return {"downloaded": False, "gb": sum(present.values()) / 1e9,
                "seconds": 0.0, "parts": sorted(present)}

    missing = [p for p in REQUIRED_PARTS if p not in present]
    print(f"downloading (missing: {missing or 'everything'}) -- about 90 GB, one time")

    started = time.time()
    snapshot_download(
        MODEL_ID, local_dir=MODEL_PATH, max_workers=16,
        allow_patterns=[
            "modular_model_index.json", "model_index.json", "*.json", "*.txt",
            "transformer/*", "text_encoder/*", "tokenizer/*", "processor/*",
            "vae/*", "audio_vae/*", "scheduler/*", "audio_scheduler/*",
        ],
    )
    weights_volume.commit()
    elapsed = time.time() - started

    final = survey()
    for part, size in sorted(final.items()):
        print(f"  {part:16} {size / 1e9:7.2f} GB")
    still_missing = [p for p in REQUIRED_PARTS if p not in final]
    return {
        "downloaded": True, "gb": sum(final.values()) / 1e9, "seconds": elapsed,
        "parts": sorted(final), "missing": still_missing,
    }


# ==========================================================================
# Step 3: tests, executed in the container
# ==========================================================================

@app.function(image=image, secrets=SECRETS, timeout=900)
def run_tests() -> dict:
    """Run the unit tests where the code will actually run.

    Passing locally is not the same as passing in the container: different
    Python, different numpy, different Pillow.
    """
    import subprocess

    result = subprocess.run(
        ["python", "-m", "pytest", "/root/tests/test_pipeline_math.py", "-q", "--no-header"],
        capture_output=True, text=True,
    )
    tail = (result.stdout + result.stderr).strip().splitlines()[-12:]
    return {"passed": result.returncode == 0, "output": "\n".join(tail)}


# ==========================================================================
# Step 4: GPU + H3 load + signature check
# ==========================================================================

@app.cls(
    image=image, gpu="B200+", volumes={WEIGHTS_DIR: weights_volume},
    secrets=SECRETS, timeout=45 * 60, memory=180 * 1024,
    min_containers=0, scaledown_window=60,
)
class Probe:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import ComponentsManager, ModularPipeline

        self.torch = torch
        self.error = None
        self.gpu_name = torch.cuda.get_device_name(0)
        self.gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        self.torch_version = torch.__version__
        self.cuda_version = torch.version.cuda

        started = time.perf_counter()
        try:
            source = MODEL_PATH if Path(MODEL_PATH).exists() else MODEL_ID
            self.manager = ComponentsManager()
            self.pipe = ModularPipeline.from_pretrained(
                source, components_manager=self.manager
            )
            self.pipe.load_components(workflow="t2va", dtype=torch.bfloat16)
            margin = "40GB" if self.gpu_vram >= 150 else "12GB"
            self.manager.enable_auto_cpu_offload(
                device="cuda", memory_reserve_margin=margin
            )
            self.loaded = True
        except Exception as exc:
            self.pipe = None
            self.loaded = False
            self.error = f"{type(exc).__name__}: {exc}"
        self.load_seconds = time.perf_counter() - started

    @modal.method()
    def inspect(self) -> dict:
        """Report the GPU, the load, and whether our call signature still fits."""
        report: dict = {
            "gpu": self.gpu_name,
            "vram_gb": round(self.gpu_vram, 1),
            "torch": self.torch_version,
            "cuda": self.cuda_version,
            "load_seconds": self.load_seconds,
            "loaded": self.loaded,
            "error": self.error,
        }
        if not self.loaded:
            return report

        doc = str(getattr(self.pipe, "doc", ""))
        report["doc_excerpt"] = doc[:1500]
        report["unrecognised_kwargs"] = [
            name for name in GENERATION_KWARGS if name not in doc
        ]

        try:
            allocated = self.torch.cuda.memory_allocated() / 1024**3
            report["vram_used_gb"] = round(allocated, 1)
            report["vram_headroom_gb"] = round(self.gpu_vram - allocated, 1)
        except Exception:
            pass
        return report


# ==========================================================================
# The one command
# ==========================================================================

@app.local_entrypoint()
def main(force_download: bool = False, skip_gpu: bool = False):
    """Set up, test, report, exit."""
    line = "=" * 74
    print(f"{line}\nGIGGSDANCE DOCTOR\n{line}")
    print("Powered by MiniMax H3. The licence excludes the EU, UK, South Korea")
    print("and USA, and covers the model's outputs as well as its weights.")
    print("See NOTICE.md. Mark anything you publish as AI-generated.\n")

    findings: list[tuple[str, bool, str]] = []
    spend = 0.0

    # -- 1. local -----------------------------------------------------
    print("[1/5] local environment")
    try:
        import giggsdance

        print(f"      giggsdance {giggsdance.__version__}")
        findings.append(("giggsdance package importable", True, giggsdance.__version__))
    except ModuleNotFoundError as exc:
        findings.append(("giggsdance package importable", False, str(exc)))
        print(f"      MISSING: {exc}")
    print(f"      modal {modal.__version__}")
    if SECRETS:
        print(f"      secrets attached: {', '.join(_SECRET_NAMES)}")
    else:
        print("      secrets: none needed (H3 is a public, ungated repo)")
    findings.append(("modal installed", True, modal.__version__))

    # -- 2. weights ---------------------------------------------------
    print("\n[2/5] weights in Modal Volume 'h3-weights'")
    started = time.time()
    weights = fetch_weights.remote(force=force_download)
    if weights["downloaded"]:
        print(f"      downloaded {weights['gb']:.1f} GB in "
              f"{weights['seconds'] / 60:.1f} min")
    else:
        print(f"      already present: {weights['gb']:.1f} GB -- skipped")
    ok = not weights.get("missing")
    findings.append((
        "H3 weights complete", ok,
        f"{weights['gb']:.1f} GB, {len(weights['parts'])}/{len(REQUIRED_PARTS)} components",
    ))
    if not ok:
        print(f"      INCOMPLETE, missing: {weights['missing']}")

    # -- 3. tests -----------------------------------------------------
    print("\n[3/5] unit tests, inside the Modal container")
    tests = run_tests.remote()
    for row in tests["output"].splitlines()[-4:]:
        print(f"      {row}")
    findings.append(("unit tests pass in container", tests["passed"],
                     tests["output"].splitlines()[-1] if tests["output"] else ""))

    # -- 4. GPU -------------------------------------------------------
    if skip_gpu:
        print("\n[4/5] GPU probe skipped (--skip-gpu)")
        findings.append(("GPU + H3 load", None, "skipped"))
        probe = None
    else:
        print("\n[4/5] GPU probe: boot B200+, load H3 (~124 GB), check signature")
        print("      this is the slow part; nothing is rendered")
        probe = Probe().inspect.remote()
        print(f"      {probe['gpu']} ({probe['vram_gb']} GB), "
              f"torch {probe['torch']}, CUDA {probe['cuda']}")
        if probe["loaded"]:
            print(f"      H3 loaded in {probe['load_seconds']:.0f}s "
                  f"(${probe['load_seconds'] * USD_PER_SECOND:.4f})")
            if "vram_used_gb" in probe:
                print(f"      VRAM {probe['vram_used_gb']} GB used, "
                      f"{probe['vram_headroom_gb']} GB free")
            findings.append(("H3 loads on GPU", True,
                             f"{probe['load_seconds']:.0f}s"))
            unknown = probe.get("unrecognised_kwargs") or []
            findings.append((
                "generation signature matches", not unknown,
                "all arguments recognised" if not unknown
                else f"not found in pipe.doc: {unknown}",
            ))
            if unknown:
                print(f"      WARNING: {unknown} absent from the pipeline's "
                      f"declared inputs")
                print("      run `modal run run.py --describe` for the full signature")
        else:
            print(f"      FAILED: {probe['error']}")
            findings.append(("H3 loads on GPU", False, probe["error"] or "unknown"))
        spend += probe["load_seconds"] * USD_PER_SECOND

    elapsed = time.time() - started

    # -- 5. report ----------------------------------------------------
    print(f"\n{line}\nRESULT\n{line}")
    width = max(len(name) for name, _, _ in findings)
    failed = 0
    for name, state, detail in findings:
        mark = "PASS" if state else ("SKIP" if state is None else "FAIL")
        if state is False:
            failed += 1
        print(f"  [{mark}] {name:<{width}}  {detail}")

    print(f"\nwall clock {elapsed / 60:.1f} min | GPU spend about ${spend:.2f}")
    print("(the weights download runs on CPU and is charged separately, "
          "roughly $0.50 once)")

    if failed:
        print(f"\n{failed} check(s) failed -- fix those before rendering.")
        if probe and not probe.get("loaded"):
            print("If loading failed on memory, try gpu=\"B300\" (288 GB) in run.py.")
        return

    print("\nEverything needed is in place. Next:")
    print("    modal run run.py                       # render, defaults under $1")
    print("    modal run run.py --resolution 1440p    # with super-resolution")
    print("\nA warm container skips the model load, so batching clips is the")
    print("single biggest saving available.")
