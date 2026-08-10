"""Notebook entry point -- for Jupyter, Colab, Kaggle, VS Code notebooks.

`modal run` is a *shell* command, so it cannot be typed into a notebook cell:
the cell is parsed as Python and you get `SyntaxError: invalid syntax`. And
Modal's `@app.local_entrypoint()` only exists for the CLI. From a notebook you
drive Modal through its Python API instead, inside an `app.run()` block, which
is what this module does for you.

Usage in a notebook -- two cells:

    # Cell 1: setup (once per session)
    !git clone https://github.com/Hvkki/minimax.git
    %cd minimax
    !pip install -q modal numpy pillow
    import os
    os.environ["MODAL_TOKEN_ID"] = "..."      # from modal.com/settings/tokens
    os.environ["MODAL_TOKEN_SECRET"] = "..."

    # Cell 2: render
    import notebook
    notebook.render()

`notebook.render()` does everything `modal run run.py` does -- fetches weights if
missing, renders, saves output.mp4, prints the measured cost -- and additionally
displays the video inline when IPython is available.

Free, no GPU and no Modal account needed, to check the pipeline first:

    import notebook
    notebook.dry_run()

Powered by MiniMax H3. Read NOTICE.md: the licence excludes the EU, UK, South
Korea and USA, and the restriction covers the model's outputs too. Mark anything
you publish as AI-generated.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

__all__ = ["check_setup", "dry_run", "render", "probe", "describe"]


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ != "TerminalInteractiveShell"
    except Exception:
        return False


def check_setup(verbose: bool = True) -> dict:
    """Report what is and is not ready, without raising.

    Checks the three things that actually go wrong in a notebook: the package is
    not importable (cloned wrongly, or wrong working directory), modal is not
    installed, and Modal credentials are absent.
    """
    status: dict[str, object] = {}

    try:
        import giggsdance  # noqa: F401

        status["package"] = True
        status["package_version"] = giggsdance.__version__
    except ModuleNotFoundError:
        status["package"] = False

    try:
        import modal  # noqa: F401

        status["modal"] = True
        status["modal_version"] = modal.__version__
    except ModuleNotFoundError:
        status["modal"] = False

    has_env = bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"))
    has_file = (Path.home() / ".modal.toml").exists()
    status["credentials"] = has_env or has_file
    status["credentials_source"] = (
        "environment" if has_env else "~/.modal.toml" if has_file else None
    )
    status["ready"] = bool(status["package"] and status["modal"] and status["credentials"])

    if verbose:
        def mark(ok):
            return "OK  " if ok else "MISSING"

        print(f"[{mark(status['package'])}] giggsdance package"
              f"{'  v' + str(status.get('package_version')) if status['package'] else ''}")
        if not status["package"]:
            print("        cd into the cloned repo:  %cd minimax")
        print(f"[{mark(status['modal'])}] modal"
              f"{'  v' + str(status.get('modal_version')) if status['modal'] else ''}")
        if not status["modal"]:
            print("        !pip install -q modal")
        print(f"[{mark(status['credentials'])}] Modal credentials"
              f"{'  (' + str(status['credentials_source']) + ')' if status['credentials'] else ''}")
        if not status["credentials"]:
            print("        get a token pair at https://modal.com/settings/tokens then:")
            print('        os.environ["MODAL_TOKEN_ID"] = "ak-..."')
            print('        os.environ["MODAL_TOKEN_SECRET"] = "as-..."')
        print()
        print("ready" if status["ready"] else "not ready -- fix the above first")
    return status


def dry_run(resolution: str = "native", fps: int = 60, duration_s: float = 5.0):
    """Validate the whole pipeline locally. No GPU, no weights, no account, $0.

    Needs numpy, pillow and ffmpeg. On Colab ffmpeg is already installed.
    """
    from bench_one_video import dry_run as _dry_run

    code = _dry_run(resolution=resolution, fps=fps, duration_s=duration_s)
    if code == 0:
        print("\npipeline is sound -- the GPU run differs only in generation "
              "and the real upscaler")
    return code


def _display(path: Path) -> None:
    """Show the clip inline, falling back quietly outside a notebook."""
    if not _in_notebook():
        return
    try:
        import base64

        from IPython.display import HTML, display

        size_mb = path.stat().st_size / 1e6
        if size_mb > 40:
            print(f"({path.name} is {size_mb:.0f} MB -- too large to embed; "
                  f"download it from the file browser)")
            return
        encoded = base64.b64encode(path.read_bytes()).decode()
        display(HTML(
            f'<video controls style="max-width:100%" '
            f'src="data:video/mp4;base64,{encoded}"></video>'
        ))
    except Exception as exc:  # pragma: no cover
        print(f"(could not embed preview: {exc})")


def render(
    prompt: str | None = None,
    duration_s: float = 5.0,
    resolution: str = "native",
    fps: int = 60,
    steps: int = 8,
    seed: int = 0,
    aspect_ratio: str = "16:9",
    budget_usd: float = 1.00,
    out: str = "output.mp4",
    force_download: bool = False,
    show: bool = True,
):
    """Render one clip on Modal from a notebook. Returns (Path, report dict).

    Equivalent to `modal run run.py`, but driven through Modal's Python API so it
    works in a cell. Idempotent: weights are downloaded only if the Volume does
    not already have them.
    """
    status = check_setup(verbose=False)
    if not status["ready"]:
        print("setup incomplete:\n")
        check_setup(verbose=True)
        raise SystemExit("fix setup above, then call notebook.render() again")

    import run as pipeline

    prompt = prompt or pipeline.DEFAULT_PROMPT
    rate = pipeline.USD_PER_SECOND
    max_seconds = int(budget_usd / rate)

    canvas = pipeline.resolve_canvas(aspect_ratio)
    num_frames = pipeline.frames_for_duration(duration_s)
    out_w, out_h = pipeline.resolve_target(resolution, canvas.width, canvas.height)
    crop_h = pipeline.plan_geometry(canvas.width, canvas.height, out_w, out_h, 2).crop_height
    scale = pipeline.pick_scale(crop_h, out_h)
    planned = pipeline.plan_interpolation(
        num_frames, pipeline.SRC_FPS, float(fps)
    ).num_dst_frames

    print(f"{num_frames} frames @24fps ({num_frames / 24:.3f}s) at {canvas}")
    print(f"  -> {planned} frames @{fps}fps at {out_w}x{out_h}")
    print(f"  -> {'no super-resolution' if scale <= 1 else f'{scale}x super-resolution'}"
          f", {steps} steps")
    print(f"budget ${budget_usd:.2f} = {max_seconds}s of B200 time, "
          f"enforced as a hard timeout\n")

    started = time.time()
    with pipeline.app.run():
        print("[1/3] weights")
        info = pipeline.ensure_weights.remote(force=force_download)
        if info["downloaded"]:
            print(f"      fetched {info['gb']:.1f} GB in {info['seconds'] / 60:.1f} min")

        print("[2/3] upscaler")
        if scale <= 1:
            print("      not needed at this resolution")
        else:
            print(f"      {pipeline.ensure_upscaler.remote(scale).get('name')} ready")

        print("[3/3] render")
        renderer = pipeline.Renderer.with_options(timeout=max_seconds)()
        try:
            result = renderer.render.remote(
                prompt=prompt, duration_s=duration_s, aspect_ratio=aspect_ratio,
                resolution=resolution, fps=fps, steps=steps, seed=seed,
            )
        except Exception as exc:
            spent = time.time() - started
            print(f"\nfailed after {spent:.0f}s (at most ${spent * rate:.2f}): {exc}")
            print("timeout? raise budget_usd or lower steps. "
                  "generation error? call notebook.describe()")
            raise

    path = Path(out)
    path.write_bytes(result.pop("video"))
    probe_data = result.pop("probe")
    pipeline._report(result, probe_data, time.time() - started, out, budget_usd)

    if show:
        _display(path)
    return path, result


def probe():
    """Measure only the cold start and model load. Returns seconds."""
    import run as pipeline

    with pipeline.app.run():
        seconds = pipeline.Renderer().report_load.remote()
    cost = seconds * pipeline.USD_PER_SECOND
    print(f"model load: {seconds:.1f}s = ${cost:.4f}")
    print(f"a $1.00 budget leaves {int(1.0 / pipeline.USD_PER_SECOND - seconds)}s "
          f"to render in")
    return seconds


def describe():
    """Print the diffusers pipeline signature. Use this if generation fails."""
    import run as pipeline

    with pipeline.app.run():
        text = pipeline.Renderer().describe.remote()
    print(text)
    return text
