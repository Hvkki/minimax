"""Fast end-to-end smoke test: correctness, not throughput.

Runs the real interpolation -> geometry -> encode path at a deliberately tiny
output size so it finishes in seconds. Verifies the properties that are easy to
get silently wrong: frame count, duration, A/V agreement, bit depth, colour
tags and frame rate.

Run: python3 tests/smoke_render.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from giggsdance.stages.encode import (  # noqa: E402
    EncodeSettings, encode_frames, probe, write_wav,
)
from giggsdance.stages.interpolate import plan_interpolation  # noqa: E402
from giggsdance.stages.upscale import (  # noqa: E402
    LanczosUpscaler, plan_geometry, process_frame,
)

OUT = Path("/tmp/gd_smoke")
SRC_W, SRC_H, SRC_N = 224, 128, 48   # stand-in canvas, same 1.75 aspect as H3
TARGET = (640, 360)                  # tiny 16:9 target so this is quick
AUDIO_SR = 32000


def synth_frame(i: int, n: int) -> np.ndarray:
    """Shallow gradient (banding bait) plus a moving bar (motion for the flow)."""
    x = np.linspace(0, 1, SRC_W, dtype=np.float32)[None, :]
    y = np.linspace(0, 1, SRC_H, dtype=np.float32)[:, None]
    g = 0.15 + 0.25 * y + 0.02 * x
    frame = np.stack([g, g * 0.95, g * 1.05], axis=-1)
    cx = int((i / max(1, n)) * SRC_W)
    frame[:, max(0, cx - 4):cx + 4, :] = 0.85
    return np.clip(frame, 0.0, 1.0)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    plan = plan_interpolation(SRC_N, 24.0, 60.0)
    geo = plan_geometry(SRC_W, SRC_H, TARGET[0], TARGET[1], model_scale=4)
    upscaler = LanczosUpscaler(4)
    source = [synth_frame(i, SRC_N) for i in range(SRC_N)]

    samples = int(round(plan.src_duration_s * AUDIO_SR))
    t = np.arange(samples) / AUDIO_SR
    audio = np.stack([
        0.2 * np.sin(2 * np.pi * 440 * t),
        0.2 * np.sin(2 * np.pi * 554 * t),
    ]).astype(np.float32)
    wav = write_wav(OUT / "audio.wav", audio, AUDIO_SR)

    def frames():
        for timing in plan.timings:
            left = source[timing.left]
            if timing.is_copy:
                frame = left
            else:
                right = source[min(timing.left + 1, SRC_N - 1)]
                frame = left * (1.0 - timing.t) + right * timing.t
            yield process_frame(frame, geo, upscaler)

    settings = EncodeSettings(codec="h265", crf=20, preset="ultrafast", fps=60.0)
    out_path = OUT / "smoke.mp4"
    encode_frames(frames(), geo.out_width, geo.out_height, out_path, settings, wav)
    elapsed = time.time() - started

    info = probe(out_path)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    audio_stream = next(s for s in info["streams"] if s["codec_type"] == "audio")

    v_dur = float(video.get("duration") or info["format"]["duration"])
    a_dur = float(audio_stream.get("duration") or info["format"]["duration"])
    nb_frames = int(video.get("nb_frames") or 0)

    checks = [
        ("output exists", out_path.exists() and out_path.stat().st_size > 0, out_path.stat().st_size),
        ("resolution", (video["width"], video["height"]) == TARGET, f"{video['width']}x{video['height']}"),
        ("frame rate is 60", video["r_frame_rate"] in ("60/1", "60000/1000"), video["r_frame_rate"]),
        ("10-bit pixel format", video["pix_fmt"] == "yuv420p10le", video["pix_fmt"]),
        ("bt709 primaries tagged", video.get("color_primaries") == "bt709", video.get("color_primaries")),
        ("bt709 transfer tagged", video.get("color_transfer") == "bt709", video.get("color_transfer")),
        ("bt709 matrix tagged", video.get("color_space") == "bt709", video.get("color_space")),
        ("frame count matches plan", abs(nb_frames - plan.num_dst_frames) <= 1,
         f"{nb_frames} vs planned {plan.num_dst_frames}"),
        ("A/V duration agree <20ms", abs(v_dur - a_dur) < 0.020, f"v={v_dur:.4f}s a={a_dur:.4f}s"),
        ("duration matches source <25ms", abs(v_dur - plan.src_duration_s) < 0.025,
         f"{v_dur:.4f}s vs src {plan.src_duration_s:.4f}s"),
        ("audio is stereo", int(audio_stream["channels"]) == 2, audio_stream["channels"]),
        ("audio at 48kHz", int(audio_stream["sample_rate"]) == 48000, audio_stream["sample_rate"]),
    ]

    print(f"\nrendered {plan.num_dst_frames} frames at {TARGET[0]}x{TARGET[1]} "
          f"in {elapsed:.1f}s ({elapsed / plan.num_dst_frames * 1000:.0f} ms/frame)\n")
    failed = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:32} {detail}")
        if not ok:
            failed += 1

    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
