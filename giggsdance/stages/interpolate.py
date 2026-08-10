"""24 fps -> 60 fps interpolation.

Why this module exists at all
----------------------------
24 -> 60 is a 2.5x ratio, which is *not* an integer. The tempting shortcut is
to run a 2x interpolator twice (24 -> 48 -> 96) and then drop frames down to
60. That produces a video whose frames are not evenly spaced in time: the
retained frames land on an irregular grid, which reads as judder even though
the container claims 60 fps. It is the single most common way a "60fps"
pipeline ends up looking worse than the 24 fps source.

The correct approach is to place every output frame at its own absolute
timestamp and synthesise exactly that moment:

    output frame i sits at  t_i = i / 60  seconds
    in source-frame units:  s   = t_i * 24 = i * 0.4
    so it interpolates between source frames floor(s) and floor(s)+1
    at fractional position  t = s - floor(s)

For 24 -> 60 the fraction cycles 0.0, 0.4, 0.8, 0.2, 0.6 with period 5, i.e.
every 5th output frame is an exact copy of a source frame and the other four
are genuine interpolations. The spacing is perfectly uniform.

Audio sync
----------
Because each frame is placed by absolute time rather than by accumulating
offsets, there is no drift: frame i is at i/60 seconds no matter how long the
clip is. The only discrepancy is at the tail, where the last output frame may
fall up to one frame short of the audio's end. We deliberately generate enough
frames to *cover* the audio and let the muxer trim with ``-shortest``, so the
final file is exactly as long as the generated soundtrack.
"""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)

FRAME_GLOB = "frame_%08d.png"


@dataclass(frozen=True)
class FrameTiming:
    """One output frame: interpolate ``left`` -> ``left + 1`` at ``t``."""

    index: int
    left: int
    t: float

    @property
    def is_copy(self) -> bool:
        """True when this frame is an exact source frame, no synthesis needed."""
        return self.t <= 1e-9


@dataclass(frozen=True)
class InterpolationPlan:
    src_fps: float
    dst_fps: float
    num_src_frames: int
    timings: tuple[FrameTiming, ...]

    @property
    def num_dst_frames(self) -> int:
        return len(self.timings)

    @property
    def src_duration_s(self) -> float:
        return self.num_src_frames / self.src_fps

    @property
    def dst_duration_s(self) -> float:
        return self.num_dst_frames / self.dst_fps

    @property
    def num_synthesised(self) -> int:
        return sum(0 if timing.is_copy else 1 for timing in self.timings)


def plan_interpolation(
    num_src_frames: int,
    src_fps: float = 24.0,
    dst_fps: float = 60.0,
) -> InterpolationPlan:
    """Build the absolute-timestamp interpolation plan.

    Generates ``ceil(duration * dst_fps)`` frames so the video always covers
    the full source duration; the muxer trims the sub-frame tail. Frames whose
    interpolation window would run past the last source frame are clamped to a
    copy of the last frame (this can only ever affect the final frame or two).
    """
    if num_src_frames < 1:
        raise ValueError("need at least one source frame")
    if src_fps <= 0 or dst_fps <= 0:
        raise ValueError("frame rates must be positive")

    if num_src_frames == 1:
        return InterpolationPlan(src_fps, dst_fps, 1, (FrameTiming(0, 0, 0.0),))

    duration = num_src_frames / src_fps
    num_dst = max(1, math.ceil(duration * dst_fps - 1e-9))
    step = src_fps / dst_fps
    last = num_src_frames - 1

    timings = []
    for i in range(num_dst):
        s = i * step
        left = int(math.floor(s + 1e-9))
        t = s - left
        if left >= last:
            left, t = last, 0.0
        timings.append(FrameTiming(index=i, left=left, t=round(t, 9)))

    return InterpolationPlan(src_fps, dst_fps, num_src_frames, tuple(timings))


# --------------------------------------------------------------------------
# Backend 1: ffmpeg minterpolate (default -- no extra model weights needed)
# --------------------------------------------------------------------------

def interpolate_with_ffmpeg(
    src_dir: Path,
    dst_dir: Path,
    src_fps: float = 24.0,
    dst_fps: float = 60.0,
    ffmpeg: str = "ffmpeg",
    pix_fmt: str = "rgb48le",
) -> int:
    """Motion-compensated interpolation via ffmpeg's ``minterpolate``.

    ffmpeg handles the non-integer 2.5x ratio correctly on its own (it works in
    absolute time, same principle as :func:`plan_interpolation`). This is the
    default backend because it needs no downloaded weights, which means the
    repository works on a fresh machine.

    Slower than RIFE on a GPU, and softer on fast motion. Prefer the RIFE
    backend when you have the weights.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    filters = (
        f"minterpolate=fps={dst_fps}:mi_mode=mci:mc_mode=aobmc"
        ":me_mode=bidir:me=epzs:vsbmc=1"
    )
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
        "-framerate", str(src_fps),
        "-i", str(src_dir / FRAME_GLOB),
        "-vf", filters,
        "-pix_fmt", pix_fmt,
        "-start_number", "0",
        str(dst_dir / FRAME_GLOB),
    ]
    LOGGER.info("interpolating with ffmpeg minterpolate: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return len(list(dst_dir.glob("frame_*.png")))


# --------------------------------------------------------------------------
# Backend 2: RIFE (optional, arbitrary-timestep)
# --------------------------------------------------------------------------

class RifeUnavailable(RuntimeError):
    """RIFE weights or module could not be loaded."""


class RifeInterpolator:
    """Adapter around a RIFE v4.x ``flownet`` that supports a timestep input.

    RIFE is not pip-installable and its weights are distributed separately, so
    this is deliberately an optional backend. v4.x is required because earlier
    versions only do the midpoint (t=0.5) and cannot hit the 0.2/0.4/0.6/0.8
    positions that a correct 24 -> 60 conversion needs.

    ``model_dir`` must contain a RIFE v4.x ``flownet.pkl`` and be importable as
    a package exposing ``Model`` (the layout used by the Practical-RIFE repo).
    """

    def __init__(self, model_dir: Path, device: str = "cuda", fp16: bool = True):
        self.model_dir = Path(model_dir)
        self.device = device
        self.fp16 = fp16
        self._model = None

    def load(self):
        if self._model is not None:
            return self._model
        try:
            import sys

            import torch
        except ImportError as exc:  # pragma: no cover
            raise RifeUnavailable("torch is required for the RIFE backend") from exc

        if not self.model_dir.exists():
            raise RifeUnavailable(
                f"RIFE model dir {self.model_dir} not found. Clone a RIFE v4.x "
                "release into it (it must expose train_log/RIFE_HDv3.py and "
                "flownet.pkl), or use the default ffmpeg backend."
            )

        sys.path.insert(0, str(self.model_dir))
        try:
            from train_log.RIFE_HDv3 import Model  # type: ignore
        except Exception as exc:
            raise RifeUnavailable(
                f"could not import RIFE from {self.model_dir}: {exc}"
            ) from exc

        model = Model()
        model.load_model(str(self.model_dir), -1)
        model.eval()
        model.device()
        self._model = model
        self._torch = torch
        return model

    def interpolate(self, frame_a, frame_b, t: float):
        """Synthesise the frame at fractional position ``t`` between two frames.

        ``frame_a`` / ``frame_b`` are float tensors shaped (3, H, W) in [0, 1].
        """
        model = self.load()
        torch = self._torch
        with torch.no_grad():
            a = frame_a.unsqueeze(0).to(self.device)
            b = frame_b.unsqueeze(0).to(self.device)
            if self.fp16:
                a, b = a.half(), b.half()
            # RIFE needs both dimensions padded to a multiple of 32.
            _, _, height, width = a.shape
            pad_h = (32 - height % 32) % 32
            pad_w = (32 - width % 32) % 32
            if pad_h or pad_w:
                a = torch.nn.functional.pad(a, (0, pad_w, 0, pad_h), mode="replicate")
                b = torch.nn.functional.pad(b, (0, pad_w, 0, pad_h), mode="replicate")
            out = model.inference(a, b, timestep=t)
            out = out[:, :, :height, :width]
            return out.float().clamp(0, 1).squeeze(0).cpu()
