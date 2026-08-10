"""MiniMax H3 generation constraints.

Every number here comes from the published H3 model card and the diffusers
integration docs, not from guesswork:

  * output frame rate is fixed at 24 fps
  * ``num_frames`` must be ``17 * n + 5`` (what the visual VAE can decode)
  * the resulting duration must stay inside 5.0 - 15.0 seconds
  * height and width must be multiples of 32
  * the canvas short edge defaults to 768 and the canvas is additionally
    capped by ``canvas_max_pixels`` (1032192 = 1344 * 768)

The pixel cap is the subtle one. For 16:9 the "nearest multiple of 32" width
for a 768 tall canvas is 1376, but 1376 * 768 = 1056768 which is over the cap,
so the real trained canvas is 1344x768 (exactly 1032192). Any code that only
rounds to a multiple of 32 without re-checking the area will ask for a canvas
H3 was not trained on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

SRC_FPS = 24
"""H3 generates at a fixed 24 fps."""

FRAME_QUANTUM = 17
FRAME_OFFSET = 5
"""``num_frames`` must equal ``FRAME_QUANTUM * n + FRAME_OFFSET``."""

MIN_DURATION_S = 5.0
MAX_DURATION_S = 15.0

CANVAS_SHORT_EDGE = 768
CANVAS_MAX_PIXELS = 1032192
SIZE_MULTIPLE = 32

AUDIO_SAMPLE_RATE = 32000
"""H3's audio VAE emits 32 kHz stereo. Retained for reference; the real value
is read back from the pipeline's ``sampling_rate`` output at runtime."""


def _valid_frame_counts() -> list[int]:
    """All ``17n + 5`` frame counts whose duration lands in [5, 15] seconds."""
    counts = []
    n = 1
    while True:
        frames = FRAME_QUANTUM * n + FRAME_OFFSET
        duration = frames / SRC_FPS
        if duration > MAX_DURATION_S:
            break
        if duration >= MIN_DURATION_S:
            counts.append(frames)
        n += 1
    return counts


VALID_FRAME_COUNTS: tuple[int, ...] = tuple(_valid_frame_counts())
MIN_FRAMES = VALID_FRAME_COUNTS[0]
MAX_FRAMES = VALID_FRAME_COUNTS[-1]


class ConstraintError(ValueError):
    """Raised when a request cannot be satisfied by H3's fixed geometry."""


def snap_num_frames(requested: int) -> int:
    """Snap ``requested`` up to the next decodable ``17n + 5`` frame count.

    Mirrors the diffusers behaviour (snap up), but raises instead of silently
    producing an out-of-range duration.
    """
    if requested > MAX_FRAMES:
        raise ConstraintError(
            f"{requested} frames is {requested / SRC_FPS:.2f}s, over H3's "
            f"{MAX_DURATION_S}s limit (max {MAX_FRAMES} frames = "
            f"{MAX_FRAMES / SRC_FPS:.3f}s). Use chained clips for longer output."
        )
    for count in VALID_FRAME_COUNTS:
        if count >= requested:
            return count
    raise ConstraintError(f"no valid frame count for {requested}")


def frames_for_duration(seconds: float) -> int:
    """Frame count for a requested duration in seconds.

    Snapped up to the next decodable count, except that a request inside the
    documented window but above the largest decodable count (anything in
    14.375s - 15.0s) clamps down to ``MAX_FRAMES`` rather than failing. Asking
    for H3's advertised 15 second maximum should not be an error just because
    ``17n + 5`` never lands exactly on 360.
    """
    if seconds > MAX_DURATION_S + 1e-9:
        raise ConstraintError(
            f"{seconds}s exceeds H3's {MAX_DURATION_S}s per-clip limit. "
            "Use chained clips (see pipeline.render_chain) for longer output."
        )
    if seconds < MIN_DURATION_S - 1e-9:
        raise ConstraintError(
            f"{seconds}s is below H3's {MIN_DURATION_S}s per-clip minimum."
        )
    requested = math.ceil(seconds * SRC_FPS - 1e-9)
    if requested > MAX_FRAMES:
        return MAX_FRAMES
    return snap_num_frames(requested)


def duration_for_frames(num_frames: int) -> float:
    """Exact clip duration in seconds for a frame count."""
    return num_frames / SRC_FPS


def _floor_to_multiple(value: float, multiple: int = SIZE_MULTIPLE) -> int:
    return max(multiple, int(math.floor(value / multiple)) * multiple)


def _round_to_multiple(value: float, multiple: int = SIZE_MULTIPLE) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def parse_aspect_ratio(ratio: str) -> Fraction:
    """Parse ``"16:9"`` into a Fraction. Accepts ``"16/9"`` too."""
    text = ratio.strip().replace("/", ":")
    if ":" not in text:
        raise ConstraintError(f"aspect ratio {ratio!r} must look like '16:9'")
    left, right = text.split(":", 1)
    try:
        num, den = float(left), float(right)
    except ValueError as exc:
        raise ConstraintError(f"bad aspect ratio {ratio!r}") from exc
    if num <= 0 or den <= 0:
        raise ConstraintError(f"aspect ratio {ratio!r} must be positive")
    return Fraction(num / den).limit_denominator(10000)


@dataclass(frozen=True)
class Canvas:
    """A generation canvas that satisfies every H3 geometry constraint."""

    width: int
    height: int

    @property
    def pixels(self) -> int:
        return self.width * self.height

    @property
    def aspect(self) -> float:
        return self.width / self.height

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.width}x{self.height}"


def resolve_canvas(
    aspect_ratio: str = "16:9",
    short_edge: int = CANVAS_SHORT_EDGE,
    max_pixels: int = CANVAS_MAX_PIXELS,
) -> Canvas:
    """Pick the best legal canvas for ``aspect_ratio``.

    Searches candidate short edges (multiples of 32, starting at the requested
    one and walking down) and for each the two nearest legal long edges. Ranks
    candidates by, in order: closeness of the short edge to the request, then
    relative aspect-ratio error, then larger area.

    For 16:9 at short edge 768 this returns 1344x768 -- the canvas H3 was
    actually trained on -- because 1376x768 would breach the pixel cap.
    """
    ratio_f = float(parse_aspect_ratio(aspect_ratio))
    landscape = ratio_f >= 1.0
    requested_short = _round_to_multiple(short_edge)

    best_key: tuple[int, float, int] | None = None
    best: Canvas | None = None

    for candidate_short in range(requested_short, SIZE_MULTIPLE - 1, -SIZE_MULTIPLE):
        raw_long = candidate_short * ratio_f if landscape else candidate_short / ratio_f
        low = _floor_to_multiple(raw_long)
        for long_edge in (low, low + SIZE_MULTIPLE):
            if long_edge < SIZE_MULTIPLE:
                continue
            width, height = (
                (long_edge, candidate_short) if landscape else (candidate_short, long_edge)
            )
            if width * height > max_pixels:
                continue
            ar_error = abs((width / height) - ratio_f) / ratio_f
            key = (
                abs(candidate_short - requested_short),
                round(ar_error, 6),
                -(width * height),
            )
            if best_key is None or key < best_key:
                best_key, best = key, Canvas(width=width, height=height)

    if best is None:
        raise ConstraintError(
            f"no canvas with a multiple-of-32 geometry fits aspect {aspect_ratio!r} "
            f"under {max_pixels} pixels"
        )
    return best


@dataclass(frozen=True)
class ClipSpec:
    """A fully resolved, legal single-clip request."""

    num_frames: int
    canvas: Canvas
    seed: int

    @property
    def duration_s(self) -> float:
        return duration_for_frames(self.num_frames)


def resolve_clip(
    duration_s: float,
    aspect_ratio: str = "16:9",
    short_edge: int = CANVAS_SHORT_EDGE,
    seed: int = 0,
) -> ClipSpec:
    """Turn a human request into something H3 will actually accept."""
    return ClipSpec(
        num_frames=frames_for_duration(duration_s),
        canvas=resolve_canvas(aspect_ratio, short_edge=short_edge),
        seed=seed,
    )
