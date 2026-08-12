"""Multimodal reference inputs for H3's ``ref2va`` workflow.

H3 accepts a mixed, *ordered* list of references: images, video clips and audio
clips. The published limits are strict, and every one of them is checked here
locally, before a GPU is ever booted -- a rejected request should cost nothing.

    images   <= 9
    videos   <= 3   each 2-15 s, total <= 15 s
    audios   <= 3   each 2-15 s, total <= 15 s, never on their own
    total    <= 12  files across all types

Two things that are easy to get wrong:

**Order is semantic.** The position of each reference decides how it is labelled
in the prompt (``<Picture 1>``, ``<Video 1>``, ``<Audio 1>``) and where it lands
on the shared audio/video rotary clock. Re-ordering the same files is a
*different request*, not a cosmetic change, so this module preserves the order
you give and never sorts.

**Audio cannot stand alone.** An audio reference has to accompany at least one
image or video. Audio never reaches the text encoder; it is encoded by the audio
VAE only, so on its own there is nothing for H3 to attach it to.

``ref2va`` also runs on a different checkpoint partition (``transformer_ref/``,
66.28 GB) from ``t2va``/``fl2va`` (``transformer/``, 66.28 GB). Wanting both
resident at once is what pushes the memory requirement past a B200.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

MAX_IMAGES = 9
MAX_VIDEOS = 3
MAX_AUDIOS = 3
MAX_TOTAL = 12

MIN_CLIP_S = 2.0
MAX_CLIP_S = 15.0
MAX_TOTAL_CLIP_S = 15.0

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus"}


class ReferenceError(ValueError):
    """A reference set that H3 would reject, caught before spending money."""


@dataclass
class ReferenceSet:
    """An ordered reference set, validated against H3's documented limits."""

    images: list[str] = field(default_factory=list)
    videos: list[str] = field(default_factory=list)
    audios: list[str] = field(default_factory=list)
    order: list[tuple[str, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.images) + len(self.videos) + len(self.audios)

    @property
    def is_empty(self) -> bool:
        return self.total == 0

    @property
    def workflow(self) -> str:
        """Which H3 workflow this set implies."""
        if self.is_empty:
            return "t2va"
        if self.videos or self.audios or len(self.images) > 2:
            return "ref2va"
        # One or two bare images are keyframes, which the cheaper partition
        # handles. Only promote to ref2va when references are genuinely needed.
        return "fl2va"

    def summary(self) -> str:
        parts = []
        if self.images:
            parts.append(f"{len(self.images)} image(s)")
        if self.videos:
            parts.append(f"{len(self.videos)} video(s)")
        if self.audios:
            parts.append(f"{len(self.audios)} audio clip(s)")
        return ", ".join(parts) if parts else "no references"


def classify(path: str) -> str:
    """Guess a reference's modality from its extension."""
    suffix = Path(str(path)).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    if suffix in AUDIO_SUFFIXES:
        return "audio"
    raise ReferenceError(
        f"cannot tell what {path!r} is from its extension. Supported: "
        f"images {sorted(IMAGE_SUFFIXES)}, videos {sorted(VIDEO_SUFFIXES)}, "
        f"audio {sorted(AUDIO_SUFFIXES)}"
    )


def build_reference_set(
    images: Sequence[str] | None = None,
    videos: Sequence[str] | None = None,
    audios: Sequence[str] | None = None,
    mixed: Sequence[str] | None = None,
) -> ReferenceSet:
    """Assemble and validate a reference set.

    ``mixed`` is a single ordered list whose modalities are inferred from file
    extensions -- convenient on a command line, and it preserves the exact order
    given, which matters because order is semantic.
    """
    images = list(images or [])
    videos = list(videos or [])
    audios = list(audios or [])
    order: list[tuple[str, str]] = []

    if mixed:
        for item in mixed:
            kind = classify(item)
            order.append((kind, str(item)))
            {"image": images, "video": videos, "audio": audios}[kind].append(str(item))
    else:
        order = (
            [("image", str(p)) for p in images]
            + [("video", str(p)) for p in videos]
            + [("audio", str(p)) for p in audios]
        )

    result = ReferenceSet(images=images, videos=videos, audios=audios, order=order)
    validate(result)
    return result


def validate(references: ReferenceSet) -> ReferenceSet:
    """Raise :class:`ReferenceError` if H3 would reject this set."""
    if len(references.images) > MAX_IMAGES:
        raise ReferenceError(
            f"{len(references.images)} images, but H3 accepts at most {MAX_IMAGES}"
        )
    if len(references.videos) > MAX_VIDEOS:
        raise ReferenceError(
            f"{len(references.videos)} video clips, but H3 accepts at most {MAX_VIDEOS}"
        )
    if len(references.audios) > MAX_AUDIOS:
        raise ReferenceError(
            f"{len(references.audios)} audio clips, but H3 accepts at most {MAX_AUDIOS}"
        )
    if references.total > MAX_TOTAL:
        raise ReferenceError(
            f"{references.total} reference files, but H3 accepts at most "
            f"{MAX_TOTAL} across all types"
        )
    if references.audios and not (references.images or references.videos):
        raise ReferenceError(
            "audio references cannot be the only input -- each must accompany at "
            "least one image or video, because audio never reaches the text "
            "encoder and has nothing to attach to on its own"
        )
    return references


def check_clip_durations(
    durations: Sequence[float],
    kind: str = "video",
) -> None:
    """Validate measured clip durations against the 2-15 s per-clip rule.

    Kept separate from :func:`validate` because it needs the media to have been
    probed. Call it once durations are known -- ideally still before booting a GPU.
    """
    for index, seconds in enumerate(durations, start=1):
        if seconds < MIN_CLIP_S - 1e-6:
            raise ReferenceError(
                f"{kind} {index} is {seconds:.2f}s; each must be at least {MIN_CLIP_S}s"
            )
        if seconds > MAX_CLIP_S + 1e-6:
            raise ReferenceError(
                f"{kind} {index} is {seconds:.2f}s; each must be at most {MAX_CLIP_S}s"
            )
    total = sum(durations)
    if total > MAX_TOTAL_CLIP_S + 1e-6:
        raise ReferenceError(
            f"{kind} clips total {total:.2f}s; the combined limit is "
            f"{MAX_TOTAL_CLIP_S}s"
        )


def probe_duration(path: str, ffprobe: str = "ffprobe") -> float:
    """Measure a media file's duration with ffprobe."""
    import json
    import subprocess

    result = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json", "-show_format", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def to_diffusers(references: ReferenceSet) -> list[Any]:
    """Turn a validated set into diffusers reference objects, in order.

    Uses each class's ``from_file`` classmethod, which carries the real frame
    rate and sample rate through. ``load_video`` would drop the frame rate, and a
    reference whose true rate was lost is conditioned on at the wrong speed with
    nothing raised about it.
    """
    from diffusers.modular_pipelines.minimax_h3 import (
        MiniMaxH3AudioReference,
        MiniMaxH3ImageReference,
        MiniMaxH3VideoReference,
    )

    builders = {
        "image": MiniMaxH3ImageReference,
        "video": MiniMaxH3VideoReference,
        "audio": MiniMaxH3AudioReference,
    }
    return [builders[kind].from_file(path) for kind, path in references.order]
