"""Pipeline stages.

The order matters and is not arbitrary:

  1. generate     H3-Base -> 768p / 24 fps frames + a jointly generated soundtrack
  2. interpolate  24 -> 60 fps at *native* resolution, where optical flow is both
                  cheap and accurate
  3. upscale      crop to the exact output aspect, super-resolve, supersample down
  4. encode       10-bit, bt709 tagged, audio as the duration authority

Interpolating before upscaling is deliberate. Flow estimation is far more reliable
at 768p than at 4K (large displacements exceed a flow network's receptive field),
and doing it first means the expensive super-resolution stage is the only thing
that ever touches full-resolution frames.

``generate`` imports torch and diffusers lazily, so the geometry, timing and
encoding stages can be imported and tested without a GPU.
"""

from __future__ import annotations

from .encode import EncodeSettings, encode_frames, probe, write_wav
from .interpolate import (
    FrameTiming,
    InterpolationPlan,
    interpolate_with_ffmpeg,
    plan_interpolation,
)
from .upscale import (
    RESOLUTIONS,
    UHD_2K,
    UHD_4K,
    Geometry,
    LanczosUpscaler,
    SpandrelUpscaler,
    TileGrid,
    blend_tiles,
    pick_scale,
    plan_geometry,
    plan_tiles,
    process_frame,
    resize_lanczos,
    resolve_target,
    target_for_source,
)

__all__ = [
    "RESOLUTIONS",
    "EncodeSettings",
    "encode_frames",
    "probe",
    "write_wav",
    "FrameTiming",
    "InterpolationPlan",
    "interpolate_with_ffmpeg",
    "plan_interpolation",
    "UHD_2K",
    "UHD_4K",
    "Geometry",
    "LanczosUpscaler",
    "SpandrelUpscaler",
    "TileGrid",
    "blend_tiles",
    "pick_scale",
    "plan_geometry",
    "plan_tiles",
    "process_frame",
    "resize_lanczos",
    "resolve_target",
    "target_for_source",
]
