"""768p -> 4K super-resolution with aspect correction and shimmer control.

Three things this stage gets right that naive upscalers do not.

1. H3's canvas is not 16:9.
   The trained canvas is 1344x768, which is 1.75:1. UHD is 3840x2160, which is
   1.7778:1. Scaling one onto the other directly stretches the image by 1.6%
   horizontally -- small, but very visible on faces. We centre-crop to the
   target aspect first, losing 12 rows out of 768, then scale.

2. Supersampling instead of direct scaling.
   768 -> 2160 is 2.8125x, which no ESRGAN-family model does natively. Running
   a 4x model and then Lanczos-downscaling to 2160 gives a *better* result than
   asking for 2.8125x directly: the downscale averages away the high-frequency
   artefacts and per-pixel noise the upscaler invents, which is also the main
   source of temporal shimmer.

3. Deterministic tiling.
   4x on a cropped 1344x756 frame is 5376x3024 = 16 megapixels, which will not
   fit in VRAM alongside the model on most cards. Tiles are cut on a fixed grid
   with a symmetric overlap and blended with a linear ramp, so the same frame
   always produces the same output and seams do not crawl between frames.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

UHD_4K = (3840, 2160)
UHD_2K = (2560, 1440)

RESOLUTIONS: dict[str, tuple[int, int]] = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": UHD_2K,
    "2160p": UHD_4K,
}


def pick_scale(crop_height: int, out_height: int) -> int:
    """Choose the cheapest super-resolution factor that reaches the target.

    Returns 1 when no upscaling is needed at all. H3's 768p canvas down to 720p is
    a *downscale*, so running a super-resolution model first would spend the most
    expensive stage in the pipeline producing detail that is then thrown away.

    2x costs roughly a quarter of 4x measured in model output pixels (4.1 vs 16.3
    megapixels per frame from H3's 1344x756 cropped canvas), so 2x is preferred
    whenever it reaches the target; a Lanczos step covers the remainder. That
    supersampling also suppresses the high-frequency artefacts and temporal
    shimmer that super-resolution models invent.
    """
    if out_height <= crop_height:
        return 1
    return 2 if out_height / crop_height <= 2.0 + 1e-6 else 4


def target_for_source(
    src_width: int,
    src_height: int,
    target: tuple[int, int] = UHD_4K,
) -> tuple[int, int]:
    """Orient the output to match the source.

    Without this, a 9:16 vertical generation (768x1344) forced into landscape
    UHD would be centre-cropped to 768x432 -- throwing away 68% of the frame.
    A portrait source should render portrait 4K (2160x3840) instead.
    """
    long_edge, short_edge = max(target), min(target)
    if src_height > src_width:
        return (short_edge, long_edge)
    return (long_edge, short_edge)


@dataclass(frozen=True)
class Geometry:
    """Resolved crop-then-scale plan for one frame."""

    src_width: int
    src_height: int
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int
    model_scale: int
    intermediate_width: int
    intermediate_height: int
    out_width: int
    out_height: int

    @property
    def crop_box(self) -> tuple[int, int, int, int]:
        """PIL-style (left, upper, right, lower)."""
        return (
            self.crop_x,
            self.crop_y,
            self.crop_x + self.crop_width,
            self.crop_y + self.crop_height,
        )

    @property
    def cropped_rows(self) -> int:
        return self.src_height - self.crop_height

    @property
    def cropped_cols(self) -> int:
        return self.src_width - self.crop_width

    @property
    def effective_scale(self) -> float:
        return self.out_height / self.crop_height

    @property
    def is_supersampled(self) -> bool:
        """True when we upscale past the target and come back down."""
        return self.intermediate_height > self.out_height


def plan_geometry(
    src_width: int,
    src_height: int,
    out_width: int = UHD_4K[0],
    out_height: int = UHD_4K[1],
    model_scale: int = 4,
    fit: str = "crop",
) -> Geometry:
    """Plan the crop/pad and scale from a source frame to the output size.

    ``fit="crop"`` (default) centre-crops the minimum number of whole even
    pixels needed to match the output aspect ratio. For H3's 1344x768 canvas
    into UHD that is 12 rows -- 1.6% of the height, invisible.

    ``fit="pad"`` keeps the entire frame and letterboxes instead. Use it when
    the aspect gap is large (generating 1:1 or 4:3 and outputting 16:9 would
    otherwise crop 25-45% away).

    Nothing is ever stretched in either mode.
    """
    if min(src_width, src_height, out_width, out_height) <= 0:
        raise ValueError("dimensions must be positive")
    if model_scale < 1:
        raise ValueError("model_scale must be >= 1")
    if fit not in ("crop", "pad"):
        raise ValueError("fit must be 'crop' or 'pad'")

    target_ar = out_width / out_height
    src_ar = src_width / src_height

    if abs(src_ar - target_ar) < 1e-9 or fit == "pad":
        crop_width, crop_height = src_width, src_height
    elif src_ar > target_ar:
        # Source is wider than target: trim columns.
        crop_height = src_height
        crop_width = min(src_width, int(round(src_height * target_ar)) & ~1)
    else:
        # Source is taller/narrower than target: trim rows.
        crop_width = src_width
        crop_height = min(src_height, int(round(src_width / target_ar)) & ~1)

    crop_x = (src_width - crop_width) // 2
    crop_y = (src_height - crop_height) // 2

    return Geometry(
        src_width=src_width,
        src_height=src_height,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_width=crop_width,
        crop_height=crop_height,
        model_scale=model_scale,
        intermediate_width=crop_width * model_scale,
        intermediate_height=crop_height * model_scale,
        out_width=out_width,
        out_height=out_height,
    )


# --------------------------------------------------------------------------
# Tiling
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Tile:
    x: int
    y: int
    width: int
    height: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass(frozen=True)
class TileGrid:
    """A uniform tile grid plus the overlap it actually achieved.

    The achieved overlap matters: naively stepping by ``tile - overlap`` and
    clamping the final tile to the frame edge produces a much larger overlap on
    that last tile than on the others. If the blend then feathers with the
    *requested* overlap width, the ramp does not span the real overlap region
    and a seam becomes visible -- and because it sits at a fixed pixel column it
    crawls as a vertical line through the whole clip. So positions are
    distributed evenly and the real overlap is reported back.
    """

    tiles: tuple[Tile, ...]
    overlap_x: int
    overlap_y: int

    def __len__(self) -> int:
        return len(self.tiles)

    def __iter__(self):
        return iter(self.tiles)


def _positions(size: int, tile: int, min_overlap: int) -> tuple[list[int], int]:
    """Evenly spaced tile origins covering ``size``, plus the real overlap."""
    if tile >= size:
        return [0], 0
    stride = tile - min_overlap
    count = max(2, -(-(size - min_overlap) // stride))  # ceil division
    last = size - tile
    if count == 1:
        return [0], 0
    origins = [int(round(i * last / (count - 1))) for i in range(count)]
    origins = sorted(set(origins))
    if len(origins) < 2:
        return origins, 0
    achieved = min(
        tile - (origins[i + 1] - origins[i]) for i in range(len(origins) - 1)
    )
    return origins, max(0, achieved)


def plan_tiles(width: int, height: int, tile: int, overlap: int) -> TileGrid:
    """Cut a deterministic, uniformly overlapping tile grid over the frame."""
    if tile <= 0:
        return TileGrid((Tile(0, 0, width, height),), 0, 0)
    if overlap < 0 or overlap >= tile:
        raise ValueError("overlap must be >= 0 and < tile")

    xs, overlap_x = _positions(width, tile, overlap)
    ys, overlap_y = _positions(height, tile, overlap)
    tile_w, tile_h = min(tile, width), min(tile, height)

    tiles = tuple(
        Tile(x, y, tile_w, tile_h) for y in ys for x in xs
    )
    return TileGrid(tiles, overlap_x, overlap_y)


def _ramp(length: int, overlap: int, at_start: bool, at_end: bool) -> np.ndarray:
    """A 1-D blend weight with linear ramps only on interior edges."""
    weights = np.ones(length, dtype=np.float32)
    if overlap <= 0:
        return weights
    ramp_len = min(overlap, length)
    ramp = np.linspace(0.0, 1.0, ramp_len + 2, dtype=np.float32)[1:-1]
    if not at_start:
        weights[:ramp_len] = np.minimum(weights[:ramp_len], ramp)
    if not at_end:
        weights[-ramp_len:] = np.minimum(weights[-ramp_len:], ramp[::-1])
    return weights


def blend_tiles(
    outputs: list[tuple[Tile, np.ndarray]],
    width: int,
    height: int,
    scale: int,
    grid: TileGrid,
) -> np.ndarray:
    """Composite upscaled tiles with linear feathering across the real overlap."""
    out_h, out_w = height * scale, width * scale
    accum = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weight = np.zeros((out_h, out_w, 1), dtype=np.float32)

    for tile, patch in outputs:
        ty, tx = tile.y * scale, tile.x * scale
        patch_h, patch_w = patch.shape[:2]
        wy = _ramp(
            patch_h, grid.overlap_y * scale,
            at_start=(tile.y == 0),
            at_end=(tile.y + tile.height >= height),
        )
        wx = _ramp(
            patch_w, grid.overlap_x * scale,
            at_start=(tile.x == 0),
            at_end=(tile.x + tile.width >= width),
        )
        w = (wy[:, None] * wx[None, :])[:, :, None]
        accum[ty:ty + patch_h, tx:tx + patch_w] += patch.astype(np.float32) * w
        weight[ty:ty + patch_h, tx:tx + patch_w] += w

    np.maximum(weight, 1e-6, out=weight)
    return accum / weight


# --------------------------------------------------------------------------
# Upscaler backends
# --------------------------------------------------------------------------

class Upscaler:
    """Interface: takes float32 HWC in [0,1], returns float32 HWC in [0,1]."""

    scale: int = 1
    name: str = "identity"

    def upscale(self, image: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class LanczosUpscaler(Upscaler):
    """Weight-free fallback. Use for smoke tests and CI, not for real output.

    Produces a correctly shaped, correctly timed 4K file so the rest of the
    pipeline can be validated without downloading an ESRGAN checkpoint, but it
    invents no detail -- it is a plain resample.
    """

    name = "lanczos"

    def __init__(self, scale: int = 4):
        self.scale = scale

    def upscale(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        return resize_lanczos(image, width * self.scale, height * self.scale)


class SpandrelUpscaler(Upscaler):
    """Real super-resolution via spandrel, which loads ESRGAN-family weights.

    spandrel is used instead of the ``realesrgan`` package because the latter
    pins an old ``basicsr`` that no longer installs cleanly. spandrel reads the
    same ``.pth`` checkpoints (Real-ESRGAN x4plus, 4x-UltraSharp, animevideo,
    SwinIR, DAT, ...) and reports each model's true scale factor.
    """

    name = "spandrel"

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda",
        dtype: str = "float16",
        tile: int = 384,
        overlap: int = 32,
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.dtype = dtype
        self.tile = tile
        self.overlap = overlap
        self._model = None
        self._torch = None

    def load(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            from spandrel import ModelLoader
        except ImportError as exc:
            raise RuntimeError(
                "the spandrel backend needs 'torch' and 'spandrel' installed "
                "(pip install spandrel)"
            ) from exc

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"upscaler checkpoint not found: {self.model_path}. "
                "Run scripts/download_weights.py --upscaler to fetch one."
            )

        descriptor = ModelLoader().load_from_file(str(self.model_path))
        model = descriptor.model.eval().to(self.device)
        if self.dtype == "float16" and self.device.startswith("cuda"):
            model = model.half()
        self.scale = int(getattr(descriptor, "scale", 4) or 4)
        self._model = model
        self._torch = torch
        LOGGER.info(
            "loaded upscaler %s (scale=%dx) on %s",
            self.model_path.name, self.scale, self.device,
        )
        return model

    def upscale(self, image: np.ndarray) -> np.ndarray:
        model = self.load()
        torch = self._torch
        height, width = image.shape[:2]
        tiles = plan_tiles(width, height, self.tile, self.overlap)

        outputs: list[tuple[Tile, np.ndarray]] = []
        with torch.no_grad():
            for tile in tiles:
                left, upper, right, lower = tile.box
                patch = np.ascontiguousarray(image[upper:lower, left:right])
                tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0)
                tensor = tensor.to(self.device)
                if self.dtype == "float16" and self.device.startswith("cuda"):
                    tensor = tensor.half()
                out = model(tensor)
                out = out.float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
                outputs.append((tile, out))

        return blend_tiles(outputs, width, height, self.scale, tiles)


def resize_lanczos(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """High-quality 16-bit-safe resample, used for the supersample downscale."""
    from PIL import Image

    channels = []
    for c in range(3):
        band = Image.fromarray(
            (np.clip(image[:, :, c], 0, 1) * 65535.0).astype(np.uint16), mode="I;16"
        )
        band = band.resize((width, height), Image.LANCZOS)
        channels.append(np.asarray(band).astype(np.float32) / 65535.0)
    return np.stack(channels, axis=-1)


def process_frame(image: np.ndarray, geometry: Geometry, upscaler: Upscaler) -> np.ndarray:
    """crop -> upscale -> Lanczos down to the exact output size (letterbox if needed)."""
    left, upper, right, lower = geometry.crop_box
    cropped = np.ascontiguousarray(image[upper:lower, left:right])
    upscaled = upscaler.upscale(cropped)

    out_w, out_h = geometry.out_width, geometry.out_height
    src_h, src_w = upscaled.shape[:2]

    # Fit inside the target box preserving aspect, then centre on a black canvas.
    scale = min(out_w / src_w, out_h / src_h)
    fit_w = max(1, int(round(src_w * scale)))
    fit_h = max(1, int(round(src_h * scale)))
    if (fit_w, fit_h) != (src_w, src_h):
        upscaled = resize_lanczos(upscaled, fit_w, fit_h)

    if (fit_w, fit_h) == (out_w, out_h):
        return np.clip(upscaled, 0.0, 1.0)

    canvas = np.zeros((out_h, out_w, 3), dtype=np.float32)
    off_x, off_y = (out_w - fit_w) // 2, (out_h - fit_h) // 2
    canvas[off_y:off_y + fit_h, off_x:off_x + fit_w] = np.clip(upscaled, 0.0, 1.0)
    return canvas
