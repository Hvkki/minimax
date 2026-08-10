"""16-bit PNG frame sequences on disk.

Frames go to disk between stages rather than staying in RAM. A 14.4 second clip
is 345 frames at 768p and 863 frames at 60 fps; as float32 in memory that is
4.3 GB and 10.7 GB respectively, and the 4K stage would be 43 GB. Round-tripping
through 16-bit PNG costs some I/O and keeps peak memory flat, which matters
because the H3 weights are already occupying most of the machine.

16-bit specifically: 8-bit would quantise the gradients that the 4x upscale then
stretches across three output pixels each, which is what produces banding.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np

LOGGER = logging.getLogger(__name__)

FRAME_PATTERN = "frame_%08d.png"
FRAME_GLOB = "frame_*.png"


def frame_path(directory: Path, index: int) -> Path:
    return Path(directory) / (FRAME_PATTERN % index)


def write_frame(path: Path, frame: np.ndarray, depth: int = 16) -> Path:
    """Write one float32 HWC [0,1] frame as a PNG.

    ``depth=16`` uses a three-panel ``I;16`` strip, because PIL has no 16-bit RGB
    mode and cannot merge 16-bit bands. These files are only meant to be read
    back by :func:`read_frame` -- ffmpeg would see one wide greyscale image.

    ``depth=8`` writes an ordinary RGB PNG that ffmpeg can read. Use it for the
    ffmpeg ``minterpolate`` path, which is 8-bit internally regardless, so
    nothing is lost by matching it.
    """
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected HWC RGB, got shape {frame.shape}")
    clipped = np.clip(frame, 0.0, 1.0)

    if depth == 8:
        array = (clipped * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(array, mode="RGB").save(path, optimize=False, compress_level=1)
        return path
    if depth != 16:
        raise ValueError("depth must be 8 or 16")

    array = (clipped * 65535.0 + 0.5).astype(np.uint16)
    strip = np.concatenate([array[:, :, c] for c in range(3)], axis=1)
    Image.fromarray(strip, mode="I;16").save(path, optimize=False, compress_level=1)
    return path


def read_frame(path: Path) -> np.ndarray:
    """Read a frame back to float32 HWC [0,1].

    Handles all three shapes we might encounter: our own 16-bit strips, ordinary
    8-bit RGB PNGs (ours or ffmpeg's), and true 16-bit RGB PNGs.
    """
    from PIL import Image

    with Image.open(path) as handle:
        array = np.asarray(handle)

    if array.ndim == 3:
        if array.shape[2] == 4:
            array = array[:, :, :3]
        scale = 255.0 if array.dtype == np.uint8 else 65535.0
        return array.astype(np.float32) / scale

    if array.ndim != 2 or array.shape[1] % 3 != 0:
        raise ValueError(f"{path} is not a frame or frame strip (shape {array.shape})")
    width = array.shape[1] // 3
    channels = [array[:, c * width:(c + 1) * width] for c in range(3)]
    return np.stack(channels, axis=-1).astype(np.float32) / 65535.0


def write_sequence(
    directory: Path,
    frames: np.ndarray | list[np.ndarray],
    depth: int = 16,
) -> int:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    count = 0
    for index, frame in enumerate(frames):
        write_frame(frame_path(directory, index), np.asarray(frame), depth=depth)
        count += 1
    LOGGER.info("wrote %d frames (%d-bit) to %s", count, depth, directory)
    return count


def list_sequence(directory: Path) -> list[Path]:
    return sorted(Path(directory).glob(FRAME_GLOB))


def count_sequence(directory: Path) -> int:
    return len(list_sequence(directory))


def iter_sequence(directory: Path) -> Iterator[np.ndarray]:
    for path in list_sequence(directory):
        yield read_frame(path)


class SequenceReader:
    """Random access to a frame sequence with a tiny LRU, for interpolation.

    Interpolation walks pairs (i, i+1) and revisits the same left frame up to
    three times in a row for 24 -> 60, so a 4-entry cache removes most re-reads
    without holding the clip in memory.
    """

    def __init__(self, directory: Path, cache_size: int = 4):
        self.paths = list_sequence(directory)
        if not self.paths:
            raise FileNotFoundError(f"no frames in {directory}")
        self.cache_size = max(1, cache_size)
        self._cache: dict[int, np.ndarray] = {}
        self._order: list[int] = []

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> np.ndarray:
        index = max(0, min(index, len(self.paths) - 1))
        hit = self._cache.get(index)
        if hit is not None:
            return hit
        frame = read_frame(self.paths[index])
        self._cache[index] = frame
        self._order.append(index)
        while len(self._order) > self.cache_size:
            self._cache.pop(self._order.pop(0), None)
        return frame

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width) of the first frame."""
        first = self[0]
        return first.shape[0], first.shape[1]
