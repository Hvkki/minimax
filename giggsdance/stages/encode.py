"""Final mux: 4K frames + H3's native soundtrack -> an upload-ready master.

Why this stage is not just "call ffmpeg"
---------------------------------------
Banding. A 768p source blown up to 4K spreads every 8-bit quantisation step
across ~3 output pixels, which turns invisible gradient steps in skies, skin and
lighting falloff into visible contour bands. Three things fight it, in order of
effect:

  1. Keep the pipeline in 16-bit up to this point (we pipe rgb48le, not rgb24).
  2. Encode to a 10-bit format so the encoder is not forced to re-quantise to
     8 bits at the very last step.
  3. Deband plus a *tiny* amount of temporally-varying noise. The noise is the
     unglamorous part that actually works: it dithers the remaining steps so
     the eye integrates them instead of seeing edges.

Memory. 863 frames of 4K 16-bit RGB is ~43 GB, so frames are never all held or
written to disk -- they are streamed into ffmpeg's stdin one at a time and the
encoder consumes them as they arrive.

Sync. The soundtrack is the duration authority. Video frames are placed at
absolute timestamps by the interpolation stage, so they cannot drift; ``-shortest``
trims the sub-frame tail so the container is exactly as long as the audio.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class EncodeSettings:
    """Output encoding options.

    Defaults target "best master I can hand YouTube", not "smallest file".
    YouTube re-encodes every upload, so the job here is to give its encoder
    clean, high-bitrate, correctly tagged input.
    """

    codec: str = "h265"          # h265 | h264 | vp9
    crf: int = 16
    preset: str = "slow"
    fps: float = 60.0
    deband: bool = True
    dither: bool = True
    dither_strength: int = 2
    audio_bitrate: str = "384k"
    audio_rate: int = 48000
    faststart: bool = True
    extra_video_args: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.codec not in ("h265", "h264", "vp9"):
            raise ValueError(f"unsupported codec {self.codec!r}")
        if not 0 <= self.crf <= 51:
            raise ValueError("crf must be 0-51")

    @property
    def is_ten_bit(self) -> bool:
        # H.264 High 10 is poorly supported by consumer players and by YouTube's
        # ingest, so h264 stays 8-bit and leans on dithering instead.
        return self.codec in ("h265", "vp9")

    @property
    def pix_fmt(self) -> str:
        return "yuv420p10le" if self.is_ten_bit else "yuv420p"

    @property
    def container(self) -> str:
        return "webm" if self.codec == "vp9" else "mp4"


def build_video_filters(settings: EncodeSettings) -> str:
    """Filter chain applied to the 4K frames before encoding."""
    chain: list[str] = []

    if settings.deband:
        # Conservative thresholds: strong enough for upscaled gradients, weak
        # enough to leave real texture alone.
        chain.append("deband=1thr=0.008:2thr=0.008:3thr=0.008:4thr=0.008:range=16:blur=true")

    if settings.dither and settings.dither_strength > 0:
        # allf=t makes the pattern change every frame. Static noise would be
        # compressed away as a fixed texture and would not dither anything.
        chain.append(f"noise=alls={settings.dither_strength}:allf=t")

    chain.append(f"format={settings.pix_fmt}")
    # Tag, do not convert: the data is already bt709 primaries/transfer. Without
    # these tags players guess, and YouTube's pipeline can shift the colours.
    chain.append(
        "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"
    )
    return ",".join(chain)


def build_codec_args(settings: EncodeSettings) -> list[str]:
    if settings.codec == "h265":
        return [
            "-c:v", "libx265",
            "-crf", str(settings.crf),
            "-preset", settings.preset,
            "-profile:v", "main10",
            "-tag:v", "hvc1",  # so QuickTime/Safari will open it
            "-x265-params", "log-level=error:aq-mode=3",
            *settings.extra_video_args,
        ]
    if settings.codec == "h264":
        return [
            "-c:v", "libx264",
            "-crf", str(settings.crf),
            "-preset", settings.preset,
            "-profile:v", "high",
            "-level", "5.2",  # required for 3840x2160 at 60 fps
            *settings.extra_video_args,
        ]
    return [
        "-c:v", "libvpx-vp9",
        "-crf", str(settings.crf),
        "-b:v", "0",
        "-row-mt", "1",
        "-tile-columns", "2",
        "-deadline", "good",
        "-cpu-used", "2",
        *settings.extra_video_args,
    ]


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> Path:
    """Write float audio as 24-bit PCM WAV.

    H3 emits 32 kHz stereo floats shaped (channels, samples) or
    (1, channels, samples). 24-bit keeps the model's output intact so the only
    lossy audio step is the final AAC/Opus encode.
    """
    array = np.asarray(audio)
    while array.ndim > 2:
        array = array[0]
    if array.ndim == 1:
        array = array[None, :]
    if array.shape[0] > array.shape[1]:  # (samples, channels) -> (channels, samples)
        array = array.T

    channels, _ = array.shape
    clipped = np.clip(array, -1.0, 1.0)
    ints = (clipped * 8388607.0).astype(np.int32)
    interleaved = ints.T.reshape(-1)
    packed = interleaved.astype("<i4").tobytes()
    # Drop the high byte of each little-endian int32 to make 24-bit samples.
    raw = bytearray()
    for i in range(0, len(packed), 4):
        raw += packed[i:i + 3]

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(3)
        handle.setframerate(int(sample_rate))
        handle.writeframes(bytes(raw))
    return path


def build_encode_command(
    width: int,
    height: int,
    output: Path,
    settings: EncodeSettings,
    audio_path: Path | None = None,
    ffmpeg: str = "ffmpeg",
) -> list[str]:
    """The full ffmpeg invocation, with raw 16-bit RGB arriving on stdin."""
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb48le",
        "-s", f"{width}x{height}",
        "-framerate", str(settings.fps),
        "-i", "-",
    ]
    if audio_path is not None:
        cmd += ["-i", str(audio_path)]

    cmd += ["-filter_complex", f"[0:v]{build_video_filters(settings)}[v]"]
    cmd += ["-map", "[v]"]
    if audio_path is not None:
        cmd += ["-map", "1:a"]

    cmd += build_codec_args(settings)

    if audio_path is not None:
        if settings.codec == "vp9":
            cmd += ["-c:a", "libopus", "-b:a", "256k"]
        else:
            cmd += ["-c:a", "aac", "-b:a", settings.audio_bitrate]
        cmd += ["-ar", str(settings.audio_rate), "-ac", "2", "-shortest"]

    cmd += ["-r", str(settings.fps)]
    if settings.faststart and settings.container == "mp4":
        cmd += ["-movflags", "+faststart"]
    cmd.append(str(output))
    return cmd


class StreamingEncoder:
    """Feed float32 HWC frames in [0,1]; get an encoded file out.

    Used as a context manager so the pipe is always closed and the exit code
    always checked, even when the producing loop raises.
    """

    def __init__(
        self,
        width: int,
        height: int,
        output: Path,
        settings: EncodeSettings,
        audio_path: Path | None = None,
        ffmpeg: str = "ffmpeg",
    ):
        self.width = width
        self.height = height
        self.output = Path(output)
        self.settings = settings
        self.audio_path = audio_path
        self.ffmpeg = ffmpeg
        self.frames_written = 0
        self._proc: subprocess.Popen | None = None

    @property
    def command(self) -> list[str]:
        return build_encode_command(
            self.width, self.height, self.output, self.settings,
            self.audio_path, self.ffmpeg,
        )

    def __enter__(self) -> "StreamingEncoder":
        if shutil.which(self.ffmpeg) is None:
            raise RuntimeError(
                f"{self.ffmpeg!r} not found on PATH. Install ffmpeg (it must be "
                "built with libx265/libx264 and the deband+noise filters)."
            )
        self.output.parent.mkdir(parents=True, exist_ok=True)
        cmd = self.command
        LOGGER.info("encoding: %s", " ".join(cmd))
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        return self

    def write(self, frame: np.ndarray) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("encoder is not running")
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"frame is {frame.shape[1]}x{frame.shape[0]}, "
                f"encoder expects {self.width}x{self.height}"
            )
        as_u16 = (np.clip(frame, 0.0, 1.0) * 65535.0 + 0.5).astype("<u2")
        self._proc.stdin.write(np.ascontiguousarray(as_u16).tobytes())
        self.frames_written += 1

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._proc is None:
            return False
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except BrokenPipeError:
            pass
        code = self._proc.wait()
        self._proc = None
        if exc_type is None and code != 0:
            raise RuntimeError(f"ffmpeg exited with code {code}")
        return False


def encode_frames(
    frames: Iterable[np.ndarray],
    width: int,
    height: int,
    output: Path,
    settings: EncodeSettings,
    audio_path: Path | None = None,
    ffmpeg: str = "ffmpeg",
) -> Path:
    """Stream ``frames`` into an encoded file and return its path."""
    with StreamingEncoder(width, height, output, settings, audio_path, ffmpeg) as encoder:
        for frame in frames:
            encoder.write(frame)
        if encoder.frames_written == 0:
            raise ValueError("no frames were written")
    return Path(output)


def probe(path: Path, ffprobe: str = "ffprobe") -> dict:
    """Read back what we actually produced. Used to verify, not to guess."""
    import json

    cmd = [
        ffprobe, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def concat_with_audio_crossfade(
    clips: Sequence[Path],
    output: Path,
    crossfade_s: float = 0.25,
    ffmpeg: str = "ffmpeg",
) -> Path:
    """Join rendered clips, crossfading the audio across each join.

    Read the docstring of ``pipeline.render_chain`` before relying on this:
    H3 generates each clip's soundtrack independently, so joins have genuinely
    discontinuous ambience. A crossfade softens the transition; it does not
    create a continuous score. This is mitigation, not a fix.
    """
    if not clips:
        raise ValueError("no clips to concatenate")
    if len(clips) == 1:
        shutil.copy(clips[0], output)
        return Path(output)

    cmd = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-y"]
    for clip in clips:
        cmd += ["-i", str(clip)]

    parts: list[str] = []
    video_labels = "".join(f"[{i}:v]" for i in range(len(clips)))
    parts.append(f"{video_labels}concat=n={len(clips)}:v=1:a=0[v]")

    current = "0:a"
    for i in range(1, len(clips)):
        label = f"a{i}"
        parts.append(
            f"[{current}][{i}:a]acrossfade=d={crossfade_s}:c1=tri:c2=tri[{label}]"
        )
        current = label

    cmd += [
        "-filter_complex", ";".join(parts),
        "-map", "[v]", "-map", f"[{current}]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "384k",
        "-movflags", "+faststart",
        str(output),
    ]
    LOGGER.info("concatenating %d clips", len(clips))
    subprocess.run(cmd, check=True)
    return Path(output)
