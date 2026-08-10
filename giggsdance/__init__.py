"""Giggsdance -- a 60 fps, high-resolution rendering pipeline on top of MiniMax H3.

Powered by MiniMax H3 (https://huggingface.co/MiniMaxAI/MiniMax-H3), used under
the MiniMax H3 Community License Agreement. See NOTICE.md, and read the territory
restriction there before running anything: the licence limits use of the weights
*and their outputs* to outside the EU, UK, South Korea and USA.

What H3 gives you:      768p, 24 fps, 4-15 s, 32 kHz stereo audio
What this adds:          correct 24->60 fps conversion, super-resolution to
                         1080p/1440p/2160p, banding control, 10-bit output,
                         exact A/V sync, and clip chaining for longer pieces
"""

from __future__ import annotations

__version__ = "0.1.0"

from .constraints import (
    MAX_DURATION_S,
    MAX_FRAMES,
    MIN_DURATION_S,
    MIN_FRAMES,
    SRC_FPS,
    VALID_FRAME_COUNTS,
    Canvas,
    ClipSpec,
    ConstraintError,
    duration_for_frames,
    frames_for_duration,
    resolve_canvas,
    resolve_clip,
    snap_num_frames,
)
from .prompt import (
    DialogueLine,
    PromptMode,
    Shot,
    Storyboard,
    resolve_prompt,
)

__all__ = [
    "__version__",
    "SRC_FPS",
    "MIN_DURATION_S",
    "MAX_DURATION_S",
    "MIN_FRAMES",
    "MAX_FRAMES",
    "VALID_FRAME_COUNTS",
    "Canvas",
    "ClipSpec",
    "ConstraintError",
    "duration_for_frames",
    "frames_for_duration",
    "resolve_canvas",
    "resolve_clip",
    "snap_num_frames",
    "DialogueLine",
    "PromptMode",
    "Shot",
    "Storyboard",
    "resolve_prompt",
]
