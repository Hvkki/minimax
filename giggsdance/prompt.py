"""Building the structured prompt H3-Base actually wants.

H3 is a three-module system: Context-IR -> Base -> Regenerate-2K. Only Base is
open-weight. Context-IR is the hosted component that turns a loose human request
into the long, structured, timestamped description Base was trained to read, and
the model card is blunt that it "is critical to the quality of the final
output".

So feeding H3-Base a bare one-liner like "stickman fight" leaves most of the
model's quality on the table. This module gives you three ways to produce a
proper prompt, in increasing order of quality:

  RAW        - pass your text through untouched. Fine for quick tests.
  STORYBOARD - assemble the documented Context-IR layout locally from shots you
               describe. Free, offline, deterministic, and a very large step up
               from a one-liner.
  API        - call the official Context-IR endpoint. Best quality, needs a
               MiniMax API key, and it is what the reference outputs were made
               with.

The STORYBOARD layout mirrors the structure published in the model card's
reproducible cases: an ``integrated_multimodal_description`` containing
``[Shot N]`` blocks with absolute timecodes, then ``overall_soundscape``, then
``non_diegetic_music``. Dialogue is wrapped in the ``<d>`` special token that H3
adds to its tokenizer, tagged with a language.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

LOGGER = logging.getLogger(__name__)

DEFAULT_LANGUAGE = "English"

SUPPORTED_DIALOGUE_LANGUAGES = (
    "Arabic", "Chinese", "English", "French", "German", "Italian",
    "Japanese", "Korean", "Portuguese", "Russian", "Spanish",
)
"""The 11 languages the model card lists as having stable dialogue support."""


class PromptMode(str, Enum):
    RAW = "raw"
    STORYBOARD = "storyboard"
    API = "api"


def format_timecode(seconds: float) -> str:
    """Format seconds as ``MM:SS.mmm``, matching the model card's timestamps."""
    if seconds < 0:
        raise ValueError("timecode cannot be negative")
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:06.3f}"


@dataclass
class DialogueLine:
    """One spoken line.

    Rendered as ``<Speaker> (S1) says, <d>[English] text</d>`` style content.
    The ``<d>`` wrapper is a real special token in H3's tokenizer, which is why
    the repo's own tokenizer files are required.
    """

    text: str
    language: str = DEFAULT_LANGUAGE
    speaker: str | None = None

    def __post_init__(self):
        if self.language not in SUPPORTED_DIALOGUE_LANGUAGES:
            LOGGER.warning(
                "language %r is outside the 11 languages with documented stable "
                "support; quality may vary", self.language,
            )

    def render(self) -> str:
        tagged = f"<d>[{self.language}] {self.text.strip()}</d>"
        if self.speaker:
            return f"{self.speaker} speaks, {tagged}"
        return tagged


@dataclass
class Shot:
    """One shot in the storyboard."""

    description: str
    start_s: float | None = None
    dialogue: list[DialogueLine] = field(default_factory=list)
    camera: str | None = None

    def render(self, index: int) -> str:
        parts = [f"[Shot {index}]"]
        if self.start_s is not None and self.start_s > 0:
            parts.append(f"At {format_timecode(self.start_s)},")
        if self.camera:
            parts.append(f"{self.camera.strip().rstrip('.')}.")
        parts.append(self.description.strip())
        body = " ".join(parts)
        for line in self.dialogue:
            body += " " + line.render()
        return body


@dataclass
class Storyboard:
    """A structured description that renders to the Context-IR layout.

    ``soundscape`` is diegetic sound (what exists in the scene) and ``music`` is
    non-diegetic score (what the audience hears but the characters do not). H3
    keeps them in separate fields, and filling both in is one of the easiest
    quality wins available -- leaving them empty makes the model invent a
    soundtrack with no direction.
    """

    shots: list[Shot]
    soundscape: str | None = None
    music: str | None = None
    style: str | None = None

    def __post_init__(self):
        if not self.shots:
            raise ValueError("a storyboard needs at least one shot")

    def render(self) -> str:
        shots = " ".join(shot.render(i + 1) for i, shot in enumerate(self.shots))
        if self.style:
            shots = f"{self.style.strip().rstrip('.')}. {shots}"

        sections = [f"integrated_multimodal_description: {shots}"]
        if self.soundscape:
            sections.append(f"overall_soundscape: {self.soundscape.strip()}")
        if self.music:
            sections.append(f"non_diegetic_music: {self.music.strip()}")
        return "\n".join(sections)


def storyboard_from_dicts(data: Sequence[dict] | dict) -> Storyboard:
    """Build a Storyboard from plain JSON-ish data (what the API receives)."""
    if isinstance(data, dict):
        shots_data = data.get("shots") or []
        style = data.get("style")
        soundscape = data.get("soundscape")
        music = data.get("music")
    else:
        shots_data, style, soundscape, music = data, None, None, None

    shots = []
    for raw in shots_data:
        if isinstance(raw, str):
            shots.append(Shot(description=raw))
            continue
        dialogue = [
            DialogueLine(
                text=line["text"] if isinstance(line, dict) else str(line),
                language=(line.get("language", DEFAULT_LANGUAGE)
                          if isinstance(line, dict) else DEFAULT_LANGUAGE),
                speaker=line.get("speaker") if isinstance(line, dict) else None,
            )
            for line in raw.get("dialogue", []) or []
        ]
        shots.append(
            Shot(
                description=raw["description"],
                start_s=raw.get("start_s"),
                dialogue=dialogue,
                camera=raw.get("camera"),
            )
        )
    return Storyboard(shots=shots, soundscape=soundscape, music=music, style=style)


# --------------------------------------------------------------------------
# Official Context-IR API
# --------------------------------------------------------------------------

CONTEXT_IR_PATH = "/v1/video_generation_v2/h3_context_ir"


class ContextIRError(RuntimeError):
    pass


class ContextIRClient:
    """Thin client for MiniMax's hosted Context-IR endpoint.

    Optional. When configured, this is the highest-quality prompt path because
    it is the same component that produced the reference outputs on the model
    card. Without it, use :class:`Storyboard`.

    The exact request/response schema is versioned by MiniMax; the response is
    searched for the enhanced prompt rather than assuming one fixed key, so a
    minor schema change degrades to a clear error instead of a wrong prompt.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str = "https://api.minimax.io",
        timeout: float = 120.0,
    ):
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY")
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def enhance(
        self,
        prompt: str,
        duration_s: int = 10,
        aspect_ratio: str = "16:9",
        poll_interval: float = 2.0,
        max_wait: float = 300.0,
    ) -> str:
        if not self.configured:
            raise ContextIRError(
                "no MiniMax API key. Set MINIMAX_API_KEY, or use "
                "PromptMode.STORYBOARD / PromptMode.RAW instead."
            )

        payload = {
            "model": "MiniMax-H3",
            "prompt": prompt,
            "duration": int(duration_s),
            "ratio": aspect_ratio,
        }
        task = self._post(CONTEXT_IR_PATH, payload)
        enhanced = self._extract_prompt(task)
        if enhanced:
            return enhanced

        task_id = self._extract_task_id(task)
        if not task_id:
            raise ContextIRError(
                f"Context-IR response had neither a prompt nor a task id: "
                f"{json.dumps(task)[:400]}"
            )

        deadline = time.time() + max_wait
        while time.time() < deadline:
            time.sleep(poll_interval)
            status = self._get(f"{CONTEXT_IR_PATH}/{task_id}")
            state = str(self._find(status, "status") or "").lower()
            if state in ("succeeded", "success", "completed"):
                enhanced = self._extract_prompt(status)
                if enhanced:
                    return enhanced
                raise ContextIRError("Context-IR succeeded but returned no prompt")
            if state in ("failed", "error"):
                raise ContextIRError(f"Context-IR failed: {json.dumps(status)[:400]}")
        raise ContextIRError(f"Context-IR timed out after {max_wait}s")

    # -- plumbing ---------------------------------------------------------

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.api_base}{path}",
            data=json.dumps(payload).encode(),
            headers=self._headers(),
            method="POST",
        )
        return self._send(request)

    def _get(self, path: str) -> dict:
        request = urllib.request.Request(
            f"{self.api_base}{path}", headers=self._headers(), method="GET"
        )
        return self._send(request)

    def _send(self, request) -> dict:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:400]
            raise ContextIRError(f"HTTP {exc.code} from Context-IR: {body}") from exc
        except urllib.error.URLError as exc:
            raise ContextIRError(f"could not reach Context-IR: {exc.reason}") from exc

    @staticmethod
    def _find(payload, key: str):
        """Depth-first search for a key, so schema nesting changes don't break us."""
        stack = [payload]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                if key in node:
                    return node[key]
                stack.extend(node.values())
            elif isinstance(node, (list, tuple)):
                stack.extend(node)
        return None

    def _extract_prompt(self, payload) -> str | None:
        content = self._find(payload, "content")
        if isinstance(content, dict) and isinstance(content.get("prompt"), str):
            return content["prompt"]
        prompt = self._find(payload, "prompt")
        return prompt if isinstance(prompt, str) and prompt.strip() else None

    def _extract_task_id(self, payload) -> str | None:
        for key in ("id", "task_id"):
            value = self._find(payload, key)
            if isinstance(value, str) and value:
                return value
        return None


def resolve_prompt(
    prompt: str | None = None,
    storyboard: Storyboard | dict | None = None,
    mode: PromptMode = PromptMode.RAW,
    duration_s: int = 10,
    aspect_ratio: str = "16:9",
    context_ir: ContextIRClient | None = None,
) -> tuple[str, str]:
    """Produce the final prompt string. Returns ``(prompt, mode_actually_used)``.

    Falls back gracefully: an API request that fails drops to the storyboard if
    one was supplied, then to the raw text, and says so in the returned mode
    rather than silently changing quality.
    """
    board: Storyboard | None = None
    if storyboard is not None:
        board = (
            storyboard if isinstance(storyboard, Storyboard)
            else storyboard_from_dicts(storyboard)
        )

    if mode == PromptMode.API:
        client = context_ir or ContextIRClient()
        source = prompt or (board.render() if board else None)
        if not source:
            raise ValueError("API prompt mode needs prompt text or a storyboard")
        try:
            return client.enhance(source, duration_s, aspect_ratio), "api"
        except ContextIRError as exc:
            LOGGER.warning("Context-IR unavailable (%s); falling back", exc)
            if board:
                return board.render(), "storyboard(api-fallback)"
            return source, "raw(api-fallback)"

    if mode == PromptMode.STORYBOARD:
        if board is None:
            raise ValueError("storyboard mode needs a storyboard")
        return board.render(), "storyboard"

    if not prompt:
        raise ValueError("raw mode needs prompt text")
    return prompt, "raw"
