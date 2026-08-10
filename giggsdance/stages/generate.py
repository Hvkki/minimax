"""H3-Base generation via the diffusers modular pipeline.

IMPORTANT, READ BEFORE TRUSTING THIS FILE
-----------------------------------------
Everything here is written against the published diffusers integration for
MiniMax-H3 and the model card, but it has **not** been executed against real
weights -- that needs ~124 GB of checkpoint and a large GPU. The pure geometry,
timing and encoding stages in this repo are unit-tested and verified; this stage
is the one to check first if something misbehaves. ``pipe.doc`` prints the exact
signature the installed diffusers version expects, and ``preflight.py`` will
compare it against what we send.

Memory, which is the thing that surprises people
------------------------------------------------
For a single workflow in bfloat16:

    transformer partition   61.7 GB
    Qwen3-VL conditioner    62.1 GB
    -------------------------------
    ~124 GB before activations

That is why ``load_components()`` is always called with an explicit
``workflow=``: leaving it out pulls *both* transformer partitions (another
61.7 GB) so one pipeline can serve all three tasks.

Chaining strategy
-----------------
Two ways to continue a clip, with very different costs:

  FL2VA  (default) - feed the previous clip's last frame in as the next clip's
                     first frame. Reuses the *same* transformer partition, so no
                     extra weights. Gives frame-exact visual continuity at the
                     cut. Does not carry audio or identity beyond that frame.

  REF2VA (better)  - pass the whole previous clip back as a video reference,
                     which the diffusers docs describe explicitly ("A generation
                     as a reference"). Much stronger identity, style and motion
                     continuity because the model sees the clip, not one frame.
                     Costs the second transformer partition (+61.7 GB).

Neither makes the soundtrack continuous. See ``pipeline.render_chain``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..constraints import Canvas, ConstraintError, snap_num_frames

LOGGER = logging.getLogger(__name__)

MODEL_ID = "MiniMaxAI/MiniMax-H3"

TRANSFORMER_GB = 61.7
CONDITIONER_GB = 62.1


class Workflow(str, Enum):
    T2VA = "t2va"      # text only
    FL2VA = "fl2va"    # first and/or last keyframe
    REF2VA = "ref2va"  # ordered multimodal references


class LoadStrategy(str, Enum):
    AUTO = "auto"
    FULL = "full"                  # everything resident, needs ~2x80GB
    OFFLOAD = "offload"            # 1x80GB, ComponentsManager auto CPU offload
    INT8_OFFLOAD = "int8_offload"  # 24-32GB consumer card, int8 + group offload
    DUAL = "dual"                  # conditioner on cuda:1, rest on cuda:0


@dataclass
class GenerationResult:
    """One generated clip: frames plus its jointly generated soundtrack."""

    frames: np.ndarray          # (F, H, W, 3) float32 in [0, 1]
    audio: np.ndarray           # (channels, samples) float32
    sample_rate: int
    prompt: str
    seed: int
    canvas: Canvas
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def duration_s(self) -> float:
        return self.num_frames / 24.0

    @property
    def last_frame(self) -> np.ndarray:
        return self.frames[-1]


def frames_to_array(videos: Any) -> np.ndarray:
    """Normalise whatever diffusers handed back into (F, H, W, 3) float32 [0,1].

    The pipeline can return a list of PIL images, a numpy array, or a torch
    tensor, and tensors may be channels-first. Rather than assume one, detect
    and convert -- an axis mix-up here shows up as a silently transposed video.
    """
    if videos is None:
        raise ValueError("pipeline returned no video")

    # torch tensor -> numpy
    if hasattr(videos, "detach") and hasattr(videos, "cpu"):
        videos = videos.detach().float().cpu().numpy()

    # list of PIL images (or list of arrays)
    if isinstance(videos, (list, tuple)):
        if len(videos) == 0:
            raise ValueError("pipeline returned an empty frame list")
        first = videos[0]
        if hasattr(first, "detach"):
            videos = np.stack([f.detach().float().cpu().numpy() for f in videos])
        elif hasattr(first, "convert"):  # PIL
            videos = np.stack([np.asarray(f.convert("RGB")) for f in videos])
        else:
            videos = np.stack([np.asarray(f) for f in videos])

    array = np.asarray(videos)

    # Drop a leading batch axis if present: (1, F, ...) -> (F, ...)
    if array.ndim == 5:
        array = array[0]
    if array.ndim != 4:
        raise ValueError(f"expected 4 dims after unbatching, got shape {array.shape}")

    # Channels-first (F, 3, H, W) -> channels-last
    if array.shape[1] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (0, 2, 3, 1))
    if array.shape[-1] != 3:
        raise ValueError(f"could not find a 3-channel axis in shape {array.shape}")

    array = array.astype(np.float32, copy=False)
    if array.max() > 1.5:  # uint8-style range
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def audio_to_array(audio: Any) -> np.ndarray:
    """Normalise the soundtrack to (channels, samples) float32."""
    if audio is None:
        raise ValueError("pipeline returned no audio")
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    array = np.asarray(audio, dtype=np.float32)
    while array.ndim > 2:
        array = array[0]
    if array.ndim == 1:
        array = array[None, :]
    if array.shape[0] > array.shape[1]:  # (samples, channels) -> (channels, samples)
        array = array.T
    return array


class H3Generator:
    """Loads H3-Base once and generates clips from it."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        workflow: Workflow | str = Workflow.T2VA,
        strategy: LoadStrategy | str = LoadStrategy.AUTO,
        device: str = "cuda",
        num_inference_steps: int = 24,
        offload_margin: str = "12GB",
        attention_backend: str | None = None,
        local_files_only: bool = False,
    ):
        self.model_id = model_id
        self.workflow = Workflow(workflow)
        self.strategy = LoadStrategy(strategy)
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.offload_margin = offload_margin
        self.attention_backend = attention_backend
        self.local_files_only = local_files_only

        self._pipe = None
        self._manager = None
        self._loaded_workflows: set[str] = set()
        self._torch = None

    # -- capability probing ------------------------------------------------

    def _probe(self) -> tuple[int, float]:
        import torch

        if not torch.cuda.is_available():
            return 0, 0.0
        count = torch.cuda.device_count()
        smallest = min(
            torch.cuda.get_device_properties(i).total_memory / 1024**3
            for i in range(count)
        )
        return count, smallest

    def resolve_strategy(self) -> LoadStrategy:
        """Pick a loading recipe from the hardware actually present."""
        if self.strategy != LoadStrategy.AUTO:
            return self.strategy

        count, vram = self._probe()
        if count == 0:
            raise RuntimeError(
                "no CUDA device found. H3-Base needs a GPU; there is no usable "
                "CPU inference path for a 33B video transformer."
            )
        if count >= 2 and vram >= 78:
            chosen = LoadStrategy.FULL
        elif count >= 2 and vram >= 46:
            chosen = LoadStrategy.DUAL
        elif vram >= 78:
            chosen = LoadStrategy.OFFLOAD
        else:
            chosen = LoadStrategy.INT8_OFFLOAD
        LOGGER.info(
            "auto-selected %s for %d GPU(s) with %.0f GB each", chosen.value, count, vram
        )
        return chosen

    # -- loading ----------------------------------------------------------

    def load(self, workflow: Workflow | str | None = None):
        """Load components for one workflow. Idempotent per workflow."""
        target = Workflow(workflow) if workflow else self.workflow
        if self._pipe is not None and target.value in self._loaded_workflows:
            return self._pipe

        import torch
        from diffusers import ComponentsManager, ModularPipeline

        self._torch = torch
        strategy = self.resolve_strategy()

        if self._pipe is None:
            LOGGER.info(
                "loading %s (workflow=%s, strategy=%s) -- expect ~%.0f GB of weights",
                self.model_id, target.value, strategy.value,
                TRANSFORMER_GB + CONDITIONER_GB,
            )
            if strategy == LoadStrategy.INT8_OFFLOAD:
                self._pipe = self._build_int8_pipeline(target)
            else:
                self._manager = ComponentsManager()
                self._pipe = ModularPipeline.from_pretrained(
                    self.model_id, components_manager=self._manager
                )
                self._pipe.load_components(
                    workflow=target.value, dtype=torch.bfloat16
                )
                if strategy in (LoadStrategy.OFFLOAD, LoadStrategy.DUAL):
                    self._manager.enable_auto_cpu_offload(
                        device=self.device, memory_reserve_margin=self.offload_margin
                    )
                elif strategy == LoadStrategy.FULL:
                    self._manager.enable_auto_cpu_offload(device=self.device)
        else:
            # Second workflow on an existing pipeline (e.g. adding ref2va).
            LOGGER.info("loading additional workflow %s (+%.0f GB)",
                        target.value, TRANSFORMER_GB)
            self._pipe.load_components(
                workflow=target.value, dtype=self._torch.bfloat16
            )

        if self.attention_backend:
            try:
                self._pipe.transformer.set_attention_backend(self.attention_backend)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("could not set attention backend %r: %s",
                               self.attention_backend, exc)

        self._loaded_workflows.add(target.value)
        return self._pipe

    def _build_int8_pipeline(self, workflow: Workflow):
        """Consumer-card recipe: int8 weights plus block-level group offloading.

        Expect ~75 GB of *host* RAM to be in use; the weights live in system
        memory and stream onto the card a block at a time.
        """
        import torch
        from diffusers import (
            MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig,
        )
        from diffusers.hooks import apply_group_offloading
        from torchao.quantization import Int8WeightOnlyConfig
        from transformers import Qwen3VLForConditionalGeneration
        from transformers import TorchAoConfig as TransformersTorchAoConfig

        pipe = ModularPipeline.from_pretrained(self.model_id)
        pipe.update_components(
            transformer=MiniMaxH3Transformer3DModel.from_pretrained(
                self.model_id, subfolder="transformer", dtype=torch.bfloat16,
                quantization_config=TorchAoConfig(
                    Int8WeightOnlyConfig(version=2),
                    modules_to_not_convert=[
                        "proj_in", "audio_proj_in", "context_embedder",
                        "time_embedder", "time_proj", "token_refiner",
                        "norm_out", "proj_out", "audio_proj_out",
                    ],
                ),
                low_cpu_mem_usage=False,
            ),
            text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id, subfolder="text_encoder", dtype=torch.bfloat16,
                quantization_config=TransformersTorchAoConfig(
                    Int8WeightOnlyConfig(version=2),
                    modules_to_not_convert=[
                        "model.visual", "model.language_model.embed_tokens",
                        "model.language_model.norm", "lm_head",
                    ],
                ),
            ),
        )
        pipe.load_components(workflow=workflow.value, dtype=torch.bfloat16)

        pipe.transformer.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)

        offload = dict(
            onload_device=torch.device(self.device),
            offload_device=torch.device("cpu"),
            use_stream=True,
        )
        pipe.transformer.enable_group_offload(
            offload_type="block_level", num_blocks_per_group=1, **offload
        )
        apply_group_offloading(pipe.text_encoder.model, offload_type="leaf_level", **offload)
        pipe.vae.to(self.device)
        pipe.audio_vae.to(self.device)
        return pipe

    def describe(self) -> str:
        """``pipe.doc`` -- the authoritative signature for the installed version."""
        pipe = self.load()
        return str(getattr(pipe, "doc", "<no doc available>"))

    # -- generation -------------------------------------------------------

    def generate(
        self,
        prompt: str,
        num_frames: int,
        canvas: Canvas | None = None,
        seed: int = 0,
        image: Any = None,
        last_image: Any = None,
        references: Sequence[Any] | None = None,
        num_inference_steps: int | None = None,
    ) -> GenerationResult:
        """Generate one clip with its soundtrack.

        ``num_frames`` is validated against H3's ``17n + 5`` rule up front so a
        bad value fails immediately instead of after a long load.
        """
        snapped = snap_num_frames(num_frames)
        if snapped != num_frames:
            LOGGER.info("snapped num_frames %d -> %d (17n+5)", num_frames, snapped)

        if references:
            workflow = Workflow.REF2VA
        elif image is not None or last_image is not None:
            workflow = Workflow.FL2VA
        else:
            workflow = Workflow.T2VA

        pipe = self.load(workflow)
        torch = self._torch

        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "num_frames": snapped,
            "num_inference_steps": num_inference_steps or self.num_inference_steps,
            "generator": torch.Generator().manual_seed(int(seed)),
            "output": ["videos", "audio", "sampling_rate"],
            "output_type": "pt",
        }
        if canvas is not None:
            kwargs["height"] = canvas.height
            kwargs["width"] = canvas.width
        if image is not None:
            kwargs["image"] = image
        if last_image is not None:
            kwargs["last_image"] = last_image
        if references:
            kwargs["references"] = list(references)

        LOGGER.info(
            "generating %d frames (%.3fs) at %s via %s, seed=%d, steps=%d",
            snapped, snapped / 24.0, canvas or "model default",
            workflow.value, seed, kwargs["num_inference_steps"],
        )
        results = pipe(**kwargs)

        frames = frames_to_array(results["videos"][0])
        audio = audio_to_array(results["audio"][0] if isinstance(
            results["audio"], (list, tuple)) else results["audio"])
        sample_rate = int(results["sampling_rate"])

        if frames.shape[0] != snapped:
            LOGGER.warning(
                "asked for %d frames, pipeline returned %d", snapped, frames.shape[0]
            )

        height, width = frames.shape[1], frames.shape[2]
        return GenerationResult(
            frames=frames,
            audio=audio,
            sample_rate=sample_rate,
            prompt=prompt,
            seed=seed,
            canvas=canvas or Canvas(width=width, height=height),
            metadata={
                "workflow": workflow.value,
                "num_inference_steps": kwargs["num_inference_steps"],
                "model_id": self.model_id,
            },
        )

    def continue_from(
        self,
        previous: GenerationResult,
        prompt: str,
        num_frames: int,
        seed: int = 0,
        mode: str = "fl2va",
        anchor_image: Any = None,
        num_inference_steps: int | None = None,
    ) -> GenerationResult:
        """Generate the next clip in a chain.

        ``mode="fl2va"`` conditions on the previous clip's final frame (cheap,
        one transformer partition). ``mode="ref2va"`` passes the entire previous
        clip back as a video reference (much better continuity, needs the second
        partition loaded).

        ``anchor_image`` is an optional fixed reference -- a character sheet, say
        -- included on every link of the chain to slow identity drift.
        """
        if mode == "ref2va":
            from diffusers.modular_pipelines.minimax_h3 import (
                MiniMaxH3ImageReference, MiniMaxH3VideoReference,
            )

            torch = self._torch or __import__("torch")
            references: list[Any] = [
                MiniMaxH3VideoReference(
                    frames=torch.from_numpy(previous.frames),
                    audio=torch.from_numpy(previous.audio),
                    sample_rate=previous.sample_rate,
                )
            ]
            if anchor_image is not None:
                references.append(MiniMaxH3ImageReference(image=anchor_image))
            return self.generate(
                prompt=prompt, num_frames=num_frames, canvas=previous.canvas,
                seed=seed, references=references,
                num_inference_steps=num_inference_steps,
            )

        if mode != "fl2va":
            raise ConstraintError(f"unknown chain mode {mode!r}; use fl2va or ref2va")

        from PIL import Image

        last = (previous.last_frame * 255.0 + 0.5).astype(np.uint8)
        return self.generate(
            prompt=prompt, num_frames=num_frames, canvas=previous.canvas,
            seed=seed, image=Image.fromarray(last),
            num_inference_steps=num_inference_steps,
        )

    def unload(self) -> None:
        """Release the model and empty the CUDA cache."""
        self._pipe = None
        self._manager = None
        self._loaded_workflows.clear()
        if self._torch is not None:
            try:
                self._torch.cuda.empty_cache()
            except Exception:  # pragma: no cover
                pass
