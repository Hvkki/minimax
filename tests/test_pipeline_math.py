"""Unit tests for the parts that must be exactly right.

These cover the pure functions -- H3's frame/canvas constraints, the 24->60 fps
timing, and the crop/scale geometry. No GPU, no weights, no ffmpeg needed, so they
run anywhere in about a second.

Run: python -m pytest tests/test_pipeline_math.py -v
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from giggsdance.constraints import (  # noqa: E402
    CANVAS_MAX_PIXELS,
    MAX_FRAMES,
    SIZE_MULTIPLE,
    VALID_FRAME_COUNTS,
    ConstraintError,
    duration_for_frames,
    frames_for_duration,
    parse_aspect_ratio,
    resolve_canvas,
    snap_num_frames,
)
from giggsdance.stages.interpolate import plan_interpolation  # noqa: E402
from giggsdance.stages.upscale import (  # noqa: E402
    RESOLUTIONS,
    blend_tiles,
    pick_scale,
    plan_geometry,
    plan_tiles,
    target_for_source,
)

# ---------------------------------------------------------------------------
# H3 frame-count rule: num_frames must be 17n + 5, duration in [5, 15] seconds
# ---------------------------------------------------------------------------


def test_every_valid_count_satisfies_the_quantum_rule():
    for count in VALID_FRAME_COUNTS:
        assert (count - 5) % 17 == 0, f"{count} is not 17n+5"
        assert 5.0 <= count / 24 <= 15.0


def test_valid_counts_are_the_expected_set():
    assert VALID_FRAME_COUNTS == (
        124, 141, 158, 175, 192, 209, 226, 243,
        260, 277, 294, 311, 328, 345,
    )
    assert MAX_FRAMES == 345
    assert duration_for_frames(345) == pytest.approx(14.375)


def test_snap_rounds_up_to_the_next_decodable_count():
    assert snap_num_frames(124) == 124
    assert snap_num_frames(125) == 141
    assert snap_num_frames(200) == 209


def test_exactly_fifteen_seconds_clamps_instead_of_raising():
    """H3 advertises 15s; 17n+5 never lands on 360, so clamp rather than fail."""
    assert frames_for_duration(15.0) == 345
    assert frames_for_duration(14.9) == 345


def test_out_of_range_durations_raise():
    with pytest.raises(ConstraintError):
        frames_for_duration(20.0)
    with pytest.raises(ConstraintError):
        frames_for_duration(2.0)


# ---------------------------------------------------------------------------
# Canvas: multiples of 32, under the pixel cap, and 16:9 must be H3's own canvas
# ---------------------------------------------------------------------------


def test_sixteen_by_nine_is_the_trained_canvas():
    """1344x768, not the naive 1376x768 -- that would breach the pixel cap."""
    canvas = resolve_canvas("16:9")
    assert (canvas.width, canvas.height) == (1344, 768)
    assert canvas.pixels == CANVAS_MAX_PIXELS


@pytest.mark.parametrize("ratio", ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "2.39:1"])
def test_canvas_always_legal(ratio):
    canvas = resolve_canvas(ratio)
    assert canvas.width % SIZE_MULTIPLE == 0
    assert canvas.height % SIZE_MULTIPLE == 0
    assert canvas.pixels <= CANVAS_MAX_PIXELS
    target = float(parse_aspect_ratio(ratio))
    assert abs(canvas.aspect - target) / target < 0.03


def test_bad_aspect_ratio_raises():
    with pytest.raises(ConstraintError):
        resolve_canvas("banana")


# ---------------------------------------------------------------------------
# 24 -> 60 fps timing
# ---------------------------------------------------------------------------


def test_frames_are_uniformly_spaced_in_time():
    """The whole point: one gap value, not the irregular grid 2x-then-drop gives."""
    plan = plan_interpolation(345, 24.0, 60.0)
    times = [t.index / 60.0 for t in plan.timings]
    gaps = {round(times[i + 1] - times[i], 10) for i in range(len(times) - 1)}
    assert len(gaps) == 1
    assert gaps.pop() == pytest.approx(1 / 60)


def test_fraction_cycle_is_the_expected_five_step_pattern():
    plan = plan_interpolation(345, 24.0, 60.0)
    histogram = Counter(round(t.t, 4) for t in plan.timings)
    assert set(histogram) == {0.0, 0.2, 0.4, 0.6, 0.8}
    # Only every 5th output frame is an exact copy of a source frame.
    assert histogram[0.0] == 175
    for fraction in (0.2, 0.4, 0.6, 0.8):
        assert histogram[fraction] == 172


def test_video_covers_the_audio_and_never_precedes_it():
    plan = plan_interpolation(345, 24.0, 60.0)
    assert plan.dst_duration_s >= plan.src_duration_s
    # Overshoot stays under one output frame; the muxer trims it.
    assert plan.dst_duration_s - plan.src_duration_s < 1 / 60


def test_source_position_is_monotonic_and_in_bounds():
    plan = plan_interpolation(124, 24.0, 60.0)
    positions = [t.left + t.t for t in plan.timings]
    assert all(b >= a for a, b in zip(positions, positions[1:]))
    assert max(positions) <= 123.0
    assert all(0 <= t.left <= 123 for t in plan.timings)


def test_same_fps_is_a_passthrough():
    plan = plan_interpolation(124, 24.0, 24.0)
    assert plan.num_dst_frames == 124
    assert all(t.is_copy for t in plan.timings)


def test_single_frame_is_handled():
    assert plan_interpolation(1).num_dst_frames == 1


def test_rejects_empty_input():
    with pytest.raises(ValueError):
        plan_interpolation(0)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_crop_hits_the_output_aspect_exactly_without_stretching():
    """1344x768 is 1.75:1; UHD is 1.7778:1. Crop 12 rows, never stretch."""
    geo = plan_geometry(1344, 768, 3840, 2160, model_scale=4)
    assert (geo.crop_width, geo.crop_height) == (1344, 756)
    assert geo.crop_width / geo.crop_height == pytest.approx(3840 / 2160, abs=1e-4)
    assert geo.cropped_rows == 12
    assert geo.is_supersampled  # 4x overshoots 2160, then Lanczos brings it down


def test_pad_mode_keeps_the_whole_frame():
    geo = plan_geometry(1344, 768, 3840, 2160, model_scale=4, fit="pad")
    assert (geo.crop_width, geo.crop_height) == (1344, 768)


def test_portrait_source_gets_a_portrait_target():
    """Without this a 9:16 generation would be cropped to a sliver of itself."""
    assert target_for_source(768, 1344) == (2160, 3840)
    assert target_for_source(1344, 768) == (3840, 2160)
    geo = plan_geometry(768, 1344, *target_for_source(768, 1344), model_scale=4)
    kept = (geo.crop_width * geo.crop_height) / (768 * 1344)
    assert kept > 0.95


def test_scale_selection_avoids_pointless_super_resolution():
    """768p -> 720p is a downscale; running a model first would be wasted money."""
    assert pick_scale(756, 720) == 1
    assert pick_scale(756, 1080) == 2
    assert pick_scale(756, 1440) == 2
    assert pick_scale(756, 2160) == 4


@pytest.mark.parametrize("name", list(RESOLUTIONS))
def test_every_resolution_produces_exact_output_dimensions(name):
    out_w, out_h = RESOLUTIONS[name]
    crop_h = plan_geometry(1344, 768, out_w, out_h, 2).crop_height
    geo = plan_geometry(1344, 768, out_w, out_h, max(1, pick_scale(crop_h, out_h)))
    assert (geo.out_width, geo.out_height) == (out_w, out_h)


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "width,height,tile,overlap",
    [(1344, 756, 384, 32), (1344, 756, 512, 64), (3840, 2160, 512, 64), (64, 64, 32, 8)],
)
def test_tiles_cover_the_frame_with_uniform_stride(width, height, tile, overlap):
    import numpy as np

    grid = plan_tiles(width, height, tile, overlap)
    coverage = np.zeros((height, width), dtype=int)
    for piece in grid:
        coverage[piece.y:piece.y + piece.height, piece.x:piece.x + piece.width] += 1
    assert coverage.min() >= 1, "a region of the frame is never upscaled"

    xs = sorted({piece.x for piece in grid})
    if len(xs) > 1:
        strides = {b - a for a, b in zip(xs, xs[1:])}
        assert len(strides) == 1, f"non-uniform stride {strides} leaves a visible seam"
    assert grid.overlap_x >= 0 and grid.overlap_y >= 0


def test_blending_a_flat_field_stays_flat():
    """A seam shows up as a deviation here, and would crawl through the clip."""
    import numpy as np

    grid = plan_tiles(1344, 756, 384, 32)
    scale = 2
    patches = [
        (piece, np.full((piece.height * scale, piece.width * scale, 3), 0.5, np.float32))
        for piece in grid
    ]
    blended = blend_tiles(patches, 1344, 756, scale, grid)
    assert np.abs(blended - 0.5).max() < 1e-5


def test_oversized_tile_degrades_to_one_tile():
    grid = plan_tiles(200, 100, 384, 32)
    assert len(grid) == 1
    assert grid.overlap_x == 0 and grid.overlap_y == 0


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        plan_tiles(512, 512, 128, 128)



# ---------------------------------------------------------------------------
# resolve_target: "native" means no super-resolution runs at all
# ---------------------------------------------------------------------------


def test_native_crops_h3s_canvas_to_exact_sixteen_by_nine():
    """1344x768 is 1.75:1, so players letterbox it. A crop is free; a scale is not."""
    from giggsdance.stages.upscale import resolve_target

    assert resolve_target("native", 1344, 768) == (1344, 756)
    width, height = resolve_target("native", 1344, 768)
    assert width / height == pytest.approx(16 / 9, abs=1e-4)


def test_native_leaves_far_from_widescreen_canvases_alone():
    """Cropping a square to 16:9 would discard 44% of the frame."""
    from giggsdance.stages.upscale import resolve_target

    assert resolve_target("native", 768, 768) == (768, 768)
    assert resolve_target("native", 1504, 640) == (1504, 640)
    assert resolve_target("native", 1024, 768) == (1024, 768)


def test_native_handles_portrait():
    from giggsdance.stages.upscale import resolve_target

    assert resolve_target("native", 768, 1344) == (756, 1344)


def test_named_resolutions_follow_canvas_orientation():
    from giggsdance.stages.upscale import resolve_target

    assert resolve_target("1440p", 1344, 768) == (2560, 1440)
    assert resolve_target("1440p", 768, 1344) == (1440, 2560)


def test_unknown_resolution_name_raises():
    from giggsdance.stages.upscale import resolve_target

    with pytest.raises(ValueError, match="unknown resolution"):
        resolve_target("4k", 1344, 768)


def test_native_never_triggers_a_super_resolution_pass():
    """The whole cost saving depends on pick_scale returning 1 for native."""
    from giggsdance.stages.upscale import resolve_target

    out_w, out_h = resolve_target("native", 1344, 768)
    crop_h = plan_geometry(1344, 768, out_w, out_h, 2).crop_height
    assert pick_scale(crop_h, out_h) == 1



# ---------------------------------------------------------------------------
# Multimodal references (ref2va): 9 images / 3 videos / 3 audio / 12 total
# ---------------------------------------------------------------------------

from giggsdance.references import (  # noqa: E402
    MAX_TOTAL,
    ReferenceError,
    ReferenceSet,
    build_reference_set,
    check_clip_durations,
    classify,
)


def test_the_documented_maximum_set_is_accepted():
    """9 + 2 + 1 = 12, the largest legal combination."""
    refs = build_reference_set(
        images=[f"i{n}.png" for n in range(9)],
        videos=["a.mp4", "b.mp4"],
        audios=["v.wav"],
    )
    assert refs.total == MAX_TOTAL == 12
    assert refs.workflow == "ref2va"


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"images": ["x.png"] * 10}, "at most 9"),
        ({"videos": ["v.mp4"] * 4}, "at most 3"),
        ({"images": ["i.png"], "audios": ["a.wav"] * 4}, "at most 3"),
        ({"images": ["i.png"] * 9, "videos": ["v.mp4"] * 3, "audios": ["a.wav"] * 3},
         "at most 12"),
    ],
)
def test_over_limit_sets_are_rejected(kwargs, reason):
    with pytest.raises(ReferenceError, match=reason):
        build_reference_set(**kwargs)


def test_audio_alone_is_rejected():
    """Audio never reaches the text encoder, so it has nothing to attach to."""
    with pytest.raises(ReferenceError, match="cannot be the only input"):
        build_reference_set(audios=["voice.wav"])
    # Paired with an image it is fine.
    assert build_reference_set(images=["face.png"], audios=["voice.wav"]).total == 2


def test_workflow_is_inferred_and_keyframes_stay_on_the_cheap_partition():
    """1-2 bare images are keyframes (fl2va, transformer/). More needs ref2va."""
    assert build_reference_set().workflow == "t2va"
    assert build_reference_set(images=["a.png"]).workflow == "fl2va"
    assert build_reference_set(images=["a.png", "b.png"]).workflow == "fl2va"
    assert build_reference_set(images=["a.png", "b.png", "c.png"]).workflow == "ref2va"
    assert build_reference_set(videos=["v.mp4"]).workflow == "ref2va"


def test_mixed_list_preserves_order_because_order_is_semantic():
    refs = build_reference_set(mixed=["b.mp4", "a.png", "s.wav", "c.png"])
    assert refs.order == [
        ("video", "b.mp4"), ("image", "a.png"),
        ("audio", "s.wav"), ("image", "c.png"),
    ]


def test_modality_is_classified_from_extension():
    assert classify("x.PNG") == "image"
    assert classify("clip.mov") == "video"
    assert classify("track.flac") == "audio"
    with pytest.raises(ReferenceError, match="cannot tell what"):
        classify("mystery.xyz")


@pytest.mark.parametrize(
    "durations,ok",
    [([2.0], True), ([15.0], True), ([1.9], False), ([15.1], False),
     ([5.0, 5.0, 4.0], True), ([8.0, 8.0], False)],
)
def test_clip_duration_rules(durations, ok):
    if ok:
        check_clip_durations(durations)
    else:
        with pytest.raises(ReferenceError):
            check_clip_durations(durations)


def test_empty_set_is_text_to_video():
    refs = ReferenceSet()
    assert refs.is_empty and refs.total == 0 and refs.workflow == "t2va"
