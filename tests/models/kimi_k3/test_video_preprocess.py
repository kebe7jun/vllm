# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Kimi-K3 video preprocessing.

These cover the pure input-side logic -- frame sampling, temporal chunking,
timestamp rendering and the per-video regrouping the model relies on -- so they
need neither a GPU nor the Kimi-K3 checkpoint.
"""

import time

import numpy as np
import pytest
import torch
from PIL import Image

from vllm.models.kimi_k3.common.mm_preprocess import (
    VIDEO_MEDIA_PROC_CFG_KEYS,
    KimiK3VideoConfig,
    chunk_num_tokens,
    format_kimi_k3_timestamp,
    frames_to_pil,
    group_frames_into_chunks,
    kimi_k3_supports_video,
    resolve_frame_timestamps,
    select_frame_indices,
    split_video_into_chunks,
)
from vllm.models.kimi_k3.common.video_inputs import (
    group_chunk_features,
    normalize_video_pixel_inputs,
)


def _reference_timestamp(seconds: float) -> str:
    """Kimi-K2.5's reference rendering, which K3 inherits."""
    return time.strftime("%H:%M:%S", time.gmtime(seconds)) + (
        f".{int(seconds % 1 * 1000):03d}"
    )


# --------------------------------------------------------------------------
# Timestamps
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seconds", [0.0, 0.5, 1.0, 3.5, 59.999, 60.0, 3599.25, 3600.0, 7325.125]
)
def test_timestamp_matches_reference_implementation(seconds):
    assert format_kimi_k3_timestamp(seconds, "hh:mm:ss.fff") == _reference_timestamp(
        seconds
    )


def test_timestamp_modes():
    assert format_kimi_k3_timestamp(3725.5, "hh:mm:ss.fff") == "01:02:05.500"
    assert format_kimi_k3_timestamp(3725.5, "mm:ss.fff") == "02:05.500"
    assert format_kimi_k3_timestamp(3725.5, "mm:ss") == "02:05"


def test_timestamp_wraps_past_a_day_like_gmtime():
    # 25h -> 01:00:00, matching time.gmtime's wrap-around.
    assert format_kimi_k3_timestamp(90000.0, "hh:mm:ss.fff") == "01:00:00.000"


def test_timestamp_clamps_negative_input():
    assert format_kimi_k3_timestamp(-5.0, "hh:mm:ss.fff") == "00:00:00.000"


def test_timestamp_rejects_unknown_mode():
    with pytest.raises(ValueError, match="timestamp mode"):
        format_kimi_k3_timestamp(1.0, "ss")


# --------------------------------------------------------------------------
# Config resolution
# --------------------------------------------------------------------------


def test_supports_video_detection():
    assert not kimi_k3_supports_video({"patch_size": 14, "in_patch_limit": 1024})
    for key in VIDEO_MEDIA_PROC_CFG_KEYS:
        assert kimi_k3_supports_video({"patch_size": 14, key: 1})


def test_video_config_defaults():
    cfg = KimiK3VideoConfig.from_media_proc_cfg({"sample_fps": 8.0})
    assert cfg.sample_fps == 8.0
    assert cfg.temporal_merge_kernel_size == 4
    assert cfg.max_num_frames_each_video is None
    assert cfg.timestamp_mode == "hh:mm:ss.fff"
    # No checkpoint cap -> profiling still needs a bounded frame count.
    assert cfg.profiling_num_frames > 0


def test_video_config_reads_checkpoint_values():
    cfg = KimiK3VideoConfig.from_media_proc_cfg(
        {
            "sample_fps": 2.5,
            "max_num_frames_each_video": 48,
            "temporal_merge_kernel_size": 2,
            "timestamp_mode": "mm:ss",
        }
    )
    assert cfg.sample_fps == 2.5
    assert cfg.max_num_frames_each_video == 48
    assert cfg.temporal_merge_kernel_size == 2
    assert cfg.timestamp_mode == "mm:ss"
    assert cfg.profiling_num_frames == 48


@pytest.mark.parametrize(
    "cfg,match",
    [
        ({"sample_fps": 0}, "sample_fps"),
        ({"sample_fps": -1.0}, "sample_fps"),
        ({"temporal_merge_kernel_size": 0}, "temporal_merge_kernel_size"),
        ({"temporal_merge_kernel_size": 5}, "temporal_merge_kernel_size"),
        ({"max_num_frames_each_video": 0}, "max_num_frames_each_video"),
    ],
)
def test_video_config_rejects_invalid_values(cfg, match):
    with pytest.raises(ValueError, match=match):
        KimiK3VideoConfig.from_media_proc_cfg(cfg)


# --------------------------------------------------------------------------
# Frame timestamps and sampling
# --------------------------------------------------------------------------


def test_resolve_timestamps_prefers_frame_indices():
    metadata = {"fps": 30.0, "frames_indices": [0, 15, 30], "total_num_frames": 31}
    assert resolve_frame_timestamps(3, metadata, 8.0) == [0.0, 0.5, 1.0]


def test_resolve_timestamps_spreads_over_original_length():
    # Loader handed us 3 of 61 frames without explicit indices.
    metadata = {"fps": 30.0, "total_num_frames": 61}
    assert resolve_frame_timestamps(3, metadata, 8.0) == [0.0, 1.0, 2.0]


def test_resolve_timestamps_falls_back_to_fps_only():
    assert resolve_frame_timestamps(3, {"fps": 2.0}, 8.0) == [0.0, 0.5, 1.0]


def test_resolve_timestamps_without_metadata_uses_sample_fps():
    assert resolve_frame_timestamps(3, None, 4.0) == [0.0, 0.25, 0.5]


def test_resolve_timestamps_ignores_mismatched_indices():
    # Index list that does not describe the frames we got must not be trusted.
    metadata = {"fps": 10.0, "frames_indices": [0, 5], "total_num_frames": 11}
    assert resolve_frame_timestamps(3, metadata, 8.0) == [0.0, 0.5, 1.0]


def test_select_keeps_everything_below_sample_fps():
    cfg = KimiK3VideoConfig(sample_fps=8.0)
    timestamps = [i / 2.0 for i in range(10)]  # 2 fps
    assert select_frame_indices(timestamps, cfg) == list(range(10))


def test_select_thins_down_to_sample_fps():
    cfg = KimiK3VideoConfig(sample_fps=2.0)
    # 4 seconds at 10 fps -> 41 frames, thinned to ~2 fps -> 9 frames.
    timestamps = [i / 10.0 for i in range(41)]
    selected = select_frame_indices(timestamps, cfg)
    assert len(selected) == 9
    assert selected[0] == 0
    assert selected[-1] == 40
    assert selected == sorted(selected)


def test_select_respects_max_frames():
    cfg = KimiK3VideoConfig(sample_fps=1000.0, max_num_frames_each_video=5)
    timestamps = [i / 10.0 for i in range(100)]
    selected = select_frame_indices(timestamps, cfg)
    assert len(selected) == 5
    assert selected[0] == 0
    assert selected[-1] == 99


def test_select_handles_degenerate_inputs():
    cfg = KimiK3VideoConfig(sample_fps=8.0)
    assert select_frame_indices([], cfg) == []
    assert select_frame_indices([0.0], cfg) == [0]
    # A zero-duration clip has no meaningful frame rate; keep every frame.
    assert select_frame_indices([1.0, 1.0, 1.0], cfg) == [0, 1, 2]


# --------------------------------------------------------------------------
# Temporal chunking
# --------------------------------------------------------------------------


def test_group_exact_multiple():
    assert group_frames_into_chunks([0, 1, 2, 3, 4, 5, 6, 7], 4) == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]


def test_group_keeps_short_trailing_chunk():
    assert group_frames_into_chunks([0, 1, 2, 3, 4], 4) == [[0, 1, 2, 3], [4]]


def test_group_with_kernel_one_is_frame_per_chunk():
    assert group_frames_into_chunks([0, 1, 2], 1) == [[0], [1], [2]]


def test_group_rejects_invalid_kernel():
    with pytest.raises(ValueError, match="temporal_merge_kernel_size"):
        group_frames_into_chunks([0, 1], 0)


# --------------------------------------------------------------------------
# Token accounting -- the reason video is cheaper than N still images
# --------------------------------------------------------------------------


@pytest.mark.parametrize("num_frames", [1, 2, 3, 4])
def test_chunk_tokens_ignore_the_temporal_axis(num_frames):
    """Temporal pooling means T never reaches the token budget."""
    assert chunk_num_tokens((num_frames, 16, 16), (2, 2)) == 64


def test_video_is_k_times_cheaper_than_the_same_frames_as_images():
    grid = (4, 16, 16)
    merge = (2, 2)
    tokens_per_chunk = chunk_num_tokens(grid, merge)
    tokens_per_frame_as_image = chunk_num_tokens((1, 16, 16), merge)
    assert tokens_per_chunk * 4 == tokens_per_frame_as_image * 4
    # 16 frames: 4 chunks vs 16 separate images.
    assert 4 * tokens_per_chunk == (16 * tokens_per_frame_as_image) // 4


# --------------------------------------------------------------------------
# Frame normalisation
# --------------------------------------------------------------------------


def test_frames_to_pil_accepts_hwc_uint8():
    frames = [np.zeros((4, 6, 3), dtype=np.uint8)]
    (converted,) = frames_to_pil(frames)
    assert isinstance(converted, Image.Image)
    assert converted.size == (6, 4)


def test_frames_to_pil_accepts_chw_and_torch():
    frames = [torch.zeros(3, 4, 6, dtype=torch.uint8)]
    (converted,) = frames_to_pil(frames)
    assert converted.size == (6, 4)


def test_frames_to_pil_clips_float_input():
    frame = np.full((2, 2, 3), 300.0, dtype=np.float32)
    (converted,) = frames_to_pil([frame])
    assert np.asarray(converted).max() == 255


def test_frames_to_pil_expands_grayscale():
    (converted,) = frames_to_pil([np.zeros((2, 2, 1), dtype=np.uint8)])
    assert np.asarray(converted).shape == (2, 2, 3)


def test_frames_to_pil_passes_through_pil():
    image = Image.new("RGB", (3, 3))
    assert frames_to_pil([image])[0] is image


def test_frames_to_pil_rejects_bad_shapes_and_types():
    with pytest.raises(ValueError, match="3D video frame"):
        frames_to_pil([np.zeros((2, 2), dtype=np.uint8)])
    with pytest.raises(TypeError, match="Unsupported"):
        frames_to_pil(["not a frame"])


# --------------------------------------------------------------------------
# End-to-end splitting
# --------------------------------------------------------------------------


def _frames(count: int, size=(8, 6)):
    width, height = size
    return [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(count)]


def test_split_video_groups_and_timestamps():
    cfg = KimiK3VideoConfig(sample_fps=8.0, temporal_merge_kernel_size=4)
    metadata = {"fps": 8.0, "frames_indices": list(range(10)), "total_num_frames": 10}

    chunks, timestamps = split_video_into_chunks(_frames(10), metadata, cfg)

    assert [len(chunk) for chunk in chunks] == [4, 4, 2]
    # Each chunk is stamped with its FIRST frame, as the reference does.
    assert timestamps == pytest.approx([0.0, 0.5, 1.0])
    assert all(isinstance(frame, Image.Image) for chunk in chunks for frame in chunk)


def test_split_video_applies_frame_cap_before_chunking():
    cfg = KimiK3VideoConfig(
        sample_fps=8.0, max_num_frames_each_video=4, temporal_merge_kernel_size=4
    )
    metadata = {"fps": 8.0, "frames_indices": list(range(16)), "total_num_frames": 16}

    chunks, timestamps = split_video_into_chunks(_frames(16), metadata, cfg)

    assert len(chunks) == 1
    assert len(chunks[0]) == 4
    assert len(timestamps) == 1


def test_split_video_single_frame():
    cfg = KimiK3VideoConfig()
    chunks, timestamps = split_video_into_chunks(_frames(1), None, cfg)
    assert [len(chunk) for chunk in chunks] == [1]
    assert timestamps == [0.0]


def test_split_video_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one frame"):
        split_video_into_chunks([], None, KimiK3VideoConfig())


# --------------------------------------------------------------------------
# Model-side regrouping
# --------------------------------------------------------------------------


def _pixels(grids):
    total = sum(t * h * w for t, h, w in grids)
    return torch.zeros(total, 3, 14, 14)


def test_normalize_video_pixel_inputs_roundtrip():
    grids = [(4, 4, 4), (2, 4, 4), (4, 2, 2)]
    grid_thws = torch.tensor(grids, dtype=torch.long)
    pixel_values, out_grids, counts = normalize_video_pixel_inputs(
        _pixels(grids),
        grid_thws,
        torch.tensor([[2], [1]], dtype=torch.long),
        torch.float32,
    )
    assert counts == [2, 1]
    assert out_grids.shape == (3, 3)
    assert pixel_values.dtype == torch.float32


def test_normalize_video_pixel_inputs_concatenates_lists():
    grids = [(4, 4, 4), (2, 4, 4)]
    _, out_grids, counts = normalize_video_pixel_inputs(
        [_pixels([grids[0]]), _pixels([grids[1]])],
        [torch.tensor([grids[0]]), torch.tensor([grids[1]])],
        [torch.tensor([1]), torch.tensor([1])],
        torch.float32,
    )
    assert counts == [1, 1]
    assert out_grids.shape == (2, 3)


def test_normalize_video_pixel_inputs_detects_count_mismatch():
    grids = [(4, 4, 4), (2, 4, 4)]
    with pytest.raises(ValueError, match="chunk counts sum"):
        normalize_video_pixel_inputs(
            _pixels(grids),
            torch.tensor(grids, dtype=torch.long),
            torch.tensor([[5]], dtype=torch.long),
            torch.float32,
        )


def test_normalize_video_pixel_inputs_detects_patch_mismatch():
    grids = [(4, 4, 4)]
    with pytest.raises(ValueError, match="patch"):
        normalize_video_pixel_inputs(
            torch.zeros(3, 3, 14, 14),
            torch.tensor(grids, dtype=torch.long),
            torch.tensor([[1]], dtype=torch.long),
            torch.float32,
        )


def test_normalize_video_pixel_inputs_requires_chunk_counts():
    grids = [(4, 4, 4)]
    with pytest.raises(ValueError, match="num_chunks_per_video"):
        normalize_video_pixel_inputs(
            _pixels(grids),
            torch.tensor(grids, dtype=torch.long),
            None,
            torch.float32,
        )


def test_group_chunk_features_concatenates_per_video():
    chunk_features = [torch.full((2, 8), float(i)) for i in range(5)]
    grouped = group_chunk_features(chunk_features, [2, 1, 2])

    assert [tensor.shape for tensor in grouped] == [(4, 8), (2, 8), (4, 8)]
    # Order within a video follows time order.
    assert torch.equal(grouped[0], torch.cat(chunk_features[:2]))
    assert grouped[1] is chunk_features[2]


def test_group_chunk_features_detects_mismatch():
    with pytest.raises(ValueError, match="chunk feature tensor"):
        group_chunk_features([torch.zeros(2, 8)], [2])
