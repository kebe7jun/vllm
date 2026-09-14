# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-side helpers for Kimi-K3 video inputs.

A source video reaches the vision tower as several temporal chunks, each one
MoonViT grid with ``T > 1``. The tower returns one feature tensor per grid, so
the chunks belonging to the same video have to be stitched back together: vLLM
expects exactly one embedding tensor per multi-modal item.

Shared by the NVIDIA and AMD model paths, which are otherwise independent.
"""

from typing import Any, cast

import torch


def _as_tensor(value: Any, name: str) -> torch.Tensor:
    """Coerce a batched multi-modal field into a single tensor."""
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"{name} must not be empty")
        value = torch.cat([torch.as_tensor(item) for item in value], dim=0)
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"{name} must be a tensor or a list of tensors, got {type(value)}"
        )
    return value


def normalize_video_pixel_inputs(
    pixel_values: Any,
    grid_thws: Any,
    num_chunks_per_video: Any,
    target_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Validate and flatten one batch of video inputs.

    Returns the patch tensor, the ``(num_chunks, 3)`` grid tensor, and how many
    chunks each source video owns (in item order).
    """
    pixel_values = _as_tensor(pixel_values, "pixel_values_videos")
    # Mirror the image path: some producers keep a leading item dimension.
    if pixel_values.ndim in (3, 5):
        pixel_values = pixel_values.reshape(
            pixel_values.shape[0] * pixel_values.shape[1], *pixel_values.shape[2:]
        )
    pixel_values = pixel_values.to(target_dtype)

    grid_thws = _as_tensor(grid_thws, "video_grid_thws")
    grid_thws = grid_thws.reshape(-1, grid_thws.shape[-1])
    if grid_thws.ndim != 2 or grid_thws.size(1) != 3:
        raise ValueError(f"unexpected shape for video_grid_thws: {grid_thws.shape}")

    if num_chunks_per_video is None:
        raise ValueError(
            "Kimi-K3 video inputs are missing `num_chunks_per_video`, so the "
            "per-chunk features cannot be regrouped per video."
        )
    chunk_counts_tensor = _as_tensor(num_chunks_per_video, "num_chunks_per_video")
    chunk_counts = [int(count) for count in chunk_counts_tensor.flatten().tolist()]

    if sum(chunk_counts) != grid_thws.size(0):
        raise ValueError(
            f"Kimi-K3 video chunk counts sum to {sum(chunk_counts)} but "
            f"{grid_thws.size(0)} grid row(s) were provided."
        )

    expected_patches = int(grid_thws.prod(-1).sum())
    if pixel_values.size(0) != expected_patches:
        raise ValueError(
            f"Kimi-K3 video grids describe {expected_patches} patch(es) but "
            f"{pixel_values.size(0)} were provided."
        )

    return pixel_values, grid_thws, chunk_counts


def group_chunk_features(
    chunk_features: list[torch.Tensor],
    chunk_counts: list[int],
) -> list[torch.Tensor]:
    """Concatenate each video's per-chunk features into one tensor.

    The concatenation order matches the prompt, where a video expands into one
    ``<|media_begin|>video ...<|media_end|>`` block per chunk, in time order.
    """
    if sum(chunk_counts) != len(chunk_features):
        raise ValueError(
            f"Expected {sum(chunk_counts)} chunk feature tensor(s) for "
            f"{len(chunk_counts)} video(s), got {len(chunk_features)}."
        )

    grouped: list[torch.Tensor] = []
    offset = 0
    for count in chunk_counts:
        if count < 1:
            raise ValueError(
                f"A Kimi-K3 video must own at least one chunk, got {count}"
            )
        features = cast(list[torch.Tensor], chunk_features[offset : offset + count])
        grouped.append(features[0] if count == 1 else torch.cat(features, dim=0))
        offset += count

    return grouped
