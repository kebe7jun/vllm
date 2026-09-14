# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared Kimi-K3 multimodal preprocessing."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from PIL import Image
from transformers import BatchFeature

from vllm.config.multimodal import (
    BaseDummyOptions,
    ImageDummyOptions,
    VideoDummyOptions,
)
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
    VideoProcessorItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    InputProcessingContext,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
    cached_encode,
)
from vllm.transformers_utils.configs.kimi_k3 import KimiK3Config
from vllm.transformers_utils.processor import cached_get_image_processor
from vllm.transformers_utils.processors.kimi_k3 import KimiK3Processor

logger = init_logger(__name__)


def navit_resize_image(
    width: int,
    height: int,
    patch_size: int,
    merge_kernel_size: int,
    in_patch_limit: int,
    patch_limit_on_one_side: int,
    fixed_output_tokens: int | None,
):
    # Apply the patch limits.
    s1 = math.sqrt(
        in_patch_limit
        / (max(1.0, width // patch_size) * max(1.0, height // patch_size))
    )
    s2 = patch_limit_on_one_side * patch_size / width
    s3 = patch_limit_on_one_side * patch_size / height
    scale = min(1.0, s1, s2, s3)
    new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
    new_w = min(new_w, patch_limit_on_one_side * patch_size)
    new_h = min(new_h, patch_limit_on_one_side * patch_size)

    factor = merge_kernel_size * patch_size

    pad_height = (factor - new_h % factor) % factor
    pad_width = (factor - new_w % factor) % factor

    if fixed_output_tokens is not None:
        num_tokens = fixed_output_tokens
    else:
        # Calculate new dimensions after padding and patching
        token_height = (new_h + pad_height) // factor
        token_width = (new_w + pad_width) // factor

        assert token_height * merge_kernel_size <= patch_limit_on_one_side, (
            f"token_height {token_height} * merge_kernel_size {merge_kernel_size} > "
            f"patch_limit_on_one_side {patch_limit_on_one_side}"
        )
        assert token_width * merge_kernel_size <= patch_limit_on_one_side, (
            f"token_width {token_width} * merge_kernel_size {merge_kernel_size} > "
            f"patch_limit_on_one_side {patch_limit_on_one_side}"
        )

        num_tokens = token_height * token_width
    return {
        "num_tokens": num_tokens,
        "new_width": new_w,
        "new_height": new_h,
        "pad_width": pad_width,
        "pad_height": pad_height,
        "sampled_nframes": 1,
    }


# ---------------------------------------------------------------------------
# Video support
# ---------------------------------------------------------------------------
#
# MoonViT-3D processes a group of consecutive frames as ONE grid: attention is
# factorised into an intra-frame spatial pass and an inter-frame temporal pass,
# and the patch merger then pools across the temporal dimension. A group of
# ``k`` frames therefore costs the same number of LLM tokens as a single frame
# -- the ``T`` dimension of ``grid_thws`` is what distinguishes a temporal
# group from a still image, and it never reaches the token budget.
#
# Prompt layout (inherited from Kimi-K2.5, whose reference processor builds
# exactly this string):
#
#     {timestamp}<|media_begin|>video<|media_content|>{pads}<|media_end|>
#
# repeated once per temporal group, versus the image form which carries the
# source resolution instead of a timestamp.

#: ``media_proc_cfg`` keys that only exist on a checkpoint whose media
#: processor understands video. If none of them is present we keep the
#: model image-only, so an older checkpoint behaves exactly as before.
VIDEO_MEDIA_PROC_CFG_KEYS = (
    "sample_fps",
    "temporal_merge_kernel_size",
    "max_num_frames_each_video",
    "in_patch_limit_video",
    "in_patch_limit_each_frame",
    "timestamp_mode",
)

_DEFAULT_SAMPLE_FPS = 8.0
_DEFAULT_TEMPORAL_MERGE_KERNEL_SIZE = 4
_DEFAULT_TIMESTAMP_MODE = "hh:mm:ss.fff"
#: Frames assumed per video during memory profiling when the checkpoint does
#: not pin ``max_num_frames_each_video``. Overridable per request through
#: ``--limit-mm-per-prompt``/dummy options.
_DEFAULT_PROFILING_NUM_FRAMES = 64


#: Fallback used when a config predates ``KimiK3Config.video_placeholder``.
DEFAULT_VIDEO_PLACEHOLDER = (
    "<|media_begin|>video<|media_content|><|media_pad|><|media_end|>"
)


def _video_placeholder(hf_config: Any) -> str:
    return getattr(hf_config, "video_placeholder", None) or DEFAULT_VIDEO_PLACEHOLDER


def kimi_k3_supports_video(media_proc_cfg: Mapping[str, Any]) -> bool:
    """Whether the checkpoint's media processor is video-capable."""
    return any(key in media_proc_cfg for key in VIDEO_MEDIA_PROC_CFG_KEYS)


@dataclass(frozen=True)
class KimiK3VideoConfig:
    """Video knobs, as shipped by the checkpoint's ``media_proc_cfg``.

    The defaults only apply to keys the checkpoint leaves out.
    """

    sample_fps: float = _DEFAULT_SAMPLE_FPS
    max_num_frames_each_video: int | None = None
    temporal_merge_kernel_size: int = _DEFAULT_TEMPORAL_MERGE_KERNEL_SIZE
    timestamp_mode: str = _DEFAULT_TIMESTAMP_MODE

    @classmethod
    def from_media_proc_cfg(
        cls, media_proc_cfg: Mapping[str, Any]
    ) -> "KimiK3VideoConfig":
        sample_fps = media_proc_cfg.get("sample_fps")
        sample_fps = (
            float(sample_fps) if sample_fps is not None else _DEFAULT_SAMPLE_FPS
        )
        if sample_fps <= 0:
            raise ValueError(
                f"Kimi-K3 media_proc_cfg['sample_fps'] must be positive, "
                f"got {sample_fps}."
            )

        max_frames = media_proc_cfg.get("max_num_frames_each_video")
        if max_frames is not None:
            max_frames = int(max_frames)
            if max_frames < 1:
                raise ValueError(
                    "Kimi-K3 media_proc_cfg['max_num_frames_each_video'] must "
                    f"be >= 1, got {max_frames}."
                )

        temporal_merge = media_proc_cfg.get("temporal_merge_kernel_size")
        temporal_merge = (
            int(temporal_merge)
            if temporal_merge is not None
            else _DEFAULT_TEMPORAL_MERGE_KERNEL_SIZE
        )
        # MoonViT-3D's temporal position table is sized by
        # `vision_config.init_pos_emb_time` (4 in every shipped K3 config), so
        # a group may never hold more frames than that.
        if not 1 <= temporal_merge <= 4:
            raise ValueError(
                "Kimi-K3 media_proc_cfg['temporal_merge_kernel_size'] must be "
                f"between 1 and 4, got {temporal_merge}."
            )

        timestamp_mode = media_proc_cfg.get("timestamp_mode") or _DEFAULT_TIMESTAMP_MODE

        return cls(
            sample_fps=sample_fps,
            max_num_frames_each_video=max_frames,
            temporal_merge_kernel_size=temporal_merge,
            timestamp_mode=str(timestamp_mode),
        )

    @property
    def profiling_num_frames(self) -> int:
        """Worst-case frame count to assume during memory profiling."""
        if self.max_num_frames_each_video is not None:
            return self.max_num_frames_each_video
        return _DEFAULT_PROFILING_NUM_FRAMES


def format_kimi_k3_timestamp(timestamp: float, mode: str) -> str:
    """Render a chunk timestamp the way the reference processor does.

    The ``hh:mm:ss.fff`` branch reproduces Kimi-K2.5's reference
    implementation, ``time.strftime("%H:%M:%S", time.gmtime(ts))`` plus
    milliseconds -- including its wrap-around past 24 hours.
    """
    timestamp = max(0.0, float(timestamp))
    total_seconds = int(timestamp)
    milliseconds = int((timestamp % 1) * 1000)
    hours = (total_seconds // 3600) % 24
    minutes = (total_seconds // 60) % 60
    seconds = total_seconds % 60

    if mode == "hh:mm:ss.fff":
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    if mode == "mm:ss.fff":
        return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    if mode == "mm:ss":
        return f"{minutes:02d}:{seconds:02d}"
    raise ValueError(
        f"Invalid Kimi-K3 timestamp mode: {mode!r}. Supported modes are "
        "'hh:mm:ss.fff', 'mm:ss.fff' and 'mm:ss'."
    )


def resolve_frame_timestamps(
    num_frames: int,
    metadata: Mapping[str, Any] | None,
    sample_fps: float,
) -> list[float]:
    """Absolute timestamp (seconds) of every frame handed to us.

    vLLM's media loader may already have sub-sampled the video, so the
    authoritative mapping back to wall-clock time is
    ``metadata["frames_indices"]`` over ``metadata["fps"]`` (the *original*
    frame rate), mirroring how Qwen3-VL derives its own timestamps.
    """
    if num_frames <= 0:
        return []

    if metadata:
        fps = metadata.get("fps")
        fps = float(fps) if fps else 0.0
        if fps > 0:
            indices = metadata.get("frames_indices")
            if indices is not None and len(indices) == num_frames:
                return [float(index) / fps for index in indices]

            # No explicit indices: the loader sampled uniformly across the
            # whole video, so reconstruct the source positions from the
            # original frame count.
            total = metadata.get("total_num_frames")
            total = int(total) if total else 0
            if total > 1 and num_frames > 1:
                step = (total - 1) / (num_frames - 1)
                return [(i * step) / fps for i in range(num_frames)]
            return [i / fps for i in range(num_frames)]

    # Last resort: assume the frames arrive at the configured sampling rate.
    rate = sample_fps if sample_fps > 0 else 1.0
    return [i / rate for i in range(num_frames)]


def select_frame_indices(
    timestamps: Sequence[float],
    video_config: KimiK3VideoConfig,
) -> list[int]:
    """Pick which of the received frames to keep, as sorted indices.

    Thins the frames down to ``sample_fps`` and then to
    ``max_num_frames_each_video``, always keeping at least one frame.
    """
    num_frames = len(timestamps)
    if num_frames <= 1:
        return list(range(num_frames))

    target = num_frames
    duration = float(timestamps[-1]) - float(timestamps[0])
    if duration > 0:
        effective_fps = (num_frames - 1) / duration
        if effective_fps > video_config.sample_fps:
            target = int(round(duration * video_config.sample_fps)) + 1

    max_frames = video_config.max_num_frames_each_video
    if max_frames is not None:
        target = min(target, max_frames)

    target = max(1, min(target, num_frames))
    if target == num_frames:
        return list(range(num_frames))

    return np.linspace(0, num_frames - 1, target).round().astype(int).tolist()


def group_frames_into_chunks(
    indices: Sequence[int],
    temporal_merge_kernel_size: int,
) -> list[list[int]]:
    """Split selected frame indices into consecutive temporal groups.

    The trailing group may be short; MoonViT-3D pools over whatever ``T`` it
    is given, so a partial group is valid and simply carries fewer frames.
    """
    if temporal_merge_kernel_size < 1:
        raise ValueError(
            "temporal_merge_kernel_size must be >= 1, got "
            f"{temporal_merge_kernel_size}."
        )
    step = temporal_merge_kernel_size
    return [list(indices[i : i + step]) for i in range(0, len(indices), step)]


def chunk_num_tokens(grid_thw: Sequence[int], merge_kernel_size: Sequence[int]) -> int:
    """LLM tokens one MoonViT grid occupies.

    The ``T`` axis is dropped on purpose: the merger pools across time, so a
    temporal group costs exactly one frame's worth of tokens. This mirrors the
    reference implementation's ``video_grid_thw[i][1:].prod() // merge_length``.
    """
    _, height, width = (int(value) for value in grid_thw)
    kernel_height, kernel_width = (int(value) for value in merge_kernel_size)
    return (height * width) // (kernel_height * kernel_width)


def frames_to_pil(frames: Any) -> list[Image.Image]:
    """Normalise one video's frames into the PIL list the processor wants."""
    pil_frames: list[Image.Image] = []
    for frame in frames:
        if isinstance(frame, Image.Image):
            pil_frames.append(frame)
            continue

        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Unsupported Kimi-K3 video frame type: {type(frame)}")

        if frame.ndim != 3:
            raise ValueError(
                f"Expected a 3D video frame (H, W, C) or (C, H, W), got shape "
                f"{frame.shape}."
            )
        # Torch-style CHW arrives from tensor inputs; PIL needs HWC.
        if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
            frame = np.transpose(frame, (1, 2, 0))
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        pil_frames.append(Image.fromarray(frame))

    return pil_frames


def split_video_into_chunks(
    frames: Any,
    metadata: Mapping[str, Any] | None,
    video_config: KimiK3VideoConfig,
) -> tuple[list[list[Image.Image]], list[float]]:
    """Sample and group one source video.

    Returns the temporal groups of PIL frames and, for each group, the
    timestamp of its first frame (the reference processor stamps groups with
    their opening frame, not their midpoint).
    """
    pil_frames = frames_to_pil(frames)
    if not pil_frames:
        raise ValueError("Kimi-K3 video input must contain at least one frame.")

    timestamps = resolve_frame_timestamps(
        len(pil_frames), metadata, video_config.sample_fps
    )
    selected = select_frame_indices(timestamps, video_config)
    groups = group_frames_into_chunks(selected, video_config.temporal_merge_kernel_size)

    chunks = [[pil_frames[index] for index in group] for group in groups]
    chunk_timestamps = [timestamps[group[0]] for group in groups]
    return chunks, chunk_timestamps


class KimiK3ProcessingInfo(BaseProcessingInfo):
    """Processing information for the image-only Kimi-K3 model.

    K3 uses the standard ``image`` modality (unlike K2.5's unified
    ``vision_chunk``), so it builds its own ``KimiK3Processor`` wrapper around
    the checkpoint's image processor and resolves the ``<|media_pad|>`` token
    id the same way K2.5 does.
    """

    def __init__(self, ctx: InputProcessingContext) -> None:
        super().__init__(ctx)

        self.hf_config = hf_config = self.get_hf_config()

        tokenizer = self.get_tokenizer()
        image_processor = cached_get_image_processor(
            self.ctx.model_config.model,
            revision=self.ctx.model_config.revision,
            trust_remote_code=self.ctx.model_config.trust_remote_code,
        )

        # Resolve token ID from the tokenizer because transformers v5
        # may remap token IDs vs config.json.
        config_token_id = hf_config.media_placeholder_token_id
        resolved_token_id = tokenizer.convert_tokens_to_ids("<|media_pad|>")
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        is_valid_resolved = isinstance(resolved_token_id, int) and (
            unk_token_id is None or resolved_token_id != unk_token_id
        )
        if is_valid_resolved and resolved_token_id != config_token_id:
            logger.warning_once(
                "Kimi-K3 config.media_placeholder_token_id (%d) disagrees "
                "with tokenizer mapping for <|media_pad|> (%d). "
                "Using tokenizer value.",
                config_token_id,
                resolved_token_id,
            )
            media_token_id = resolved_token_id
            # Patch config so downstream code also sees the correct ID.
            hf_config.media_placeholder_token_id = resolved_token_id
        else:
            media_token_id = config_token_id

        self.media_token_id = media_token_id
        self.media_token = tokenizer.decode(media_token_id)

        self.image_processor = image_processor
        self.hf_processor = KimiK3Processor(
            tokenizer=tokenizer,
            image_processor=image_processor,
        )
        self.media_tokens_calculator = image_processor.media_tokens_calculator
        self.media_proc_cfg = image_processor.media_proc_cfg

        # Video is opt-in on the checkpoint: a media processor that ships no
        # video keys keeps this model image-only, exactly as before.
        self.supports_video = kimi_k3_supports_video(self.media_proc_cfg)
        self._video_config: KimiK3VideoConfig | None = None
        logger.info_once(
            "Kimi-K3 video input is %s (checkpoint media_proc_cfg %s video keys).",
            "enabled" if self.supports_video else "disabled",
            "declares" if self.supports_video else "declares no",
        )

    def get_video_config(self) -> KimiK3VideoConfig:
        """Video knobs from the checkpoint, resolved on first use.

        Deliberately lazy: a malformed video section must never break
        image-only serving.
        """
        if not self.supports_video:
            raise ValueError(
                "This Kimi-K3 checkpoint's media processor declares no video "
                "configuration, so video input is not available. Expected at "
                f"least one of {list(VIDEO_MEDIA_PROC_CFG_KEYS)} in "
                "`media_proc_cfg`."
            )
        if self._video_config is None:
            self._video_config = KimiK3VideoConfig.from_media_proc_cfg(
                self.media_proc_cfg
            )
        return self._video_config

    def get_merge_kernel_size(self) -> tuple[int, int]:
        kernel = self.hf_config.vision_config.merge_kernel_size
        return int(kernel[0]), int(kernel[1])

    def get_hf_processor(self, **kwargs: object) -> KimiK3Processor:
        return self.hf_processor

    def get_hf_config(self) -> KimiK3Config:
        return self.ctx.get_hf_config(KimiK3Config)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # None means unlimited
        limits: dict[str, int | None] = {"image": None}
        if self.supports_video:
            limits["video"] = None
        return limits

    @classmethod
    def get_max_image_size(
        cls,
        patch_size: int,
        merge_kernel_size: int,
        in_patch_limit: int,
        patch_limit_on_one_side: int,
        fixed_output_tokens: int | None,
    ) -> ImageSize:
        max_side = patch_limit_on_one_side * patch_size
        best_score = (-1, -1)
        best_size = (max_side, max_side)

        for width_patches in range(patch_limit_on_one_side + 1):
            width = min((width_patches + 1) * patch_size - 1, max_side)
            for height_patches in range(width_patches, patch_limit_on_one_side + 1):
                height = min((height_patches + 1) * patch_size - 1, max_side)
                resize_config = navit_resize_image(
                    width,
                    height,
                    patch_size,
                    merge_kernel_size,
                    in_patch_limit,
                    patch_limit_on_one_side,
                    fixed_output_tokens,
                )
                padded_width = resize_config["new_width"] + resize_config["pad_width"]
                padded_height = (
                    resize_config["new_height"] + resize_config["pad_height"]
                )
                num_patches = padded_width // patch_size * (padded_height // patch_size)
                score = (resize_config["num_tokens"], num_patches)
                if score > best_score:
                    best_score = score
                    best_size = (width, height)
        return ImageSize(width=best_size[0], height=best_size[1])


class KimiK3DummyInputsBuilder(BaseDummyInputsBuilder[KimiK3ProcessingInfo]):
    """Builds dummy inputs for K3 profiling.

    The dummy text is made of the ``<|kimi_image_placeholder|>`` /
    ``<|kimi_video_placeholder|>`` markers that K3's ``_get_prompt_updates``
    expands, and the dummy mm data is a plain list of PIL images under the
    ``image`` key plus frame arrays under ``video``.
    """

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        hf_config = self.info.get_hf_config()
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        return (
            hf_config.image_placeholder * num_images
            + _video_placeholder(hf_config) * num_videos
        )

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        media_proc_cfg = self.info.image_processor.media_proc_cfg
        max_size = self.info.get_max_image_size(
            media_proc_cfg["patch_size"],
            media_proc_cfg["merge_kernel_size"],
            media_proc_cfg["in_patch_limit"],
            media_proc_cfg["patch_limit_on_one_side"],
            media_proc_cfg["fixed_output_tokens"],
        )
        num_images = mm_counts.get("image", 0)
        image_overrides = cast(
            ImageDummyOptions | None,
            mm_options.get("image") if mm_options else None,
        )
        dummy_data: MultiModalDataDict = {
            "image": self._get_dummy_images(
                width=max_size.width,
                height=max_size.height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }

        num_videos = mm_counts.get("video", 0)
        if num_videos:
            video_config = self.info.get_video_config()
            # A video frame is bounded by the per-frame patch budget, which is
            # tighter than a still image's whenever the checkpoint sets it.
            frame_size = self.info.get_max_image_size(
                media_proc_cfg["patch_size"],
                media_proc_cfg["merge_kernel_size"],
                media_proc_cfg.get("in_patch_limit_each_frame")
                or media_proc_cfg["in_patch_limit"],
                media_proc_cfg["patch_limit_on_one_side"],
                media_proc_cfg["fixed_output_tokens"],
            )
            video_overrides = cast(
                VideoDummyOptions | None,
                mm_options.get("video") if mm_options else None,
            )
            dummy_data["video"] = self._get_dummy_videos(
                width=frame_size.width,
                height=frame_size.height,
                num_frames=video_config.profiling_num_frames,
                num_videos=num_videos,
                overrides=video_overrides,
            )

        return dummy_data


def _unpack_video_item(item: Any) -> tuple[Any, Mapping[str, Any] | None]:
    """Split a video item into frames and (optional) loader metadata."""
    if isinstance(item, tuple):
        if len(item) >= 2:
            return item[0], item[1]
        return item[0], None
    return item, None


class KimiK3MultiModalProcessor(BaseMultiModalProcessor[KimiK3ProcessingInfo]):
    """Multi-modal processor for Kimi-K3.

    Images use the standard ``image`` modality. Videos are sampled and grouped
    into temporal chunks here, then handed to the checkpoint's media processor
    as ``video_chunk`` media; one source video expands into as many
    ``<|media_begin|>video ...<|media_end|>`` blocks as it has chunks.
    """

    def _apply_hf_processor_main(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Run the HF processor, sampling and chunking videos on the way in.

        Mirrors the base implementation, with the single difference that
        ``videos`` are converted into flat ``video_chunks`` before the call and
        the per-video bookkeeping (chunk counts, chunk timestamps) is attached
        to the result.
        """
        valid_mm_items = mm_items.select(
            {k for k, c in mm_items.get_all_counts().items() if c > 0}
        )
        processor_data, passthrough_data = self._get_hf_mm_data(valid_mm_items)
        processor_data = dict(processor_data)

        video_extras: dict[str, Any] = {}
        if videos := processor_data.pop("videos", None):
            video_config = self.info.get_video_config()

            flat_chunks: list[list[Image.Image]] = []
            num_chunks_per_video: list[int] = []
            timestamps_per_video: list[list[float]] = []
            for item in videos:
                frames, metadata = _unpack_video_item(item)
                chunks, timestamps = split_video_into_chunks(
                    frames, metadata, video_config
                )
                flat_chunks.extend(chunks)
                num_chunks_per_video.append(len(chunks))
                timestamps_per_video.append(timestamps)

            processor_data["video_chunks"] = flat_chunks
            video_extras = {
                # Shaped (num_videos, 1) so that batching several videos keeps
                # one row per video: the model needs these counts to split the
                # flat per-chunk features back into one tensor per video.
                "num_chunks_per_video": torch.tensor(
                    num_chunks_per_video, dtype=torch.long
                ).unsqueeze(-1),
                "video_timestamps": timestamps_per_video,
            }

        if processor_data:
            processor_data, hf_processor_mm_kwargs = self._preprocess_hf_mm_data(
                processor_data, hf_processor_mm_kwargs
            )

            prompt_text = self._get_hf_processor_text(mm_items.get_all_counts())
            if prompt_text is not None:
                processor_data = dict(text=prompt_text, **processor_data)

            processed_data = self.info.ctx.call_hf_processor(
                self.info.get_hf_processor(**hf_processor_mm_kwargs),
                processor_data,
                hf_processor_mm_kwargs,
            )
            processed_data.update(passthrough_data)
        else:
            processed_data = BatchFeature(dict(passthrough_data))

        processed_data.update(video_extras)

        return self._postprocess_hf_mm_data(
            processor_data,
            hf_processor_mm_kwargs,
            processed_data,
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Slice the flattened patch tensor back into per-item items.

        ``pixel_values`` holds all patches from every image concatenated; each
        image's patch count is ``prod(grid_thws[i])``. ``grid_thws`` is one
        ``[N_t, N_h, N_w]`` row per image.

        Video is grouped one level deeper: one source video owns several
        consecutive chunk rows, so both its patches and its grid rows are
        sliced by ``num_chunks_per_video`` rather than one row per item.
        """
        grid_thws = hf_inputs.get("grid_thws", torch.empty((0, 3)))
        grid_sizes = grid_thws.prod(-1)

        fields = dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes("image", grid_sizes),
            grid_thws=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        )

        video_grid_thws = hf_inputs.get("video_grid_thws")
        if video_grid_thws is None:
            return fields

        chunk_counts_raw = hf_inputs.get("num_chunks_per_video")
        if chunk_counts_raw is None:
            raise ValueError(
                "Kimi-K3 produced video grids without per-video chunk counts."
            )
        if isinstance(chunk_counts_raw, torch.Tensor):
            chunk_counts_raw = chunk_counts_raw.flatten().tolist()
        num_chunks_per_video = [int(count) for count in chunk_counts_raw]
        video_grid_sizes = video_grid_thws.prod(-1)
        if int(sum(num_chunks_per_video)) != int(video_grid_sizes.shape[0]):
            raise ValueError(
                "Kimi-K3 video chunk bookkeeping is inconsistent: "
                f"{sum(num_chunks_per_video)} chunk(s) recorded across "
                f"{len(num_chunks_per_video)} video(s), but the processor "
                f"returned {int(video_grid_sizes.shape[0])} grid row(s)."
            )

        # Patches belonging to each source video = its chunks' patches summed.
        video_patch_sizes = torch.tensor(
            [
                int(chunk.sum())
                for chunk in torch.split(video_grid_sizes, num_chunks_per_video)
            ],
            dtype=torch.long,
        )
        chunk_counts = torch.tensor(num_chunks_per_video, dtype=torch.long)

        fields.update(
            pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
                "video", video_patch_sizes
            ),
            video_grid_thws=MultiModalFieldConfig.flat_from_sizes(
                "video", chunk_counts, keep_on_cpu=True
            ),
            num_chunks_per_video=MultiModalFieldConfig.batched(
                "video", keep_on_cpu=True
            ),
            video_timestamps=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
        )
        return fields

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """Expand each K3 media placeholder into a media block.

        An image placeholder becomes
        ``<|media_begin|>image {w}x{h}<|media_content|>{pads}<|media_end|>``,
        embedding the per-image resolution in the prompt.

        A video placeholder becomes one
        ``{timestamp}<|media_begin|>video<|media_content|>{pads}<|media_end|>``
        block per temporal chunk. Only the ``<|media_pad|>`` positions are
        marked as embedding slots.
        """
        media_token_id = self.info.media_token_id
        media_token = self.info.media_token
        hf_config = self.info.get_hf_config()
        image_placeholder = hf_config.image_placeholder
        tokenizer = self.info.get_tokenizer()

        def get_replacement(item_idx: int) -> PromptUpdateDetails:
            images = mm_items.get_items("image", ImageProcessorItems)
            image = images.get(item_idx)
            if image is None:
                raise ValueError(f"Missing image data at index {item_idx}")

            # The checkpoint image processor works on media dicts, so wrap the
            # PIL image before asking it for the token count.
            num_media_token = self.info.media_tokens_calculator(
                {"type": "image", "image": image}
            )
            pads = media_token * num_media_token

            # NOTE: `width`/`height` are the ORIGINAL upload dimensions, not the
            # post-preprocess (smart-resized) ones. `image` comes from the
            # untouched parsed `mm_items`; the checkpoint image processor
            # (`KimiK3VisionProcessor.preprocess`) only produces new tensors via
            # `image.resize(...)` and never mutates the stored PIL. This matches
            # the reference HF processor (`KimiK3Processor.preprocess_medias`),
            # which also builds the prompt from the original `img.size`. The
            # resize is reflected only in the pad count above.
            width, height = images.get_image_size(item_idx)
            full = (
                f"<|media_begin|>image {width}x{height}<|media_content|>"
                f"{pads}<|media_end|>"
            )

            return PromptUpdateDetails.select_token_id(
                cached_encode(tokenizer, full, add_special_tokens=False),
                media_token_id,
            )

        updates = [
            PromptReplacement(
                modality="image",
                target=cached_encode(
                    tokenizer, image_placeholder, add_special_tokens=False
                ),
                replacement=get_replacement,
            ),
        ]

        if not mm_items.get_all_counts().get("video", 0):
            return updates

        merge_kernel_size = self.info.get_merge_kernel_size()
        timestamp_mode = self.info.get_video_config().timestamp_mode

        def get_video_replacement(item_idx: int) -> PromptUpdateDetails:
            videos = mm_items.get_items("video", VideoProcessorItems)
            if videos.get(item_idx) is None:
                raise ValueError(f"Missing video data at index {item_idx}")

            out_item = out_mm_kwargs["video"][item_idx]
            grid_thws = out_item["video_grid_thws"].data
            timestamps = out_item["video_timestamps"].data

            num_chunks = len(grid_thws)
            if len(timestamps) != num_chunks:
                raise ValueError(
                    f"Kimi-K3 video {item_idx} has {num_chunks} chunk(s) but "
                    f"{len(timestamps)} timestamp(s)."
                )

            blocks = []
            for chunk_idx in range(num_chunks):
                stamp = format_kimi_k3_timestamp(
                    float(timestamps[chunk_idx]), timestamp_mode
                )
                num_media_token = chunk_num_tokens(
                    grid_thws[chunk_idx], merge_kernel_size
                )
                pads = media_token * num_media_token
                blocks.append(
                    f"{stamp}<|media_begin|>video<|media_content|>{pads}<|media_end|>"
                )

            return PromptUpdateDetails.select_token_id(
                cached_encode(tokenizer, "".join(blocks), add_special_tokens=False),
                media_token_id,
            )

        updates.append(
            PromptReplacement(
                modality="video",
                target=cached_encode(
                    tokenizer,
                    _video_placeholder(hf_config),
                    add_special_tokens=False,
                ),
                replacement=get_video_replacement,
            )
        )
        return updates
