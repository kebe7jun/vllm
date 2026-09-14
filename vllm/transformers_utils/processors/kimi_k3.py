# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import BaseImageProcessor, BatchFeature, TensorType
from transformers.processing_utils import ProcessorMixin

from vllm.tokenizers.hf import HfTokenizer

# Media-dict ``type`` the checkpoint vision processor uses for a temporal
# group of frames. This is the Kimi-K2.5 convention
# (``vllm.multimodal.inputs.VisionChunkVideo``), which Kimi-K3 inherits: its
# vision tower is the same MoonViT-3D stack and its ``media_proc_cfg`` ships
# the video keys (``sample_fps``, ``temporal_merge_kernel_size``, ...).
VIDEO_CHUNK_MEDIA_TYPE = "video_chunk"


class KimiK3Processor(ProcessorMixin):
    """HF-style processor wrapper for Kimi-K3.

    K3 exposes the standard ``image`` modality, so vLLM calls this processor
    with ``images=[PIL, ...]``. The underlying checkpoint image processor
    (``KimiK3VisionProcessor``) works on ``{"type": "image", "image": PIL}``
    media dicts, so this wrapper adapts bare PIL images into that shape before
    delegating to ``preprocess``.

    Video is passed separately as ``video_chunks``: a flat list of temporal
    groups (``list[list[PIL]]``), already sampled and grouped by the vLLM-side
    processor. Each group becomes one ``{"type": "video_chunk",
    "video_chunk": [PIL, ...]}`` media dict, i.e. one MoonViT grid whose ``T``
    dimension is the group's frame count. Video outputs are returned under
    ``pixel_values_videos``/``video_grid_thws`` so the image path stays
    untouched.

    Text is only tokenized here; the ``<|kimi_image_placeholder|>`` /
    ``<|kimi_video_placeholder|>`` markers are expanded into media blocks by
    the model's ``_get_prompt_updates`` on the vLLM side.
    """

    attributes = ["image_processor", "tokenizer"]

    def __init__(
        self,
        image_processor: BaseImageProcessor,
        tokenizer: HfTokenizer,
    ) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(
        self,
        text: str | list[str] | None = None,
        images: object | list[object] | None = None,
        video_chunks: list[list[object]] | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            medias = [{"type": "image", "image": image} for image in images]
            mm_inputs = self.image_processor.preprocess(
                medias,
                return_tensors=return_tensors,
            )
        else:
            mm_inputs = {}

        if video_chunks:
            video_medias = [
                {
                    "type": VIDEO_CHUNK_MEDIA_TYPE,
                    VIDEO_CHUNK_MEDIA_TYPE: list(chunk),
                }
                for chunk in video_chunks
            ]
            try:
                video_outputs = self.image_processor.preprocess(
                    video_medias,
                    return_tensors=return_tensors,
                )
            except Exception as exc:  # noqa: BLE001 - re-raised with context
                raise ValueError(
                    "The Kimi-K3 checkpoint's vision processor rejected a "
                    f"{VIDEO_CHUNK_MEDIA_TYPE!r} media item. Video input "
                    "requires a checkpoint whose `preprocess` accepts the "
                    "Kimi-K2.5 video-chunk media dict "
                    f"(`{{'type': '{VIDEO_CHUNK_MEDIA_TYPE}', "
                    f"'{VIDEO_CHUNK_MEDIA_TYPE}': [PIL, ...]}}`). Run with "
                    "images only, or update "
                    "`vllm.transformers_utils.processors.kimi_k3."
                    "VIDEO_CHUNK_MEDIA_TYPE` to the key this checkpoint uses."
                ) from exc

            # Keep the video tensors under their own keys so the image path
            # (`pixel_values`/`grid_thws`) is byte-for-byte unchanged.
            mm_inputs = {
                **mm_inputs,
                "pixel_values_videos": video_outputs["pixel_values"],
                "video_grid_thws": video_outputs["grid_thws"],
            }

        if text is not None:
            if not isinstance(text, list):
                text = [text]
            text_inputs = self.tokenizer(text)
        else:
            text_inputs = {}

        return BatchFeature(
            data={**text_inputs, **mm_inputs},
            tensor_type=return_tensors,
        )
