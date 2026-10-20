# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch
import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.utils import get_draft_load_config
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
    maybe_share_target_embed,
)
from vllm.v1.worker.gpu.spec_decode.utils import get_pp_safe_draft_load_config


logger = init_logger(__name__)


def _maybe_restore_dflash2_bf16_linears(
    draft_model: nn.Module, hf_config
) -> list[tuple[str, tuple[int, ...]]]:
    """Disable optional WO FP8 copies only on the DFlash2 inner linears."""
    if os.getenv("VLLM_DFLASH2_BF16_LINEAR", "0") != "1" or (
        "DFlash2DraftModel" not in (getattr(hf_config, "architectures", None) or ())
    ):
        return []
    restored = []
    for name, module in draft_model.model.named_modules():
        if (
            isinstance(module, LinearBase)
            and getattr(module, "_wo_fp8_weight", None) is not None
        ):
            restored.append((name, tuple(module.weight.shape)))
            del module._wo_fp8_weight
            if hasattr(module, "_wo_fp8_scale"):
                del module._wo_fp8_scale
    return restored


def _maybe_restore_dflash2_bf16_shared_head(
    draft_model: nn.Module, target_lm_head: nn.Module | None, hf_config
) -> tuple[int, ...] | None:
    """Restore the shared target/draft head's BF16 math before graph capture."""
    if (
        os.getenv("VLLM_DFLASH2_BF16_SHARED_HEAD", "0") != "1"
        or "DFlash2DraftModel" not in (getattr(hf_config, "architectures", None) or ())
        or target_lm_head is None
        or getattr(draft_model, "lm_head", None) is not target_lm_head
    ):
        return None
    weight = getattr(target_lm_head, "weight", None)
    if (
        weight is None
        or weight.dtype != torch.bfloat16
        or getattr(target_lm_head, "_wo_fp8_weight", None) is None
    ):
        return None
    shape = tuple(weight.shape)
    del target_lm_head._wo_fp8_weight
    if hasattr(target_lm_head, "_wo_fp8_scale"):
        del target_lm_head._wo_fp8_scale
    return shape


def load_dflash_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.models.qwen3_dflash import (
        dflash_has_any_non_causal,
    )

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    # Select an attention backend that supports the drafter's attention: mixing
    # a non-causal layer onto a causal-only backend would fail.
    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=speculative_config.attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else vllm_config.cache_config
        ),
        load_config=get_pp_safe_draft_load_config(get_draft_load_config(vllm_config)),
    )
    with set_model_tag("dflash_head"):
        dflash_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    restored = _maybe_restore_dflash2_bf16_linears(
        dflash_model, draft_model_config.hf_config
    )
    if restored:
        logger.info(
            "DFlash2 BF16 linear experiment: disabled %d WO FP8 copies: %s",
            len(restored),
            restored,
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    # MuseGlimmerForCausalLM marks its inner MuseGlimmerModel as the language
    # model, so get_language_model() already returns the inner module and has
    # no .model of its own.
    target_inner = getattr(target_language_model, "model", target_language_model)
    draft_inner = dflash_model.model

    maybe_share_target_embed(dflash_model, draft_inner, target_inner)

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(dflash_model, "lm_head", None)
    if (
        target_lm_head is not None
        and not getattr(dflash_model.config, "has_own_lm_head", False)
        and _should_share(
            dflash_model, "has_own_lm_head", draft_lm_head, target_lm_head
        )
    ):
        if draft_lm_head is not None:
            del dflash_model.lm_head
        dflash_model.lm_head = target_lm_head

    shared_head_shape = _maybe_restore_dflash2_bf16_shared_head(
        dflash_model, target_lm_head, draft_model_config.hf_config
    )
    if shared_head_shape is not None:
        logger.info(
            "DFlash2 BF16 shared-head experiment: disabled shared target/draft "
            "WO FP8 copy, weight shape=%s; target head math changed",
            shared_head_shape,
        )

    return dflash_model
