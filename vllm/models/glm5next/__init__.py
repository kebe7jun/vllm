# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lazy model exports keep model-owned kernels importable by shared hooks."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .common.mtp import Glm5NextMTP
    from .nvidia.model import Glm5NextForCausalLM, Glm5NextForConditionalGeneration


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from vllm.platforms import current_platform

    if current_platform.is_xpu():
        raise NotImplementedError("GLM-5.3-Flash does not currently support XPU.")
    if name == "Glm5NextMTP":
        from .common.mtp import Glm5NextMTP

        return Glm5NextMTP
    if current_platform.is_cuda():
        from .nvidia.model import (
            Glm5NextForCausalLM,
            Glm5NextForConditionalGeneration,
        )
    else:
        from .common.model import (  # type: ignore[assignment]
            Glm5NextForCausalLM,
            Glm5NextForConditionalGeneration,
        )
    return {
        "Glm5NextForCausalLM": Glm5NextForCausalLM,
        "Glm5NextForConditionalGeneration": Glm5NextForConditionalGeneration,
    }[name]


__all__ = [
    "Glm5NextForCausalLM",
    "Glm5NextForConditionalGeneration",
    "Glm5NextMTP",
]
