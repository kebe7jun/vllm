# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility import for the model-owned GLM decode implementation."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module(
    "vllm.models.glm5next.nvidia.specialized_decode.ops.topk_sigmoid"
)
