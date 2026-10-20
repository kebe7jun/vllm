# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA model ownership for the optional GLM specialized decode executor."""

from ..common.model import Glm5NextForCausalLM as StockCausalLM
from ..common.model import (
    Glm5NextForConditionalGeneration as StockConditionalGeneration,
)
from .specialized_decode.runtime import (
    enable_measured_profile,
    specialized_decode_config,
)


class Glm5NextForCausalLM(StockCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        decode_config = specialized_decode_config(vllm_config)
        if decode_config is not None:
            enable_measured_profile()
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.specialized_decode_config = decode_config
        self.specialized_decode = None
        self._specialized_decode_selected = True

    def build_specialized_decode(self):
        from .specialized_decode.runtime.decode import SpecializedDecode

        if self.specialized_decode is not None:
            raise RuntimeError("GLM specialized executor is already built")
        self.specialized_decode = SpecializedDecode(
            self, self.specialized_decode_config
        )

    def get_model_state_cls(self):
        if self.specialized_decode_config is not None:
            from .model_state import Glm5NextModelState

            return Glm5NextModelState
        from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

        return MambaHybridModelState

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        decode = self.specialized_decode
        if decode is not None and self._specialized_decode_selected:
            step = decode.plan(positions, intermediate_tensors)
            if step is not None:
                return decode.run(step, input_ids, positions, inputs_embeds)
        return self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )


class Glm5NextForConditionalGeneration(StockConditionalGeneration):
    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def get_model_state_cls(self):
        return self.language_model.get_model_state_cls()
