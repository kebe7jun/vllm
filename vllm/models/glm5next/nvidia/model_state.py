# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 lifetime of the model-owned executor and its profiling/real KV bindings."""

from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState


class Glm5NextModelState(MambaHybridModelState):
    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        if hasattr(model, "get_language_model"):
            model = model.get_language_model()
        self.language_model = model
        model.build_specialized_decode()

    def initialize_kv_cache(self, kv_cache_config, block_tables):
        super().initialize_kv_cache(kv_cache_config, block_tables)
        # MRV2 replaces its graph manager before this hook; profiling graphs
        # have been discarded before the real cache is allocated.
        self.language_model.specialized_decode.prepare_kv(graphs_discarded=True)
