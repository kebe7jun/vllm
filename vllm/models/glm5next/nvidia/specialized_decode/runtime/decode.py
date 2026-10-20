# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One per-step plan over the existing GLM layer sequence and prepared Q4 cores."""

from dataclasses import dataclass

import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.models.glm5next.nvidia.specialized_decode.ops.fused_allreduce import (
    get_mhc_fused_allreduce,
)

from ..ops.kda_q4_parallel_history_fused_norm import (
    kda_q4_parallel_history_fused_norm,
)
from .contracts import CacheBinding, verification_metadata


@dataclass(frozen=True)
class Step:
    metadata: tuple
    cache_generation: int


class PreparedKDA:
    def __init__(self, layer, ordinal, owner):
        self.layer, self.ordinal, self.owner = layer, ordinal, owner
        conv, self.recurrent = layer.kv_cache
        self.conv = conv if layer._conv_state_dim_first else conv.transpose(-1, -2)
        if (
            tuple(self.conv.shape[1:]) != (3072, 6)
            or self.conv.dtype != torch.bfloat16
            or tuple(self.recurrent.shape[1:]) != (8, 128, 128)
            or self.recurrent.dtype != torch.float32
        ):
            raise ValueError(f"Unsupported KDA cache at {layer.prefix}")
        self.weight = layer._get_merged_conv_weight()
        arrivals = getattr(layer, "_q4_history_arrivals", None)
        if arrivals is None:
            arrivals = torch.zeros(2, 8, dtype=torch.int32, device=self.conv.device)
            layer._q4_history_arrivals = arrivals
        self.arrivals = arrivals
        self.output = torch.empty(
            4, 1024, dtype=torch.bfloat16, device=self.conv.device
        )

    def run(self, projected, gate_projected):
        md = self.owner.step.metadata[self.ordinal]
        layer = self.layer
        kda_q4_parallel_history_fused_norm(
            projected,
            gate_projected,
            self.weight,
            self.conv,
            md.spec_state_indices_tensor,
            md.num_accepted_tokens,
            layer.A_log,
            layer.dt_bias,
            self.recurrent,
            layer.o_norm.weight,
            self.output,
            self.arrivals,
            eps=layer.o_norm.eps,
        )
        layer._q4_capture_calls = getattr(layer, "_q4_capture_calls", 0) + 1
        return self.output


class SpecializedDecode:
    def __init__(self, language_model, config):
        if config is None:
            raise ValueError("SpecializedDecode requires an enabled DecodeConfig")
        self.language_model, self.backbone, self.config = (
            language_model,
            language_model.model,
            config,
        )
        self.layers = tuple(self.backbone._active_layers)
        self.kda = tuple(
            layer.self_attn
            for layer in self.layers
            if type(layer.self_attn).__name__ == "Glm5NextLinearAttention"
        )
        reasons = []
        if (
            len(self.layers) != 45
            or len(self.kda) != 34
            or self.backbone.config.hidden_size != 4096
            or self.backbone.config.hc_mult != 4
        ):
            reasons.append("full45 GLM-5.3-Flash with34 KDA/H4096/hc4 required")
        if tuple(self.backbone.aux_hidden_state_layers) != (6, 15, 25, 34, 43):
            reasons.append("DFlash aux taps must be (6,15,25,34,43)")
        if any(layer.is_sequence_parallel for layer in self.layers):
            reasons.append("sequence parallelism is unsupported")
        for layer in self.kda:
            if (
                layer.conv_size != 4
                or layer.head_dim != 128
                or layer.local_num_heads != 8
                or layer.q_conv1d.bias is not None
                or layer.kda_lower_bound not in (None, -5.0)
            ):
                reasons.append(f"unsupported KDA geometry: {layer.prefix}")
        count = sum(
            getattr(m, "_wo_fp8_weight", None) is not None
            for m in language_model.modules()
        )
        if count != 178:
            reasons.append(f"expected178 target WO copies, found{count}")
        if reasons:
            raise ValueError("GLM specialized model: " + "; ".join(reasons))
        self.names = tuple(layer.prefix for layer in self.kda)
        self.binding = CacheBinding()
        self.active, self.step = False, None
        self.prepared: tuple[PreparedKDA, ...] = ()
        self.fused_ar = get_mhc_fused_allreduce()
        self.plan_calls = self.specialized_calls = self.stock_calls = 0
        self.cache_bind_count = 0
        self._cache_modules = tuple(
            (name, m)
            for name, m in self.backbone.named_modules()
            if hasattr(m, "kv_cache")
        )

    def _fingerprint(self):
        rows = []

        def add(name, value):
            if isinstance(value, torch.Tensor):
                rows.append(
                    (
                        name,
                        value.data_ptr(),
                        tuple(value.shape),
                        tuple(value.stride()),
                        str(value.dtype),
                        str(value.device),
                    )
                )
            elif isinstance(value, (list, tuple)):
                for i, x in enumerate(value):
                    add(f"{name}:{i}", x)

        for name, module in self._cache_modules:
            add(name, module.kv_cache)
        return tuple(rows)

    def prepare_kv(self, *, graphs_discarded=False):
        if self.active:
            raise RuntimeError("Cannot rebind an active specialized decode")
        identity = self._fingerprint()
        if not identity:
            raise ValueError("No bound GLM caches")
        if self.binding.bind(identity, graphs_discarded=graphs_discarded):
            self.prepared = tuple(
                PreparedKDA(layer, i, self) for i, layer in enumerate(self.kda)
            )
            for layer, prepared in zip(self.kda, self.prepared):
                layer._glm53_prepared_q4 = prepared
            self.cache_bind_count += 1

    def plan(self, positions, intermediate_tensors):
        self.plan_calls += 1
        if self.binding.identity is None:
            raise RuntimeError("Specialized decode has no bound KV cache")
        if intermediate_tensors is not None or positions.shape[0] != 4:
            self.stock_calls += 1
            return None
        if not is_forward_context_available():
            self.stock_calls += 1
            return None
        context = get_forward_context()
        if context.cudagraph_runtime_mode not in (
            CUDAGraphMode.NONE,
            CUDAGraphMode.FULL,
        ):
            self.stock_calls += 1
            return None
        md = verification_metadata(
            context.attn_metadata, self.names, positions.shape[0]
        )
        if md is None:
            self.stock_calls += 1
            return None
        if torch.cuda.is_current_stream_capturing():
            self.binding.captured = True
        return Step(md, self.binding.generation)

    def run(self, step, input_ids, positions, inputs_embeds=None):
        if self.active or step.cache_generation != self.binding.generation:
            raise RuntimeError("Overlapping or stale specialized decode step")
        self.active, self.step = True, step
        self.specialized_calls += 1
        try:
            return self._run_layers(input_ids, positions, inputs_embeds)
        finally:
            self.active, self.step = False, None

    def _run_layers(self, input_ids, positions, inputs_embeds):
        backbone = self.backbone
        hidden_states = (
            inputs_embeds
            if inputs_embeds is not None
            else backbone.embed_input_ids(input_ids)
        )
        residual = post = comb = None
        aux_hidden_states = []
        for idx, layer in enumerate(self.layers, start=backbone.start_layer):
            if idx in backbone.aux_hidden_state_layers:
                aux_hidden_states.append(
                    backbone._aux_hidden_state(hidden_states, residual, post, comb)
                )
            hidden_states, residual, post, comb = layer(
                positions, hidden_states, residual, post, comb
            )
        if backbone.end_layer in backbone.aux_hidden_state_layers:
            aux_hidden_states.append(
                backbone._aux_hidden_state(hidden_states, residual, post, comb)
            )
        hidden_states = backbone.norm(hidden_states)
        return (
            (hidden_states, aux_hidden_states) if aux_hidden_states else hidden_states
        )

    def audit(self):
        return {
            "plan_calls": self.plan_calls,
            "specialized_calls": self.specialized_calls,
            "stock_calls": self.stock_calls,
            "cache_generation": self.binding.generation,
            "cache_bind_count": self.cache_bind_count,
            "captured": self.binding.captured,
            "kda_layers": list(self.names),
            "active": self.active,
            "cache_identity": self.binding.identity,
            "prepared_outputs": [x.output.data_ptr() for x in self.prepared],
            "arrivals": [x.arrivals.data_ptr() for x in self.prepared],
        }
