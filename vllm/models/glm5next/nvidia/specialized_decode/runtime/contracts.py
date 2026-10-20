# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only configuration, eligibility and cache-lifetime contracts."""

from dataclasses import dataclass
from typing import Any


def enabled(additional_config: dict[str, Any]) -> bool:
    backend = additional_config.get("decode_backend", "auto")
    if backend not in ("auto", "latency_specialized"):
        raise ValueError("decode_backend must be auto or latency_specialized")
    return backend == "latency_specialized"


@dataclass(frozen=True)
class Deployment:
    capability: int = 90
    device_name: str = "NVIDIA H200"
    tp: int = 8
    pp: int = 1
    dp: int = 1
    dcp: int = 1
    pcp: int = 1
    max_num_seqs: int = 1
    max_model_len: int = 8192
    quantization: str = "fp8"
    moe_backend: str = "triton"
    runner_v2: bool = True
    full_decode_graphs: bool = True
    compilation_mode: int = 0
    speculative_method: str = "dflash"
    speculative_tokens: int = 3
    features: tuple[str, ...] = ()


def unsupported(deployment: Deployment) -> tuple[str, ...]:
    d = deployment
    checks = (
        (d.capability != 90 or "H200" not in d.device_name, "H200 / SM90 required"),
        (d.tp != 8, "TP8 required"),
        (any(x != 1 for x in (d.pp, d.dp, d.dcp, d.pcp)), "PP/DP/DCP/PCP must be 1"),
        (d.max_num_seqs != 1, "max_num_seqs=1 required"),
        (d.max_model_len > 8192, "max_model_len<=8192 required"),
        (d.quantization != "fp8", "FP8 target checkpoint required"),
        (d.moe_backend != "triton", "triton MoE backend required"),
        (not d.runner_v2, "Model Runner V2 required"),
        (not d.full_decode_graphs, "FULL decode CUDA graphs required"),
        (d.compilation_mode != 0, "compilation mode NONE required"),
        (
            d.speculative_method != "dflash" or d.speculative_tokens != 3,
            "DFlash K3 required",
        ),
    )
    return tuple(reason for failed, reason in checks if failed) + tuple(
        f"unsupported feature: {name}" for name in d.features
    )


def validate(deployment: Deployment) -> None:
    reasons = unsupported(deployment)
    if reasons:
        raise ValueError("GLM specialized decode: " + "; ".join(reasons))


def verification_metadata(
    metadata: Any, layer_names: tuple[str, ...], num_tokens: int
) -> tuple[Any, ...] | None:
    if num_tokens != 4 or not isinstance(metadata, dict):
        return None
    selected = []
    for name in layer_names:
        md = metadata.get(name)
        if md is None or any(
            getattr(md, field, None) != expected
            for field, expected in (
                ("num_prefills", 0),
                ("num_decodes", 0),
                ("num_spec_decodes", 1),
                ("num_spec_decode_tokens", 4),
                ("num_actual_tokens", 4),
            )
        ):
            return None
        indices = getattr(md, "spec_state_indices_tensor", None)
        accepted = getattr(md, "num_accepted_tokens", None)
        if (
            indices is None
            or tuple(indices.shape) != (1, 4)
            or accepted is None
            or tuple(accepted.shape) != (1,)
        ):
            return None
        selected.append(md)
    return tuple(selected)


@dataclass
class CacheBinding:
    identity: tuple[Any, ...] | None = None
    generation: int = 0
    captured: bool = False

    def bind(
        self, identity: tuple[Any, ...], *, graphs_discarded: bool = False
    ) -> bool:
        if identity == self.identity:
            return False
        if self.captured and not graphs_discarded:
            raise RuntimeError("Cannot rebind captured specialized-decode caches")
        self.captured = False
        self.identity = identity
        self.generation += 1
        return True


def deployment_from_vllm(config: Any, capability: int, device_name: str) -> Deployment:
    p, m, c = config.parallel_config, config.model_config, config.cache_config
    spec, compilation = config.speculative_config, config.compilation_config
    offload = config.offload_config
    flags = (
        (getattr(p, "enable_expert_parallel", False), "EP"),
        (getattr(p, "enable_eplb", False), "EPLB"),
        (getattr(p, "use_ubatching", False), "DBO"),
        (config.lora_config is not None, "LoRA"),
        (config.kv_transfer_config is not None, "KV connector"),
        (config.weight_transfer_config is not None, "weight updates"),
        (getattr(m, "enable_sleep_mode", False), "sleep mode"),
        (getattr(m, "enable_prompt_embeds", False), "prompt embeddings"),
        (
            getattr(m, "multimodal_config", None) is not None
            and not m.multimodal_config.language_model_only,
            "multimodal execution",
        ),
        (c.enable_prefix_caching, "prefix caching"),
        (getattr(c, "use_replayssm", False), "ReplaySSM"),
        (getattr(c, "use_kda_recoverssm", False), "RecoverSSM"),
        (getattr(c, "cache_dtype", "auto") not in ("auto", "bfloat16"), "KV dtype"),
        (getattr(spec, "enable_adaptive_verification", False), "adaptive verification"),
        (
            offload.uva.cpu_offload_gb > 0 or offload.prefetch.offload_group_size > 0,
            "CPU weight offload",
        ),
        (
            getattr(config.aux_output_config, "enable_return_routed_experts", False),
            "routed-expert capture",
        ),
    )
    graphs = compilation.cudagraph_mode
    return Deployment(
        capability=capability,
        device_name=device_name,
        tp=p.tensor_parallel_size,
        pp=p.pipeline_parallel_size,
        dp=p.data_parallel_size,
        dcp=p.decode_context_parallel_size,
        pcp=getattr(p, "prefill_context_parallel_size", 1),
        max_num_seqs=config.scheduler_config.max_num_seqs,
        max_model_len=m.max_model_len,
        quantization=m.quantization,
        moe_backend=config.kernel_config.moe_backend,
        runner_v2=config.use_v2_model_runner,
        full_decode_graphs=graphs is not None and graphs.decode_mode().name == "FULL",
        compilation_mode=int(compilation.mode),
        speculative_method=getattr(spec, "method", ""),
        speculative_tokens=getattr(spec, "num_speculative_tokens", 0),
        features=tuple(name for active, name in flags if active),
    )
