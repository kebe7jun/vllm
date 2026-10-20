# GLM-5.3-Flash specialized decode on H200

This opt-in backend packages the measured GLM decode implementation under the
model. Enable it with:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve /gpfs/rd/models/GLM-5.3-Flash \
  --tensor-parallel-size 8 --max-num-seqs 1 --max-model-len 8192 \
  --language-model-only --moe-backend triton --no-enable-prefix-caching \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,4]}' \
  --additional-config '{"decode_backend":"latency_specialized"}' \
  --speculative-config '{"method":"dflash","num_speculative_tokens":3,"model":"/home/jovyan/models/incoai/GLM-5.3-Flash-DFlash2","draft_sample_method":"greedy","rejection_sample_method":"standard","attention_backend":"FLASH_ATTN"}'
```

Model Runner V2, compilation mode NONE and FULL decode graphs are required.
The validated target is the full 45-layer FP8 checkpoint on eight unpartitioned
H200 GPUs. The first version supports c1 DFlash K3 verification (four target
rows), BF16 convolution caches and FP32 recurrent states. Other deployments
raise at startup. Prefill, mixed steps, partial drafts and other widths use
the inherited model forward.

`auto`, the default, does not build an executor or change optimization flags.
It preserves this branch's existing behavior, including legacy opt-in flags.

## Ownership and lifetime

The NVIDIA model owns `SpecializedDecode`. Its MRV2 ModelState builds the
executor after weight loading and auxiliary-layer selection, then validates
and binds each profiling or real KV cache. MRV2 discards profiling graphs
before the real cache bind. Rebinding a captured cache outside this lifecycle
is rejected. One `plan()` per model forward checks every KDA layer's uniform
verification metadata without reading GPU values. `run()` preserves the
existing layer order, auxiliary outputs and final normalization.

Weights and KV caches are reused. Q4 output buffers and counters are prepared
before capture. Graph replay adds no routing kernels, allocations or host
synchronization. A specialized failure raises; it is not caught and retried
through another implementation.

## Numerical profile and compatibility

The opt-in selects the measured profile: Q4 BV8 fused KDA, the existing small-M
MoE/GEMV and prefetch paths, a WO-FP8 copy of eligible BF16 target projections,
private BF16 draft linears, a shared WO head, and fused all-reduce with
rank-ordered **BF16 rounding after every rank**. Extra WO quantization and the
draft precision are numerical choices, not a claim of universal equivalence.
Conflicting legacy flags are rejected before weight loading.

The profile adapter binds inherited branch hooks once per target worker
process. Those generic linear/MoE, sampler and runner hooks remain separate
foundation changes; this package does not claim to remove every historical
public-layer change. Specialized implementation sources live in `ops/`;
legacy module paths remain compatibility aliases to the same module objects.
The ordinary shared mHC kernels remain in their original package; only the
measured single-kernel implementation is model-owned.

No claim of hardware portability, MTP/DSpark support, arbitrary batch sizes,
code correctness or population accuracy equivalence follows from the c1
validation. Performance claims require same-source, same-precision A/B results.

The backend follows model-owned specialized-decode integration. It retains
the existing multi-kernel layer execution; it is not a single persistent
CUDA launch for the whole model. The packaging alone has no claimed speedup.
