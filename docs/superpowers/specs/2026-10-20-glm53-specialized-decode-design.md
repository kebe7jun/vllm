# GLM-5.3-Flash NVIDIA Specialized Decode

User-approved scope: implement the model-owned specialized-decode structure discussed in this chat, with no performance regression. This is a refactor of the measured stack, not a new kernel or numerical experiment.

## Contract
- Keep frozen v41 and v50 namespaces unchanged. Use an isolated worktree and new remote namespace.
- Reference is v50 BF16-compatible fused AR, full45 FP8/TP8/c1/DFlashK3/Q4/private-draftBF16/shared-WO. Compare identical precision and kernels; no numerical mode changes in the timed A/B.
- `additional_config={"decode_backend":"latency_specialized"}` explicitly opts in. `auto` preserves the inherited path; this branch's inherited generic optimization hooks are not claimed to be a new public framework.
- NVIDIA GLM wrapper owns SpecializedDecode. MRV2 ModelState builds after weights/aux configuration and binds profiling/real caches. A single plan per model forward selects eligible M4 uniform DFlash verification. Prefill, mixed, partial drafts, and other widths retain inherited forward.
- Configuration validation collects all unsupported features before load: CUDA SM90 H200, TP8/PP1/DP1/DCP1/PCP1, c1, FP8 target, triton MoE, FULL decode graphs, no LoRA/EP/EPLB/DBO/compile/sleep/offload/weight updates/routed capture/nondefault KV connector/ReplaySSM/RecoverSSM. Only DFlash K3 is declared supported in this first version.
- Post-load checks verify full45 model geometry, 34 KDA layers and aux-tap contracts, WO copy coverage, fusion/precision configuration. Cache bind validates BF16 conv [3072,6], FP32 recurrence [8,128,128], all static buffers and immutable graph-era binding. No GPU values are read for routing.
- Runtime prepares scratch before CUDA graph capture, reuses loaded weights and existing cache tensors, and uses the same layer sequence/aux handling/final norm as the baseline. No per-token GPU allocation or synchronization is added to graph replay.
- Model-specialized kernels move under glm5next/nvidia/specialized_decode; old import locations remain compatibility re-exports where needed by the inherited branch. Generic sampler/runner optimizations stay separate. Do not claim all inherited public-layer changes have been eliminated.
- No runtime exception fallback; errors fail the request/engine. Only planned eligibility rejection returns inherited forward.

## Validation and acceptance
CPU tests cover default/invalid options, aggregate refusals, uniform all-layer routing, partial/mixed/prefill declines, lifecycle and cache identity. Tests assert behavior with lightweight input records; GPU-backed tests validate actual imports and metadata.
On idle H200s run a bounded exact-source audit and full-model old/new comparison, preferably one load with separately retained graph banks. Use the same 64 unique GSM8K questions, natural EOS, and 3 excluded code warmups+9 formal 4096 ignoreEOS requests per cohort. If thermal/order variance makes a regression uncertain, reverse the order before any promotion. Main no-regression gate: pooled decode TPS no lower than baseline by >1% and estimated cycle no worse by >1%, with per-task/AL/trajectory caveats. An uncertain/failed gate keeps baseline recommended; never relabel a failed result as success.
Keep mean, whole-request max/min, AL/per-position acceptance, estimated cycle, all8 rank source/graph/cache audit and telemetry in the existing Site. Baseline itself has long-code run-to-run trajectory variation; exact complete-code equality is evidence, not an assumed invariant. No population accuracy-equivalence claim from64 samples.
