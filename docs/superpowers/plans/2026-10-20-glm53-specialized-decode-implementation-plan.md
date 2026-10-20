# GLM Specialized Decode Implementation Plan

> Execute inline with superpowers:executing-plans. User authorized implementation in this chat; no additional approval checkpoint is needed for reversible implementation/testing.

**Goal:** Package the measured GLM H200 decode stack as a model-owned opt-in backend without regression.
**Architecture:** NVIDIA wrapper + MRV2 lifecycle + one plan/run over the existing layer sequence. Preserve kernel arithmetic and runtime buffers; move specialized source with compatibility imports.
**Tech stack:** Python, Triton/TileLang existing kernels, vLLM MRV2, CUDA graphs.
**Spec:** design.md

## Global constraints
Frozen sources unchanged; v50 BF16-compatible arithmetic; full45/TP8/c1/DFlashK3/Q4; no external process termination; idle gate before all8 model load; all Python through .venv; no PR/push without separate authorization.

## Review focus
- Default-disabled backend must preserve model registry and inherited behavior.
- Every rank must agree on routing for mixed/partial/padded steps.
- Profiling and real cache rebinding must not leave stale pointers in captured graphs.
- Auxiliary FFN taps must continue to consume reduced outputs.
- Numerical profile and kernel geometry must not change during a packaging-only A/B.

## Tasks
1. Checkpoint the exact measured stack in an isolated worktree. Add CPU refusal/routing tests first and observe failure. Implement lazy config and metadata planning; run focused CPU suite.
2. Implement NVIDIA model/ModelState ownership, cache/model validation and prepared decode run. Copy/move existing model kernels without arithmetic changes; retain compatibility imports. Test routing/lifecycle/default behavior and static import/hash invariants. Run relevant linters and GPU correctness checks on actual source.
3. Prepare one-shot gated full-model A/B using unchanged numerical flags, fixed prompts and audited graph/cache identity. Run64math and9code requests per cohort, retain raw data and source hashes; reverse order if needed to resolve performance gate. No automatic promotion on incomplete/noisy data.
4. Fresh whole-change review, fix important findings with regression tests, update README and Site, archive final recommendation and no-regression evidence. Do not open a PR or modify shared branches.
