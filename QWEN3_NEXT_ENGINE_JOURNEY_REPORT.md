# Qwen3-Next Hybrid LLM Engine Journey Report

Date: 2026-03-07  
Repo: `rlray`  
Primary implementation: `xlmlib/qwen3_next_engine.py`

## Executive Summary
- The engine evolved from initial HF adaptation into a stable hybrid runtime (full attention + linear attention + MoE + TP).
- Recent stability fixes focused on linear-attention state mapping and prefill/decode branch correctness.
- User-reported quality milestone: AIME24 pass@1 = 90% (30 problems).
- Next focus is throughput, especially decode-heavy optimization.

## Key Changes Across the Journey
1. Added engine-compatible Qwen3-Next model wrappers and unified cache_params flow.
2. Implemented paged KV cache for full-attention layers.
3. Implemented GatedDeltaNet linear-attention path with recurrent decode + chunked prefill.
4. Added varlen packed prefill support via cu_seqlens (FLA path).
5. Added MoE backend strategy (triton/bmm/loop/auto) and pretransposed weight preparation.
6. Added TP support, collectives, and safety barriers around cache allocation.
7. Hardened stop-token and scheduler termination behavior.
8. Fixed linear state slot mapping semantics and constrained slot remap to decode path to avoid prefill hangs.

## Important Runtime Semantics
- `use_precomputed_states` is true only when:
  - `cache_params.has_previous_state` is true, and
  - `seq_len == 1` (decode shape contract).
- `linear_slots` lifecycle:
  - allocated per `seq_id` in runner,
  - passed via context in prefill/decode preparation,
  - consumed in linear-attention state read/write,
  - released when sequence finishes.
- `last_recurrent_state` is written back each step and is the linear-attention memory carried into the next decode token.

## Throughput Bottlenecks (Current)
- Decode loop dominates wall time for long generations.
- Effective decode batch size/coalescing determines token throughput.
- TP collectives and dynamic-kernel behavior can limit graph-style acceleration.
- Logging and host-side prep overhead in hot loops can be non-trivial.

## Throughput Speedup Plan (Prioritized)
### Phase A: Baseline metrics (first)
Track on fixed benchmark prompts:
- decode tok/s
- prefill tok/s
- TTFT
- average decode batch size
- GPU utilization and TP communication ratio

### Phase B: Low-risk wins
1. Reduce hot-path logging frequency in decode.
2. Improve scheduler decode coalescing under latency budget.
3. Ensure fast kernel paths are active (FLA/FlashAttention/Triton where supported).
4. Reuse decode prep buffers to reduce host->device overhead.

Expected gain: typically 5%–20% depending on current debug overhead and batching efficiency.

### Phase C: CUDA Graph (conditional)
- Use primarily for decode and stable shapes.
- Best suited to TP=1 and graph-friendly kernels.
- Keep TP graph capture guarded due collective mismatch/deadlock risk if capture differs across ranks.

Expected gain when applicable: typically 10%–30% decode throughput.

### Phase D: Workload controls
- Tune max completion length and stop behavior for long-generation tasks.
- This can reduce tail latency substantially in decode-dominant workloads.

### Phase E: Advanced optimization
- Speculative decoding exploration.
- Quantization experiments with quality gates.
- TP/NCCL overlap and communication tuning.

## Selected Milestone Commits
- `9dc1dd1` limit linear slot mapping to decode path to avoid prefill hang
- `5995edb` fix linear attention state slot mapping and engine generation semantics
- `ca68d3e` remove hardcoded legacy eos fallback in scheduler
- `a4ef149` fix engine termination checks for eos lists and prefill stop tokens
- `12ef2c1` align full-attn RoPE with HF and add diagnostics

## Next Concrete Steps
1. Build a reproducible benchmark script for decode/prefill throughput and TTFT.
2. Apply Phase B optimizations one-by-one and measure deltas.
3. Evaluate CUDA Graph only in eligible configs and compare replay-hit effectiveness.
4. Finalize default runtime profile for quality-first vs speed-first modes.
