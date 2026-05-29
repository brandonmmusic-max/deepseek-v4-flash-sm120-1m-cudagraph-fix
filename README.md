# DeepSeek-V4-Flash sm120: CUDA-graph IMA at large `--max-model-len` is a decode-scratch reservation bug (fix inside) — can anyone repro?

**TL;DR** — On the unholy-fusion vLLM stack, V4-Flash crashes with `CUDA error: an illegal memory access` **during CUDA graph capture** whenever you boot with a large `--max-model-len` (e.g. 1048576), cudagraph ON (no `--enforce-eager`), **and `--max-num-seqs > 64`**. It's *not* really about 1M context and it's *not* the mixed-SKU autotune key — it's the sparse-MLA decode scratch being reserved for only 64 tokens while CUDA-graph capture issues uniform-decode batches up to `max_num_seqs`. I have a 2-line-of-logic, pure-Python, no-recompile fix that boots full 1M on TP2 with **cudagraph + flashinfer autotune both ON**. Looking for a repro/refute before I PR it.

## Stack
`local-inference-lab/vllm@dev/unholy-fusion` (`v0.1.dev1+gbfad804ed`) + b12x@master + lucifer flashinfer sparse-MLA sm120 (PR#3395) + lucifer DeepGEMM@sm120 + cutlass-dsl 4.5.1 + PR#42784. Rig: 4× RTX PRO 6000 Blackwell (sm120, mixed Max-Q + full Workstation Edition, PCIe no-NVLink), 300 W cap.

## Symptom
```
(EngineCore) RuntimeError: Worker failed with error 'CUDA error: an illegal memory access was encountered'
```
…thrown while "Capturing CUDA graphs (decode, FULL)". Add `--enforce-eager` and it runs fine — which is exactly why it *looks* like "1M just needs eager." It doesn't.

## Root cause
The sparse-MLA sm120 decode path reserves a shared split-K **decode scratch** during memory profiling (`_reserve_decode_workspace` → `_get_decode_scratch(...)`). The scratch's **leading (token) dim** is sized by `_max_decode_workspace_tokens(...)`, which is **hard-capped at `_DECODE_MAX_TOKENS = 64`**.

But the **uniform-decode FULL CUDA graphs** are captured for batch sizes up to `max_num_seqs` (× `(1 + num_speculative_tokens)` under MTP). The moment capture issues a decode batch **> 64 tokens**, the shared `WorkspaceManager` must grow → it calls `torch.accelerator.empty_cache()` + realloc → **illegal during graph capture → IMA**.

So the trigger is "captured decode batch > 64", not "context = 1M". Large `--max-model-len` just makes people boot with a big default `max_num_seqs`, so they hit it.

**Red herring I chased first:** flashinfer's AutoTuner keys its config cache by `torch.cuda.get_device_name()`, and my rig mixes "…Max-Q Workstation Edition" with "…Workstation Edition" (same silicon, different power bin). I patched the tuner to normalize the name so all TP ranks share one config — **the IMA persisted**. So it's not a cross-rank autotune mismatch. (Still a reasonable hygiene patch for mixed-SKU boxes, but not this bug.)

## Fix (pure-Python, no recompile)
Reserve the decode scratch's leading token dim during profiling for the **largest decode batch capture will actually issue**, instead of the 64-token dispatch bound:

`attention.py` (`DeepseekV4MLAAttention.__init__`):
```python
_num_spec_tokens = (vllm_config.speculative_config.num_speculative_tokens
                    if vllm_config.speculative_config is not None else 0)
self.max_decode_tokens = min(
    self.max_num_batched_tokens,
    vllm_config.scheduler_config.max_num_seqs * (1 + _num_spec_tokens),
)
```
`nvidia/sm120.py` (`_reserve_decode_workspace`): pass `min(max_decode_tokens, max_num_batched_tokens)` as the scratch's leading dim instead of `_max_decode_workspace_tokens(max_num_batched_tokens)`.

Only the **leading token dim** grows — `num_heads / num_splits / d_v` and all trailing strides are unchanged, so every per-call decode view is **byte-identical** (kernel still guards `t_idx >= num_tokens`). No decode-math change; the workspace is fully sized during warmup and never reallocs mid-capture. Full patch attached.

**No-patch workaround:** boot with `--max-num-seqs 64`. Every uniform-decode FULL graph then stays ≤ the 64-token reserve → no growth → no IMA. Costs you decode-batch concurrency.

## Evidence (this fix, TP2, full 1M, cudagraph + autotune ON)
```
--tensor-parallel-size 2 --max-model-len 1048576 --kv-cache-dtype fp8
--gpu-memory-utilization 0.90 --max-num-batched-tokens 2048 --max-num-seqs 128
(enforce_eager=False, cudagraph_mode=FULL_AND_PIECEWISE, enable_flashinfer_autotune=True)
```
- Boots to READY ~310 s. Captures **all 19 decode-FULL graphs (sizes 1…128)** + 35 mixed-PIECEWISE graphs (…256) with **zero IMA**.
- KV pool: **8.26 GiB = 1,807,465 tokens = 1.72× concurrency at 1M.**
- Serves a **162,019-token-context** request cleanly (the exact path that IMA'd before). 0 IMA lines post-serve.
- ~100–107 tok/s single-user decode (TP2, no-MTP — this boot is a 1M/cudagraph correctness proof, not the throughput-max config).

## Bonus: same fix unlocks MTP at 1M (no extra work)
Because the reservation includes the `(1 + num_speculative_tokens)` factor, turning on MTP "just works" — the decode scratch auto-sizes for the 3-token draft batch and capture still survives:
```
--speculative-config '{"method":"mtp","num_speculative_tokens":2,"draft_sample_method":"probabilistic","moe_backend":"b12x"}'
(TP2, --max-model-len 1048576, --gpu-memory-utilization 0.92, --max-num-seqs 128, cudagraph + autotune ON)
```
- READY, cudagraph capture survives, **0 IMA**.
- KV pool 6.96 GiB = 1,533,447 tokens = **1.46× at 1M** (full 1M still fits with MTP).
- **149.5 tok/s @256 / 170.2 tok/s @512** single-user decode (vs ~100–107 no-MTP).
- MTP acceptance: mean length 2.29/3, avg draft acceptance ~65%, per-position [0.851, 0.439].
- (Logs flag `max_num_scheduled_tokens=2048` capping draft slots — raising `--max-num-batched-tokens` should push this further.)

## Ask
If you're on sm120 (single-SKU is fine — even better, since it rules out my mixed-SKU box) or any other arch: can you repro the IMA at `--max-num-seqs > 64` + cudagraph + large `--max-model-len`, and confirm either the patch **or** `--max-num-seqs 64` clears it? If it reproduces off my rig I'll open the PR. Repro script + patch attached.
