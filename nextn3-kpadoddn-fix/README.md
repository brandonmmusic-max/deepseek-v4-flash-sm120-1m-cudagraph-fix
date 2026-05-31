# Bug report + fix: DeepGEMM SM120 `paged_mqa_logits` — native odd-`next_n` (`kPadOddN`) out-of-bounds access

**Component:** DeepSeek-V4 Lightning Indexer logits kernel, SM120 (sm_120a / RTX PRO 6000 Blackwell workstation) specialization
**Files (5):** `deep_gemm/include/deep_gemm/impls/sm120_fp8_paged_mqa_logits.cuh` + the FP4 twin `sm120_fp4_paged_mqa_logits.cuh` (kernels), `csrc/jit_kernels/impls/smxx_fp8_fp4_paged_mqa_logits.hpp` (host descriptor/launch setup), `csrc/apis/attention.hpp` (host gate), `tests/test_attention.py` (regression test)
**Severity:** Hard crash (CUDA `illegal memory access`) — makes the native `next_n=3` (MTP k=2 verify) path unusable on SM120.
**Status:** Root-caused and fixed; gold-standard validation = `compute-sanitizer` **34 errors → 0 errors** (FP8 + FP4) on the clean PR-target branch. End-to-end estonia 30/30 on a downstream build. Validated on real SM120 hardware.
**Validated diff:** `sm120_nextn3_kpadoddn.patch` (5 files — the authoritative patch; the kernel hunks below are quoted from it).

---

## TL;DR

The SM120 port of `paged_mqa_logits` **incompletely ported SM100's odd-`next_n` (`kPadOddN`) handling**. The odd `next_n` path (`next_n` odd and ≥ 3, e.g. `next_n=3` for MTP `num_speculative_tokens=2`) is the `# TODO (matt): integrate kernel with next_n = 4 support` in `vllm/v1/attention/backends/mla/indexer.py`, with `natively_supported_next_n_fp4 = [1, 2]`. Forcing the native `next_n=3` path (instead of the flatten-to-`next_n=1` fallback) crashes.

The odd leftover atom — the **last atom of each next_n group**, which holds **1 real token**, not `kNextNAtom = 2` — is mishandled in **three coordinated places in the kernel, plus a loop bound and a host gate**:

1. **Atom count (`kNumNextNAtoms`).** The SM120 compute kernel computes `kNumNextNAtoms = ceil_div(next_n,2) + 1` (= 3 for `next_n=3`) via a spurious `+ 1` in the `kPadOddN` branch, but the **metadata kernel** `smxx_paged_mqa_logits_metadata` builds the schedule with stride `ceil_div(next_n,2)` (= 2). The consumer then decodes the stride-2 schedule on the wrong stride → wrong `block_table` row / `context_lens` slot.
2. **Q load (`issue_tma_q`).** The odd atom's Q is TMA-copied as `kNextNAtom * kNumHeads` (2 tokens) even though only 1 is valid, so the kernel later **reads the never-populated 2nd-token SMEM slot** → **`Invalid __shared__ read of size 16 bytes`** — this is the actual sanitizer error. The fix loads only `kNumHeads` (1 token) for the odd atom via new `tensor_map_q_odd` / `tensor_map_weights_odd` (FP4 also `tensor_map_sf_q_odd`) TMA descriptors and zeroes the padding SMEM.
3. **Store (non-varlen dispatch).** The `else if constexpr (kPadOddN)` arm is missing, so the odd atom stores `Int<kNextNAtom>` (2 rows) instead of `Int<1>` — the 2nd row lands one past the `[B*next_n, max_model_len]` logits tensor.

Plus the **`q_idx` loop end-bound**: `batch_size` → `batch_size * kNumNextNAtoms` (the previous-q sentinel must count atoms, not requests). Plus a **host gate**: `csrc/apis/attention.hpp` asserts `next_n <= 2` for SM120 — widened to `next_n <= 3` to dispatch the path at all, and `csrc/jit_kernels/impls/smxx_fp8_fp4_paged_mqa_logits.hpp` builds the new `_odd` TMA descriptors host-side and passes them into the SM120 launch.

This is a genuine **odd-atom-path completion** across both the FP8 and FP4 SM120 kernels (~5 files) — **not** a 2-liner. It includes a real host (`.hpp`) change, not pure-header.

**Honest scope note:** the `+1` and the missing store arm DO match a static SM100-vs-SM120 divergence (verified by diffing the kernels). The **Q-load fix (the `_odd` descriptors) is the validated working solution for the SMEM-read OOB** — whether SM100 handles the odd-atom Q load by this exact mechanism is **not** verified, so treat it as a working fix the maintainer may want to reshape, not "restore the SM100 form."

---

## Environment

- 2× NVIDIA RTX PRO 6000 (Blackwell, **sm_120a**), PCIe (no NVLink), TP=2
- DeepSeek-V4-Flash, native Lightning Indexer, `index_topk=512`, 1M context, `--kv-cache-dtype fp8`
- MTP speculative decoding, `num_speculative_tokens=2` → verify pass over `1+k = 3` tokens → `next_n=3`
- SM120 DeepGEMM kernels (fork/community build — these kernels do **not** exist in upstream DeepGEMM `main`; see "Upstream status")

## Symptom

With the native `next_n=3` verify path enabled (i.e. bypassing the flatten fallback), both TP workers die during warmup. The user-visible error surfaces asynchronously — often at the next **synchronous** CUDA call after the async `fp8_fp4_paged_mqa_logits` launch (e.g. a downstream `persistent_topk` occupancy query, which is itself innocent and correctly sized for `num_rows = B*next_n`):

```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

`compute-sanitizer --tool memcheck` pins the offending kernel and gives the **actual** diagnostic — a shared-memory **read** out of bounds in `sm120_*_paged_mqa_logits`:

```
========= Invalid __shared__ read of size 16 bytes
=========     at void deep_gemm::sm120_fp8_paged_mqa_logits<(unsigned int)3, ...>(...)
========= ERROR SUMMARY: 34 errors
```

Note: the original triage guessed an OOB *write* on the store; the sanitizer shows the primary fault is an OOB *read* of shared memory on the Q-load path (the 2nd-token SMEM slot of the odd atom, which is never populated). The missing store arm is a real divergence too, but the SMEM-read OOB is what the sanitizer reports first.

## Root cause (detail)

`next_n` query tokens are packed into "atoms" of 2 (`kNextNAtom = 2`) so the tensor cores aren't fed a single row. Even `next_n` packs cleanly; **odd** `next_n` (≥3) yields one full pair + one leftover ("odd") atom — the **last atom of each next_n group** — that holds a single real token. Producer (metadata kernel) and consumer (compute kernel) must agree on the atom layout, the odd atom's Q must be loaded as one token, and its store must be clamped to one row. The SM120 port got all three wrong (plus a loop bound and the host gate).

**Place 1 — `kNumNextNAtoms` `+1` (atom count):**
```cpp
// sm120_fp8_paged_mqa_logits.cuh:151  (FP4 :157)  (BROKEN)
static constexpr uint32_t kNumNextNAtoms =
    kPadOddN ? (kNextN + kNextNAtom - 1) / kNextNAtom + 1
             : (kNextN + kNextNAtom - 1) / kNextNAtom;        // = 3 for next_n=3
```
vs. the metadata kernel (non-varlen branch), which builds the schedule with stride `ceil_div(next_n, 2) = 2`. With `kNumNextNAtoms=3`, the scheduler's `atom_to_token_idx` / `atom_to_block_table_row` / `get_num_kv` decode the metadata's stride-2 schedule on the wrong stride → wrong `block_table` row and out-of-range `context_lens` slot. SM100 uses `constexpr_ceil_div(kNextN, kNextNAtom)` with no `+1`.

**Place 2 — `issue_tma_q` loads 2 tokens for the odd atom (THE sanitizer error):**
```cpp
// sm120_fp8_paged_mqa_logits.cuh issue_tma_q (BROKEN — same path for every atom)
tma::copy<kHeadDim, kNextNAtom * kNumHeads, kHeadDim>(   // copies 2 tokens of Q
    &tensor_map_q, full_q_barriers[stage_idx], smem_q[stage_idx],
    0, q_token_idx * kNumHeads);
tma::copy<kNextNAtom * kNumHeads, 1, 0>(
    &tensor_map_weights, full_q_barriers[stage_idx], smem_weights[stage_idx],
    0, q_token_idx);
full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
```
The odd atom only has **1** real token, but the unconditional copy advances the TMA source coordinate as if 2 valid tokens exist and the kernel's later compute reads the 2nd-token SMEM slot, which was never filled → **`Invalid __shared__ read of size 16 bytes`**. The fix special-cases the odd atom: zero the padding SMEM, then TMA-copy only `kNumHeads` (1 token) of Q / weights / (FP4) scale-factors via dedicated one-token descriptors `tensor_map_q_odd` / `tensor_map_weights_odd` / `tensor_map_sf_q_odd`, with the barrier expecting the reduced byte count.

**Place 3 — missing odd-atom store clamp:**
```cpp
// sm120_fp8_paged_mqa_logits.cuh store dispatch (BROKEN)
if constexpr (kIsVarlen) {
    if (is_paired_atom) compute_and_store(cute::Int<kNextNAtom>{});
    else                compute_and_store(cute::Int<1>{});
} else {
    compute_and_store(cute::Int<kNextNAtom>{});   // unconditional Int<2> — no kPadOddN arm
}
```
The odd atom stores 2 rows; for the last request that second row is logits row `B*next_n` (one past the `[B*next_n, max_model_len]` tensor) → OOB write. SM100 has an `else if constexpr (kPadOddN)` arm that stores `Int<1>` for the trailing atom; SM120 dropped it.

**Plus — `q_idx` loop end-bound:** the previous-q sentinel is initialized to `batch_size` but must count **atoms**, so it becomes `batch_size * kNumNextNAtoms` (2 sites per kernel — the TMA-issue warp and the math warp).

**Plus — host gate + descriptor setup:** `csrc/apis/attention.hpp` asserts `next_n <= 2` on `arch_major == 12` (two asserts) — widened to `next_n <= 3` so the path dispatches at all; `csrc/jit_kernels/impls/smxx_fp8_fp4_paged_mqa_logits.hpp` constructs the new `_odd` one-token TMA descriptors host-side (guarded by `pad_odd_n = !is_varlen && next_n%2==1 && next_n>=3`) and passes them into the SM120 launch path.

## Why it was latent

On SM120, `next_n=3` normally takes the **flatten** path (`use_flattening=True`): each verify token is expanded into its own `next_n=1` row, so `kPadOddN` is never true and the broken arms are never reached. Only forcing the native `next_n=3` path exposes the defects. The native odd/large-`next_n` path was never finished for SM120 (`# TODO (matt)`, `natively_supported_next_n_fp4=[1,2]`).

## The fix (validated diff: `sm120_nextn3_kpadoddn.patch`, 5 files)

This is an **odd-atom-path completion**, not a 2-liner. The kernel hunks below are quoted directly from the validated diff (FP8 shown; the FP4 twin carries the identical structure with the extra `sf_q_odd` descriptor). Both FP8 and FP4 SM120 kernels are patched.

**(1) Atom count — drop the `+1`** (`sm120_fp8_paged_mqa_logits.cuh:151`, FP4 `:157`):
```diff
- static constexpr uint32_t kNumNextNAtoms = kPadOddN ? (kNextN + kNextNAtom - 1) / kNextNAtom + 1 : (kNextN + kNextNAtom - 1) / kNextNAtom;
+ static constexpr uint32_t kNumNextNAtoms = (kNextN + kNextNAtom - 1) / kNextNAtom;
```

**(2) Q load — one-token copy for the odd atom** (new `if constexpr (kPadOddN)` block in `issue_tma_q`):
```cpp
if constexpr (kPadOddN) {
    if (q_idx % kNumNextNAtoms == kNumNextNAtoms - 1) {
        auto smem_q_bytes = reinterpret_cast<uint8_t*>(smem_q[stage_idx]);
        for (uint32_t i = SMEM_Q_SIZE_PER_STAGE / kNextNAtom; i < SMEM_Q_SIZE_PER_STAGE; ++i)
            smem_q_bytes[i] = 0;                                  // zero the padded 2nd-token slot
        for (uint32_t i = kNumHeads; i < kNextNAtom * kNumHeads; ++i)
            smem_weights[stage_idx][i] = 0.0f;
        tma::copy<kHeadDim, kNumHeads, kHeadDim>(                 // ONE token, not kNextNAtom
            &tensor_map_q_odd, full_q_barriers[stage_idx], smem_q[stage_idx],
            0, q_token_idx * kNumHeads);
        tma::copy<kNumHeads, 1, 0>(
            &tensor_map_weights_odd, full_q_barriers[stage_idx], smem_weights[stage_idx],
            0, q_token_idx);
        full_q_barriers[stage_idx]->arrive_and_expect_tx(
            SMEM_Q_SIZE_PER_STAGE / kNextNAtom + SMEM_WEIGHT_SIZE_PER_STAGE / kNextNAtom);
    } else { /* full kNextNAtom pair copy */ }
} else { /* full kNextNAtom pair copy */ }
```
(FP4 adds the matching `tma::copy<kNumHeads,1,0>(&tensor_map_sf_q_odd, ...)` and zeroes `smem_sf_q`.)

**(3) Store — clamp the odd atom to one row** (new `else if` arm in the non-varlen store dispatch):
```diff
  } else if constexpr (kPadOddN) {
+     if (q_idx % kNumNextNAtoms == kNumNextNAtoms - 1)
+         compute_and_store(cute::Int<1>{});
+     else
+         compute_and_store(cute::Int<kNextNAtom>{});
  } else {
      compute_and_store(cute::Int<kNextNAtom>{});
  }
```

**(4) Loop end-bound** — count atoms, not requests (both the TMA-issue and math warps):
```diff
- uint32_t q_idx = batch_size, ...;
+ uint32_t q_idx = batch_size * kNumNextNAtoms, ...;
```

**(5) Host gate + `_odd` descriptors** — `csrc/apis/attention.hpp`:
```diff
- DG_HOST_ASSERT(next_n <= 2 and "SM120 paged MQA currently supports next_n <= 2");
+ DG_HOST_ASSERT(next_n <= 3 and "SM120 paged MQA currently supports next_n <= 3");
```
and `csrc/jit_kernels/impls/smxx_fp8_fp4_paged_mqa_logits.hpp` adds the `tensor_map_q_odd` / `tensor_map_weights_odd` (+ FP4 `tensor_map_sf_q_odd`) struct fields, builds them host-side with `make_tma_2d_desc(..., num_heads, ...)` (extent = 1 token) when `pad_odd_n`, and routes them into an SM120-specific `launch_kernel` arm.

The SM120 math-warp atom index is named `q_idx` (SM100 calls it `q_atom_idx`). The metadata kernel needs **no** change — it already computes `2`; the fix realigns the consumer (and the per-atom load/store). The headers are JIT-compiled, so clear the DeepGEMM JIT cache to force recompilation before testing.

**Honest caveat on the Q-load mechanism:** the `+1` (place 1) and the missing store arm (place 3) match a static SM100-vs-SM120 divergence I verified by diffing the kernels. The **Q-load fix (the `_odd` one-token descriptors) is a validated working solution for the SMEM-read OOB**, but whether SM100 handles the odd-atom load by this exact mechanism is **not** verified — the maintainer may prefer a different odd-atom load (e.g. reusing the main descriptor with an OOB-fill flag). Present it as a working fix, not as "restore the SM100 form."

(The authoritative patch is `sm120_nextn3_kpadoddn.patch` — 5 files, applied and validated as below.)

## Verification

**Gold-standard (authoritative evidence) — clean PR-target branch.** `compute-sanitizer --tool memcheck` on the clean `leavelet/DeepGEMM:sm120 @ 1845516035dcee28239fe8d638e363188133e028` checkout (the PR #324 head), `next_n=3`, both kernels:

| Check (clean branch) | Before | After |
|---|---|---|
| `compute-sanitizer` memcheck, FP8 `next_n=3` | `Invalid __shared__ read of 16 bytes` in `sm120_fp8_paged_mqa_logits`, **`34 errors`** | **`ERROR SUMMARY: 0 errors`** |
| `compute-sanitizer` memcheck, FP4 `next_n=3` | OOB in `sm120_fp4_paged_mqa_logits` | **`ERROR SUMMARY: 0 errors`** |
| `tests/test_attention.py::test_paged_mqa_logits`, `next_n ∈ {1,2,3}` × FP8/FP4 | crash at `next_n=3` | **all `VALIDATION_OK`** (FP8 `calc_diff ≈ 1.2e-6`, FP4 `≈ 4e-15`) |

Raw logs: `dg_clean_unpatched_nextn3_sanitizer.log` (34 errors), `dg_clean_patched_final_nextn3_sanitizer.log` (FP8 0), `dg_clean_patched_final_nextn3_fp4_sanitizer.log` (FP4 0), `dg_clean_patched_final_nextn123_sweep.log` (validation sweep). The clean-branch before/after is the **authoritative evidence** for leavelet's exact kernel.

**End-to-end (downstream build — context, not validation of leavelet's exact kernel).** The estonia 30/30 + ~185 tok/s figures below were measured on a **downstream build (unholy-fusion vLLM, Luke Alonso's lineage)** whose SM120 kernel differs slightly from leavelet's PR head — a *smaller* fix sufficed on that build. So treat these as "the native `next_n=3` path works end-to-end on a downstream build," **not** as validation of leavelet's exact kernel (that's what the clean-branch sanitizer numbers above are for):

| Check (downstream build) | Before | After |
|---|---|---|
| estonia 30-shot retrieval (greedy, byte-identical prompts) | n/a (crash) | **30 / 30 PASS** |
| single-user decode (TP2, MTP k=2, native `next_n=3`) | n/a (crash) | **~185–188 tok/s** |

Test harness: `llm_decode_bench.py --test-profile estonia --completion-stats-temperature 0` (greedy) + a single-user Marbury-style decode probe.
**estonia test + repro writeup (my repo):** `https://github.com/brandonmmusic-max/deepseek-sm120`

## Reproduce it yourself (any sm_120 GPU — RTX 5090 / RTX PRO 6000)

DeepGEMM JIT-compiles kernels on the GPU at runtime, so this reproduces directly from the maintainer's branch — no special build of the inference stack needed.

```bash
# 1. Clone the maintainer's SM120 branch (PR #324 head — the one with the bug)
git clone --recursive https://github.com/leavelet/DeepGEMM && cd DeepGEMM && git checkout sm120
git rev-parse HEAD            # record the commit you validated against

# 2. Build the JIT module (needs CUDA toolkit + torch)
./develop.sh                  # editable install of `deep_gemm`; kernels compile on first GPU call

# 3. Enable next_n=3 in the test. In tests/test_attention.py, the SM120 (arch_major==12)
#    next_n tuple is (1, 2) — change it to (1, 2, 3):
#      ... else (1, 2) if arch_major == 12 else ...   ->   ... else (1, 2, 3) if arch_major == 12 else ...

# 4. UNPATCHED -> reproduce the OOB
python tests/test_attention.py                                    # test_paged_mqa_logits FAILS at next_n=3
compute-sanitizer --tool memcheck python tests/test_attention.py  # illegal access in sm120_*_paged_mqa_logits

# 5. Apply the validated fix — the full 5-file odd-atom-completion diff, NOT a 2-edit tweak.
#    Files touched (sm120_nextn3_kpadoddn.patch):
#      csrc/apis/attention.hpp                                       (host gate next_n<=2 -> <=3)
#      csrc/jit_kernels/impls/smxx_fp8_fp4_paged_mqa_logits.hpp      (host: build _odd descriptors + SM120 launch arm)
#      deep_gemm/include/deep_gemm/impls/sm120_fp8_paged_mqa_logits.cuh
#      deep_gemm/include/deep_gemm/impls/sm120_fp4_paged_mqa_logits.cuh
#      tests/test_attention.py                                       (SM120 next_n arm (1,2) -> (1,2,3))
#    Kernel changes per .cuh: drop the +1 in kNumNextNAtoms; add the kPadOddN one-token
#    issue_tma_q block (tensor_map_q_odd / _weights_odd / FP4 _sf_q_odd); add the kPadOddN
#    Int<1> store arm; init q_idx sentinel to batch_size*kNumNextNAtoms.
git apply sm120_nextn3_kpadoddn.patch        # then clear the DeepGEMM JIT cache so it recompiles

# 6. PATCHED -> confirm
python tests/test_attention.py                                    # PASS (next_n=3, and 1/2 still pass)
compute-sanitizer --tool memcheck python tests/test_attention.py  # ERROR SUMMARY: 0 errors
```

Expected: step 4 faults inside `sm120_fp8_paged_mqa_logits` / `sm120_fp4_paged_mqa_logits` (`Invalid __shared__ read of 16 bytes`, 34 errors); step 6 is clean (0 errors, FP8 + FP4) and `test_paged_mqa_logits` passes at `next_n ∈ {1, 2, 3}`. The validated diff (`sm120_nextn3_kpadoddn.patch`) and the before/after sanitizer captures are in this repo.

## Upstream status (verified against source)

- vLLM **v0.22.0** (latest release) still gates `next_n` to `{1,2}` — `natively_supported_next_n_fp4 = [1, 2]` with `# TODO (matt): integrate kernel with next_n = 4 support` open — confirmed against the `v0.22.0` tag.
- There are **two** SM120 `paged_mqa_logits` kernels in the DeepGEMM forks (same author, `jasl` = `leavelet`):
  - **PR #318** (`jasl/DeepGEMM:sm120` @ `7a7a41a1`, base `main`) — a **row-major** rewrite with no atom logic. **Does NOT have this bug.** This is the kernel vLLM issue #41063 references.
  - **PR #324** (`leavelet/DeepGEMM:sm120` @ `1845516…`, base `deepseek-ai/DeepGEMM:nv_dev`) — an **atom-based port of the SM100 kernel** (ports `kPadOddN`/`compute_and_store`/`kNextNAtom`). **This is the kernel with the bug**, and the FP4 twin too.
- The #324 port **intends** odd-`next_n` support (it defines `kPadOddN = (!kIsVarlen) && (kNextN%2==1) && (kNextN>=3)`) but the odd-atom path is **incompletely ported** — the atom count (`+1`), the odd-atom Q load, and the odd-atom store are all wrong, plus the loop bound and the host gate → a **latent OOB**, dormant at `next_n ∈ {1,2}`, firing at `next_n=3`. The SM120 host gate (`next_n <= 2` in `attention.hpp`) makes the native path unreachable until widened; once widened, the kernel faults.
- Active context: **PR #342** (`liji-nv`, "Fix IMA guard in paged MQA logits scheduler") was **merged into `nv_dev` on 2026-05-29** — a sibling IMA fix in the same scheduler, good precedent.

## Suggested PR target

1. **DeepGEMM PR #324 — `leavelet/DeepGEMM:sm120`** (base `deepseek-ai/DeepGEMM:nv_dev`; open, mergeable). Apply the full validated 5-file odd-atom-completion diff (`sm120_nextn3_kpadoddn.patch`): the kernel changes to `sm120_fp8_paged_mqa_logits.cuh` + `sm120_fp4_paged_mqa_logits.cuh` (atom count, odd-atom one-token Q load via `_odd` descriptors, odd-atom store clamp, loop bound), the host `_odd` descriptor setup in `smxx_fp8_fp4_paged_mqa_logits.hpp`, the `next_n <= 3` gate in `attention.hpp`, and the `next_n=3` test. Frame as an **incomplete odd-`next_n` port → latent OOB** fix; note the atom-count + store divergences match SM100 but the `_odd` Q-load is a working solution the maintainer may reshape. Address `leavelet`; reference PR #342 as the sibling-fix precedent. **Do NOT target PR #318** — its row-major kernel is clean.
2. **vLLM** (separate, follow-on) — lift `natively_supported_next_n_fp4` to `[1, 2, 3]` and delete `# TODO (matt)`, once the kernel fix lands in an `nv_dev` DeepGEMM that vLLM pins.
