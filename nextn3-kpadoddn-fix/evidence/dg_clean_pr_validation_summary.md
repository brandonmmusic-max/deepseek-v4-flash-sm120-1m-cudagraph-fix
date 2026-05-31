# DeepGEMM SM120 next_n=3 PR validation

Clean checkout:

```text
repo: https://github.com/leavelet/DeepGEMM
branch: sm120
HEAD: 1845516035dcee28239fe8d638e363188133e028
GPU: NVIDIA RTX PRO 6000 Blackwell, sm_120a visible as CUDA capability (12, 0)
container: klc/vllm-v4-unholy:wsfix-think
```

Important note: the branch has host-side SM120 guards rejecting `next_n > 2`.
The first unmodified next_n=3 run therefore stops at:

```text
RuntimeError: Assertion error (csrc/apis/attention.hpp:218): next_n <= 2 and "SM120 paged MQA currently supports next_n <= 2"
```

For kernel repro, the guard was widened to `next_n <= 3` and
`tests/test_attention.py` was changed to include SM120 `next_n=3`.

## Before

Raw logs:

```text
/tmp/codex_b12x/results/dg_clean_unpatched_nextn3_repro.log
/tmp/codex_b12x/results/dg_clean_unpatched_nextn3_sanitizer.log
```

Normal targeted next_n=3 repro:

```text
EXCEPTION AcceleratorError: CUDA error: an illegal memory access was encountered
...
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

Memcheck:

```text
========= Invalid __shared__ read of size 16 bytes
=========     at void deep_gemm::sm120_fp8_paged_mqa_logits<(unsigned int)3, (unsigned int)64, (unsigned int)128, (unsigned int)64, (bool)1, (bool)0, (unsigned int)2, (unsigned int)3, (unsigned int)128, (unsigned int)128, (unsigned int)256, float>(...)
...
========= ERROR SUMMARY: 34 errors
```

## After

Raw logs:

```text
/tmp/codex_b12x/results/dg_clean_patched_final_nextn123_sweep.log
/tmp/codex_b12x/results/dg_clean_patched_final_nextn3_sanitizer.log
/tmp/codex_b12x/results/dg_clean_patched_final_nextn3_fp4_sanitizer.log
/tmp/codex_b12x/results/dg_clean_nextn3_fix.diff
```

Targeted validation sweep:

```text
LAUNCH_OK fp4=False next_n=1 logits_shape=(256, 2048)
VALIDATION_OK fp4=False next_n=1 calc_diff=1.22033e-06
LAUNCH_OK fp4=False next_n=2 logits_shape=(512, 2048)
VALIDATION_OK fp4=False next_n=2 calc_diff=1.1859e-06
LAUNCH_OK fp4=False next_n=3 logits_shape=(768, 2048)
VALIDATION_OK fp4=False next_n=3 calc_diff=1.20595e-06
LAUNCH_OK fp4=True next_n=1 logits_shape=(256, 2048)
VALIDATION_OK fp4=True next_n=1 calc_diff=4.10783e-15
LAUNCH_OK fp4=True next_n=2 logits_shape=(512, 2048)
VALIDATION_OK fp4=True next_n=2 calc_diff=3.9968e-15
LAUNCH_OK fp4=True next_n=3 logits_shape=(768, 2048)
VALIDATION_OK fp4=True next_n=3 calc_diff=3.9968e-15
```

FP8 memcheck:

```text
========= COMPUTE-SANITIZER
LAUNCH_OK fp4=False next_n=3 logits_shape=(768, 2048)
========= ERROR SUMMARY: 0 errors
```

FP4 memcheck:

```text
========= COMPUTE-SANITIZER
LAUNCH_OK fp4=True next_n=3 logits_shape=(768, 2048)
========= ERROR SUMMARY: 0 errors
```

## Patch scope

The final diff is:

```text
csrc/apis/attention.hpp
csrc/jit_kernels/impls/smxx_fp8_fp4_paged_mqa_logits.hpp
deep_gemm/include/deep_gemm/impls/sm120_fp8_paged_mqa_logits.cuh
deep_gemm/include/deep_gemm/impls/sm120_fp4_paged_mqa_logits.cuh
tests/test_attention.py
```

It includes:

```text
1. Enable SM120 next_n=3 host-side dispatch.
2. Match SM120 kNumNextNAtoms to the scheduler: ceil_div(kNextN, kNextNAtom), no odd +1.
3. Use one-token odd TMA descriptors for the leftover next_n atom.
4. Initialize the previous-q sentinel to batch_size * kNumNextNAtoms.
5. Store the odd leftover atom with compute_and_store(cute::Int<1>{}).
```
