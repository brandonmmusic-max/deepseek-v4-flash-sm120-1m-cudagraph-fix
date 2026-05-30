# DeepSeek-V4-Flash on 2× RTX PRO 6000 Blackwell (SM120): full **1M context on TP=2** *and* **30/30 on estonia** — ready-to-pull image + the one env flag that was silently killing long-context accuracy

Two results on 2× RTX PRO 6000 Blackwell (SM120, PCIe, **no NVLink**), official `deepseek-ai/DeepSeek-V4-Flash` (fp8), vLLM `dev/unholy-fusion` stack (b12x + lucifer1004 flashinfer sparse-MLA PR#3395 + DeepGEMM@sm120):

1. **Full `--max-model-len 1048576` (1M) on TP=2**, with **cudagraph + flashinfer autotune ON** (no `--enforce-eager`). KV pool **1.32× at 1M**, **~108 tok/s** single-user decode at any context.
2. **29–30/30 on the `estonia` long-context reasoning test** (`llm-inference-bench`), at greedy, benchmark unmodified — **30/30 on a clean run**; it wobbles by ~1 at concurrency 30 (batching nondeterminism, *not* a config issue — see *Results* below).

Pull it: `docker pull verdictai/deepseek-v4-flash-sm120-1m` · launch script + clickable estonia TUI + the patch: **github.com/brandonmmusic-max/deepseek-v4-flash-sm120-1m-cudagraph-fix**

---

## ⚠️ The estonia fix everyone on this stack needs: **do NOT set `VLLM_USE_B12X_SPARSE_INDEXER`**

This cost me a full day. On the official fp8 model I was getting **0–9/30** on estonia and chased *everything* — temperature (it must be **greedy**, the model default temp 1.0 scatters answers), reasoning mode, the reasoning parser, even the PCIe all-reduce. All red herrings.

The actual culprit: **`VLLM_USE_B12X_SPARSE_INDEXER=1`**. That kernel is the thing that picks **which 512 tokens** the model attends to (`index_topk=512`). Its fast top-k was selecting the **wrong** tokens — so on the estonia needle (a multi-hop chain `bench → cassette MX-88 → vendor V-441 → Mirel **Instrument** → Estonia`, with a near-identical-name distractor `V-447 → Mirel **Industrial** → Latvia`) the model **literally never saw the Estonia link** and deterministically answered "Latvia," often looping to the token cap.

**Unset that env → native "Lightning Indexer" → 29–30/30** (30/30 on a clean run, up from **0–9/30**), and reasoning *collapses* from p50 6,839 → **~1,750–2,060** tokens with **0 runs hitting the token cap** (it commits fast and correct once it can see the right context). Same model, same prompt, one env flag.

This matters way beyond estonia: the b12x sparse indexer **silently corrupts any long-context retrieval**. If you serve V4-Flash for RAG / long documents / agents, **drop it.** Its real win is high-concurrency throughput, not single-user — at single-user the native indexer costs ~nothing (see speed below).

The other two non-obvious knobs:
- **Reasoning must be ON.** vLLM's `deepseek_v4` chat encoder defaults `thinking=false`. Either `--default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"high"}'` (+ `--reasoning-parser deepseek_v4`), or the equivalent I used: a server-side default + `--reasoning-parser glm45` (= `DeepSeekV3ReasoningWithThinkingParser`, which forces thinking so it actually splits `<think>`). Note `--reasoning-parser deepseek_v4` **without** the default-kwargs falls back to the *identity* parser and never splits — a real footgun.
- **`--disable-custom-all-reduce`.** vLLM's CUSTOM all-reduce errors on PCIe-no-NVLink (`invalid argument`); this forces pure NCCL. (b12x PCIe all-reduce also works — it was **not** the bug.)

---

## Results & run-to-run variance

- **30/30 on a clean run; 29–30/30 across runs** (greedy, concurrency 30, 30 measured requests).
- **0 runs hit the token cap** — no thinking-loops. Reasoning is short and decisive: **p50 ~1,750–2,060 tokens** (vs 6,800+ *before* the fix, when it ruminated its way into the distractor).
- The ~1-run wobble is **batching nondeterminism**, not a regression: at greedy + concurrency 30, fp8 + MoE + sparse-attention reduction order shifts with how the 30 requests happen to batch together, so once in a while a single request commits to the "Latvia" distractor early (a short ~880-token trace). Want a rock-solid 30/30 for a demo? Drop the concurrency (e.g. `--profile-concurrency 8`) — fewer requests in flight = more deterministic.
- For contrast, the **same config scored 0–9/30** before dropping the b12x indexer.

---

## How the 1M-on-TP=2 works (the cudagraph patch)

Out of the box, V4-Flash's sparse-MLA SM120 decode **IMAs during CUDA-graph capture** at large `--max-model-len` — so people run `--enforce-eager` (slow) or cap context ~128K.

Root cause: the decode path reserves a shared split-K **decode scratch** during memory profiling, but sizes its leading (token) dim from `_max_decode_workspace_tokens(...)` which is **hard-capped at `_DECODE_MAX_TOKENS = 64`**. CUDA-graph capture then issues uniform-decode batches up to `max_num_seqs` tokens — and the moment a captured batch exceeds 64, the shared `WorkspaceManager` has to **grow → `torch.accelerator.empty_cache()` mid-capture**, which is illegal during graph capture → `CUDA error: an illegal memory access`.

Fix (2-file, pure-python, no recompile): reserve the scratch's leading token dim during profiling for `min(max_num_batched_tokens, max_num_seqs * (1 + num_speculative_tokens))` — the largest decode batch capture will actually issue. Only the leading token dim grows (`num_heads / num_splits / d_v` unchanged), so per-call decode views are **byte-identical** — no math change, no speed nerf. Capture survives → full 1M on TP2 with cudagraph + autotune ON.

KV at 1M on TP2: weights ~74 GiB/GPU, KV pool **9.16 GiB = 1.39M token-slots = 1.32× of 1,048,576** (fits a full-1M request with headroom). No-patch workaround if you can't apply it: `--max-num-seqs 64` (keeps every uniform-decode graph within the 64-token reserve). Patch + repro in the repo.

---

## Speed — the native indexer is basically free at single-user

Streaming, TTFT-corrected decode rate (greedy, single user):

| context | TTFT (prefill) | decode |
|---|---|---|
| short | 0.1 s | **109.1 tok/s** |
| ~120K | 22.0 s | **104.4 tok/s** |

Decode is **flat ~104–109 tok/s from short context to 120K** — the native Lightning Indexer's per-token cost at single-user is negligible. The only long-context cost is **prefill** (~5,450 tok/s ingest at 120K), which is inherent, not the indexer.

**Want more speed?** None of these touch greedy correctness (the indexer was the only correctness bug):
- **b12x PCIe one-shot all-reduce** — faster TP comm than NCCL on PCIe. Swap `--disable-custom-all-reduce` + the NCCL envs for `VLLM_ENABLE_PCIE_ALLREDUCE=1 VLLM_PCIE_ALLREDUCE_BACKEND=b12x`.
- **MTP greedy spec-decode — *don't bother for single-user, it's a net slowdown here.*** With the native indexer (the one you need for 30/30 estonia), MTP k=2 measured **82 tok/s vs 111 no-MTP at an identical config**, despite a healthy **2.48 accept-len**. The verify pass runs the Lightning Indexer over (1+k) tokens (k=2 flattens to 3 single-token launches), and the indexer is the decode bottleneck — so each MTP step costs ~3.3× a plain decode, more than the 2.48× acceptance buys back. (Any "~1.6× MTP" figure on this stack was measured with the **b12x sparse indexer** — i.e. the broken-retrieval path.) MTP only wins here with a *trained* multi-token drafter (accept-len > ~3.3) or a faster-but-correct indexer.
- The real high-concurrency win would be **fixing the b12x indexer's top-k** so you get its speed *and* correct retrieval — that's the open upstream problem.

---

## Config (full, copy-paste)

```bash
docker run -d --name dsv4 --gpus all --ipc host --shm-size 32g \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_USE_B12X_MOE=1 \
  `# NOTE: VLLM_USE_B12X_SPARSE_INDEXER intentionally NOT set -- it corrupts long-ctx retrieval` \
  -e VLLM_ALLREDUCE_USE_SYMM_MEM=0 -e NCCL_P2P_LEVEL=SYS -e NCCL_NET_GDR_LEVEL=SYS \
  -e CUTE_DSL_ARCH=sm_120a -e TORCH_CUDA_ARCH_LIST=12.0a \
  -p 127.0.0.1:9201:9201 -v /path/to/DeepSeek-V4-Flash:/model:ro \
  verdictai/deepseek-v4-flash-sm120-1m:latest \
  vllm serve /model --served-model-name deepseek-v4-flash --trust-remote-code \
    --kv-cache-dtype fp8 --block-size 256 --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.92 --max-model-len 1048576 \
    --max-num-batched-tokens 8192 --max-num-seqs 128 --enable-prefix-caching \
    --tokenizer-mode deepseek_v4 --reasoning-parser glm45 --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice --disable-custom-all-reduce --host 0.0.0.0 --port 9201
```
Run estonia at **greedy**: `llm_decode_bench.py --port 9201 --test-profile estonia --completion-stats-temperature 0`

---

## Can anyone reproduce?

If you've got 2× (or more) RTX PRO 6000 Blackwell / other SM120:
1. Does `docker pull verdictai/deepseek-v4-flash-sm120-1m` + the config above give you **~29–30/30 estonia** at greedy? (And do you see the same ~1-run wobble at concurrency 30, or does your stack hold a flat 30?)
2. Does adding `VLLM_USE_B12X_SPARSE_INDEXER=1` tank your estonia score (confirming the indexer is the culprit)?
3. Does `--max-model-len 1048576` boot on **TP=2** for you with cudagraph (no enforce-eager)?

Curious whether the b12x-indexer long-context regression reproduces on single-SKU rigs and other models (GLM-5.1, etc.) — and whether anyone's already root-caused *why* its top-k diverges. Image + patch + launch script + a clickable estonia TUI are all in the repo.
