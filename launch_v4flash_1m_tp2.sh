#!/usr/bin/env bash
# =============================================================================
# DeepSeek-V4-Flash on 2x RTX PRO 6000 Blackwell (SM120, PCIe, no NVLink)
#   * FULL 1,048,576-token context on TP=2  (cudagraph + flashinfer autotune ON, NO --enforce-eager)
#   * 30/30 on the llm-inference-bench "estonia" long-context reasoning test (at GREEDY)
#   * ~108 tok/s single-user decode at any context length
#
# IMAGE:  verdictai/deepseek-v4-flash-sm120-1m   (docker pull)
#   = vLLM dev/unholy-fusion + lucifer1004 flashinfer sparse-MLA PR#3395 + b12x MoE
#     + DeepGEMM@sm120 + the 1M-context cudagraph decode-scratch patch
#     + a patch defaulting the chat encoder to thinking=true / reasoning_effort=high.
#
# MODEL:  huggingface.co/deepseek-ai/DeepSeek-V4-Flash  (download, mount read-only)
# =============================================================================
set -u
MODEL="${MODEL:-/path/to/DeepSeek-V4-Flash}"     # <-- EDIT: path to the downloaded model
PORT="${PORT:-9201}"
MAXLEN="${MAXLEN:-1048576}"                       # full 1M on TP2.  262144 = faster boot / more KV headroom
IMG="${IMG:-verdictai/deepseek-v4-flash-sm120-1m:latest}"
NAME="${NAME:-dsv4}"

docker rm -f "$NAME" 2>/dev/null
docker run -d --name "$NAME" --gpus all --ipc host --shm-size 32g \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_USE_B12X_MOE=1 \
  -e VLLM_ALLREDUCE_USE_SYMM_MEM=0 -e NCCL_P2P_LEVEL=SYS -e NCCL_NET_GDR_LEVEL=SYS \
  -e CUTE_DSL_ARCH=sm_120a -e TORCH_CUDA_ARCH_LIST=12.0a \
  -p 127.0.0.1:"$PORT":"$PORT" \
  -v "$MODEL":/model:ro \
  "$IMG" \
  vllm serve /model --served-model-name deepseek-v4-flash --trust-remote-code \
    --kv-cache-dtype fp8 --block-size 256 --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.92 --max-model-len "$MAXLEN" \
    --max-num-batched-tokens 8192 --max-num-seqs 128 --enable-prefix-caching \
    --tokenizer-mode deepseek_v4 --reasoning-parser glm45 --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice --disable-custom-all-reduce \
    --host 0.0.0.0 --port "$PORT"

cat <<'NOTE'
booted "dsv4" -> http://127.0.0.1:9201 . Watch: docker logs -f dsv4

  *** THE ONE THING THAT MATTERS FOR LONG-CONTEXT CORRECTNESS ***
  DO NOT set VLLM_USE_B12X_SPARSE_INDEXER. Its fast top-k selects the WRONG tokens and
  SILENTLY wrecks long-context retrieval (0-9/30 on estonia). Leaving it unset uses the
  native "Lightning Indexer" -> 30/30. This was the entire bug.

  * Reasoning is ON by default (thinking / reasoning_effort=high). Run reasoning evals at
    GREEDY (temperature 0) -- estonia is deterministic; the model default temp 1.0 scatters answers.
  * --disable-custom-all-reduce forces pure NCCL (vLLM's CUSTOM all-reduce errors on PCIe-no-NVLink).

  OPTIONAL SPEED LEVERS (do NOT affect greedy correctness -- the indexer was the only bug):
    - b12x PCIe one-shot all-reduce (faster TP comm): remove the NCCL_*/SYMM_MEM envs and
      --disable-custom-all-reduce; add  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x
    - MTP greedy spec-decode (~1.6x decode, lossless at temp 0; use MAXLEN<=262144 for the memory):
      add  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"attention_backend":"flashinfer","draft_sample_method":"greedy"}'
NOTE
