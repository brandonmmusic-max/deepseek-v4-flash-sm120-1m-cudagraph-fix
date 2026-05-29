#!/bin/bash
# Repro: DeepSeek-V4-Flash @ full 1M context on TP2, CUDA graph + flashinfer autotune
# BOTH ON (no --enforce-eager). Without the workspace patch this IMAs during CUDA graph
# capture whenever --max-num-seqs > 64. With the patch (or with --max-num-seqs 64) it boots.
# Stack: local-inference-lab/vllm@dev/unholy-fusion + b12x + lucifer flashinfer sparse-MLA
# sm120 (PR#3395) + DeepGEMM@sm120 + cutlass-dsl 4.5.1 + PR#42784. vLLM v0.1.dev1+gbfad804ed.
set -e
MODEL=${MODEL:-/path/to/deepseek-v4-flash}
IMG=${IMG:-your/vllm-v4-unholy:wsfix}   # base image + the 2-file patch applied (git apply -p1)
docker run -d --name vllm-v4-1m --gpus all --ipc host --shm-size 64m \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e VLLM_USE_B12X_MOE=1 -e VLLM_USE_B12X_SPARSE_INDEXER=1 -e CUTE_DSL_ARCH=sm_120a -e TORCH_CUDA_ARCH_LIST=12.0a \
  -p 127.0.0.1:9201:9201 -v "$MODEL":"$MODEL":ro "$IMG" \
  vllm serve "$MODEL" --served-model-name deepseek-v4-flash --trust-remote-code \
    --tensor-parallel-size 2 --kv-cache-dtype fp8 \
    --max-model-len 1048576 --gpu-memory-utilization 0.90 \
    --max-num-batched-tokens 2048 --max-num-seqs 128 \
    --host 0.0.0.0 --port 9201
# cudagraph FULL_AND_PIECEWISE + flashinfer autotune are ON by default (no flags needed).
# NO-PATCH WORKAROUND: drop --max-num-seqs to 64 (keeps every uniform-decode FULL graph
# within the 64-token decode-scratch reserve -> no mid-capture workspace growth -> no IMA).
