#!/usr/bin/env bash
# Judge both E4 pools with Llama-3.1-70B-Instruct (CRAG's official judge), bf16,
# TP=2 across both A100s (lossless). Download already complete.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
# The two A100s lack working GPU P2P (separate PCIe complexes) -> NCCL's default
# peer transport fails at init (ncclAllReduce). Force shared-memory/host transport.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
MODEL=meta-llama/Llama-3.1-70B-Instruct
echo "=== judge e5 (Llama-70B TP=2) ==="
on_gpus 2 78000 uv run python -m scdn.e4_judge \
  --preds data/e4/granularity_quality.preds.jsonl --tag e5-llama70b --model "$MODEL" \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.92 --max-model-len 2048 \
  --out data/e4/judge_e5_llama70b.jsonl
echo "=== judge qwen3 (Llama-70B TP=2) ==="
on_gpus 2 78000 uv run python -m scdn.e4_judge \
  --preds data/e4/granularity_quality_qwen3.preds.jsonl --tag qwen3-llama70b --model "$MODEL" \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.92 --max-model-len 2048 \
  --out data/e4/judge_qwen3_llama70b.jsonl
echo "=== E4 LLAMA JUDGE DONE ==="
