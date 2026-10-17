#!/usr/bin/env bash
# Judge the CE (cross-encoder) pool predictions with Llama-3.1-70B bf16 TP=2.
# Socket-transport all-reduce is slow and occasionally stalls past the default
# 300s execute-model deadline -> raise the timeouts so a hiccup doesn't kill the
# engine (the same judge completed 1470 prompts in an earlier run).
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export VLLM_RPC_TIMEOUT=1800000
on_gpus 2 78000 uv run python -m semdn.e4_judge \
  --preds data/e4/granularity_quality_ce.preds.jsonl \
  --tags ce-llama70b --outs data/e4/judge_ce_llama70b.jsonl \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.95 --max-model-len 2048
echo "=== E4 CE JUDGE DONE ==="
