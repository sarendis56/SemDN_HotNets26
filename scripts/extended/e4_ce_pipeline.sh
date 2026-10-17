#!/usr/bin/env bash
# Cross-encoder rerank pipeline: does utility-aware ranking (bge-reranker-v2-m3)
# rescue the deep pool? rerank (cuda:1) -> reader (cuda:1) -> Llama-70B bf16 judge
# (TP=2, socket NCCL). All sequential so GPUs free between stages.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
echo "=== rerank full pool with cross-encoder (cuda:1) ==="
on_gpus 1 uv run python -m scdn.e4_granularity --stage rerank --n 150 --device cuda
echo "=== reader on CE pool (cuda:1) ==="
on_gpus 1 uv run python -m scdn.e4_answer \
  --pool data/e4/pool_ce.jsonl --out data/e4/granularity_quality_ce.jsonl
echo "=== judge CE pool with Llama-3.1-70B bf16 TP=2 ==="
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1
on_gpus 2 78000 uv run python -m scdn.e4_judge \
  --preds data/e4/granularity_quality_ce.preds.jsonl \
  --tags ce-llama70b --outs data/e4/judge_ce_llama70b.jsonl \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.95 --max-model-len 2048
echo "=== E4 CE PIPELINE DONE ==="
