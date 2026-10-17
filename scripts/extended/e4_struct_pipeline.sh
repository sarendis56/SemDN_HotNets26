#!/usr/bin/env bash
# Structure-aware chunking pipeline: re-select (e5) + re-rerank (CE) over the SAME
# cached pages but with structure-aware chunks (atomic tables, boilerplate removed,
# title-prefixed), then reader + Llama-3.1-70B bf16 judge. Compares against the
# fixed-256-token results to test whether cleaner units cut hallucination.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
echo "=== select e5 / structure (cuda:1) ==="
on_gpus 1 uv run python -m scdn.e4_granularity --stage select --encoder e5 --chunker structure --n 150 --device cuda
echo "=== rerank CE / structure (cuda:1) ==="
on_gpus 1 uv run python -m scdn.e4_granularity --stage rerank --chunker structure --n 150 --device cuda
echo "=== reader: e5-structure pool (cuda:1) ==="
on_gpus 1 uv run python -m scdn.e4_answer \
  --pool data/e4/pool_struct.jsonl --out data/e4/granularity_quality_struct.jsonl
echo "=== reader: CE-structure pool (cuda:1) ==="
on_gpus 1 uv run python -m scdn.e4_answer \
  --pool data/e4/pool_ce_struct.jsonl --out data/e4/granularity_quality_ce_struct.jsonl
echo "=== judge both (Llama-70B bf16 TP=2, raised timeouts) ==="
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 VLLM_RPC_TIMEOUT=1800000
on_gpus 2 78000 uv run python -m scdn.e4_judge \
  --preds data/e4/granularity_quality_struct.preds.jsonl,data/e4/granularity_quality_ce_struct.preds.jsonl \
  --tags e5-struct-llama70b,ce-struct-llama70b \
  --outs data/e4/judge_e5_struct_llama70b.jsonl,data/e4/judge_ce_struct_llama70b.jsonl \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.95 --max-model-len 2048
echo "=== E4 STRUCT PIPELINE DONE ==="
