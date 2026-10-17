#!/usr/bin/env bash
# Ablation: structure-aware chunking WITHOUT the title prefix (tests whether the
# per-chunk title prefix is what hurt the e5 bi-encoder). select e5 -> reader ->
# Llama-3.1-70B bf16 judge.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
echo "=== select e5 / structnt (cuda:1) ==="
on_gpus 1 uv run python -m scdn.e4_granularity --stage select --encoder e5 --chunker structnt --n 150 --device cuda
echo "=== reader: e5-structnt pool (cuda:1) ==="
on_gpus 1 uv run python -m scdn.e4_answer \
  --pool data/e4/pool_structnt.jsonl --out data/e4/granularity_quality_structnt.jsonl
echo "=== judge (Llama-70B bf16 TP=2, raised timeouts) ==="
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 VLLM_RPC_TIMEOUT=1800000
on_gpus 2 78000 uv run python -m scdn.e4_judge \
  --preds data/e4/granularity_quality_structnt.preds.jsonl \
  --tags e5-structnt-llama70b --outs data/e4/judge_e5_structnt_llama70b.jsonl \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.95 --max-model-len 2048
echo "=== E4 STRUCTNT ABLATION DONE ==="
