#!/usr/bin/env bash
# Judge both E4 pools with Llama-3.1-70B-Instruct on ONE A100 via on-the-fly FP8
# (near-lossless, ~71GB). Single model load for both pools (the box's multi-GPU
# NCCL is broken, so no TP). Prints [judge health] so a broken judge is visible.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
on_gpus 1 79000 uv run python -m semdn.e4_judge \
  --preds data/e4/granularity_quality.preds.jsonl,data/e4/granularity_quality_qwen3.preds.jsonl \
  --tags e5-llama70b,qwen3-llama70b \
  --outs data/e4/judge_e5_llama70b.jsonl,data/e4/judge_qwen3_llama70b.jsonl \
  --model meta-llama/Llama-3.1-70B-Instruct --quantization fp8 \
  --gpu-memory-utilization 0.96 --max-model-len 2048
echo "=== E4 LLAMA JUDGE DONE ==="
