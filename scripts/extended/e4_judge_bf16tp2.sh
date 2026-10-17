#!/usr/bin/env bash
# Judge both pools with Llama-3.1-70B-Instruct in bf16 (LOSSLESS) on TP=2. The two
# A100s are SYS-connected (cross-NUMA, no NVLink/PCIe-switch), so NCCL's P2P and
# SHM transports fail -> force socket transport by disabling both. bf16 70B is
# ~70.6GB/GPU, so util 0.95 (KV is tiny: max_model_len 2048, short judge prompts).
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
on_gpus 2 78000 uv run python -m scdn.e4_judge \
  --preds data/e4/granularity_quality.preds.jsonl,data/e4/granularity_quality_qwen3.preds.jsonl \
  --tags e5-llama70b,qwen3-llama70b \
  --outs data/e4/judge_e5_llama70b.jsonl,data/e4/judge_qwen3_llama70b.jsonl \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.95 --max-model-len 2048
echo "=== E4 LLAMA JUDGE DONE ==="
