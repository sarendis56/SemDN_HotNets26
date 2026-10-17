#!/usr/bin/env bash
# Re-run E4 select+reader with Qwen3-Embedding-0.6B (no e5 pre-filter) to test
# whether page@10 > chunk@all is an e5-ranker artifact. Pages cached -> no credits.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
echo "=== qwen3 select (cuda:1, scores ALL pool chunks) ==="
on_gpus 1 uv run python -m scdn.e4_granularity --stage select --encoder qwen3 --device cuda
echo "=== qwen3 reader (cuda:1) ==="
on_gpus 1 uv run python -m scdn.e4_answer \
  --pool data/e4/pool_qwen3.jsonl --out data/e4/granularity_quality_qwen3.jsonl
echo "=== E4 QWEN3 DONE ==="
