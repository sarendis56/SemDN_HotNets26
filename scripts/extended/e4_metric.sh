#!/usr/bin/env bash
# Re-evaluate E4 with the proper CRAG metric (LLM judge -> truthfulness). Regenerates
# reader predictions (saving them) for both encoder pools, then judges each.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
echo "=== reader e5 (save preds) ==="
on_gpus 1 uv run python -m semdn.e4_answer \
  --pool data/e4/pool.jsonl --out data/e4/granularity_quality.jsonl
echo "=== reader qwen3 (save preds) ==="
on_gpus 1 uv run python -m semdn.e4_answer \
  --pool data/e4/pool_qwen3.jsonl --out data/e4/granularity_quality_qwen3.jsonl
echo "=== judge e5 ==="
on_gpus 1 uv run python -m semdn.e4_judge \
  --preds data/e4/granularity_quality.preds.jsonl --tag e5 --out data/e4/judge_e5.jsonl
echo "=== judge qwen3 ==="
on_gpus 1 uv run python -m semdn.e4_judge \
  --preds data/e4/granularity_quality_qwen3.preds.jsonl --tag qwen3 --out data/e4/judge_qwen3.jsonl
echo "=== E4 METRIC DONE ==="
