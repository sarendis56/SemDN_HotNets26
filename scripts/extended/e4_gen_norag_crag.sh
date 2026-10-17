#!/usr/bin/env bash
# GENERATION only (judge deferred): builds the controlled CRAG Task-1 experiment and
# the no-RAG baselines, saving every prediction to .preds.jsonl for later Llama-70B
# judging. Single free GPU (cuda:1).
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
echo "=== CRAG build (n=300, structnt) ==="
on_gpus 1 uv run python -m scdn.e4_crag --stage build --n 300 --chunker structnt --device cuda
echo "=== CRAG reader: pageret@1/@3 + chunkret (saves preds) ==="
on_gpus 1 uv run python -m scdn.e4_answer \
  --pool data/e4/pool_crag.jsonl --out data/e4/granularity_quality_crag.jsonl
echo "=== CRAG no-RAG baseline (closed book, saves preds) ==="
on_gpus 1 uv run python -m scdn.e4_answer \
  --pool data/e4/pool_crag.jsonl --out data/e4/granularity_quality_crag_norag.jsonl --closed-book
echo "=== live-web no-RAG baseline (closed book, 147 Q, saves preds) ==="
on_gpus 1 uv run python -m scdn.e4_answer \
  --pool data/e4/pool.jsonl --out data/e4/granularity_quality_live_norag.jsonl --closed-book
echo "=== E4 GEN (no-RAG + CRAG) DONE — preds for deferred judging: ==="
ls -la data/e4/granularity_quality_crag.preds.jsonl \
       data/e4/granularity_quality_crag_norag.preds.jsonl \
       data/e4/granularity_quality_live_norag.preds.jsonl
