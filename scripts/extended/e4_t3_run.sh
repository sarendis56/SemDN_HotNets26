#!/usr/bin/env bash
# CRAG Task-3 pool-DEPTH sweep end-to-end. build (e5, structnt, page@{1,3,5,10,20,50}
# over ≤50 pages; ~1.5h extraction, resumable) → readers (Qwen2.5-7B & Qwen3-32B) on
# the sweep + no-RAG → Llama-3.1-70B judge. Shows the page-depth TREND in a clean corpus.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
M32=Qwen/Qwen3-32B
P=data/e4/pool_crag_t3.jsonl

echo "=== build depth sweep (e5, cuda:1) ==="
on_gpus 1 uv run python -m scdn.e4_crag_t3 --stage build --n 300 --device cuda

echo "=== reader Qwen2.5-7B: depth sweep ==="
on_gpus 1 uv run python -m scdn.e4_answer --pool "$P" --out data/e4/granularity_quality_crag_t3.jsonl
echo "=== reader Qwen2.5-7B: no-RAG ==="
on_gpus 1 uv run python -m scdn.e4_answer --pool "$P" --out data/e4/granularity_quality_crag_t3_norag.jsonl --closed-book
echo "=== reader Qwen3-32B: depth sweep ==="
on_gpus 1 uv run python -m scdn.e4_answer --model "$M32" --pool "$P" --out data/e4/granularity_quality_crag_t3_q3.jsonl
echo "=== reader Qwen3-32B: no-RAG ==="
on_gpus 1 uv run python -m scdn.e4_answer --model "$M32" --pool "$P" --out data/e4/granularity_quality_crag_t3_norag_q3.jsonl --closed-book

echo "=== judge all 4 with Llama-3.1-70B (TP=2) ==="
bash scripts/lib/judge_robust.sh \
  "data/e4/granularity_quality_crag_t3.preds.jsonl,data/e4/granularity_quality_crag_t3_norag.preds.jsonl,data/e4/granularity_quality_crag_t3_q3.preds.jsonl,data/e4/granularity_quality_crag_t3_norag_q3.preds.jsonl" \
  "t3-7b,t3-norag-7b,t3-q3,t3-norag-q3" \
  "data/e4/judge_t3_7b.jsonl,data/e4/judge_t3_norag_7b.jsonl,data/e4/judge_t3_q3.jsonl,data/e4/judge_t3_norag_q3.jsonl"
echo "=== T3 RUN DONE ==="
