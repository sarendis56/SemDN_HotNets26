#!/usr/bin/env bash
# Task 3 — utility-aware (LLM-answerability) reranking vs e5 similarity, on the SAME
# top-20 e5 candidates (coherent chunks). Tests O1: does ranking by usefulness beat
# ranking by similarity on CRAG quality?
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
M32=Qwen/Qwen3-32B
G=data/e4

echo "=== stage candidates: e5 top-20 over coherent chunks ==="
on_gpus 1 uv run python -m scdn.e4_utility_rank --stage candidates --chunker coherent
echo "=== stage rank: LLM utility scoring + rerank ==="
on_gpus 1 uv run python -m scdn.e4_utility_rank --stage rank
P=$G/pool_crag_t3_utility.jsonl

echo "=== readers (e5 vs util conds) ==="
on_gpus 1 uv run python -m scdn.e4_answer --pool "$P" --out $G/granularity_quality_crag_t3_utility.jsonl
on_gpus 1 uv run python -m scdn.e4_answer --model "$M32" --pool "$P" --out $G/granularity_quality_crag_t3_utility_q3.jsonl

echo "=== judge ==="
bash scripts/lib/judge_robust.sh \
  "$G/granularity_quality_crag_t3_utility.preds.jsonl,$G/granularity_quality_crag_t3_utility_q3.preds.jsonl" \
  "t3-utility-7b,t3-utility-q3" \
  "$G/judge_t3_utility_7b.jsonl,$G/judge_t3_utility_q3.jsonl"
echo "=== UTILITY DONE ==="
