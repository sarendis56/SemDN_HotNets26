#!/usr/bin/env bash
# Task 1 (E4-Q) lock: shallow lexical index (BM25) and search-rank floor (provided)
# vs the dense e5 index, on the CRAG Task-3 50-page corpus (n=300, reproducible).
# Pools built by e4_crag_t3.py --stage rank --encoder {bm25,provided}; e5 dense pool
# (pool_crag_t3.jsonl) + its judge labels (judge_t3_7b/q3) already exist, so we only
# read+judge the two NEW shallow pools, then e4_stats pairs them against e5 by qid.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
M32=Qwen/Qwen3-32B
G=data/e4

echo "=== reader Qwen2.5-7B: BM25 + provided ==="
on_gpus 1 uv run python -m semdn.e4_answer --pool "$G/pool_crag_t3_bm25.jsonl"     --out "$G/granularity_quality_crag_t3_bm25.jsonl"
on_gpus 1 uv run python -m semdn.e4_answer --pool "$G/pool_crag_t3_provided.jsonl" --out "$G/granularity_quality_crag_t3_provided.jsonl"
echo "=== reader Qwen3-32B: BM25 + provided ==="
on_gpus 1 uv run python -m semdn.e4_answer --model "$M32" --pool "$G/pool_crag_t3_bm25.jsonl"     --out "$G/granularity_quality_crag_t3_bm25_q3.jsonl"
on_gpus 1 uv run python -m semdn.e4_answer --model "$M32" --pool "$G/pool_crag_t3_provided.jsonl" --out "$G/granularity_quality_crag_t3_provided_q3.jsonl"

echo "=== judge all 4 with Llama-3.1-70B (TP=2) ==="
bash scripts/lib/judge_robust.sh \
  "$G/granularity_quality_crag_t3_bm25.preds.jsonl,$G/granularity_quality_crag_t3_provided.preds.jsonl,$G/granularity_quality_crag_t3_bm25_q3.preds.jsonl,$G/granularity_quality_crag_t3_provided_q3.preds.jsonl" \
  "t3-bm25-7b,t3-provided-7b,t3-bm25-q3,t3-provided-q3" \
  "$G/judge_t3_bm25_7b.jsonl,$G/judge_t3_provided_7b.jsonl,$G/judge_t3_bm25_q3.jsonl,$G/judge_t3_provided_q3.jsonl"
echo "=== E4-Q T3 READ+JUDGE DONE ==="
