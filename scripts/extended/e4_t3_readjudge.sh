#!/usr/bin/env bash
# Re-run Task-3 readers + judge on the (intact) depth-sweep pool, after the int-gold
# metric fix. Build is cached (pool_crag_t3.jsonl, all 6 depths × 300); skip it.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
M32=Qwen/Qwen3-32B
P=data/e4/pool_crag_t3.jsonl
echo "=== reader Qwen2.5-7B: depth sweep ==="
on_gpus 1 uv run python -m semdn.e4_answer --pool "$P" --out data/e4/granularity_quality_crag_t3.jsonl
echo "=== reader Qwen2.5-7B: no-RAG ==="
on_gpus 1 uv run python -m semdn.e4_answer --pool "$P" --out data/e4/granularity_quality_crag_t3_norag.jsonl --closed-book
echo "=== reader Qwen3-32B: depth sweep ==="
on_gpus 1 uv run python -m semdn.e4_answer --model "$M32" --pool "$P" --out data/e4/granularity_quality_crag_t3_q3.jsonl
echo "=== reader Qwen3-32B: no-RAG ==="
on_gpus 1 uv run python -m semdn.e4_answer --model "$M32" --pool "$P" --out data/e4/granularity_quality_crag_t3_norag_q3.jsonl --closed-book
echo "=== judge all 4 with Llama-3.1-70B (TP=2) ==="
bash scripts/lib/judge_robust.sh \
  "data/e4/granularity_quality_crag_t3.preds.jsonl,data/e4/granularity_quality_crag_t3_norag.preds.jsonl,data/e4/granularity_quality_crag_t3_q3.preds.jsonl,data/e4/granularity_quality_crag_t3_norag_q3.preds.jsonl" \
  "t3-7b,t3-norag-7b,t3-q3,t3-norag-q3" \
  "data/e4/judge_t3_7b.jsonl,data/e4/judge_t3_norag_7b.jsonl,data/e4/judge_t3_q3.jsonl,data/e4/judge_t3_norag_q3.jsonl"
echo "=== T3 READJUDGE DONE ==="
