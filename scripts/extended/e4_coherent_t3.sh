#!/usr/bin/env bash
# Task 4: measure the coherent chunker's utility on CRAG Task-3. Cache built by
# `e4_crag_t3 --stage cache --chunker coherent`. e5-rank -> reader(7B+32B) -> Llama-70B
# judge, then e4q_stats compares coherent vs structnt (the incumbent) at matched depth.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
M32=Qwen/Qwen3-32B
G=data/e4

echo "=== e5-rank coherent chunks ==="
on_gpus 1 uv run python -m scdn.e4_crag_t3 --stage rank --chunker coherent --encoder e5
P=$G/pool_crag_t3_coherent_e5.jsonl

echo "=== reader Qwen2.5-7B ==="
on_gpus 1 uv run python -m scdn.e4_answer --pool "$P" --out $G/granularity_quality_crag_t3_coherent.jsonl
echo "=== reader Qwen3-32B ==="
on_gpus 1 uv run python -m scdn.e4_answer --model "$M32" --pool "$P" --out $G/granularity_quality_crag_t3_coherent_q3.jsonl

echo "=== judge (Llama-70B TP=2) ==="
bash scripts/lib/judge_robust.sh \
  "$G/granularity_quality_crag_t3_coherent.preds.jsonl,$G/granularity_quality_crag_t3_coherent_q3.preds.jsonl" \
  "t3-coherent-7b,t3-coherent-q3" \
  "$G/judge_t3_coherent_7b.jsonl,$G/judge_t3_coherent_q3.jsonl"
echo "=== COHERENT T3 DONE ==="
