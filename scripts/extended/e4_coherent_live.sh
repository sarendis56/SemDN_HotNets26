#!/usr/bin/env bash
# Generalize the coherent chunker to the LIVE-WEB setting (where it hasn't been tested;
# the noisiest, most realistic E4 setting, so the deep-pool win should show most). Reuses
# the cached serp_deep + pages_cache; re-chunks with coherent, e5-ranks, reader+judge.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
M32=Qwen/Qwen3-32B
G=data/e4

echo "=== select: coherent chunks, e5, live web ==="
on_gpus 1 uv run python -m semdn.e4_granularity --stage select --encoder e5 --chunker coherent
P=$G/pool_coherent.jsonl

echo "=== readers ==="
on_gpus 1 uv run python -m semdn.e4_answer --pool "$P" --out $G/granularity_quality_coherent.jsonl
on_gpus 1 uv run python -m semdn.e4_answer --model "$M32" --pool "$P" --out $G/granularity_quality_coherent_q3.jsonl

echo "=== judge ==="
bash scripts/lib/judge_robust.sh \
  "$G/granularity_quality_coherent.preds.jsonl,$G/granularity_quality_coherent_q3.preds.jsonl" \
  "live-coherent-7b,live-coherent-q3" \
  "$G/judge_coherent_7b.jsonl,$G/judge_coherent_q3.jsonl"
echo "=== COHERENT LIVE DONE ==="
