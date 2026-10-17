#!/usr/bin/env bash
# Supplement §E's QUALITY depth sweep with the two rankers that were structural-only:
# BM25 (lexical) and the Qwen3-Reranker. Coherent chunks; reader 7B+32B; Llama-70B judge.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
export HF_HUB_OFFLINE=1
M32=Qwen/Qwen3-32B
G=data/e4

echo "=== rank coherent: bm25 (CPU) + qwen3rr (GPU) ==="
on_gpus 1 uv run python -m scdn.e4_crag_t3 --stage rank --chunker coherent --encoder bm25
on_gpus 1 uv run python -m scdn.e4_crag_t3 --stage rank --chunker coherent --encoder qwen3rr

for enc in bm25 qwen3rr; do
  P=$G/pool_crag_t3_coherent_${enc}.jsonl
  echo "=== reader 7B: coherent/$enc ==="
  on_gpus 1 uv run python -m scdn.e4_answer --pool "$P" --out $G/granularity_quality_crag_t3_coherent_${enc}.jsonl
  echo "=== reader 32B: coherent/$enc ==="
  on_gpus 1 uv run python -m scdn.e4_answer --model "$M32" --pool "$P" --out $G/granularity_quality_crag_t3_coherent_${enc}_q3.jsonl
done

echo "=== judge all 4 (Llama-70B TP=2) ==="
bash scripts/lib/judge_robust.sh \
  "$G/granularity_quality_crag_t3_coherent_bm25.preds.jsonl,$G/granularity_quality_crag_t3_coherent_bm25_q3.preds.jsonl,$G/granularity_quality_crag_t3_coherent_qwen3rr.preds.jsonl,$G/granularity_quality_crag_t3_coherent_qwen3rr_q3.preds.jsonl" \
  "t3-coh-bm25-7b,t3-coh-bm25-q3,t3-coh-qwen3rr-7b,t3-coh-qwen3rr-q3" \
  "$G/judge_t3_coherent_bm25_7b.jsonl,$G/judge_t3_coherent_bm25_q3.jsonl,$G/judge_t3_coherent_qwen3rr_7b.jsonl,$G/judge_t3_coherent_qwen3rr_q3.jsonl"
echo "=== COHERENT RR+BM25 DONE ==="
