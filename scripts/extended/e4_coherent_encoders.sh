#!/usr/bin/env bash
# Q3: is the coherent chunker's depth trend (quality rising with pages, esp. weak reader)
# clear for OTHER encoder pipelines, not just e5? Reader+judge over the coherent chunks
# ranked by qwen3-embedding and the BGE cross-encoder (pools built by e4_crag_t3 --stage
# rank --chunker coherent --encoder {qwen3,ce}). e5 coherent already done (judge_t3_coherent_*).
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
M32=Qwen/Qwen3-32B
G=data/e4

for enc in qwen3 ce; do
  P=$G/pool_crag_t3_coherent_${enc}.jsonl
  echo "=== reader 7B: coherent/$enc ==="
  on_gpus 1 uv run python -m scdn.e4_answer --pool "$P" --out $G/granularity_quality_crag_t3_coherent_${enc}.jsonl
  echo "=== reader 32B: coherent/$enc ==="
  on_gpus 1 uv run python -m scdn.e4_answer --model "$M32" --pool "$P" --out $G/granularity_quality_crag_t3_coherent_${enc}_q3.jsonl
done

echo "=== judge all 4 (Llama-70B TP=2) ==="
bash scripts/lib/judge_robust.sh \
  "$G/granularity_quality_crag_t3_coherent_qwen3.preds.jsonl,$G/granularity_quality_crag_t3_coherent_qwen3_q3.preds.jsonl,$G/granularity_quality_crag_t3_coherent_ce.preds.jsonl,$G/granularity_quality_crag_t3_coherent_ce_q3.preds.jsonl" \
  "t3-coh-qwen3-7b,t3-coh-qwen3-q3,t3-coh-ce-7b,t3-coh-ce-q3" \
  "$G/judge_t3_coherent_qwen3_7b.jsonl,$G/judge_t3_coherent_qwen3_q3.jsonl,$G/judge_t3_coherent_ce_7b.jsonl,$G/judge_t3_coherent_ce_q3.jsonl"
echo "=== COHERENT ENCODERS DONE ==="
