#!/usr/bin/env bash
# Unified consistency batch (commit once after all of it):
#  (A) CRAG Task-1 (5 pages) on the Task-3 pipeline: coherent chunks, single-stage rankers
#      {e5, bm25, provided, qwen3rr}, depth page@1/3/5, readers 7B+32B, Llama-70B judge,
#      + structural answer-location decomposition. Re-validates F8a without the CE confound.
#  (B) Live web: coherent chunks ranked by {bm25, provided, qwen3rr} (e5 done in §C),
#      reader 32B, Llama-70B judge. Adds the search-engine baselines + strong reranker at 32B.
# Assumes the Task-1 coherent cache (chunks_crag_t1_coherent.jsonl) is already built.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
export HF_HUB_OFFLINE=1
M32=Qwen/Qwen3-32B
G=data/e4
PK=1,3,5

echo "########## (A) TASK-1 ##########"
echo "=== rank t1 coherent: e5 bm25 provided qwen3rr ==="
for enc in e5 bm25 provided qwen3rr; do
  on_gpus 1 uv run python -m semdn.e4_crag_t3 --stage rank --corpus t1 --chunker coherent --encoder "$enc" --page-ks "$PK"
done

echo "=== t1 readers (7B + 32B) ==="
for enc in e5 bm25 provided qwen3rr; do
  P=$G/pool_crag_t1_coherent_${enc}.jsonl
  on_gpus 1 uv run python -m semdn.e4_answer            --pool "$P" --out $G/gq_t1_coherent_${enc}.jsonl
  on_gpus 1 uv run python -m semdn.e4_answer --model "$M32" --pool "$P" --out $G/gq_t1_coherent_${enc}_q3.jsonl
done

echo "=== t1 structural decomposition ==="
on_gpus 1 uv run python -m semdn.e4_locality_strong --corpus t1 --chunker coherent \
  --rankers e5,bm25,qwen3,ce,qwen3rr --out $G/locality_strong_t1.json

echo "=== t1 judge (8 preds, Llama-70B) ==="
P=""; T=""; O=""
for enc in e5 bm25 provided qwen3rr; do for s in "" "_q3"; do
  P+=",$G/gq_t1_coherent_${enc}${s}.preds.jsonl"; T+=",t1-${enc}${s}"; O+=",$G/judge_t1_coherent_${enc}${s}.jsonl"
done; done
bash scripts/lib/judge_robust.sh "${P#,}" "${T#,}" "${O#,}"

echo "########## (B) LIVE WEB ##########"
echo "=== live select: coherent x {bm25, provided, qwen3rr} ==="
for enc in bm25 provided qwen3rr; do
  on_gpus 1 uv run python -m semdn.e4_granularity --stage select --encoder "$enc" --chunker coherent
done

echo "=== live readers (32B) ==="
for enc in bm25 provided qwen3rr; do
  P=$G/pool_${enc}_coherent.jsonl
  on_gpus 1 uv run python -m semdn.e4_answer --model "$M32" --pool "$P" --out $G/gq_live_coherent_${enc}_q3.jsonl
done

echo "=== live judge (3 preds, Llama-70B) ==="
bash scripts/lib/judge_robust.sh \
  "$G/gq_live_coherent_bm25_q3.preds.jsonl,$G/gq_live_coherent_provided_q3.preds.jsonl,$G/gq_live_coherent_qwen3rr_q3.preds.jsonl" \
  "live-bm25-q3,live-provided-q3,live-qwen3rr-q3" \
  "$G/judge_live_coherent_bm25_q3.jsonl,$G/judge_live_coherent_provided_q3.jsonl,$G/judge_live_coherent_qwen3rr_q3.jsonl"

echo "########## UNIFIED BATCH DONE ##########"
