#!/usr/bin/env bash
# N4: larger-n (600) CRAG Task-3 comparison to firm up the chunking-vs-baselines trends.
# Needs the t3b coherent cache (chunks_crag_t3b_coherent.jsonl) already built. Ranks e5 + CE,
# builds the t3b baselines (snippet/fullpage/summary-v2), reads with 32B, judges with Llama-70B.
# Arms: chunk-e5, chunk-CE(reranker), snippet, fullpage, summary, no-RAG. RUN WHEN GPU FREE.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
export HF_HUB_OFFLINE=1
M32=Qwen/Qwen3-32B
G=data/e4
T3=data/datasets/crag/crag_task_3_dev_v4
PK=1,3,5,10,20,50

echo "=== rank t3b coherent: e5 + CE ==="
on_gpus 1 uv run python -m semdn.e4_crag_t3 --stage rank --corpus t3b --chunker coherent --encoder e5 --page-ks "$PK"
on_gpus 1 uv run python -m semdn.e4_crag_t3 --stage rank --corpus t3b --chunker coherent --encoder ce --page-ks "$PK"

echo "=== build t3b baselines (snippet/fullpage CPU; summary v2 GPU) ==="
uv run python -m semdn.e4_baselines --mode snippet  --corpus t3b --src "$T3" --n 600 --page-ks 1,3,5,10
uv run python -m semdn.e4_baselines --mode fullpage --corpus t3b --page-ks 1,3,5
on_gpus 1 uv run python -m semdn.e4_baselines --mode summary --corpus t3b --page-ks 1,3,5,10 --summarizer "$M32"

echo "=== readers (Qwen3-32B) on all arms ==="
for P in pool_crag_t3b_coherent_e5 pool_crag_t3b_coherent_ce pool_crag_t3b_snippet pool_crag_t3b_fullpage pool_crag_t3b_summary; do
  on_gpus 1 uv run python -m semdn.e4_answer --model "$M32" --pool $G/$P.jsonl --out $G/gq_${P}_q3.jsonl
done
on_gpus 1 uv run python -m semdn.e4_answer --model "$M32" --pool $G/pool_crag_t3b_coherent_e5.jsonl --out $G/gq_t3b_norag_q3.jsonl --closed-book

echo "=== judge all 6 (Llama-70B) ==="
bash scripts/lib/judge_robust.sh \
  "$G/gq_pool_crag_t3b_coherent_e5_q3.preds.jsonl,$G/gq_pool_crag_t3b_coherent_ce_q3.preds.jsonl,$G/gq_pool_crag_t3b_snippet_q3.preds.jsonl,$G/gq_pool_crag_t3b_fullpage_q3.preds.jsonl,$G/gq_pool_crag_t3b_summary_q3.preds.jsonl,$G/gq_t3b_norag_q3.preds.jsonl" \
  "t3b-e5,t3b-ce,t3b-snippet,t3b-fullpage,t3b-summary,t3b-norag" \
  "$G/judge_t3b_e5_q3.jsonl,$G/judge_t3b_ce_q3.jsonl,$G/judge_t3b_snippet_q3.jsonl,$G/judge_t3b_fullpage_q3.jsonl,$G/judge_t3b_summary_q3.jsonl,$G/judge_t3b_norag_q3.jsonl"
echo "=== N600 DONE ==="
