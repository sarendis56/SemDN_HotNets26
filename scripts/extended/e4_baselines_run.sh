#!/usr/bin/env bash
# No-chunking RAG baselines (ZeroSearch-style snippet, R1-Searcher++-style summary) as new
# rows next to chunking and no-RAG. CRAG Task-3 (50 pages) + Task-1 (5 pages), both readers,
# Llama-70B judge. RUN ONLY AFTER the unified batch (bb49ynmak) finishes and its results are
# reported (it shares the GPUs).
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
export HF_HUB_OFFLINE=1
M32=Qwen/Qwen3-32B
G=data/e4

echo "=== build pools: snippet (CPU) + summary (GPU summarizer) ==="
uv run python -m semdn.e4_baselines --mode snippet --corpus t3 --page-ks 1,3,5,10
uv run python -m semdn.e4_baselines --mode snippet --corpus t1 --page-ks 1,3,5
on_gpus 1 uv run python -m semdn.e4_baselines --mode summary --corpus t3 --page-ks 1,3,5,10
on_gpus 1 uv run python -m semdn.e4_baselines --mode summary --corpus t1 --page-ks 1,3,5

echo "=== readers (7B + 32B) on the 4 baseline pools ==="
for cp in t3 t1; do for mode in snippet summary; do
  P=$G/pool_crag_${cp}_${mode}.jsonl
  on_gpus 1 uv run python -m semdn.e4_answer            --pool "$P" --out $G/gq_${cp}_${mode}.jsonl
  on_gpus 1 uv run python -m semdn.e4_answer --model "$M32" --pool "$P" --out $G/gq_${cp}_${mode}_q3.jsonl
done; done

echo "=== judge all 8 (Llama-70B) ==="
P=""; T=""; O=""
for cp in t3 t1; do for mode in snippet summary; do for s in "" "_q3"; do
  P+=",$G/gq_${cp}_${mode}${s}.preds.jsonl"; T+=",${cp}-${mode}${s}"; O+=",$G/judge_${cp}_${mode}${s}.jsonl"
done; done; done
bash scripts/lib/judge_robust.sh "${P#,}" "${T#,}" "${O#,}"
echo "=== BASELINES DONE ==="
