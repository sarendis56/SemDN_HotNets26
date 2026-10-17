#!/usr/bin/env bash
# Summary baseline v2: the first version was worse than no-RAG because a 7B summarizer with
# a question-LEADING prompt guessed answers (hallucination). Fix = faithful non-leading
# prompt (e4_baselines SUM_SYS) + a stronger summarizer (Qwen3-32B, thinking off). Rebuild
# the summary pools, re-run the summary readers, re-judge. snippet/chunking unchanged.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
export HF_HUB_OFFLINE=1
M32=Qwen/Qwen3-32B
G=data/e4

echo "=== rebuild summary pools (Qwen3-32B faithful summarizer) ==="
on_gpus 1 uv run python -m scdn.e4_baselines --mode summary --corpus t3 --page-ks 1,3,5,10 --summarizer "$M32"
on_gpus 1 uv run python -m scdn.e4_baselines --mode summary --corpus t1 --page-ks 1,3,5   --summarizer "$M32"

echo "=== summary readers (7B + 32B) ==="
for cp in t3 t1; do
  P=$G/pool_crag_${cp}_summary.jsonl
  on_gpus 1 uv run python -m scdn.e4_answer            --pool "$P" --out $G/gq_${cp}_summary.jsonl
  on_gpus 1 uv run python -m scdn.e4_answer --model "$M32" --pool "$P" --out $G/gq_${cp}_summary_q3.jsonl
done

echo "=== re-judge the 4 summary preds ==="
P=""; T=""; O=""
for cp in t3 t1; do for s in "" "_q3"; do
  P+=",$G/gq_${cp}_summary${s}.preds.jsonl"; T+=",${cp}-summary${s}"; O+=",$G/judge_${cp}_summary${s}.jsonl"
done; done
bash scripts/lib/judge_robust.sh "${P#,}" "${T#,}" "${O#,}"
echo "=== SUMMARY V2 DONE ==="
