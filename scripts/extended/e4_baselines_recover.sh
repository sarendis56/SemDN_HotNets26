#!/usr/bin/env bash
# Recover the summary baseline (the summarizer crashed on over-long pages; fixed with
# max_model_len 8192 + truncate_prompt_tokens). Snippet pools+preds already exist, so we
# only (re)build summary pools, run summary readers, and re-judge all 8 baseline preds.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
export HF_HUB_OFFLINE=1
M32=Qwen/Qwen3-32B
G=data/e4

echo "=== build summary pools (fixed) ==="
on_gpus 1 uv run python -m scdn.e4_baselines --mode summary --corpus t3 --page-ks 1,3,5,10
on_gpus 1 uv run python -m scdn.e4_baselines --mode summary --corpus t1 --page-ks 1,3,5

echo "=== summary readers (7B + 32B) ==="
for cp in t3 t1; do
  P=$G/pool_crag_${cp}_summary.jsonl
  on_gpus 1 uv run python -m scdn.e4_answer            --pool "$P" --out $G/gq_${cp}_summary.jsonl
  on_gpus 1 uv run python -m scdn.e4_answer --model "$M32" --pool "$P" --out $G/gq_${cp}_summary_q3.jsonl
done

echo "=== judge all 8 baseline preds (snippet + summary, both readers) ==="
P=""; T=""; O=""
for cp in t3 t1; do for mode in snippet summary; do for s in "" "_q3"; do
  P+=",$G/gq_${cp}_${mode}${s}.preds.jsonl"; T+=",${cp}-${mode}${s}"; O+=",$G/judge_${cp}_${mode}${s}.jsonl"
done; done; done
bash scripts/lib/judge_robust.sh "${P#,}" "${T#,}" "${O#,}"
echo "=== BASELINES RECOVER DONE ==="
