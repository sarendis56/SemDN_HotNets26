#!/usr/bin/env bash
# Fetcher comparison (trafilatura GET vs Cloudflare cloud clean-fetch) on the live web.
# Build pools (e5), read with Qwen3-32B, judge with Llama-3.1-70B (CRAG official).
# Focused arm set: traf vs cf x {structnt, coherent} chunkers + snippet / full-page(cf) /
# hybrid(cf) / no-RAG baselines. All on the SAME n questions, same top-K SERP depth.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
G=data/e4
M32=Qwen/Qwen3-32B
N=${N:-120}
TOPK=${TOPK:-12}

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  echo "=== [1/3] build pools (e5, GPU0) ==="
  on_gpus 1 uv run python -m scdn.e4_cf_eval build --n "$N" --topk "$TOPK"
else
  echo "=== [1/3] build SKIPPED (SKIP_BUILD=1) ==="
fi

echo "=== [2/3] reader Qwen3-32B (GPU0) ==="
declare -A POOLS=(
  [traf_structnt]=pool_cf_traf_structnt
  [cf_structnt]=pool_cf_cf_structnt
  [traf_coherent]=pool_cf_traf_coherent
  [cf_coherent]=pool_cf_cf_coherent
  [snippet]=pool_cf_snippet
  [fullpage_cf]=pool_cf_fullpage_cf
  [fullpage_traf]=pool_cf_fullpage_traf
  [cf_hybrid]=pool_cf_cf_coherent_hybrid
)
PREDS=""; TAGS=""; OUTS=""
for tag in "${!POOLS[@]}"; do
  pool="$G/${POOLS[$tag]}.jsonl"
  out="$G/ans_cf_${tag}_q3.jsonl"
  echo "--- reader: $tag ($pool) ---"
  on_gpus 1 uv run python -m scdn.e4_answer --model "$M32" --pool "$pool" --out "$out"
  PREDS="$PREDS,${out%.jsonl}.preds.jsonl"; TAGS="$TAGS,cf_${tag}"; OUTS="$OUTS,$G/judge_cf_${tag}_q3.jsonl"
done
# no-RAG closed-book (parametric only), once
on_gpus 1 uv run python -m scdn.e4_answer --model "$M32" \
  --pool "$G/pool_cf_cf_structnt.jsonl" --out "$G/ans_cf_norag_q3.jsonl" --closed-book
PREDS="$PREDS,$G/ans_cf_norag_q3.preds.jsonl"; TAGS="$TAGS,cf_norag"; OUTS="$OUTS,$G/judge_cf_norag_q3.jsonl"

echo "=== [3/3] judge Llama-3.1-70B (TP=2) ==="
bash scripts/lib/judge_robust.sh "${PREDS#,}" "${TAGS#,}" "${OUTS#,}"
echo "=== CF EVAL DONE ==="
