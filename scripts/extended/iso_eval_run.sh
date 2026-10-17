#!/usr/bin/env bash
# Iso-budget quality sweep: at fixed reader token budgets, chunk vs whole-page vs snippet.
# build (e5) -> Qwen3-32B reader -> Llama-3.1-70B CRAG judge. Arms: chunk / page_rel /
# page_seq / snippet; conds: b256..b4096.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
G=data/e4
M32=Qwen/Qwen3-32B
N=${N:-300}

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  echo "=== [1/3] build iso-budget pools (e5, GPU0) ==="
  on_gpus 1 uv run python -m scdn.e4_isobudget build --n "$N"
fi

echo "=== [2/3] reader Qwen3-32B (GPU0) ==="
PREDS=""; TAGS=""; OUTS=""
for arm in chunk page_rel page_seq snippet; do
  pool="$G/pool_iso_${arm}.jsonl"
  out="$G/ans_iso_${arm}_q3.jsonl"
  echo "--- reader: $arm ---"
  on_gpus 1 uv run python -m scdn.e4_answer --model "$M32" --pool "$pool" --out "$out"
  PREDS="$PREDS,${out%.jsonl}.preds.jsonl"; TAGS="$TAGS,iso_${arm}"; OUTS="$OUTS,$G/judge_iso_${arm}_q3.jsonl"
done

echo "=== [3/3] judge Llama-3.1-70B (TP=2) ==="
bash scripts/lib/judge_robust.sh "${PREDS#,}" "${TAGS#,}" "${OUTS#,}"
echo "=== ISO-BUDGET EVAL DONE ==="
