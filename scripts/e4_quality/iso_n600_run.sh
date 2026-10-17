#!/usr/bin/env bash
# Iso-budget sweep at n=600 (CRAG t3b), dense budgets (128..6144), with a cross-encoder
# chunk arm. build (e5+CE) + snippet_full (CPU stream) -> Qwen3-32B reader -> Llama-70B judge.
# Arms: chunk / chunk_ce / page_rel / page_seq / snippet_full; conds b128..b6144.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
G=data/e4
M32=Qwen/Qwen3-32B
TAG=t3b_

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  echo "=== [1/4] build iso pools: chunk/chunk_ce/page (e5+CE, GPU0) ==="
  on_gpus 1 uv run python -m semdn.e4_isobudget build --corpus t3b --n 600 --tag "$TAG"
  echo "=== [2/4] build snippet_full pool (stream CRAG t3 source) ==="
  uv run python -m semdn.e4_isobudget snippet_full --n 600 --tag "$TAG"
fi

echo "=== [3/4] reader Qwen3-32B (GPU0) ==="
PREDS=""; TAGS=""; OUTS=""
for arm in chunk chunk_ce page_rel page_seq snippet_full; do
  pool="$G/pool_iso_${TAG}${arm}.jsonl"
  out="$G/ans_iso_${TAG}${arm}_q3.jsonl"
  echo "--- reader: $arm ---"
  on_gpus 1 uv run python -m semdn.e4_answer --model "$M32" --pool "$pool" --out "$out"
  PREDS="$PREDS,${out%.jsonl}.preds.jsonl"; TAGS="$TAGS,iso_${TAG}${arm}"; OUTS="$OUTS,$G/judge_iso_${TAG}${arm}_q3.jsonl"
done

echo "=== [4/4] judge Llama-3.1-70B (TP=2) ==="
bash scripts/lib/judge_robust.sh "${PREDS#,}" "${TAGS#,}" "${OUTS#,}"
echo "=== ISO N600 EVAL DONE ==="
