#!/usr/bin/env bash
# Iso-budget sweep, round 2 (CRAG t3b, n=600), capped at 12K (larger budgets hold most of
# the ~46 pages per question, so arms converge by construction):
#   (a) extend the five judged arms (b128..b6144) to 8K and 12K       -> *_t3b_xl_* files
#   (b) LLM-reranker 2x2 + diagnostic at all 12 budgets: chunk_llm, page_llm (Qwen3-Reranker-4B
#       on whole pages), page_oracle                                  -> *_t3b_* files
# Stages: build (40GB card OK) -> reader Qwen3-32B on ONE 80GB card (16K window, gpu-mem 0.95)
#         -> judge Llama-3.1-70B, which needs two 80GB cards: runs only with JUDGE=1.
# GPUs are picked at each stage by on_gpus (scripts/lib/gpus.sh); GPUS=... restricts the choice.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
G=data/e4
M32=Qwen/Qwen3-32B
ALL=128,256,512,768,1024,1536,2048,3072,4096,6144,8192,12288
XL=8192,12288
XL_ARMS=${XL_ARMS-"chunk_ce chunk page_rel page_seq snippet_full"}   # page_seq: full table only
LLM_ARMS=${LLM_ARMS-"chunk_llm page_llm page_oracle"}
# Run the two halves in parallel:  LLM_ARMS= bash ...   and   XL_ARMS= ON_GPUS_WAIT=1 bash ...
# BUILD_GPUS restricts the build stages (e.g. to 40GB cards, keeping the 80GB one for readers).
done600() { [ "$(wc -l < "$1" 2>/dev/null || echo 0)" -ge 600 ]; }   # complete pool, not a partial one
build_on() { GPUS=${BUILD_GPUS:-${GPUS:-}} on_gpus "$@"; }

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  [ -z "$XL_ARMS" ] || done600 "$G/pool_iso_t3b_xl_page_seq.jsonl" || {
    echo "=== [build] XL pools: chunk/chunk_ce/page_rel/page_seq (e5+BGE) ==="
    build_on 1 20000 uv run python -m scdn.e4_isobudget build --corpus t3b --n 600 \
      --tag t3b_xl_ --budgets "$XL" || exit 1; }
  [ -z "$XL_ARMS" ] || done600 "$G/pool_iso_t3b_xl_snippet_full.jsonl" || {
    echo "=== [build] XL snippet_full pool (CPU) ==="
    uv run python -m scdn.e4_isobudget snippet_full --n 600 --tag t3b_xl_ --budgets "$XL" || exit 1; }
  [ -z "$LLM_ARMS" ] || done600 "$G/pool_iso_t3b_page_oracle.jsonl" || {
    echo "=== [build] LLM-reranker pools (Qwen3-Reranker-4B, ~6h) ==="
    build_on 1 30000 uv run python -m scdn.e4_isobudget llm --corpus t3b --n 600 \
      --tag t3b_ --budgets "$ALL" || exit 1; }   # never read a partial pool
fi

PREDS=""; TAGS=""; OUTS=""
read_arm() {   # $1=tag $2=arm ; resumable: skips arms already answered
  local out="$G/ans_iso_${1}${2}_q3.jsonl"
  if [ ! -s "$out" ]; then
    echo "--- reader: $1$2 ---"
    on_gpus 1 78000 uv run python -m scdn.e4_answer --model "$M32" \
      --pool "$G/pool_iso_${1}${2}.jsonl" --out "$out" --max-model-len 16384 --gpu-mem 0.95
  fi
  PREDS="$PREDS,${out%.jsonl}.preds.jsonl"; TAGS="$TAGS,iso_${1}${2}"; OUTS="$OUTS,$G/judge_iso_${1}${2}_q3.jsonl"
}
echo "=== [reader] Qwen3-32B, one 80GB card, 16K window ==="
for arm in $LLM_ARMS; do read_arm t3b_ "$arm"; done
for arm in $XL_ARMS; do read_arm t3b_xl_ "$arm"; done

if [ "${JUDGE:-0}" = "1" ]; then
  echo "=== [judge] Llama-3.1-70B (TP=2, two 80GB cards) ==="
  bash scripts/lib/judge_robust.sh "${PREDS#,}" "${TAGS#,}" "${OUTS#,}"
else
  echo "=== reader done; judge later with: SKIP_BUILD=1 JUDGE=1 bash scripts/e4_quality/iso_xl_run.sh ==="
fi
echo "=== ISO ROUND 2 DONE ==="
