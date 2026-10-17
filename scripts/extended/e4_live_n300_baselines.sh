#!/usr/bin/env bash
# Fix-up: snippet + full-page were stuck at n=147 (e4_live_baselines sourced its question
# set from the un-rebuilt pool_coherent). Now sourced from pool_structnt (287); re-read + re-judge.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
export HF_HUB_OFFLINE=1
M32=Qwen/Qwen3-32B
G=data/e4

on_gpus 1 uv run python -m scdn.e4_answer --model "$M32" --pool $G/pool_live_snippet.jsonl  --out $G/gq_pool_live_snippet_q3.jsonl
on_gpus 1 uv run python -m scdn.e4_answer --model "$M32" --pool $G/pool_live_fullpage.jsonl --out $G/gq_pool_live_fullpage_q3.jsonl
bash scripts/lib/judge_robust.sh \
  "$G/gq_pool_live_snippet_q3.preds.jsonl,$G/gq_pool_live_fullpage_q3.preds.jsonl" \
  "snippet,fullpage" \
  "$G/judge_pool_live_snippet_q3.jsonl,$G/judge_pool_live_fullpage_q3.jsonl"
echo "=== SNIPPET/FULLPAGE n=287 DONE ==="
