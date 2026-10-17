#!/usr/bin/env bash
# Corrected snippet arm for the iso-budget sweep: pack ALL provided CRAG snippets (~47/q)
# so the snippet channel can fill the budget. Build (CPU, streams source) -> Qwen3-32B
# reader -> Llama-70B judge. Pairs against the chunk arm at every budget in e4_iso_stats.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
G=data/e4
M32=Qwen/Qwen3-32B

echo "=== [1/3] build snippet_full pool (stream CRAG t3 source ~50GB) ==="
uv run python -m semdn.e4_isobudget snippet_full --n 300

echo "=== [2/3] reader Qwen3-32B ==="
on_gpus 1 uv run python -m semdn.e4_answer --model "$M32" \
  --pool "$G/pool_iso_snippet_full.jsonl" --out "$G/ans_iso_snippet_full_q3.jsonl"

echo "=== [3/3] judge Llama-3.1-70B (TP=2) ==="
bash scripts/lib/judge_robust.sh "$G/ans_iso_snippet_full_q3.preds.jsonl" \
  "iso_snippet_full" "$G/judge_iso_snippet_full_q3.jsonl"
echo "=== ISO SNIPPET_FULL DONE ==="
