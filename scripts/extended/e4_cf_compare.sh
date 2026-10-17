#!/usr/bin/env bash
# Experiment (a) downstream — GPU. Build orig vs augmented (Cloudflare-recovered) live-web
# pools for the 25 recovered questions, read with 7B+32B, judge with Llama-70B, paired-compare.
# RUN ONLY WHEN GPUs ARE FREE.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
export HF_HUB_OFFLINE=1
M32=Qwen/Qwen3-32B
G=data/e4

echo "=== build orig vs aug pools (e5 rank, coherent chunks) ==="
on_gpus 1 uv run python -m scdn.e4_cf_compare

echo "=== readers (7B + 32B) on both arms ==="
for arm in orig aug; do
  P=$G/pool_live_cf_${arm}.jsonl
  on_gpus 1 uv run python -m scdn.e4_answer            --pool "$P" --out $G/gq_live_cf_${arm}.jsonl
  on_gpus 1 uv run python -m scdn.e4_answer --model "$M32" --pool "$P" --out $G/gq_live_cf_${arm}_q3.jsonl
done

echo "=== judge all 4 (Llama-70B) ==="
bash scripts/lib/judge_robust.sh \
  "$G/gq_live_cf_orig.preds.jsonl,$G/gq_live_cf_aug.preds.jsonl,$G/gq_live_cf_orig_q3.preds.jsonl,$G/gq_live_cf_aug_q3.preds.jsonl" \
  "live-cf-orig-7b,live-cf-aug-7b,live-cf-orig-q3,live-cf-aug-q3" \
  "$G/judge_live_cf_orig_7b.jsonl,$G/judge_live_cf_aug_7b.jsonl,$G/judge_live_cf_orig_q3.jsonl,$G/judge_live_cf_aug_q3.jsonl"
echo "=== CF COMPARE DONE ==="
