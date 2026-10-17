#!/usr/bin/env bash
# Re-run the Cloudflare AUGMENTED-pool readers (they crashed after page@1 on a 17k-char
# nav-heavy markdown chunk; fixed by context truncation in e4_answer) and re-judge aug.
# orig is already complete. RUN AFTER the GPU queue.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
export HF_HUB_OFFLINE=1
M32=Qwen/Qwen3-32B
G=data/e4
P=$G/pool_live_cf_aug.jsonl

on_gpus 1 uv run python -m semdn.e4_answer            --pool "$P" --out $G/gq_live_cf_aug.jsonl
on_gpus 1 uv run python -m semdn.e4_answer --model "$M32" --pool "$P" --out $G/gq_live_cf_aug_q3.jsonl
bash scripts/lib/judge_robust.sh \
  "$G/gq_live_cf_aug.preds.jsonl,$G/gq_live_cf_aug_q3.preds.jsonl" \
  "live-cf-aug-7b,live-cf-aug-q3" \
  "$G/judge_live_cf_aug_7b.jsonl,$G/judge_live_cf_aug_q3.jsonl"
echo "=== CF AUG FIX DONE ==="
