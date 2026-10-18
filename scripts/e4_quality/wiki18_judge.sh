#!/usr/bin/env bash
# Grade the wiki-18 matrix answers of both readers (Qwen2.5-7B, Qwen3-32B) with the CRAG judge
# (Llama-3.1-70B bf16, two 80 GB GPUs), then paired McNemar tests. Waits for two IDLE >=78 GB cards.
# Outputs: data/e1/judge_matrix_llm_{qwen25_7b_,qwen3_32b_}{.jsonl,.raw.jsonl,significance.json}
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh
export ON_GPUS_IDLE=1
for t in "" qwen3_32b_; do uv run python -m semdn.e1_matrix_judge prep --reader-tag "$t" || exit 1; done
until on_gpus 2 78000 true 2>/dev/null; do echo "waiting for 2 idle GPUs ($(date +%H:%M))"; sleep 300; done
D=data/e1
bash scripts/lib/judge_robust.sh \
  "$D/judge_in_matrix_llm_qwen25_7b_.jsonl,$D/judge_in_matrix_llm_qwen3_32b_.jsonl" "qwen25_7b,qwen3_32b" \
  "$D/judge_matrix_llm_qwen25_7b_.jsonl,$D/judge_matrix_llm_qwen3_32b_.jsonl"
for t in "" qwen3_32b_; do uv run python -m semdn.e1_matrix_judge stats --reader-tag "$t"; done
