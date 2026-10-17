#!/usr/bin/env bash
# Task-3 depth sweep for the OTHER encoders (Qwen3-emb, cross-encoder). Extract+chunk
# ONCE into a shared cache (encoder-independent, ~1.5h), then rank each encoder
# cheaply, read with Qwen3-32B, judge with Llama-70B. (e5 already done in §5D.3;
# no-RAG is encoder-independent and reused from §5D.3.)
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
M32=Qwen/Qwen3-32B
echo "=== cache: extract+chunk ≤50 pages/q once (CPU, ~1.5h, resumable) ==="
uv run python -m semdn.e4_crag_t3 --stage cache --n 300
echo "=== rank: Qwen3-embedding ==="
on_gpus 1 uv run python -m semdn.e4_crag_t3 --stage rank --encoder qwen3 --device cuda
echo "=== rank: cross-encoder (bge-reranker) ==="
on_gpus 1 uv run python -m semdn.e4_crag_t3 --stage rank --encoder ce --device cuda
echo "=== reader Qwen3-32B: Qwen3-emb depth sweep ==="
on_gpus 1 uv run python -m semdn.e4_answer --model "$M32" \
  --pool data/e4/pool_crag_t3_qwen3.jsonl --out data/e4/granularity_quality_crag_t3_qwen3_q3.jsonl
echo "=== reader Qwen3-32B: cross-encoder depth sweep ==="
on_gpus 1 uv run python -m semdn.e4_answer --model "$M32" \
  --pool data/e4/pool_crag_t3_ce.jsonl --out data/e4/granularity_quality_crag_t3_ce_q3.jsonl
echo "=== judge both (Llama-3.1-70B TP=2) ==="
bash scripts/lib/judge_robust.sh \
  "data/e4/granularity_quality_crag_t3_qwen3_q3.preds.jsonl,data/e4/granularity_quality_crag_t3_ce_q3.preds.jsonl" \
  "t3-qwen3emb-q3,t3-ce-q3" \
  "data/e4/judge_t3_qwen3_q3.jsonl,data/e4/judge_t3_ce_q3.jsonl"
echo "=== T3 ENCODERS DONE ==="
