#!/usr/bin/env bash
# Supplement §5A.2 / §5D: structure(no-title) chunker with the Qwen3-Embedding-0.6B
# encoder and with the cross-encoder reranker, under BOTH readers (Qwen2.5-7B and
# Qwen3-32B), judged by Llama-3.1-70B. Reuses cached pages (no serper credits).
# Naming: encoder qwen3 = Qwen3-Embedding-0.6B; reader q3 = Qwen3-32B.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
M32=Qwen/Qwen3-32B
P_QE=data/e4/pool_qwen3_structnt.jsonl   # qwen3-emb encoder + structnt
P_CE=data/e4/pool_ce_structnt.jsonl      # cross-encoder reranker + structnt

echo "=== retrieval: qwen3-emb encoder / structnt ==="
on_gpus 1 uv run python -m semdn.e4_granularity --stage select --encoder qwen3 --chunker structnt --n 150 --device cuda
echo "=== retrieval: cross-encoder reranker / structnt ==="
on_gpus 1 uv run python -m semdn.e4_granularity --stage rerank --chunker structnt --n 150 --device cuda

echo "=== reader Qwen2.5-7B: qwen3-emb/structnt ==="
on_gpus 1 uv run python -m semdn.e4_answer --pool "$P_QE" --out data/e4/granularity_quality_qwen3_structnt.jsonl
echo "=== reader Qwen2.5-7B: CE/structnt ==="
on_gpus 1 uv run python -m semdn.e4_answer --pool "$P_CE" --out data/e4/granularity_quality_ce_structnt.jsonl
echo "=== reader Qwen3-32B: qwen3-emb/structnt ==="
on_gpus 1 uv run python -m semdn.e4_answer --model "$M32" --pool "$P_QE" --out data/e4/granularity_quality_qwen3_structnt_q3.jsonl
echo "=== reader Qwen3-32B: CE/structnt ==="
on_gpus 1 uv run python -m semdn.e4_answer --model "$M32" --pool "$P_CE" --out data/e4/granularity_quality_ce_structnt_q3.jsonl

echo "=== judge all 4 with Llama-3.1-70B (TP=2) ==="
bash scripts/lib/judge_robust.sh \
  "data/e4/granularity_quality_qwen3_structnt.preds.jsonl,data/e4/granularity_quality_ce_structnt.preds.jsonl,data/e4/granularity_quality_qwen3_structnt_q3.preds.jsonl,data/e4/granularity_quality_ce_structnt_q3.preds.jsonl" \
  "qwen3emb-structnt-7b,ce-structnt-7b,qwen3emb-structnt-q3,ce-structnt-q3" \
  "data/e4/judge_qwen3structnt_7b.jsonl,data/e4/judge_cestructnt_7b.jsonl,data/e4/judge_qwen3structnt_q3.jsonl,data/e4/judge_cestructnt_q3.jsonl"
echo "=== STRUCTNT EXTRA DONE ==="
