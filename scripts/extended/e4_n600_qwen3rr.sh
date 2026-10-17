#!/usr/bin/env bash
# Complete H.1's reranker coverage: Qwen3-Reranker on the SAME n=600 coherent cache
# (e5 + cross-encoder already done). Rank -> 32B reader -> Llama-70B judge.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
export HF_HUB_OFFLINE=1
M32=Qwen/Qwen3-32B
G=data/e4
PK=1,3,5,10,20,50

echo "=== rank t3b coherent: qwen3rr ==="
on_gpus 1 uv run python -m semdn.e4_crag_t3 --stage rank --corpus t3b --chunker coherent --encoder qwen3rr --page-ks "$PK"

echo "=== reader (Qwen3-32B) ==="
on_gpus 1 uv run python -m semdn.e4_answer --model "$M32" \
  --pool $G/pool_crag_t3b_coherent_qwen3rr.jsonl --out $G/gq_pool_crag_t3b_coherent_qwen3rr_q3.jsonl

echo "=== judge (Llama-70B) ==="
bash scripts/lib/judge_robust.sh \
  "$G/gq_pool_crag_t3b_coherent_qwen3rr_q3.preds.jsonl" \
  "t3b-qwen3rr" \
  "$G/judge_t3b_qwen3rr_q3.jsonl"
echo "=== QWEN3RR N600 DONE ==="
