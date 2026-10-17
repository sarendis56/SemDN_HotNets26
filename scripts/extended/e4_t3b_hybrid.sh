#!/usr/bin/env bash
# Scale-up (free, no Serper): snippet+chunk HYBRID on the n=600 CRAG provided-pages set.
# Inject the CRAG page snippets as rank-0 co-candidates next to the coherent chunks, rerank
# with e5, depth-sweep. Tests the O1 ranking-bottleneck finding at n=600 (vs the live n=147).
# Cache + snippet pool already exist. RUN WHEN GPU FREE.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
export HF_HUB_OFFLINE=1
M32=Qwen/Qwen3-32B
G=data/e4
PK=1,3,5,10,20,50

echo "=== rank t3b coherent e5 + hybrid snippets ==="
on_gpus 1 uv run python -m scdn.e4_crag_t3 --stage rank --corpus t3b \
  --chunker coherent --encoder e5 --hybrid --page-ks "$PK"

echo "=== reader (Qwen3-32B) ==="
on_gpus 1 uv run python -m scdn.e4_answer --model "$M32" \
  --pool $G/pool_crag_t3b_coherent_e5_hybrid.jsonl --out $G/gq_t3b_hybrid_q3.jsonl

echo "=== judge (Llama-70B) ==="
bash scripts/lib/judge_robust.sh \
  "$G/gq_t3b_hybrid_q3.preds.jsonl" "t3b-hybrid" "$G/judge_t3b_hybrid_q3.jsonl"
echo "=== T3B HYBRID DONE ==="
