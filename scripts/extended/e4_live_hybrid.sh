#!/usr/bin/env bash
# H.3 follow-up: snippet+chunk HYBRID on live web. Inject the SERP snippets as ranking
# co-candidates next to the structnt deep chunks, rerank with e5, take top-3. Tests whether
# the hybrid captures BOTH strengths (snippet precision on factoids + deep reach on
# aggregation). Waits for the qwen3rr n=600 GPU job to finish first (one GPU job at a time).
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
export HF_HUB_OFFLINE=1
M32=Qwen/Qwen3-32B
G=data/e4

echo "########## waiting for qwen3rr n=600 to free the GPU ##########"
until grep -q "QWEN3RR N600 DONE" $G/n600_qwen3rr.log 2>/dev/null; do sleep 60; done

echo "=== build hybrid pool (e5 + structnt + SERP snippets) ==="
on_gpus 1 uv run python -m semdn.e4_granularity --stage select --encoder e5 --chunker structnt --hybrid

echo "=== how often did chunk@all pick a snippet (src_rank 0) vs a deep chunk? ==="
uv run python -c "
import json
recs=[json.loads(l) for l in open('$G/pool_structnt_hybrid.jsonl') if json.loads(l).get('n_usable',0)>0]
tot=sum(len(r['conds']['chunk@all']['src_ranks']) for r in recs)
snip=sum(1 for r in recs for s in r['conds']['chunk@all']['src_ranks'] if s==0)
print(f'  {len(recs)} questions; chunk@all picks: {snip}/{tot} are SERP snippets ({100*snip/tot:.0f}%)')
"

echo "=== reader (Qwen3-32B) ==="
on_gpus 1 uv run python -m semdn.e4_answer --model "$M32" \
  --pool $G/pool_structnt_hybrid.jsonl --out $G/gq_live_hybrid_structnt_e5_q3.jsonl

echo "=== judge (Llama-70B) ==="
bash scripts/lib/judge_robust.sh \
  "$G/gq_live_hybrid_structnt_e5_q3.preds.jsonl" \
  "live-hybrid" \
  "$G/judge_live_hybrid_q3.jsonl"
echo "=== HYBRID DONE ==="
