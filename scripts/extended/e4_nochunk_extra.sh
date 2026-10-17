#!/usr/bin/env bash
# N2+N3 GPU half: full-page baseline (CRAG t3/t1) + live-web no-chunking (snippet, full-page).
# Pools are prebuilt (CPU). Reader 7B+32B, one Llama-70B judge load for all 8. RUN WHEN GPU FREE.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
export HF_HUB_OFFLINE=1
M32=Qwen/Qwen3-32B
G=data/e4

echo "=== readers: CRAG fullpage (t3,t1) + live snippet/fullpage ==="
for P in pool_crag_t3_fullpage pool_crag_t1_fullpage pool_live_snippet pool_live_fullpage; do
  on_gpus 1 uv run python -m semdn.e4_answer            --pool $G/$P.jsonl --out $G/gq_${P}.jsonl
  on_gpus 1 uv run python -m semdn.e4_answer --model "$M32" --pool $G/$P.jsonl --out $G/gq_${P}_q3.jsonl
done

echo "=== judge all 8 (Llama-70B) ==="
P=""; T=""; O=""
for pool in pool_crag_t3_fullpage pool_crag_t1_fullpage pool_live_snippet pool_live_fullpage; do for s in "" "_q3"; do
  P+=",$G/gq_${pool}${s}.preds.jsonl"; T+=",${pool}${s}"; O+=",$G/judge_${pool}${s}.jsonl"
done; done
bash scripts/lib/judge_robust.sh "${P#,}" "${T#,}" "${O#,}"
echo "=== NOCHUNK EXTRA DONE ==="
