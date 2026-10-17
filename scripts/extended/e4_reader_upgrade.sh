#!/usr/bin/env bash
# Reader upgrade: re-generate predictions with Qwen3-32B (newer, ~32B, non-thinking)
# on the EXISTING pools (retrieval unchanged), saving _q3 preds (kept separate from
# the Qwen2.5-7B preds for comparison), then judge with Llama-3.1-70B. Generation
# needs 1 GPU (cuda:1; 32B bf16 ~65GB fits); judge needs both (TP=2).
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
DLPID="$1"
M=models/Qwen/Qwen3-32B
MODEL=Qwen/Qwen3-32B

echo "waiting for Qwen3-32B download (pid $DLPID)..."
while kill -0 "$DLPID" 2>/dev/null; do sleep 30; done
ns=$(find "$M" -name "*.safetensors" | wc -l)
echo "download exited: $ns safetensors"
if [ "$ns" -lt 17 ]; then echo "INCOMPLETE ($ns/17) — aborting reader upgrade"; exit 1; fi

gen() { on_gpus 1 uv run python -m semdn.e4_answer --model "$MODEL" "$@"; }
echo "=== [q3] reader: fixed/e5 pool (page@1..10, chunk@all) ==="
gen --pool data/e4/pool.jsonl          --out data/e4/granularity_quality_q3.jsonl
echo "=== [q3] reader: structnt pool ==="
gen --pool data/e4/pool_structnt.jsonl --out data/e4/granularity_quality_structnt_q3.jsonl
echo "=== [q3] reader: CRAG arms (pageret/chunkret) ==="
gen --pool data/e4/pool_crag.jsonl     --out data/e4/granularity_quality_crag_q3.jsonl
echo "=== [q3] reader: CRAG no-RAG ==="
gen --pool data/e4/pool_crag.jsonl     --out data/e4/granularity_quality_crag_norag_q3.jsonl --closed-book
echo "=== [q3] reader: live-web no-RAG ==="
gen --pool data/e4/pool.jsonl          --out data/e4/granularity_quality_live_norag_q3.jsonl --closed-book
echo "=== Q3 GENERATION DONE — preds: ==="
ls -la data/e4/*_q3.preds.jsonl

echo "=== [q3] judging with Llama-3.1-70B (TP=2) ==="
bash scripts/lib/judge_robust.sh \
  "data/e4/granularity_quality_q3.preds.jsonl,data/e4/granularity_quality_structnt_q3.preds.jsonl,data/e4/granularity_quality_crag_q3.preds.jsonl,data/e4/granularity_quality_crag_norag_q3.preds.jsonl,data/e4/granularity_quality_live_norag_q3.preds.jsonl" \
  "q3-fixed-e5,q3-structnt-e5,q3-crag,q3-crag-norag,q3-live-norag" \
  "data/e4/judge_q3_fixed_e5.jsonl,data/e4/judge_q3_structnt.jsonl,data/e4/judge_q3_crag.jsonl,data/e4/judge_q3_crag_norag.jsonl,data/e4/judge_q3_live_norag.jsonl"
echo "=== Q3 PIPELINE DONE ==="
