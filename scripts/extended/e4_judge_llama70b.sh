#!/usr/bin/env bash
# Wait for the Llama-3.1-70B-Instruct download (CRAG's official judge) to finish,
# then re-judge both pools with it in bf16 on TP=2 (lossless, both A100s). Resolves
# whether the ~50% hallucination is real or a 7B-judge artifact.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
DL_PID="$1"
MODELDIR=models/meta-llama/Llama-3.1-70B-Instruct

while kill -0 "$DL_PID" 2>/dev/null; do sleep 60; done
nshards=$(find "$MODELDIR" -name "*.safetensors" | wc -l)
sz=$(du -sh "$MODELDIR" | cut -f1)
echo "=== download process exited: $nshards safetensors, $sz total ==="
if [ "$nshards" -lt 30 ]; then
  echo "INCOMPLETE (expected 30 shards) — aborting judge; rerun download."; exit 1
fi
MODEL=meta-llama/Llama-3.1-70B-Instruct
echo "=== judge e5 (Llama-70B TP=2) ==="
on_gpus 2 78000 uv run python -m scdn.e4_judge \
  --preds data/e4/granularity_quality.preds.jsonl --tag e5-llama70b --model "$MODEL" \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.95 --max-model-len 2048 \
  --out data/e4/judge_e5_llama70b.jsonl
echo "=== judge qwen3 (Llama-70B TP=2) ==="
on_gpus 2 78000 uv run python -m scdn.e4_judge \
  --preds data/e4/granularity_quality_qwen3.preds.jsonl --tag qwen3-llama70b --model "$MODEL" \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.95 --max-model-len 2048 \
  --out data/e4/judge_qwen3_llama70b.jsonl
echo "=== E4 LLAMA JUDGE DONE ==="
