#!/usr/bin/env bash
# Corrected E2 trace generation: exact Flat retrieval + Search-R1's reference
# sampling (infer.py: temperature 0.7), plus a greedy/exact control run.
#
# Usage: bash scripts/e2_locality/gen_traces.sh     (picks a free GPU per run; GPUS=... restricts)
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh

# Start a retrieval server on the exact Flat index. Never kill an existing one by name
# (shared machine): if port 8000 is taken, stop and let the user free it.
if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "port 8000 already serves something; stop that server first"; exit 1
fi
nohup uv run python -m scdn.retrieval_server \
    --index data/index/wiki-18-e5-flat/e5_Flat.index \
    --device cpu --port 8000 > data/traces/retrieval_flat.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
echo "waiting for Flat retrieval server ..."
until curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; do
    sleep 15
    kill -0 $SRV 2>/dev/null || {
        echo "SERVER DIED"; tail -5 data/traces/retrieval_flat.log; exit 1; }
done
echo "Flat server up"

# Control: greedy decoding + exact retrieval (isolates the retrieval effect
# against the original greedy+HNSW run).
on_gpus 1 40000 uv run python -m scdn.gen_traces \
    --temperature 0.0 --seed 0 --tag full-g-exact

# Conventional protocol: temperature 0.7 (infer.py), three sampling seeds
# = three trajectory variants per question for de-correlated Zipf replays.
for s in 0 1 2; do
    on_gpus 1 40000 uv run python -m scdn.gen_traces \
        --temperature 0.7 --seed "$s" --tag "full-t07-s$s"
done

echo "ALL CORRECTED RUNS DONE"
