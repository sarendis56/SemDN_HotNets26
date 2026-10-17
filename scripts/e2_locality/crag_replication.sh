#!/usr/bin/env bash
# E2-CRAG (Task 2): stand up the shared CRAG e5 index as a retrieval server, drive
# Search-R1 over it (same agent/protocol as E2 wiki-18), then run the locality +
# cache-sim analysis. Server e5 on GPU 0; Search-R1 on GPU 1. Index built by
# semdn.e2_crag_index (data/e2_crag/{index.faiss,contents.jsonl,questions.jsonl,
# sizes.npy,page_ids.npy}).
#   usage: bash scripts/e2_locality/crag_replication.sh [n_variants=1]
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
D=data/e2_crag
NV="${1:-1}"

echo "=== launch CRAG retrieval server (e5 on a free GPU, Flat index) ==="
# exec: $! is then the server itself, so the EXIT trap kills it
on_gpus 1 20000 exec uv run python -m semdn.retrieval_server \
  --index "$D/index.faiss" --contents "$D/contents.jsonl" --device cuda --port 8000 \
  > "$D/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

echo "=== wait for /health ==="
for i in $(seq 1 120); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then echo "server up"; break; fi
  sleep 5
  if ! kill -0 $SRV 2>/dev/null; then echo "SERVER DIED"; tail -20 "$D/server.log"; exit 1; fi
done

echo "=== generate Search-R1 traces over CRAG ($NV variant(s)) ==="
for s in $(seq 0 $((NV-1))); do
  echo "--- variant seed $s ---"
  on_gpus 1 uv run python -m semdn.gen_traces \
    --questions-file "$D/questions.jsonl" --questions-tag crag \
    --n-per-dataset 0 --retrieval-url http://127.0.0.1:8000 \
    --seed "$s" --tag "crag-s$s" --out "$D/crag-s$s.jsonl"
done

echo "=== locality + cache-sim analysis ==="
uv run python -m semdn.analyze --traces $D/crag-s*.jsonl \
  --meta-dir "$D" --out-tag e2_crag

echo "=== E2-CRAG DONE ==="
