#!/usr/bin/env bash
# Wait for the resilient fetch to finish, then run the GPU stages (e5 select +
# vLLM reader) on cuda:1. Standalone script so it tracks as one process.
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
until grep -q "RESILIENT FETCH DONE" data/e4/logs/e4g_fetch_driver.log 2>/dev/null; do
  pgrep -f "e4_fetch_resilient" >/dev/null || { echo "driver died before marker"; }
  sleep 30
done
echo "=== fetch done -> select (cuda:1) ==="
on_gpus 1 uv run python -m semdn.e4_granularity --stage select --device cuda
echo "=== select done -> reader (cuda:1) ==="
on_gpus 1 uv run python -m semdn.e4_answer
echo "=== E4 ANSWER DONE ==="
