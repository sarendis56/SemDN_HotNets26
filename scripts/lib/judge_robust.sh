#!/usr/bin/env bash
# Robust Llama-3.1-70B bf16 TP=2 judge. Written when the box had 2 SYS-connected A100s, where
# NCCL over socket was intermittently flaky at init AND slow (occasional mid-gen stall).
# This wrapper: pins loopback for stability, raises timeouts, and retries until the output
# is written. Each attempt carries a unique SCDN_RUN_TAG env var (inherited by every vLLM
# worker); a failed attempt's orphans are killed by that tag only -- never by name, since
# the machine is shared and other vLLM jobs (ours or other users') must survive.
# Args: $1=preds(csv) $2=tags(csv) $3=outs(csv)   (first out file used as success check)
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
PREDS="$1"; TAGS="$2"; OUTS="$3"
IFS=',' read -ra OUT_ARR <<< "$OUTS"
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo NCCL_CUMEM_ENABLE=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 VLLM_RPC_TIMEOUT=1800000
for attempt in 1 2 3 4 5; do
  echo "=== judge attempt $attempt ==="
  tag="judge-$$-$attempt"
  SCDN_RUN_TAG=$tag on_gpus 2 78000 uv run python -m scdn.e4_judge \
    --preds "$PREDS" --tags "$TAGS" --outs "$OUTS" \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.95 --max-model-len 2048
  kill_tagged "$tag"; sleep 5   # orphaned workers of THIS attempt only
  ok=1; for o in "${OUT_ARR[@]}"; do [ -s "$o" ] || ok=0; done
  if [ "$ok" = 1 ]; then echo "=== JUDGE SUCCESS (attempt $attempt) ==="; break; fi
  echo "attempt $attempt incomplete; retrying"; sleep 10
done
