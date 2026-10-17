#!/usr/bin/env bash
# wiki-18 matrix with a larger reader (Qwen3-32B, the CRAG reader) on the SAME saved contexts as
# wiki18_matrix.sh. Waits for idle GPUs (no other user computing on them): one >=76 GB card, or two
# >=38.5 GB cards split tensor-parallel. Retries if the job dies (e.g. a card got taken at load).
# Outputs: data/e1/matrix_llm_qwen3_32b_{quality,preds}.jsonl, matrix_llm_qwen3_32b_significance.json
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh
export ON_GPUS_IDLE=1
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo NCCL_CUMEM_ENABLE=0
# DATASETS=nq RTAG=qwen3_32b_nq_ redoes one dataset into separate files (merge them afterwards).
RUN=(uv run python -m semdn.e1_matrix_answer --tag llm_ --model Qwen/Qwen3-32B
     --reader-tag "${RTAG:-qwen3_32b_}" --datasets "${DATASETS:-hotpotqa,2wikimultihopqa,musique,bamboogle,nq}")
for attempt in 1 2 3 4 5; do
  while :; do
    if on_gpus 1 76000 true 2>/dev/null; then n=1 min=76000; break; fi
    if on_gpus 2 38500 true 2>/dev/null; then n=2 min=38500; break; fi
    echo "waiting for idle GPUs ($(date +%H:%M))"; sleep 300
  done
  on_gpus $n $min "${RUN[@]}" && exit 0
  echo "attempt $attempt failed; waiting again"; sleep 60
done
exit 1
