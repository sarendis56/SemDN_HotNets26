#!/usr/bin/env bash
# wiki-18 granularity matrix (paper §6.3, first paragraph): chunk vs page x (e5 | e5 -> Qwen3-Reranker-4B)
# on HotpotQA / 2Wiki / MuSiQue / Bamboogle / NQ, Qwen2.5-7B reader, paired permutation tests.
#   retrieval: one process per dataset, each on its own GPU (>=30 GB free); the flat index is
#              memory-mapped, so the processes share one copy in RAM
#   reader:    one GPU, all datasets; then the paired tests
# Outputs: data/e1/matrix_llm_<ds>.jsonl, matrix_llm_{quality,preds}.jsonl, matrix_llm_significance.json
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16   # 192-core box: BLAS arena limit
DATASETS=${DATASETS:-hotpotqa 2wikimultihopqa musique bamboogle nq}
N=${N:-500}
TAG=${TAG:-llm_}
mkdir -p data/e1/logs
pids=()
for ds in $DATASETS; do
  grep -q "^wrote " "data/e1/logs/matrix_${TAG}${ds}.log" 2>/dev/null && continue   # finished earlier
  ON_GPUS_WAIT=1 on_gpus 1 30000 uv run python -m scdn.e1_matrix --datasets "$ds" --n "$N" --tag "$TAG" \
    > "data/e1/logs/matrix_${TAG}${ds}.log" 2>&1 &
  pids+=($!)
  [ "$ds" = "${DATASETS##* }" ] || sleep 240   # let it load the reranker before the next picks a GPU
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" = 0 ] || { echo "a retrieval process failed; see data/e1/logs/"; exit 1; }
on_gpus 1 40000 uv run python -m scdn.e1_matrix_answer --tag "$TAG" --datasets "$(echo $DATASETS | tr ' ' ,)"
