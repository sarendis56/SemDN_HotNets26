#!/usr/bin/env bash
# Download every model and dataset the experiments use into models/ and data/
# (both may be symlinks to a large disk). Sizes and what needs what: README.md "Artifacts".
#
#   bash scripts/setup/download_data.sh            # everything (~520 GB)
#   bash scripts/setup/download_data.sh wiki18     # only one group: models | wiki18 | crag
#
# Llama-3.1-70B-Instruct is gated: accept its license on Hugging Face and `hf auth login` first.
set -euo pipefail
cd "$(dirname "$0")/../.."
GROUP=${1:-all}
want() { [ "$GROUP" = all ] || [ "$GROUP" = "$1" ]; }

# hf_transfer accelerates downloads but is an optional dependency.
if python3 -c "import hf_transfer" 2>/dev/null; then export HF_HUB_ENABLE_HF_TRANSFER=1; fi
get_model() { hf download "$1" --local-dir "models/$1"; }

if want models; then
  get_model intfloat/e5-base-v2                                           # 0.4 GB  retriever (all experiments)
  get_model PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.3  # 15 GB   agent (E2 traces)
  get_model Qwen/Qwen2.5-7B-Instruct                                      # 15 GB   reader (wiki-18)
  get_model Qwen/Qwen3-32B                                                # 62 GB   reader (CRAG)
  get_model Qwen/Qwen3-Reranker-4B                                        # 8 GB    LLM reranker
  get_model BAAI/bge-reranker-v2-m3                                       # 2.2 GB  cross-encoder reranker
  get_model meta-llama/Llama-3.1-70B-Instruct                             # 132 GB  CRAG judge (gated)
  # only for experiments outside the paper:
  get_model Qwen/Qwen3-Embedding-0.6B                                     # 1.2 GB
  get_model Qwen/Qwen3-Reranker-0.6B                                      # 1.2 GB
fi

if want wiki18; then
  # wiki-18 corpus (21M passages, 2018 Wikipedia dump). The .jsonl.gz is a tar.gz despite its name.
  hf download PeterJinGo/wiki-18-corpus --repo-type dataset --local-dir data/corpus/wiki-18
  [ -f data/corpus/wiki-18/wiki-18.jsonl ] ||
    tar -xzf data/corpus/wiki-18/wiki-18.jsonl.gz -O > data/corpus/wiki-18/wiki-18.jsonl
  uv run python -m semdn.corpus build          # offsets / sizes / page ids -> data/corpus/wiki-18-meta

  # e5 indexes over wiki-18, shipped as split parts; join, then drop the parts.
  for ix in "wiki-18-e5-index wiki-18-e5-flat e5_Flat.index" \
            "wiki-18-e5-index-HNSW64 wiki-18-e5-hnsw64 e5_HNSW64.index"; do
    set -- $ix
    hf download "PeterJinGo/$1" --repo-type dataset --local-dir "data/index/$2"
    if [ ! -f "data/index/$2/$3" ]; then
      cat "data/index/$2"/part_* > "data/index/$2/$3" && rm "data/index/$2"/part_*
    fi
  done

  # QA sets (FlashRAG format, as used by Search-R1)
  hf download RUC-NLPIR/FlashRAG_datasets --repo-type dataset \
    --include "hotpotqa/*" "2wikimultihopqa/*" "musique/*" "bamboogle/*" "nq/*" \
    --local-dir data/datasets/flashrag
fi

if want crag; then
  # CRAG Task 1 (5 pages/question) and Task 3 (50 pages/question, 7.2 GB bz2 -> 48 GB).
  # The mirror drops long transfers; hf download resumes, so retry.
  D=data/datasets/crag
  for f in crag_task_1_dev_v4_release.jsonl.bz2 crag_task_3_dev_v4.tar.bz2; do
    for a in $(seq 1 10); do
      HF_HUB_DISABLE_XET=1 hf download Rickyoung0221/crag "$f" --repo-type dataset --local-dir "$D" && break
      echo "download of $f failed (attempt $a); resuming in 10 s"; sleep 10
    done
  done
  [ -d "$D/crag_task_3_dev_v4" ] || tar -xjf "$D/crag_task_3_dev_v4.tar.bz2" -C "$D"
fi

echo "done."
du -shL models/*/* data/*/* 2>/dev/null || true
