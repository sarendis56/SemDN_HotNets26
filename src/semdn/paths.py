"""Canonical locations of large artifacts, all relative to the repo root:

  data/    datasets, corpora, indexes, traces, and experiment outputs (may be a symlink)
  models/  model weights, one folder per Hugging Face id, e.g. models/Qwen/Qwen3-32B

See README.md ("Artifacts") for where to download or how to build each one.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
MODELS = REPO_ROOT / "models"


def model(name: str) -> str:
    """Hugging Face id -> models/<id> when downloaded there, else the id itself
    (transformers/vLLM then fetch it into the Hugging Face cache). Paths pass through."""
    local = MODELS / name
    return str(local) if local.exists() else str(name)


SEARCH_R1_7B = MODELS / "PeterJinGo" / "SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.3"
E5_BASE_V2 = MODELS / "intfloat" / "e5-base-v2"
QWEN25_7B = MODELS / "Qwen" / "Qwen2.5-7B-Instruct"
QWEN3_EMB_06B = MODELS / "Qwen" / "Qwen3-Embedding-0.6B"

WIKI18_CORPUS = DATA_ROOT / "corpus" / "wiki-18" / "wiki-18.jsonl.gz"
WIKI18_FLAT_INDEX = DATA_ROOT / "index" / "wiki-18-e5-flat" / "e5_Flat.index"
WIKI18_HNSW64_INDEX = DATA_ROOT / "index" / "wiki-18-e5-hnsw64" / "e5_HNSW64.index"

FLASHRAG_DATASETS = DATA_ROOT / "datasets" / "flashrag"
TRACES = DATA_ROOT / "traces"
