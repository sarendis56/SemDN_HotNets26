"""Validate HNSW64 retrieval against exact Flat search on logged trace queries.

Samples N queries from a trace file, re-embeds them with e5-base-v2 (CPU),
searches the exact IndexFlatIP (~60 GB RAM), and reports recall of the logged
(HNSW64, efSearch=256) results against exact ground truth.

Run:  uv run python scripts/setup/validate_recall.py --traces data/traces/full.jsonl
"""

import argparse
import json
import random
from pathlib import Path

import faiss
import numpy as np

from semdn import paths
from semdn.retrieval_server import E5Encoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=Path, default=paths.TRACES / "full.jsonl")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    turns = []
    with open(args.traces) as f:
        for line in f:
            t = json.loads(line)
            turns.extend((u["query"], u["ids"]) for u in t["turns"])
    random.Random(args.seed).shuffle(turns)
    turns = turns[: args.n]
    print(f"validating {len(turns)} logged queries", flush=True)

    enc = E5Encoder(paths.E5_BASE_V2, device="cpu")
    embs = np.concatenate(
        [enc.encode_queries([q for q, _ in turns[i : i + 64]]) for i in range(0, len(turns), 64)]
    )
    print("queries embedded; loading Flat index (~60 GB RAM) ...", flush=True)
    index = faiss.read_index(str(paths.WIKI18_FLAT_INDEX))
    print(f"flat ntotal={index.ntotal:,}; exact search ...", flush=True)
    _, exact = index.search(embs, 50)

    r3, r50, top1 = [], [], []
    for (q, logged), ex in zip(turns, exact):
        ex3, ex50 = set(ex[:3].tolist()), set(ex[:50].tolist())
        r3.append(len(set(logged[:3]) & ex3) / 3)
        r50.append(len(set(logged[:50]) & ex50) / 50)
        top1.append(int(logged[0] == ex[0]))
    print(f"recall@3  (HNSW top-3 vs exact top-3):   {np.mean(r3):.4f}")
    print(f"recall@50 (HNSW top-50 vs exact top-50): {np.mean(r50):.4f}")
    print(f"top-1 agreement:                         {np.mean(top1):.4f}")


if __name__ == "__main__":
    main()
