"""Disambiguate the graph-prefetch negative result: HNSW level-0 rows are a
*diversified* subset of similarity (RNG pruning), not true nearest neighbors.

Builds a prefetch table from the TRUE exact top-M ANN of each served chunk
(vector reconstructed from the Flat index, exact search), then re-runs the
same-page fraction, reachability, and cache-sim comparisons of graph_prefetch.py.
"""

import json

import faiss
import numpy as np

from scdn import paths
from scdn.analyze import dedup_repeat_queries
from scdn.cachesim import simulate
from scdn.stream import load_trace_variants, load_traces, synthesize_stream

TRACE_FILES = [paths.TRACES / f"full-t07-s{s}.jsonl" for s in (0, 1, 2)]
M_TABLE = 16

traces = [t for p in TRACE_FILES for t in load_traces(p)]
needed = sorted({c for t in traces for u in t["turns"] for c in u["ids_shown"]})
print(f"chunks needing true-ANN lists: {len(needed):,}", flush=True)

print("loading Flat index (~60 GB) ...", flush=True)
index = faiss.read_index(str(paths.WIKI18_FLAT_INDEX))
vecs = np.vstack([index.reconstruct(int(c)) for c in needed])
print("vectors reconstructed; exact search (this is the slow part) ...", flush=True)
_, nn = index.search(vecs, M_TABLE + 1)
table = {}
for c, row in zip(needed, nn):
    table[int(c)] = [int(x) for x in row if x != c][:M_TABLE]
del index, vecs
with open(paths.REPO_ROOT / "outputs" / "ann_neighbors.json", "w") as f:
    json.dump(table, f)

page_ids = np.load(paths.DATA_ROOT / "corpus" / "wiki-18-meta" / "page_ids.npy")
same_page = [np.mean([page_ids[n] == page_ids[c] for n in ns]) for c, ns in table.items() if ns]
print(f"TRUE-ANN neighbors on the same page as their source: {np.mean(same_page):.3f}")

graph = {int(k): v for k, v in json.load(
    open(paths.REPO_ROOT / "outputs" / "hnsw_neighbors.json")).items()}
ov = [len(set(table[c]) & set(graph.get(c, []))) / max(1, len(table[c])) for c in table]
print(f"overlap of true-ANN top-16 with HNSW level-0 rows: {np.mean(ov):.3f}")

for M in (4, 8, 16):
    hit = total = 0
    for t in dedup_repeat_queries(traces):
        seen: set = set()
        reach: set = set()
        for u in t["turns"]:
            if u["turn"] > 1:
                fresh = [c for c in u["ids_shown"] if c not in seen]
                hit += sum(c in reach for c in fresh)
                total += len(fresh)
            for c in u["ids_shown"]:
                seen.add(c)
                reach.update(table.get(c, [])[:M])
    print(f"true-ANN reachability (M={M}): {hit / total:.3f}")

sizes = np.load(paths.DATA_ROOT / "corpus" / "wiki-18-meta" / "sizes.npy")
total_bytes = float(sizes.sum())
variants = load_trace_variants(TRACE_FILES)
for sname, kw in (("distinct", dict(n_sessions=None)),
                  ("zipf0.8", dict(n_sessions=5000, alpha=0.8))):
    ev = synthesize_stream(variants, arrival_rate=10.0, seed=0, **kw)
    print(f"### stream={sname}")
    for frac in (1e-3, 1e-2):
        cap = frac * total_bytes
        r = simulate(ev, "chunk_sem", cap, sizes, prefetch_budget=3, neighbors=table)
        print(f"  {cap/1e6:7.2f}MB sem-trueANN(b=3/req) hit={r['hit_ratio']:.3f} "
              f"o/d={r['origin_to_demand']:.2f} prec={r['prefetch_precision']:.3f}")
