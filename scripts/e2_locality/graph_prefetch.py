"""Test HNSW-graph-neighbor prefetch against deep-rank (trajectory) prefetch.

Extracts level-0 adjacency from the wiki-18 HNSW64 index for all chunks served
in the t0.7 traces, then measures (a) graph-neighbor reachability of future
chunks, (b) same-page fraction of graph neighbors, (c) cache simulation with
graph-neighbor prefetch (chunk_sem) vs deep-rank prefetch (chunk_traj).
"""

import json

import faiss
import numpy as np

from scdn import paths
from scdn.analyze import dedup_repeat_queries
from scdn.cachesim import simulate
from scdn.stream import load_trace_variants, load_traces, synthesize_stream

TRACE_FILES = [paths.TRACES / f"full-t07-s{s}.jsonl" for s in (0, 1, 2)]
M_TABLE = 16  # neighbors kept per chunk in the prefetch table

traces = [t for p in TRACE_FILES for t in load_traces(p)]
needed = sorted({c for t in traces for u in t["turns"] for c in u["ids_shown"]})
print(f"chunks needing neighbor lists: {len(needed):,}")

print("loading HNSW index structure ...", flush=True)
try:
    index = faiss.read_index(str(paths.WIKI18_HNSW64_INDEX), faiss.IO_FLAG_SKIP_STORAGE)
    print("loaded graph only (storage skipped)")
except Exception as e:
    print(f"SKIP_STORAGE failed ({e}); full load ...", flush=True)
    index = faiss.read_index(str(paths.WIKI18_HNSW64_INDEX))
hnsw = index.hnsw
nb0 = hnsw.cum_nneighbor_per_level.at(1) - hnsw.cum_nneighbor_per_level.at(0)
offsets = faiss.vector_to_array(hnsw.offsets)
neigh = faiss.vector_to_array(hnsw.neighbors)
print(f"level-0 slots per node: {nb0}; neighbor array: {neigh.nbytes / 1e9:.1f} GB")

table = {}
for c in needed:
    row = neigh[offsets[c] : offsets[c] + nb0]
    row = row[row >= 0][:M_TABLE]
    table[int(c)] = [int(x) for x in row]
del neigh, offsets, index
(paths.REPO_ROOT / "outputs").mkdir(exist_ok=True)
with open(paths.REPO_ROOT / "outputs" / "hnsw_neighbors.json", "w") as f:
    json.dump(table, f)

page_ids = np.load(paths.DATA_ROOT / "corpus" / "wiki-18-meta" / "page_ids.npy")
same_page = [
    np.mean([page_ids[n] == page_ids[c] for n in ns]) for c, ns in table.items() if ns
]
print(f"graph neighbors on the same page as their source: {np.mean(same_page):.3f}")

# --- (a) reachability: fresh chunk at turn t+1 in graph-neighbors of seen chunks
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
    print(f"graph-neighbor reachability (M={M}): {hit / total:.3f}")

# --- (c) cache simulation
sizes = np.load(paths.DATA_ROOT / "corpus" / "wiki-18-meta" / "sizes.npy")
total_bytes = float(sizes.sum())
variants = load_trace_variants(TRACE_FILES)
for sname, kw in (("distinct", dict(n_sessions=None)),
                  ("zipf0.8", dict(n_sessions=5000, alpha=0.8))):
    ev = synthesize_stream(variants, arrival_rate=10.0, seed=0, **kw)
    print(f"\n### stream={sname}")
    for frac in (1e-4, 1e-3, 1e-2):
        cap = frac * total_bytes
        rows = [
            simulate(ev, "chunk_lru", cap, sizes),
            simulate(ev, "chunk_traj", cap, sizes, prefetch_budget=8),
            simulate(ev, "chunk_sem", cap, sizes, prefetch_budget=3, neighbors=table),
            simulate(ev, "chunk_sem", cap, sizes, prefetch_budget=8, neighbors=table),
        ]
        names = ["chunk_lru", "traj(b=8)", "sem(b=3/req)", "sem(b=8/req)"]
        for n, r in zip(names, rows):
            pp = f" prec={r['prefetch_precision']:.3f}" if r["prefetch_count"] else ""
            print(f"  {cap/1e6:7.2f}MB {n:12s} hit={r['hit_ratio']:.3f} "
                  f"o/d={r['origin_to_demand']:.2f}{pp}")
