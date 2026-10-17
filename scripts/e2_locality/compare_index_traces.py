"""Does the ANN index (HNSW vs exact) change the traffic the cache sees?

Compares paired greedy trace sets (same questions, same decoding; only the
retrieval index differs) on locality and cacheability.
"""

import numpy as np

from semdn import paths
from semdn.analyze import dedup_repeat_queries, duplicate_query_rate
from semdn.cachesim import simulate
from semdn.stream import load_traces, synthesize_stream, within_task_overlap

sizes = np.load(paths.DATA_ROOT / "corpus" / "wiki-18-meta" / "sizes.npy")
page_ids = np.load(paths.DATA_ROOT / "corpus" / "wiki-18-meta" / "page_ids.npy")
total = float(sizes.sum())

SETS = {
    "greedy+HNSW": paths.TRACES / "full.jsonl",
    "greedy+exact": paths.TRACES / "full-g-exact.jsonl",
}
traces = {k: load_traces(p) for k, p in SETS.items()}

# turn-1 queries are identical across the pair (same prompt, greedy):
# direct per-query divergence measure between the indexes.
t1 = {}
for k, tr in traces.items():
    t1[k] = {t["task_id"]: t["turns"][0] for t in tr if t["turns"]}
common = sorted(set(t1["greedy+HNSW"]) & set(t1["greedy+exact"]))
same_q = [tid for tid in common
          if t1["greedy+HNSW"][tid]["query"] == t1["greedy+exact"][tid]["query"]]
ov3 = [len(set(t1["greedy+HNSW"][tid]["ids"][:3]) & set(t1["greedy+exact"][tid]["ids"][:3])) / 3
       for tid in same_q]
ov50 = [len(set(t1["greedy+HNSW"][tid]["ids"][:50]) & set(t1["greedy+exact"][tid]["ids"][:50])) / 50
        for tid in same_q]
print(f"turn-1 identical queries: {len(same_q)}/{len(common)}")
print(f"  top-3 overlap HNSW vs exact: {np.mean(ov3):.3f}   top-50 overlap: {np.mean(ov50):.3f}")

for k, tr in traces.items():
    shown = [c for t in tr for u in t["turns"] for c in u["ids_shown"]]
    distinct = len(set(shown))
    dup = duplicate_query_rate(tr)["dup_rate_after_turn1"]
    wt = within_task_overlap(dedup_repeat_queries(tr))["seen_before_frac"]
    # popularity concentration of the chunk accesses themselves
    cnt = np.array(sorted(np.unique(shown, return_counts=True)[1])[::-1])
    top1pct = cnt[: max(1, len(cnt) // 100)].sum() / cnt.sum()
    print(f"\n[{k}] accesses={len(shown)}  distinct_chunks={distinct} "
          f"(working set {distinct * 639 / 1e6:.1f} MB)")
    print(f"  dup_rate={dup:.3f}  seen@t2/t3/t4 (dedup)={wt[2]:.3f}/{wt[3]:.3f}/{wt[4]:.3f}")
    print(f"  top-1%-chunks share of accesses: {top1pct:.3f}")
    for stream_name, kw in (("distinct", dict(n_sessions=None)),
                            ("zipf0.8", dict(n_sessions=5000, alpha=0.8))):
        ev = synthesize_stream(tr, arrival_rate=10.0, seed=0, **kw)
        for frac in (1e-4, 1e-3):
            r = simulate(ev, "chunk_lru", frac * total, sizes)
            print(f"  {stream_name:8s} chunk_lru @{frac * total / 1e6:.2f}MB: "
                  f"hit={r['hit_ratio']:.3f}")
