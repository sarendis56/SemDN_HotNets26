"""Dump exact numbers for the E2 report (markdown tables)."""

import numpy as np

from semdn import paths
from semdn.cachesim import simulate
from semdn.stream import load_trace_variants, synthesize_stream

TRACE_FILES = [paths.TRACES / f"full-t07-s{s}.jsonl" for s in (0, 1, 2)]
variants = load_trace_variants(TRACE_FILES)
n_turns = sum(len(t["turns"]) for vs in variants for t in vs) // len(TRACE_FILES)
print(f"tasks={len(variants)}  mean turns/variant set={n_turns}")

streams = {
    "distinct": synthesize_stream(variants, n_sessions=None, arrival_rate=10.0, seed=0),
    "zipf0.8": synthesize_stream(variants, n_sessions=5000, alpha=0.8, arrival_rate=10.0, seed=0),
}
for name, ev in streams.items():
    print(f"{name}: sessions={len(set(e.session for e in ev))}  turn_events={len(ev)}")

sizes = np.load(paths.DATA_ROOT / "corpus" / "wiki-18-meta" / "sizes.npy")
page_ids = np.load(paths.DATA_ROOT / "corpus" / "wiki-18-meta" / "page_ids.npy")
page_sizes = np.zeros(page_ids.max() + 1, dtype=np.int64)
np.add.at(page_sizes, page_ids, sizes)
total = float(sizes.sum())
print(f"corpus: chunks={len(sizes):,} pages={len(page_sizes):,} bytes={int(total):,} "
      f"mean_chunk={sizes.mean():.1f}B mean_page={page_sizes.mean():.1f}B "
      f"chunks_per_page={len(sizes)/len(page_sizes):.2f}")

FRACS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
for sname, ev in streams.items():
    print(f"\n### stream={sname}")
    print("| capacity | policy | hit ratio | origin/demand | prefetch precision |")
    print("|---|---|---|---|---|")
    for frac in FRACS:
        cap = frac * total
        for pol in ("page_lru", "chunk_lru", "chunk_traj"):
            r = simulate(ev, pol, cap, sizes, page_ids=page_ids,
                         page_sizes=page_sizes, prefetch_budget=8)
            pp = f"{r['prefetch_precision']:.3f}" if r["prefetch_count"] else "--"
            print(f"| {cap/1e6:.2f} MB | {pol} | {r['hit_ratio']:.3f} "
                  f"| {r['origin_to_demand']:.2f} | {pp} |")
    print(f"demand_bytes={simulate(ev, 'chunk_lru', total, sizes)['demand_bytes']:,}")
