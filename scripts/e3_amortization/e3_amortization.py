"""E3 — redundant work across operators (O4, claim iv).

Today each operator runs its own fetch->clean->chunk->embed pipeline, so popular
content is processed once per operator. A shared SCDN processes each unique page
once (fetch/clean/chunk) and embeds it once per *distinct encoder*. We quantify
the amortization with measured inputs:

  (1) REAL cross-tenant overlap: the 4 QA workloads (hotpotqa/2wiki/musique/
      bamboogle) as 4 heterogeneous tenants — how many wiki pages do they share?
  (2) REAL embed throughput: e5-base-v2 chunks/GPU-second on this A100.
  (3) per-page fetch bytes from E1 (lightweight no-JS baseline median).

Then a multi-tenant model (Zipf task popularity, the same skew as E2) gives the
origin-fetch and embedding-compute reduction vs number of tenants T, for a shared
canonical encoder vs bring-your-own (BYO) per-tenant encoders.

Local-only. Writes paper/figures/e3_amortization.pdf + outputs/e3_stats.json.
"""

import collections
import json
import time

import numpy as np

from scdn import paths
from scdn.corpus import Corpus

# E1 median HTML doc is ~760 KB/query over 5 pages -> ~152 KB/page (lightweight fetch)
E1_PAGE_BYTES = 760_616 // 5

corpus = Corpus()
chunks_per_page = np.bincount(corpus.page_ids).astype(np.int64)

# --- load tasks: each task -> set of wiki pages it touches (via served chunks) ---
tasks = []  # (dataset, frozenset(page_ids))
for line in open(paths.TRACES / "full-t07-s0.jsonl"):
    t = json.loads(line)
    chunkset = {x for u in t["turns"] for x in u["ids"][:3]}
    pages = {int(corpus.page_ids[c]) for c in chunkset}
    if pages:
        tasks.append((t["dataset"], frozenset(pages)))
print(f"{len(tasks)} tasks with page sets", flush=True)

# --- (1) real cross-tenant (cross-dataset) overlap ---
by_ds = collections.defaultdict(set)
for ds, pg in tasks:
    by_ds[ds] |= pg
total_pages = sum(len(v) for v in by_ds.values())
union_pages = len(set().union(*by_ds.values()))
pc = collections.Counter()
for v in by_ds.values():
    pc.update(v)
shared2 = sum(1 for c in pc.values() if c >= 2)
overlap = {
    "datasets": {k: len(v) for k, v in by_ds.items()},
    "sum_over_datasets": total_pages,
    "unique_union": union_pages,
    "redundancy_factor_4tenants": total_pages / union_pages,
    "pages_shared_by_2plus": shared2,
    "frac_pages_shared": shared2 / union_pages,
}
print("cross-dataset overlap:", json.dumps(overlap, indent=2), flush=True)

# --- (2) measured embed throughput (e5 on GPU) ---
from scdn.retrieval_server import E5Encoder

enc = E5Encoder(paths.E5_BASE_V2, device="cuda")
sample = [corpus.contents(int(c)) for c in range(2000)]
enc.encode_passages(sample[:128])  # warmup
t0 = time.time()
enc.encode_passages(sample)
thr = len(sample) / (time.time() - t0)  # chunks/sec
print(f"e5 embed throughput: {thr:.0f} chunks/GPU-s", flush=True)

# --- (3) multi-tenant amortization model (Zipf popularity, like E2) ---
rng = np.random.default_rng(0)


def model(T, alpha, m=150, reps=8):
    fa, ea_shared = [], []
    for _ in range(reps):
        ranks = rng.permutation(len(tasks)) + 1
        w = (ranks.astype(float) ** -alpha)
        w /= w.sum()
        per_op_pages = per_op_chunks = 0
        union = set()
        for _t in range(T):
            draw = rng.choice(len(tasks), size=m, replace=True, p=w)
            tp = set().union(*(tasks[d][1] for d in draw))
            per_op_pages += len(tp)
            per_op_chunks += int(chunks_per_page[list(tp)].sum())
            union |= tp
        u = np.fromiter(union, dtype=np.int64)
        union_chunks = int(chunks_per_page[u].sum())
        fa.append(per_op_pages / len(union))
        ea_shared.append(per_op_chunks / union_chunks)
    return float(np.mean(fa)), float(np.mean(ea_shared))


Ts = [1, 2, 3, 5, 8, 13, 21, 34, 50]
curves = {a: {"fetch": [], "embed_shared": []} for a in (0.6, 0.8, 1.0)}
for a in curves:
    for T in Ts:
        fa, ea = model(T, a)
        curves[a]["fetch"].append(fa)
        curves[a]["embed_shared"].append(ea)
    print(f"alpha={a}: fetch_amort@T=50={curves[a]['fetch'][-1]:.1f}x "
          f"embed_shared@T=50={curves[a]['embed_shared'][-1]:.1f}x", flush=True)

# headline @ T=21, alpha=0.8
fa21, ea21 = model(21, 0.8)
per_op_pages_est = None
stats = {
    "cross_dataset_overlap": overlap,
    "embed_throughput_chunks_per_gpu_s": thr,
    "per_page_fetch_bytes_e1": E1_PAGE_BYTES,
    "headline_T21_alpha0.8": {
        "fetch_amortization_x": fa21,
        "embed_amortization_shared_encoder_x": ea21,
        "embed_amortization_BYO_encoder_x": 1.0,
    },
    "Ts": Ts,
    "curves": curves,
}
(paths.REPO_ROOT / "outputs").mkdir(exist_ok=True)
(paths.REPO_ROOT / "outputs" / "e3_stats.json").write_text(json.dumps(stats, indent=2))

# --- figure ---
import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update({
    "pdf.fonttype": 42, "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
    "legend.fontsize": 6.8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.4, "xtick.direction": "in",
    "ytick.direction": "in", "legend.frameon": False, "savefig.bbox": "tight",
})
import matplotlib.pyplot as plt
import os as _os
import sys as _sys

_sys.path.insert(0, str(paths.REPO_ROOT / "figures"))   # shared figure style
try:
    from _style import legend_below, save as _save, style
except ImportError:   # figures/ not present: the numbers above are already written
    _sys.exit("figures/_style.py not found; skipping the figure")

style()

fig, ax = plt.subplots(figsize=(3.35, 2.4))
ax.plot(Ts, curves[0.8]["fetch"], "-o", color="#0072B2", ms=3, mew=0,
        label="fetch+chunk, or embed w/ shared encoder")
for a, ls in ((0.6, ":"), (1.0, "--")):
    ax.plot(Ts, curves[a]["fetch"], ls, color="#0072B2", lw=0.9, alpha=0.6,
            label=f"  (α={a})")
ax.axhline(1.0, color="#D55E00", lw=1.2, label="embed, BYO per-tenant encoders")
ax.set_xlabel("number of tenants (operators)")
ax.set_ylabel("work reduction vs per-operator (×)")
legend_below(fig, ncol=2, y=-0.04)
_save(fig, "e3_amortization")
print("wrote e3_amortization.pdf")
