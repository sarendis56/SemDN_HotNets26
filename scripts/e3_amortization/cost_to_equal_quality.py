"""E1-D — cost-to-equal-quality: the baseline pays fetch+chunk+embed per query
per client (no sharing); the SemDN delivers chunks and amortizes acquisition at
E2's measured cache hit rate. Fuses E1 (per-page bytes), E2 (hit rate), E3
(embed throughput). Analytical, no GPU.
"""

import json

import numpy as np

from semdn import paths

# E1 measured (decompressed content bytes, lightweight no-JS baseline; per query, k=5)
CLIENT_LINK_HTML = 760_616        # bytes the client pulls (5 pages of HTML)
CLIENT_LINK_RENDER = 5_752_861    # bytes if it must full-render (transfer)
SEMDN_LINK = 7_138                 # bytes the SemDN returns (top-5 chunks)
EMBED_THR = 392                   # chunks/GPU-s (E3 measured)

# E2 measured cache hit rate (zipf0.8, chunk_lru) from cachesim.csv
rows = [r.split(",") for r in open(paths.REPO_ROOT / "outputs" / "cachesim.csv")]
hdr = rows[0]
hit_by_cap = []
for r in rows[1:]:
    d = dict(zip(hdr, r))
    if d["stream"] == "zipf0.8" and d["policy"] == "chunk_lru":
        hit_by_cap.append((float(d["capacity_bytes"]) / 1e6, float(d["hit_ratio"])))
hit_by_cap.sort()

print("=== Communication (client link), per query ===")
print(f"  client pulls pages : {CLIENT_LINK_HTML/1e3:.0f} KB (lightweight) / "
      f"{CLIENT_LINK_RENDER/1e6:.2f} MB (full render)")
print(f"  SemDN returns chunks: {SEMDN_LINK/1e3:.1f} KB")
print(f"  reduction          : {CLIENT_LINK_HTML/SEMDN_LINK:.0f}x (lightweight) / "
      f"{CLIENT_LINK_RENDER/SEMDN_LINK:.0f}x (render)  -- unconditional")

print("\n=== Acquisition amortization vs edge cache size (from E2 hit rate) ===")
for cap, h in hit_by_cap:
    if cap in (1.343676, 13.436760808) or abs(cap - 1.34) < 0.01 or abs(cap - 13.4) < 0.1:
        pass
print("  cap(MB)  hit    origin-fetch & embed amortization = 1/(1-hit)")
for cap, h in hit_by_cap:
    print(f"  {cap:8.2f}  {h:.3f}   {1/(1-h):.1f}x")

# headline at two operating points
def amort(capmb):
    for cap, h in hit_by_cap:
        if abs(cap - capmb) / capmb < 0.05:
            return h, 1 / (1 - h)
    return None
print("\n=== Headline ===")
for capmb in (1.34, 13.44):
    h, a = amort(capmb)
    print(f"  at {capmb} MB edge cache: hit={h:.3f} -> SemDN does {a:.1f}x fewer origin "
          f"fetches AND (shared encoder) {a:.1f}x less embedding than per-query-per-client")
print(f"  embedding absolute: at {EMBED_THR} chunks/GPU-s, the saved fraction is "
      f"(hit) x (chunks embedded); only shareable across a common encoder (E3).")

# figure
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams.update({"pdf.fonttype": 42, "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": 0.6, "xtick.direction": "in",
    "ytick.direction": "in", "legend.frameon": False, "savefig.bbox": "tight"})
import matplotlib.pyplot as plt
import os as _os
import sys as _sys

_sys.path.insert(0, str(paths.REPO_ROOT / "figures"))   # shared figure style
try:
    from _style import legend_below, save as _save, style
except ImportError:   # figures/ not present: the numbers above are already written
    _sys.exit("figures/_style.py not found; skipping the figure")

style()

caps = [c for c, _ in hit_by_cap]
amorts = [1 / (1 - h) for _, h in hit_by_cap]
fig, ax = plt.subplots(figsize=(3.35, 2.3))
ax.plot(caps, amorts, "-o", color="#0072B2", ms=3, mew=0)
ax.set_xscale("log")
ax.set_xlabel("edge cache size (MB)")
ax.set_ylabel("acquisition amortization (×)\nfetch, and embed w/ shared encoder")
ax.axhline(1, color="0.6", lw=0.6)
_save(fig, "e1_cost")
print("\nwrote e1_cost.pdf")
