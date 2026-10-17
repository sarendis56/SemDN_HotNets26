"""E2 analysis: locality characterization + cache simulation -> figure + CSV.

Run:  uv run python -m semdn.analyze --traces data/traces/full.jsonl
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from semdn import paths
from semdn.cachesim import simulate
from semdn.stream import (
    chunk_access_seq,
    deep_rank_reachability,
    load_trace_variants,
    load_traces,
    reuse_distances,
    shuffled_events,
    synthesize_stream,
    within_task_overlap,
)

OUT = paths.REPO_ROOT / "outputs"


def duplicate_query_rate(traces: list[dict]) -> dict:
    """Turns whose query text exactly repeats an earlier turn of the same task."""
    dup = total = 0
    for t in traces:
        seen: set = set()
        for turn in t["turns"]:
            q = turn["query"].strip().lower()
            if turn["turn"] > 1:
                total += 1
                dup += q in seen
            seen.add(q)
    return {"dup_rate_after_turn1": dup / total if total else 0.0}


def dedup_repeat_queries(traces: list[dict]) -> list[dict]:
    """Drop turns that repeat an earlier query of the same task (conservative
    variant: locality not driven by verbatim re-asking)."""
    out = []
    for t in traces:
        seen: set = set()
        turns = []
        for turn in t["turns"]:
            q = turn["query"].strip().lower()
            if q not in seen:
                seen.add(q)
                turns.append(turn)
        if turns:
            out.append({**t, "turns": turns})
    return out


def reuse_cdf(events) -> tuple[np.ndarray, np.ndarray, float]:
    d = reuse_distances(chunk_access_seq(events))
    finite = np.sort(d[np.isfinite(d)])
    cold_frac = float(np.mean(~np.isfinite(d)))
    y = (np.arange(1, len(finite) + 1)) / len(d)  # CDF over ALL accesses
    return finite, y, cold_frac


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--traces",
        type=Path,
        nargs="+",
        default=[paths.TRACES / "full.jsonl"],
        help="one or more trace files; multiple files = sampling variants of the same questions",
    )
    ap.add_argument("--k-shown", type=int, default=3)
    ap.add_argument("--zipf-alpha", type=float, default=0.8)
    ap.add_argument("--n-sessions", type=int, default=5000)
    ap.add_argument("--arrival-rate", type=float, default=10.0)
    ap.add_argument("--prefetch-budget", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-shuffles", type=int, default=50,
                    help="number of shuffled controls for the reuse-interval robustness band")
    ap.add_argument("--meta-dir", type=Path, default=paths.DATA_ROOT / "corpus" / "wiki-18-meta",
                    help="dir holding sizes.npy + page_ids.npy (E2-CRAG: data/e2_crag)")
    ap.add_argument("--out-tag", default="e2", help="prefix for output files (use 'e2_crag' for CRAG)")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    # pooled traces for per-task metrics; variant structure for stream synthesis
    traces = [t for p in args.traces for t in load_traces(p, k_shown=args.k_shown)]
    variants = load_trace_variants(args.traces, k_shown=args.k_shown)
    stats: dict = {
        "trace_files": [str(p) for p in args.traces],
        "n_traces": len(traces),
        "n_tasks": len(variants),
    }
    stats |= duplicate_query_rate(traces)
    stats["within_task"] = within_task_overlap(traces)
    traces_dedup = dedup_repeat_queries(traces)
    stats["within_task_dedup"] = within_task_overlap(traces_dedup)
    stats["deep_rank_reachability"] = deep_rank_reachability(traces_dedup, [5, 10, 20, 47])

    sizes = np.load(args.meta_dir / "sizes.npy")
    page_ids = np.load(args.meta_dir / "page_ids.npy")
    page_sizes = np.zeros(page_ids.max() + 1, dtype=np.int64)
    np.add.at(page_sizes, page_ids, sizes)
    total_bytes = int(sizes.sum())

    streams = {
        "nodup": synthesize_stream(
            variants, n_sessions=None, arrival_rate=args.arrival_rate, seed=args.seed
        ),
        f"zipf{args.zipf_alpha}": synthesize_stream(
            variants,
            n_sessions=args.n_sessions,
            alpha=args.zipf_alpha,
            arrival_rate=args.arrival_rate,
            seed=args.seed,
        ),
    }

    # --- panel (a): reuse-interval CDF, distinct-task stream only ---
    cdf_real = reuse_cdf(streams["nodup"])
    shuffle_cdfs = [reuse_cdf(shuffled_events(streams["nodup"], seed=args.seed + i))
                    for i in range(args.n_shuffles)]
    stats["cold_frac/nodup"] = cdf_real[2]
    stats[f"cold_frac/zipf{args.zipf_alpha}"] = reuse_cdf(streams[f"zipf{args.zipf_alpha}"])[2]

    # --- panel (b): cache simulation ---
    fracs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    rows = []
    for sname, ev in streams.items():
        for frac in fracs:
            cap = frac * total_bytes
            for policy in ("page_lru", "chunk_lru", "chunk_traj"):
                r = simulate(
                    ev,
                    policy,
                    cap,
                    sizes,
                    page_ids=page_ids,
                    page_sizes=page_sizes,
                    prefetch_budget=args.prefetch_budget,
                )
                r["stream"] = sname
                r["capacity_frac"] = frac
                rows.append(r)
                print(
                    f"{sname:9s} {policy:10s} cap={frac:.0e} "
                    f"hit={r['hit_ratio']:.3f} origin/demand={r['origin_to_demand']:.2f}",
                    flush=True,
                )
    # preserve the original wiki-18 output names (out_tag default "e2"); prefix for CRAG
    tag = args.out_tag
    csv_name = "cachesim.csv" if tag == "e2" else f"{tag}_cachesim.csv"
    json_name = "locality_stats.json" if tag == "e2" else f"{tag}_locality_stats.json"
    png_name = "e2_locality.png" if tag == "e2" else f"{tag}_locality.png"
    with open(OUT / csv_name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUT / json_name).write_text(json.dumps(stats, indent=2))

    # --- figure ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    colors = {"nodup": "tab:blue", f"zipf{args.zipf_alpha}": "tab:red"}
    x, y, _ = cdf_real
    ax1.plot(x, y, color="tab:blue", label="real")
    grid = np.unique(np.concatenate([sx for sx, _, _ in shuffle_cdfs]))
    ys = np.vstack([np.interp(grid, sx, sy, left=0.0, right=sy[-1])
                    for sx, sy, _ in shuffle_cdfs])
    lo, med, hi = np.percentile(ys, [5, 50, 95], axis=0)
    ax1.fill_between(grid, lo, hi, color="tab:orange", alpha=0.16, linewidth=0)
    ax1.plot(grid, med, color="tab:orange", linestyle=":", label="shuffled median")
    ax1.set_xscale("log")
    ax1.set_xlabel("reuse interval (distinct chunks)")
    ax1.set_ylabel("CDF")
    ax1.legend(fontsize=7)
    ax1.set_title("(a) Chunk reuse interval")

    style = {"page_lru": ":", "chunk_lru": "-", "chunk_traj": "--"}
    for sname in streams:
        for policy in ("page_lru", "chunk_lru", "chunk_traj"):
            sub = [r for r in rows if r["stream"] == sname and r["policy"] == policy]
            ax2.plot(
                [r["capacity_frac"] for r in sub],
                [r["hit_ratio"] for r in sub],
                style[policy],
                color=colors[sname],
                marker="o",
                ms=3,
                label=f"{sname}/{policy}",
            )
    ax2.set_xscale("log")
    ax2.set_xlabel("cache capacity (fraction of corpus bytes)")
    ax2.set_ylabel("chunk hit ratio")
    ax2.legend(fontsize=6)
    ax2.set_title("(b) Edge cache hit ratio")
    fig.tight_layout()
    fig.savefig(OUT / png_name, dpi=180)
    print(f"wrote {OUT}/{png_name}, {csv_name}, {json_name}")
    print(json.dumps({k: v for k, v in stats.items() if not isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
