"""Paired significance for the fetcher comparison (trafilatura GET vs Cloudflare cloud
clean-fetch), live web, Qwen3-32B reader, Llama-3.1-70B CRAG judge. Reuses e4_stats'
McNemar / paired-bootstrap / Holm, as its own per-family Holm groups. Δ = A - B.

Labels: data/e4/judge_cf_*_q3.raw.jsonl (config = page@K / chunk@all / no_rag).

Run:  uv run python -m semdn.e4_cf_stats
"""
import json

from semdn import paths
from semdn.e4_stats import CORRECT, aligned, holm, load_labels, mcnemar, truthful_test

E4 = paths.DATA_ROOT / "e4"

# (family-prefixed name, (runA,cfgA), (runB,cfgB), metrics)
COMPARISONS = [
    # A — does the cloud clean-fetch help CHUNKING vs our trafilatura GET? (same chunker)
    ("A1 cf vs traf  structnt chunk@all", ("judge_cf_cf_structnt_q3", "chunk@all"), ("judge_cf_traf_structnt_q3", "chunk@all"), ["acc", "truthful", "halluc"]),
    ("A2 cf vs traf  coherent chunk@all", ("judge_cf_cf_coherent_q3", "chunk@all"), ("judge_cf_traf_coherent_q3", "chunk@all"), ["acc", "truthful", "halluc"]),
    ("A3 cf vs traf  structnt page@5",    ("judge_cf_cf_structnt_q3", "page@5"),    ("judge_cf_traf_structnt_q3", "page@5"),    ["acc", "truthful"]),
    ("A4 cf vs traf  coherent page@5",    ("judge_cf_cf_coherent_q3", "page@5"),    ("judge_cf_traf_coherent_q3", "page@5"),    ["acc", "truthful"]),

    # B — on the CF (clean+complete) pool, does CHUNK beat/tie SNIPPET? (the paper question)
    ("B1 cf-chunk-structnt@all vs snippet@5", ("judge_cf_cf_structnt_q3", "chunk@all"), ("judge_cf_snippet_q3", "page@5"), ["acc", "truthful"]),
    ("B2 cf-chunk-coherent@all vs snippet@5", ("judge_cf_cf_coherent_q3", "chunk@all"), ("judge_cf_snippet_q3", "page@5"), ["acc", "truthful"]),
    ("B3 cf-hybrid@all vs snippet@5",         ("judge_cf_cf_hybrid_q3", "chunk@all"),   ("judge_cf_snippet_q3", "page@5"), ["acc", "truthful"]),
    # the OLD muddy result re-derived on these n: traf chunk vs snippet
    ("B4 traf-chunk-structnt@all vs snippet@5", ("judge_cf_traf_structnt_q3", "chunk@all"), ("judge_cf_snippet_q3", "page@5"), ["acc", "truthful"]),
    # matched-k control (snippet@3 vs the 3-block chunk arms)
    ("B5 cf-chunk-structnt@all vs snippet@3", ("judge_cf_cf_structnt_q3", "chunk@all"), ("judge_cf_snippet_q3", "page@3"), ["acc", "truthful"]),
    ("B6 cf-hybrid@all vs snippet@3",         ("judge_cf_cf_hybrid_q3", "chunk@all"),   ("judge_cf_snippet_q3", "page@3"), ["acc", "truthful"]),

    # C — granularity on the CF pool: chunk vs whole clean page
    ("C1 cf-chunk-structnt@all vs cf-fullpage@3", ("judge_cf_cf_structnt_q3", "chunk@all"), ("judge_cf_fullpage_cf_q3", "page@3"), ["acc", "truthful"]),
    ("C2 cf-fullpage@3 vs snippet@5",             ("judge_cf_fullpage_cf_q3", "page@3"),    ("judge_cf_snippet_q3", "page@5"),     ["acc", "truthful"]),

    # D — RAG vs no-RAG sanity on this set
    ("D1 snippet@5 vs no-RAG",            ("judge_cf_snippet_q3", "page@5"),        ("judge_cf_norag_q3", "no_rag"), ["acc", "truthful"]),
    ("D2 cf-chunk-structnt@all vs no-RAG",("judge_cf_cf_structnt_q3", "chunk@all"), ("judge_cf_norag_q3", "no_rag"), ["acc", "truthful"]),
]


def family(name):
    return "".join(c for c in name.split()[0] if c.isalpha())


def main():
    cache = {}

    def L(run, cfg):
        if run not in cache:
            path = E4 / f"{run}.raw.jsonl"
            cache[run] = load_labels(run) if path.exists() else None
        if cache[run] is None:
            raise FileNotFoundError(run)
        return cache[run][cfg]

    rows = []
    for name, (rA, cA), (rB, cB), metrics in COMPARISONS:
        try:
            la, lb, n = aligned(L(rA, cA), L(rB, cB))
        except (FileNotFoundError, KeyError) as e:
            print(f"{name:38s} [skipped — {type(e).__name__}: {e}]")
            continue
        for m in metrics:
            if m == "acc":
                r = mcnemar(la, lb, lambda x: int(x in CORRECT))
                desc = f"Δacc={r['delta']:+.3f} b/c={r['b']}/{r['c']}"; p = r["p"]; test = "McNemar"
            elif m == "halluc":
                r = mcnemar(la, lb, lambda x: int(x == "incorrect"))
                desc = f"Δhal={r['delta']:+.3f} b/c={r['b']}/{r['c']}"; p = r["p"]; test = "McNemar"
            else:
                r = truthful_test(la, lb)
                desc = f"Δtru={r['delta']:+.3f} CI[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]"; p = r["p"]; test = "boot"
            rows.append([family(name), name, m, n, test, desc, p])

    by_fam = {}
    for i, row in enumerate(rows):
        by_fam.setdefault(row[0], []).append(i)
    adj = [1.0] * len(rows)
    for idxs in by_fam.values():
        for i, pa in zip(idxs, holm([rows[i][6] for i in idxs])):
            adj[i] = pa

    print(f"\n{'comparison':38s} {'metric':8s} {'n':>4} {'test':8s} {'detail':34s} {'p':>7} {'pHolm':>7}  verdict")
    out = []
    for fam, idxs in by_fam.items():
        print(f"\n--- family {fam}  (Holm over {len(idxs)} tests) ---")
        for i in idxs:
            _, name, m, n, test, desc, p = rows[i]
            pa = adj[i]
            verdict = "LOCKED" if pa < 0.05 else ("trend" if p < 0.05 else "null")
            print(f"{name:38s} {m:8s} {n:>4} {test:8s} {desc:34s} {p:7.4f} {pa:7.4f}  {verdict}")
            out.append({"family": fam, "comparison": name, "metric": m, "n": n,
                        "p_raw": p, "p_holm": pa, "verdict": verdict, "detail": desc})
    json.dump(out, open(E4 / "significance_cf.json", "w"), indent=1)
    print(f"\nwrote {E4/'significance_cf.json'}")


if __name__ == "__main__":
    main()
