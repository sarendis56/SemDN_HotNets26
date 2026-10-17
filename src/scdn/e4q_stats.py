"""E4-Q paired significance: dense semantic index (e5) vs shallow lexical (BM25) and
search-rank floor (provided order), on the CRAG Task-3 50-page corpus (n=300). Reuses
e4_stats' McNemar / paired-bootstrap / Holm machinery, but as its OWN Holm family so it
does not perturb the locked F1-F9 multiplicity correction in e4_stats.

Labels from data/e4/judge_t3_{,bm25_,provided_}{7b,q3}.raw.jsonl (config = page@K).
Δ = A - B (A listed first). e5 page@5 = the SCDN deep-semantic representative (§5D.3
sweet spot); provided page@1 = read the top search result top-down (shallow floor).

Run:  uv run python -m scdn.e4q_stats
"""

import json

from scdn import paths
from scdn.e4_stats import CORRECT, aligned, holm, load_labels, mcnemar, truthful_test

E4 = paths.DATA_ROOT / "e4"

COMPARISONS = [
    ("Q1 e5 vs bm25  page@5  (32B)",   ("judge_t3_q3", "page@5"),   ("judge_t3_bm25_q3", "page@5"),   ["acc", "truthful"]),
    ("Q2 e5 vs bm25  page@50 (32B)",   ("judge_t3_q3", "page@50"),  ("judge_t3_bm25_q3", "page@50"),  ["acc", "truthful"]),
    ("Q3 e5 p@5 vs provided p@1 (32B)",("judge_t3_q3", "page@5"),   ("judge_t3_provided_q3", "page@1"), ["acc", "truthful"]),
    ("Q4 bm25 p@5 vs provided p@1(32B)",("judge_t3_bm25_q3", "page@5"), ("judge_t3_provided_q3", "page@1"), ["acc", "truthful"]),
    ("Q5 e5 vs bm25  page@5  (7B)",    ("judge_t3_7b", "page@5"),   ("judge_t3_bm25_7b", "page@5"),   ["acc", "truthful"]),
    ("Q6 e5 p@5 vs provided p@1 (7B)", ("judge_t3_7b", "page@5"),   ("judge_t3_provided_7b", "page@1"), ["acc", "truthful"]),
    # Task 4 — coherent chunker vs the incumbent structure chunker (both e5), matched depth
    ("C1 coherent vs structnt p@5 (32B)",  ("judge_t3_coherent_q3", "page@5"),  ("judge_t3_q3", "page@5"),  ["acc", "truthful"]),
    ("C2 coherent vs structnt p@50 (32B)", ("judge_t3_coherent_q3", "page@50"), ("judge_t3_q3", "page@50"), ["acc", "truthful"]),
    ("C3 coherent vs structnt p@5 (7B)",   ("judge_t3_coherent_7b", "page@5"),  ("judge_t3_7b", "page@5"),  ["acc", "truthful"]),
    ("C4 coherent vs structnt p@50 (7B)",  ("judge_t3_coherent_7b", "page@50"), ("judge_t3_7b", "page@50"), ["acc", "truthful", "halluc"]),
    # Task 3 — utility (LLM-answerability) reranking vs e5 similarity, SAME top-20 candidates
    ("U1 util vs e5 rerank (32B)", ("judge_t3_utility_q3", "util"), ("judge_t3_utility_q3", "e5"), ["acc", "truthful", "halluc"]),
    ("U2 util vs e5 rerank (7B)",  ("judge_t3_utility_7b", "util"), ("judge_t3_utility_7b", "e5"), ["acc", "truthful", "halluc"]),
    # Task 4 generalization — coherent chunker on the LIVE web (n=147), deep pool (chunk@all)
    ("L1 coherent vs structnt chunk@all (live 7B)",  ("judge_coherent_7b", "chunk@all"), ("judge_e5_structnt_llama70b", "chunk@all"), ["acc", "truthful", "halluc"]),
    ("L2 coherent vs structnt chunk@all (live 32B)", ("judge_coherent_q3", "chunk@all"), ("judge_q3_structnt", "chunk@all"),          ["acc", "truthful", "halluc"]),
    ("L3 coherent vs fixed chunk@all (live 7B)",     ("judge_coherent_7b", "chunk@all"), ("judge_e5_llama70b", "chunk@all"),          ["acc", "truthful", "halluc"]),
    # Q3 — does a stronger ranker beat e5 on QUALITY (coherent chunks, deep pool)?
    ("X1 ce vs e5  page@50 (coh 32B)", ("judge_t3_coherent_ce_q3", "page@50"), ("judge_t3_coherent_q3", "page@50"), ["acc", "truthful", "halluc"]),
    ("X2 ce vs e5  page@5  (coh 32B)", ("judge_t3_coherent_ce_q3", "page@5"),  ("judge_t3_coherent_q3", "page@5"),  ["acc", "truthful"]),
    ("X3 ce vs e5  page@50 (coh 7B)",  ("judge_t3_coherent_ce_7b", "page@50"), ("judge_t3_coherent_7b", "page@50"), ["acc", "truthful", "halluc"]),
    ("X4 qwen3 vs e5 page@5 (coh 32B)",("judge_t3_coherent_qwen3_q3","page@5"), ("judge_t3_coherent_q3", "page@5"), ["acc", "truthful"]),
    ("X5 qwen3rr vs e5 page@50 (coh 32B)", ("judge_t3_coherent_qwen3rr_q3", "page@50"), ("judge_t3_coherent_q3", "page@50"), ["acc", "truthful", "halluc"]),
    ("X6 qwen3rr vs ce page@50 (coh 32B)", ("judge_t3_coherent_qwen3rr_q3", "page@50"), ("judge_t3_coherent_ce_q3", "page@50"), ["acc", "truthful"]),
    ("X7 bm25 vs e5  page@50 (coh 32B)",    ("judge_t3_coherent_bm25_q3", "page@50"),   ("judge_t3_coherent_q3", "page@50"),    ["acc", "truthful"]),
    ("X8 bm25 vs e5  page@5  (coh 32B)",    ("judge_t3_coherent_bm25_q3", "page@5"),    ("judge_t3_coherent_q3", "page@5"),     ["acc", "truthful", "halluc"]),
    # F8a re-validation — Task-1 unified single-stage pipeline (coherent, no CE confound):
    # page@5 (all 5 pages, chunk-level) vs page@1 (top page only) = chunkret vs pageret@1
    ("T1 e5 page@5 vs page@1 (t1 7B)",  ("judge_t1_coherent_e5", "page@5"),    ("judge_t1_coherent_e5", "page@1"),    ["acc", "truthful"]),
    ("T1 e5 page@5 vs page@1 (t1 32B)", ("judge_t1_coherent_e5_q3", "page@5"), ("judge_t1_coherent_e5_q3", "page@1"), ["acc", "truthful"]),
    ("T1 qwen3rr page@5 vs page@1(32B)",("judge_t1_coherent_qwen3rr_q3","page@5"),("judge_t1_coherent_qwen3rr_q3","page@1"),["acc","truthful"]),
    # No-chunking baselines (CRAG Task-3, 32B, page@5) — do snippet/summary beat chunking + no-RAG?
    ("N1 snippet vs chunk(e5) (t3 32B)", ("judge_t3_snippet_q3", "page@5"), ("judge_t3_coherent_q3", "page@5"), ["acc", "truthful", "halluc"]),
    ("N2 snippet vs no-RAG (t3 32B)",    ("judge_t3_snippet_q3", "page@5"), ("judge_t3_norag_q3", "no_rag"),    ["acc", "truthful"]),
    ("N3 summary vs chunk(e5) (t3 32B)", ("judge_t3_summary_q3", "page@5"), ("judge_t3_coherent_q3", "page@5"), ["acc", "truthful", "halluc"]),
    ("N4 snippet vs chunk(e5) (t3 7B)",  ("judge_t3_snippet", "page@5"),    ("judge_t3_coherent_7b", "page@5"), ["acc", "truthful"]),
    # Best chunking (coherent + RERANKER over the deep pool) vs the best snippet — does
    # chunking, done well, beat just reading snippets? (CRAG Task-3, 32B)
    ("M1 chunk+CE@50 vs snippet@10 (32B)",      ("judge_t3_coherent_ce_q3", "page@50"),      ("judge_t3_snippet_q3", "page@10"), ["acc", "truthful"]),
    ("M2 chunk+qwen3rr@50 vs snippet@10 (32B)", ("judge_t3_coherent_qwen3rr_q3", "page@50"), ("judge_t3_snippet_q3", "page@10"), ["acc", "truthful"]),
    ("M3 chunk+CE@50 vs snippet@5 (32B)",       ("judge_t3_coherent_ce_q3", "page@50"),      ("judge_t3_snippet_q3", "page@5"),  ["acc", "truthful"]),
    # Experiment (a) — does adding Cloudflare-recovered (network/JS-blocked) pages help? (live, n=25)
    ("CF1 aug vs orig chunk@all (32B)", ("judge_live_cf_aug_q3", "chunk@all"), ("judge_live_cf_orig_q3", "chunk@all"), ["acc", "truthful"]),
    ("CF2 aug vs orig page@5 (32B)",    ("judge_live_cf_aug_q3", "page@5"),    ("judge_live_cf_orig_q3", "page@5"),    ["acc", "truthful"]),
    ("CF3 aug vs orig chunk@all (7B)",  ("judge_live_cf_aug_7b", "chunk@all"), ("judge_live_cf_orig_7b", "chunk@all"), ["acc", "truthful"]),
    # P — full-page baseline (CRAG t3, 32B): does reranker chunking beat reading whole pages?
    ("P1 chunk+CE@50 vs fullpage@3 (t3 32B)", ("judge_t3_coherent_ce_q3", "page@50"), ("judge_pool_crag_t3_fullpage_q3", "page@3"), ["acc", "truthful"]),
    ("P2 fullpage@3 vs snippet@5 (t3 32B)",   ("judge_pool_crag_t3_fullpage_q3", "page@3"), ("judge_t3_snippet_q3", "page@5"), ["acc", "truthful"]),
    ("P3 fullpage@3 vs chunk-e5@5 (t3 32B)",  ("judge_pool_crag_t3_fullpage_q3", "page@3"), ("judge_t3_coherent_q3", "page@5"), ["acc", "truthful"]),
    # V — live-web no-chunking (32B): chunking vs snippet / full-page on the live setting.
    # V1/V2 use the coherent chunker (worst on live); V3/V4 use the BEST live chunker
    # (structnt + e5, per §C's L-family) so the comparison is fair to chunking.
    ("V1 chunk-coh-qwen3rr@all vs snippet@5 (live 32B)",  ("judge_live_coherent_qwen3rr_q3", "chunk@all"), ("judge_pool_live_snippet_q3", "page@5"), ["acc", "truthful"]),
    ("V2 chunk-coh-qwen3rr@all vs fullpage@3 (live 32B)", ("judge_live_coherent_qwen3rr_q3", "chunk@all"), ("judge_pool_live_fullpage_q3", "page@3"), ["acc", "truthful"]),
    ("V3 chunk-struct-e5@all vs snippet@5 (live 32B)",  ("judge_q3_structnt", "chunk@all"), ("judge_pool_live_snippet_q3", "page@5"), ["acc", "truthful"]),
    ("V4 chunk-struct-e5@all vs fullpage@3 (live 32B)", ("judge_q3_structnt", "chunk@all"), ("judge_pool_live_fullpage_q3", "page@3"), ["acc", "truthful"]),
    # V5-V7 — snippet+chunk HYBRID (snippet injected as a ranking co-candidate, e5 rerank)
    ("V5 hybrid@all vs snippet@5 (live 32B)",        ("judge_live_hybrid_q3", "chunk@all"), ("judge_pool_live_snippet_q3", "page@5"), ["acc", "truthful"]),
    ("V6 hybrid@all vs chunk-struct-e5@all (live 32B)", ("judge_live_hybrid_q3", "chunk@all"), ("judge_q3_structnt", "chunk@all"), ["acc", "truthful"]),
    ("V7 hybrid@all vs fullpage@3 (live 32B)",       ("judge_live_hybrid_q3", "chunk@all"), ("judge_pool_live_fullpage_q3", "page@3"), ["acc", "truthful"]),
    # V8-V9 — MATCHED k=3 (snippet@3 vs the top-3 chunk/hybrid arms): controls for the block-count
    # confound (snippet@5 fed 5 blocks vs the chunk arms' 3).
    ("V8 snippet@3 vs chunk-struct-e5@all (live 32B)", ("judge_pool_live_snippet_q3", "page@3"), ("judge_q3_structnt", "chunk@all"), ["acc", "truthful"]),
    ("V9 snippet@3 vs hybrid@all (live 32B)",          ("judge_pool_live_snippet_q3", "page@3"), ("judge_live_hybrid_q3", "chunk@all"), ["acc", "truthful"]),
    # B — larger-n (600) CRAG t3b: lock chunk(reranker) vs snippet / full-page, and RAG vs no-RAG
    ("B1 chunk-CE@50 vs snippet@5 (t3b 32B)",  ("judge_t3b_ce_q3", "page@50"), ("judge_t3b_snippet_q3", "page@5"), ["acc", "truthful"]),
    ("B2 chunk-CE@50 vs fullpage@3 (t3b 32B)", ("judge_t3b_ce_q3", "page@50"), ("judge_t3b_fullpage_q3", "page@3"), ["acc", "truthful"]),
    ("B3 snippet@5 vs no-RAG (t3b 32B)",       ("judge_t3b_snippet_q3", "page@5"), ("judge_t3b_norag_q3", "no_rag"), ["acc", "truthful"]),
    # B4-B7 — does the snippet+chunk HYBRID beat the single-granularity arms at n=600? (e5 rerank)
    ("B4 hybrid@5 vs chunk-e5@5 (t3b 32B)",   ("judge_t3b_hybrid_q3", "page@5"), ("judge_t3b_e5_q3", "page@5"), ["acc", "truthful"]),
    ("B5 hybrid@5 vs snippet@5 (t3b 32B)",    ("judge_t3b_hybrid_q3", "page@5"), ("judge_t3b_snippet_q3", "page@5"), ["acc", "truthful"]),
    ("B6 hybrid@5 vs chunk-CE@5 (t3b 32B)",   ("judge_t3b_hybrid_q3", "page@5"), ("judge_t3b_ce_q3", "page@5"), ["acc", "truthful"]),
    ("B7 hybrid@5 vs fullpage@3 (t3b 32B)",   ("judge_t3b_hybrid_q3", "page@5"), ("judge_t3b_fullpage_q3", "page@3"), ["acc", "truthful"]),
]


def main():
    cache = {}

    def L(run, cfg):
        if run not in cache:
            path = E4 / f"{run}.raw.jsonl"
            cache[run] = load_labels(run) if path.exists() else None
        if cache[run] is None:
            raise FileNotFoundError(run)
        return cache[run][cfg]

    def family(name):  # group by the comparison-id prefix (Q/X/L/U/T/M/N/CF)
        return "".join(c for c in name.split()[0] if c.isalpha())

    rows = []  # [family, name, metric, n, test, desc, p]
    for name, (rA, cA), (rB, cB), metrics in COMPARISONS:
        try:
            la, lb, n = aligned(L(rA, cA), L(rB, cB))
        except (FileNotFoundError, KeyError) as e:
            print(f"{name:34s} [skipped — labels not ready: {type(e).__name__} {e}]")
            continue
        for m in metrics:
            if m == "acc":
                r = mcnemar(la, lb, lambda x: int(x in CORRECT))
                desc = f"Δacc={r['delta']:+.3f}  b/c={r['b']}/{r['c']}"; p = r["p"]; test = "McNemar"
            elif m == "halluc":
                r = mcnemar(la, lb, lambda x: int(x == "incorrect"))
                desc = f"Δhal={r['delta']:+.3f}  b/c={r['b']}/{r['c']}"; p = r["p"]; test = "McNemar"
            else:
                r = truthful_test(la, lb)
                desc = f"Δtru={r['delta']:+.3f}  CI[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]"; p = r["p"]; test = "bootstrap"
            rows.append([family(name), name, m, n, test, desc, p])

    # Holm WITHIN each pre-registered family, not across the whole sprawl — one giant family
    # over-corrects (a Bonferroni penalty for unrelated tests). Each finding is corrected only
    # against its own family of tests.
    FAM = {"Q": "shallow-vs-deep (§B)", "C": "coherent-chunker (§C)", "X": "encoders/rerankers (§E)", "L": "coherent-live (§C)",
           "U": "utility-ranking (§D)", "T": "task1-unified F8a (§F)", "M": "reranker-vs-snippet (§G)",
           "N": "no-chunking baselines (§G)", "CF": "cloudflare-recovery (a)"}
    by_fam = {}
    for i, row in enumerate(rows):
        by_fam.setdefault(row[0], []).append(i)
    adj = [1.0] * len(rows)
    for idxs in by_fam.values():
        for i, pa in zip(idxs, holm([rows[i][6] for i in idxs])):
            adj[i] = pa

    print(f"{'comparison':34s} {'metric':8s} {'n':>4} {'test':9s} {'detail':40s} {'p':>7} {'p_holm':>7}  verdict")
    out = []
    for fam, idxs in by_fam.items():
        print(f"\n--- family {fam}: {FAM.get(fam, fam)}  (Holm over {len(idxs)} tests) ---")
        for i in idxs:
            _, name, m, n, test, desc, p = rows[i]
            pa = adj[i]
            verdict = "LOCKED" if pa < 0.05 else ("trend" if p < 0.05 else "null")
            print(f"{name:34s} {m:8s} {n:>4} {test:9s} {desc:40s} {p:7.4f} {pa:7.4f}  {verdict}")
            out.append({"family": fam, "comparison": name, "metric": m, "n": n, "p_raw": p,
                        "p_holm": pa, "verdict": verdict, "detail": desc})
    json.dump(out, open(E4 / "significance_e4q.json", "w"), indent=1)
    print(f"\nwrote {E4/'significance_e4q.json'}  (Holm applied per-family across {len(by_fam)} families)")


if __name__ == "__main__":
    main()
