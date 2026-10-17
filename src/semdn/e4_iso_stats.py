"""Paired significance for the iso-budget sweep: at each fixed token budget, does delivering
relevance-selected CHUNKS (e5 or cross-encoder ranked) beat delivering whole PAGES (realistic
search-order and fair relevance-order) and the full SNIPPET pool? Δ = first - second.
Reuses e4_stats machinery; per-family Holm over budgets.

Labels: data/e4/judge_iso_<tag><arm>_q3.raw.jsonl (config = b128..b6144).

Run:  uv run python -m semdn.e4_iso_stats --tag t3b_            (or --tag t3b_,t3b_xl_ to merge)
"""
import argparse
import json

import numpy as np

from semdn import paths
from semdn.e4_stats import CORRECT, SCORE, aligned, holm, load_labels, mcnemar, truthful_test

E4 = paths.DATA_ROOT / "e4"


def trio(labs):
    return (float(np.mean([SCORE[x] for x in labs])),
            float(np.mean([x in CORRECT for x in labs])),
            float(np.mean([x == "incorrect" for x in labs])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="t3b_")
    args = ap.parse_args()

    CORE = ["chunk", "page_rel", "page_seq"]
    OPT = ["chunk_ce", "snippet_full", "chunk_llm", "page_llm", "page_oracle"]
    tags = args.tag.split(",")   # "t3b_,t3b_xl_" merges the long-context run into one table
    cache = {}

    def L(a):
        if a not in cache:
            runs = [f"judge_iso_{t}{a}_q3" for t in tags]
            runs = [r for r in runs if (E4 / f"{r}.raw.jsonl").exists()]
            cache[a] = {} if runs else None   # an arm may live in one run only (e.g. *_llm)
            for r in runs:
                cache[a].update(load_labels(r))
        return cache[a]

    if any(L(a) is None for a in CORE):
        print(f"missing core judge files: {[a for a in CORE if L(a) is None]} (tag={args.tag!r})")
        return
    arms = ["chunk", "chunk_ce", "chunk_llm", "page_rel", "page_llm", "page_oracle", "page_seq",
            "snippet_full"]
    arms = [a for a in arms if L(a) is not None]
    BUDGETS = sorted(L("chunk"), key=lambda b: int(b[1:]))

    print(f"=== iso-budget answer quality (truthful / acc / halluc), tag={args.tag!r} ===")
    print(f"{'arm':13} " + " ".join(f"{b:>15}" for b in BUDGETS))
    for a in arms:
        d = L(a)
        cells = []
        for b in BUDGETS:
            cells.append(f"{trio(list(d[b].values()))[0]:+.2f}/{trio(list(d[b].values()))[1]:.2f}/"
                         f"{trio(list(d[b].values()))[2]:.2f}" if b in d else "-")
        print(f"{a:13} " + " ".join(f"{c:>15}" for c in cells))

    # accuracy-only compact curve (for the figure)
    print(f"\n=== accuracy curve ===\n{'arm':13} " + " ".join(f"{b[1:]:>6}" for b in BUDGETS))
    for a in arms:
        d = L(a)
        print(f"{a:13} " + " ".join(f"{trio(list(d[b].values()))[1]:6.3f}" if b in d else "  -  "
                                    for b in BUDGETS))

    # paired families (Δ = A − B), per-family Holm over budgets
    fams = [("R", "chunk", "page_rel", "chunk vs fair page"),
            ("P", "chunk", "page_seq", "chunk vs realistic page"),
            ("SF", "chunk", "snippet_full", "chunk vs full snippet")]
    if "chunk_ce" in arms:
        fams += [("CE", "chunk_ce", "chunk", "reranker vs e5 chunk"),
                 ("CB", "chunk_ce", "page_rel", "reranker-chunk vs fair page")]
        if "page_seq" in arms:
            fams.append(("CP", "chunk_ce", "page_seq", "reranker-chunk vs realistic page"))
        if "snippet_full" in arms:
            fams.append(("CS", "chunk_ce", "snippet_full", "reranker-chunk vs full snippet"))

    # LLM-reranker 2x2 (chunk/page x e5/Qwen3-Reranker) + page-ranking diagnostic
    fams += [("LG", "chunk_llm", "page_llm", "granularity under the LLM reranker"),
             ("LC", "chunk_llm", "chunk", "LLM reranker vs e5, chunks"),
             ("LP", "page_llm", "page_rel", "LLM reranker vs MaxP-e5, pages"),
             ("LO", "chunk_llm", "page_oracle", "LLM-reranked chunks vs oracle-ordered pages"),
             ("LS", "chunk_llm", "snippet_full", "LLM-reranked chunks vs full snippet")]

    rows = []
    for fam, A, B, _ in fams:
        if L(A) is None or L(B) is None:
            continue
        for b in BUDGETS:
            la, lb, n = aligned(L(A)[b], L(B)[b])
            acc = mcnemar(la, lb, lambda x: int(x in CORRECT))
            tru = truthful_test(la, lb)
            rows.append([fam, f"{fam} {b}", n, acc, tru])

    by_fam = {}
    for i, r in enumerate(rows):
        by_fam.setdefault(r[0], []).append(i)
    adj = [1.0] * len(rows)
    for idxs in by_fam.values():
        for i, pa in zip(idxs, holm([rows[i][3]["p"] for i in idxs])):
            adj[i] = pa

    famdesc = {f[0]: f[3] for f in fams}
    print(f"\n{'comparison':12} {'n':>4} {'Δacc':>8} {'b/c':>9} {'pHolm':>7} {'verdict':7} | {'Δtruth':>8} {'truthCI':>17}")
    out = []
    for fam, idxs in by_fam.items():
        print(f"\n--- family {fam}: {famdesc.get(fam, fam)} (Holm/{len(idxs)}) ---")
        for i in idxs:
            _, name, n, acc, tru = rows[i]
            pa = adj[i]
            verdict = "LOCKED" if pa < 0.05 else ("trend" if acc["p"] < 0.05 else "null")
            print(f"{name:12} {n:>4} {acc['delta']:+8.3f} {acc['b']:>4}/{acc['c']:<4} "
                  f"{pa:7.4f} {verdict:7} | {tru['delta']:+8.3f} [{tru['ci'][0]:+.2f},{tru['ci'][1]:+.2f}]")
            out.append({"comparison": name, "n": n, "dacc": acc["delta"], "p_acc_holm": pa,
                        "verdict": verdict, "dtruth": tru["delta"], "truth_ci": tru["ci"]})
    name = f"significance_iso_{'+'.join(t.rstrip('_') for t in tags) or 't3'}.json"
    json.dump(out, open(E4 / name, "w"), indent=1)
    print(f"\nwrote {name}")


if __name__ == "__main__":
    main()
