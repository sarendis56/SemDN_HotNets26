"""Paired significance tests for E4, on the saved per-question judge labels
(data/e4/judge_*.raw.jsonl). All comparisons are PAIRED by qid (same questions
under both conditions), which is the correct, more powerful test than single-condition
CIs. No GPU; numpy-only (no scipy):
  - accuracy / hallucination (binary): McNemar exact (two-sided binomial on discordant pairs)
  - truthfulness (score in {-1,0,.5,1}): paired bootstrap 95% CI + bootstrap p, sign-test cross-check
  - multiplicity: Holm-Bonferroni across the pre-registered family
Pre-registered comparisons map 1:1 to report findings F1,F2,F4,F7,F8,F9.

Run:  uv run python -m semdn.e4_stats
"""

import json
from math import comb

import numpy as np

from semdn import paths

E4 = paths.DATA_ROOT / "e4"
SCORE = {"perfect": 1.0, "acceptable": 0.5, "missing": 0.0, "incorrect": -1.0}
CORRECT = {"perfect", "acceptable"}
B_BOOT = 10000
RNG = np.random.default_rng(0)


def load_labels(run):
    """raw file basename -> {config: {qid: label}}"""
    d = {}
    for l in open(E4 / f"{run}.raw.jsonl"):
        r = json.loads(l)
        d.setdefault(r["config"], {})[r["qid"]] = r["label"]
    return d


def binom_two_sided(k, n, p=0.5):
    """Exact two-sided binomial test p-value (small n)."""
    if n == 0:
        return 1.0
    pmf = np.array([comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(n + 1)])
    return float(min(1.0, pmf[pmf <= pmf[k] + 1e-12].sum()))


def aligned(labA, labB):
    qs = sorted(set(labA) & set(labB))
    return [labA[q] for q in qs], [labB[q] for q in qs], len(qs)


def mcnemar(la, lb, positive):
    """McNemar exact on a binary outcome (`positive(label)`->1). A vs B; Δ = pA - pB."""
    a = np.array([positive(x) for x in la]); b = np.array([positive(x) for x in lb])
    bb = int(((a == 1) & (b == 0)).sum())   # A yes, B no
    cc = int(((a == 0) & (b == 1)).sum())   # A no, B yes
    return {"delta": float(a.mean() - b.mean()), "b": bb, "c": cc,
            "p": binom_two_sided(bb, bb + cc)}


def truthful_test(la, lb):
    """Paired bootstrap on mean truthfulness difference (A - B) + sign test."""
    d = np.array([SCORE[x] for x in la]) - np.array([SCORE[x] for x in lb])
    n = len(d); obs = float(d.mean())
    idx = RNG.integers(0, n, size=(B_BOOT, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_boot = float(2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    npos, nneg = int((d > 0).sum()), int((d < 0).sum())
    p_sign = binom_two_sided(npos, npos + nneg)
    return {"delta": obs, "ci": [float(lo), float(hi)], "p": p_boot, "p_sign": p_sign}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals); adj = [0.0] * m; running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


# (finding, A run/config, B run/config, metrics)  Δ = A − B
COMPARISONS = [
    ("F1  page@10 vs page@1 (LIVE fixed/e5)",      ("judge_e5_llama70b", "page@10"),   ("judge_e5_llama70b", "page@1"),     ["acc", "truthful"]),
    ("F2  chunk@all vs page@10 (LIVE fixed/e5)",   ("judge_e5_llama70b", "chunk@all"), ("judge_e5_llama70b", "page@10"),    ["truthful", "acc"]),
    ("F4a structnt vs fixed chunk@all (LIVE e5)",  ("judge_e5_structnt_llama70b", "chunk@all"), ("judge_e5_llama70b", "chunk@all"), ["truthful"]),
    ("F4b structnt vs +title chunk@all (LIVE e5)", ("judge_e5_structnt_llama70b", "chunk@all"), ("judge_e5_struct_llama70b", "chunk@all"), ["truthful"]),
    ("F7  chunkret vs no-RAG (CRAG)",              ("judge_crag_llama70b", "chunkret"), ("judge_crag_norag_llama70b", "no_rag"), ["acc", "truthful"]),
    ("F8a chunkret vs pageret@1 (CRAG)",           ("judge_crag_llama70b", "chunkret"), ("judge_crag_llama70b", "pageret@1"), ["acc", "truthful"]),
    ("F8b chunkret vs pageret@3 (CRAG)",           ("judge_crag_llama70b", "chunkret"), ("judge_crag_llama70b", "pageret@3"), ["acc", "truthful"]),
    ("F9  Qwen3-32B vs Qwen2.5-7B (CRAG chunkret)",("judge_q3_crag", "chunkret"),       ("judge_crag_llama70b", "chunkret"), ["acc", "truthful"]),
]


def main():
    cache = {}
    def L(run, cfg):
        cache.setdefault(run, load_labels(run))
        return cache[run][cfg]

    rows, pvals = [], []
    for name, (rA, cA), (rB, cB), metrics in COMPARISONS:
        la, lb, n = aligned(L(rA, cA), L(rB, cB))
        for m in metrics:
            if m == "acc":
                r = mcnemar(la, lb, lambda x: int(x in CORRECT))
                desc = f"Δacc={r['delta']:+.3f}  b/c={r['b']}/{r['c']}"; p = r["p"]; test = "McNemar"
            elif m == "halluc":
                r = mcnemar(la, lb, lambda x: int(x == "incorrect"))
                desc = f"Δhal={r['delta']:+.3f}  b/c={r['b']}/{r['c']}"; p = r["p"]; test = "McNemar"
            else:
                r = truthful_test(la, lb)
                desc = f"Δtru={r['delta']:+.3f}  CI[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]  sign-p={r['p_sign']:.3f}"; p = r["p"]; test = "bootstrap"
            rows.append([name, m, n, test, desc, p])
            pvals.append(p)

    adj = holm(pvals)
    print(f"{'comparison':46s} {'metric':8s} {'n':>4} {'test':9s} {'detail':46s} {'p':>7} {'p_holm':>7}  verdict")
    out = []
    for (name, m, n, test, desc, p), pa in zip(rows, adj):
        verdict = "LOCKED" if pa < 0.05 else ("trend" if p < 0.05 else "null")
        print(f"{name:46s} {m:8s} {n:>4} {test:9s} {desc:46s} {p:7.4f} {pa:7.4f}  {verdict}")
        out.append({"comparison": name, "metric": m, "n": n, "test": test,
                    "p_raw": p, "p_holm": pa, "verdict": verdict, "detail": desc})
    json.dump(out, open(E4 / "significance.json", "w"), indent=1)
    print(f"\nwrote {E4/'significance.json'}  (Holm family size = {len(pvals)}, bootstrap B={B_BOOT})")


if __name__ == "__main__":
    main()
