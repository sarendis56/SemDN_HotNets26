"""Task-3 intuition: WHERE can a deep semantic index beat CRAG's few-page baseline?

CRAG's provided pages come in search-rank order (the simulated Web API's ranking =
"actual top web search results"). A deeper index adds RECALL but also NOISE, and E4
found depth saturates ~top-5. Before chasing a quality win we measure the structural
headroom: for each question, does the gold answer even live in the shallow window, and
when it lives deeper, does dense (e5) retrieval surface it where lexical (BM25) /
provided-order cannot?

Runs over the encoder-independent chunk cache (chunks_crag_t3.jsonl: chunks + provided
page rank + golden answers). One e5 pass (cheap). BM25 reuses e4_crag_t3.bm25_scores.

Decomposition (n=300):
  - recall ceiling      : % q with >=1 gold-bearing chunk anywhere in the <=50 pages
  - shallow coverage    : % q with a gold-bearing chunk within the top-K PROVIDED pages
  - deep-only answers    : gold NOT in top-K pages but present deeper (the recall the
                          few-page baseline structurally misses)
  - dense surfacing      : of deep-only q, does e5 rank a gold chunk into the top-3 fed
                          to the reader?  (BM25 / provided as shallow contrasts)
  -> ADDRESSABLE HEADROOM = deep-only AND dense-surfaces: questions the SemDN's dense
     index could win that no shallow read of the top-K pages could.

Run:  CUDA_VISIBLE_DEVICES=1 uv run python -m semdn.e4_answer_locality --k 5 --topk 3
"""

import argparse
import json

import numpy as np

from semdn import paths
from semdn.e4_crag_t3 import CHUNKS_CACHE, bm25_scores
from semdn.metrics import normalize_answer

TOPK = 3                       # chunks fed to the reader
SHALLOW_KS = [1, 3, 5]         # "few-page" windows to treat as the baseline


def gold_in_text(text: str, golden: list[str]) -> bool:
    """Gold present in a chunk. Handles CRAG list answers ('a, b, c'): a chunk rarely
    contains the whole comma-joined list, so count it present if the full normalized
    gold is a substring OR (for comma-lists) at least half the items appear."""
    t = normalize_answer(text)
    for g in golden:
        gn = normalize_answer(g)
        if gn and gn in t:
            return True
        items = [normalize_answer(x) for x in str(g).split(",")]
        items = [x for x in items if len(x) >= 2]
        if len(items) >= 2 and sum(x in t for x in items) >= (len(items) + 1) // 2:
            return True
    return False


def best_gold_rank(scores: np.ndarray, gold_mask: np.ndarray) -> int:
    """1-based rank of the highest-scoring gold-bearing chunk (len+1 if none)."""
    order = np.argsort(-scores)
    for pos, j in enumerate(order):
        if gold_mask[j]:
            return pos + 1
    return len(scores) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--out", default=str(paths.DATA_ROOT / "e4" / "answer_locality_t3.json"))
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(CHUNKS_CACHE)]
    print(f"{len(recs)} questions", flush=True)

    from semdn.retrieval_server import E5Encoder
    enc = E5Encoder(str(paths.E5_BASE_V2), device=args.device)

    per = []
    for i, r in enumerate(recs):
        chunks, ranks = r["chunks"], np.array(r["ranks"])
        golden = r["golden_answers"]
        if not chunks:
            per.append(None)
            continue
        gold_mask = np.array([gold_in_text(c, golden) for c in chunks])
        # dense (e5) and lexical (bm25) global rankings over the SAME pool
        cemb = enc.encode_passages(chunks).astype(np.float32)
        qv = enc.encode_queries([r["question"]]).astype(np.float32)[0]
        e5 = cemb @ qv
        bm = bm25_scores(r["question"], chunks)
        pv = -np.arange(len(chunks), dtype=np.float32)   # provided order
        # shallow-window gold coverage: any gold chunk within the first K provided pages
        shallow_cov = {K: bool((gold_mask & (ranks <= K)).any()) for K in SHALLOW_KS}
        # does the ranker put a gold chunk in the top-3 it would feed the reader?
        def surfaces(scores):
            top = np.argsort(-scores)[:args.topk]
            return bool(gold_mask[top].any())
        per.append({
            "qid": r["qid"], "n_usable": r["n_usable"], "n_chunks": len(chunks),
            "has_gold": bool(gold_mask.any()),
            "gold_min_page": int(ranks[gold_mask].min()) if gold_mask.any() else 0,
            "shallow_cov": shallow_cov,
            "e5_top": surfaces(e5), "bm25_top": surfaces(bm), "provided_top": surfaces(pv),
            "e5_gold_rank": best_gold_rank(e5, gold_mask),
            "bm25_gold_rank": best_gold_rank(bm, gold_mask),
            "dynamism": r["metadata"].get("dynamism", "?"),
            "domain": r["metadata"].get("domain", "?"),
        })
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(recs)}]", flush=True)

    P = [p for p in per if p]
    n = len(P)

    def frac(pred):
        return sum(1 for p in P if pred(p)) / n

    print(f"\n=== ANSWER-LOCALITY DECOMPOSITION (CRAG Task-3, n={n}, top-{args.topk} to reader) ===")
    rc = frac(lambda p: p["has_gold"])
    print(f"recall ceiling: {rc:.3f} of q have a gold-bearing chunk in the <=50 pages "
          f"({1-rc:.3f} unanswerable from the provided corpus at all)")
    print("\nshallow-window coverage (gold present within top-K provided pages) and the "
          "deep-only residue:")
    print(f"  {'K':>3} {'shallow_cov':>12} {'deep_only':>10} {'e5_recovers':>12} "
          f"{'bm25_recovers':>14} {'HEADROOM(e5)':>13}")
    summary = {"n": n, "recall_ceiling": rc, "by_k": {}}
    for K in SHALLOW_KS:
        cov = frac(lambda p: p["shallow_cov"][K])
        # deep_only: has gold, but NOT within the top-K provided pages
        deep_only = [p for p in P if p["has_gold"] and not p["shallow_cov"][K]]
        do = len(deep_only) / n
        # of those, does dense / lexical surface a gold chunk into the reader's top-3?
        e5_rec = (sum(p["e5_top"] for p in deep_only) / len(deep_only)) if deep_only else 0.0
        bm_rec = (sum(p["bm25_top"] for p in deep_only) / len(deep_only)) if deep_only else 0.0
        headroom = len([p for p in deep_only if p["e5_top"]]) / n
        print(f"  {K:>3} {cov:>12.3f} {do:>10.3f} {e5_rec:>12.3f} {bm_rec:>14.3f} {headroom:>13.3f}")
        summary["by_k"][K] = {"shallow_cov": cov, "deep_only": do,
                              "e5_recovers": e5_rec, "bm25_recovers": bm_rec,
                              "addressable_headroom": headroom}

    # where does the gold live, and dense-vs-lexical surfacing on the WHOLE answerable set
    ans = [p for p in P if p["has_gold"]]
    e5_top = sum(p["e5_top"] for p in ans) / len(ans)
    bm_top = sum(p["bm25_top"] for p in ans) / len(ans)
    pv_top = sum(p["provided_top"] for p in ans) / len(ans)
    print(f"\non the {len(ans)} answerable q, gold chunk reaches the reader's top-{args.topk}: "
          f"e5={e5_top:.3f}  bm25={bm_top:.3f}  provided={pv_top:.3f}")
    e5r = np.array([p["e5_gold_rank"] for p in ans])
    bmr = np.array([p["bm25_gold_rank"] for p in ans])
    print(f"median rank of best gold chunk: e5={np.median(e5r):.0f}  bm25={np.median(bmr):.0f}  "
          f"(e5 better on {(e5r < bmr).mean():.2f} of q, bm25 better on {(bmr < e5r).mean():.2f})")
    summary["answerable"] = {"n": len(ans), "e5_top": e5_top, "bm25_top": bm_top,
                             "provided_top": pv_top,
                             "e5_median_gold_rank": float(np.median(e5r)),
                             "bm25_median_gold_rank": float(np.median(bmr)),
                             "e5_beats_bm25_frac": float((e5r < bmr).mean())}
    json.dump(summary, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
