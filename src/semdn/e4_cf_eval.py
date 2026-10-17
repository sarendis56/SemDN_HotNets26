"""Fetcher comparison (our trafilatura GET vs the Cloudflare cloud clean-fetch) on the
live web, holding everything else fixed. For a reservoir subset of the live questions,
over the SAME top-K SERP urls, build retrieval pools two ways:

  traf : trafilatura text from pages_cache.jsonl     (our fetcher; misses ~56%, no JS)
  cf   : CF /markdown text from cf_pages_cache.jsonl  (cloud render; gets past network failures/bot-403)

across chunkers {fixed, structnt, coherent}, e5-ranked, with snippet / full-page / no-RAG
baselines. Both fetchers feed the SAME chunkers + ranker + reader + judge, so the only
variable is the fetch (completeness + cleanliness). Writes pools the existing
e4_answer.py reader consumes unchanged.

Run (GPU for e5):  CUDA_VISIBLE_DEVICES=0 uv run python -m semdn.e4_cf_eval build --n 120 --topk 12
"""
import argparse
import json

import numpy as np

from semdn import paths
from semdn.e4_granularity import (
    _build_chunks, _make_chunker, _select_conditions, _snippet_candidates, load_crag,
)

E4 = paths.DATA_ROOT / "e4"
SERP = E4 / "serp_deep.jsonl"
CHUNKERS = ["fixed", "structnt", "coherent"]


def _keyed(path, key):
    return {json.loads(l)[key]: json.loads(l) for l in open(path)} if path.exists() else {}


def _topk_results(results, topk):
    return results[:topk]


def build(args):
    """Build all pools in one e5 session."""
    from semdn.e4_granularity import _make_encoder
    embed_docs, embed_queries = _make_encoder("e5", args.device)

    def score_fn(question, chunks):
        cemb = embed_docs(chunks).astype(np.float32)
        qv = embed_queries([question]).astype(np.float32)[0]
        return cemb @ qv

    qs = load_crag(args.n, 0)
    serp = _keyed(SERP, "qid")
    qs = [q for q in qs if q["qid"] in serp]
    print(f"{len(qs)} questions; topk={args.topk}", flush=True)

    pages = {
        "traf": _keyed(E4 / "pages_cache.jsonl", "url"),
        "cf": _keyed(E4 / "cf_pages_cache.jsonl", "url"),
    }
    for fetcher in ("traf", "cf"):
        ok = sum(1 for r in pages[fetcher].values() if r.get("ok"))
        print(f"  {fetcher}: {len(pages[fetcher])} cached pages, {ok} ok", flush=True)

    def write_chunk_pool(fetcher, chunker_name):
        chunker = _make_chunker(chunker_name)
        out = E4 / f"pool_cf_{fetcher}_{chunker_name}.jsonl"
        n_written = usable_tot = 0
        with open(out, "w") as f:
            for q in qs:
                results = _topk_results(serp[q["qid"]]["results"], args.topk)
                ctext, crank, n_usable = _build_chunks(results, pages[fetcher], chunker)
                rec = {"qid": q["qid"], "question": q["question"],
                       "golden_answers": q["golden_answers"], "metadata": q["metadata"],
                       "n_usable": n_usable, "n_chunks": len(ctext),
                       "conds": {}, "deep_topk_max_rank": 0}
                if ctext:
                    scores = score_fn(q["question"], ctext)
                    conds, deep_max = _select_conditions(ctext, crank, scores)
                    rec["conds"] = conds
                    rec["deep_topk_max_rank"] = deep_max
                f.write(json.dumps(rec) + "\n")
                n_written += 1
                usable_tot += n_usable
        print(f"  wrote {out.name}: {n_written} q, mean usable pages {usable_tot/max(1,n_written):.1f}",
              flush=True)

    # chunk pools: 2 fetchers x chunkers
    for fetcher in ("traf", "cf"):
        for ch in args.chunkers:
            write_chunk_pool(fetcher, ch)

    # snippet baseline (fetcher-independent: title + SERP snippet of top-K results)
    def write_snippet():
        out = E4 / "pool_cf_snippet.jsonl"
        pks = [1, 3, 5]
        with open(out, "w") as f:
            for q in qs:
                results = _topk_results(serp[q["qid"]]["results"], max(pks))
                blocks = [f"{r.get('title','')}\n{r.get('snippet','')}".strip()
                          for r in results if (r.get("snippet") or "").strip()]
                conds = {f"page@{K}": {"chunk_texts": blocks[:K]} for K in pks}
                f.write(json.dumps({"qid": q["qid"], "question": q["question"],
                    "golden_answers": q["golden_answers"], "metadata": q["metadata"],
                    "n_usable": len(blocks), "conds": conds}) + "\n")
        print(f"  wrote {out.name}", flush=True)

    # full-page baseline per fetcher (whole pages, no chunking, capped per page)
    def write_fullpage(fetcher):
        out = E4 / f"pool_cf_fullpage_{fetcher}.jsonl"
        pks = [1, 3, 5]
        cap = args.page_chars
        with open(out, "w") as f:
            for q in qs:
                results = _topk_results(serp[q["qid"]]["results"], max(pks))
                blocks = []
                for r in results:
                    pg = pages[fetcher].get(r["url"])
                    if pg and pg.get("ok"):
                        blocks.append(pg["text"][:cap])
                conds = {f"page@{K}": {"chunk_texts": blocks[:K]} for K in pks}
                f.write(json.dumps({"qid": q["qid"], "question": q["question"],
                    "golden_answers": q["golden_answers"], "metadata": q["metadata"],
                    "n_usable": len(blocks), "conds": conds}) + "\n")
        print(f"  wrote {out.name}", flush=True)

    # hybrid (cf chunks + snippets as ranking co-candidates), best chunker
    def write_hybrid(fetcher, chunker_name):
        chunker = _make_chunker(chunker_name)
        out = E4 / f"pool_cf_{fetcher}_{chunker_name}_hybrid.jsonl"
        with open(out, "w") as f:
            for q in qs:
                results = _topk_results(serp[q["qid"]]["results"], args.topk)
                ctext, crank, n_usable = _build_chunks(results, pages[fetcher], chunker)
                stext, srank = _snippet_candidates(results)
                ctext, crank = ctext + stext, crank + srank
                rec = {"qid": q["qid"], "question": q["question"],
                       "golden_answers": q["golden_answers"], "metadata": q["metadata"],
                       "n_usable": n_usable, "n_chunks": len(ctext),
                       "conds": {}, "deep_topk_max_rank": 0}
                if ctext:
                    scores = score_fn(q["question"], ctext)
                    conds, deep_max = _select_conditions(ctext, crank, scores)
                    rec["conds"], rec["deep_topk_max_rank"] = conds, deep_max
                f.write(json.dumps(rec) + "\n")
        print(f"  wrote {out.name}", flush=True)

    write_snippet()
    for fetcher in ("traf", "cf"):
        write_fullpage(fetcher)
    write_hybrid("cf", "coherent")
    print("build done.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--n", type=int, default=120)
    b.add_argument("--topk", type=int, default=12)
    b.add_argument("--device", default="cuda")
    b.add_argument("--page-chars", type=int, default=3500)
    b.add_argument("--chunkers", nargs="+", default=CHUNKERS)
    args = ap.parse_args()
    if args.cmd == "build":
        build(args)


if __name__ == "__main__":
    main()
