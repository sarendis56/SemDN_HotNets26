"""Live-web no-chunking baselines for the §G/N3 comparison on the realistic setting:
  snippet  : title + SERP snippet of the top-K results (no fetch).
  fullpage : the top-K reachable whole pages (trafilatura text, no chunking), capped per page.
Pairs against the live chunk arms (chunk e5 = pool_coherent; chunk reranker = pool_qwen3rr_coherent).
Questions/metadata come from pool_coherent.jsonl (the live coherent pool).

Run:  uv run python -m scdn.e4_live_baselines --mode snippet
      uv run python -m scdn.e4_live_baselines --mode fullpage
"""
import argparse
import json

from scdn import paths

E4 = paths.DATA_ROOT / "e4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["snippet", "fullpage"])
    ap.add_argument("--page-ks", default="1,3,5")
    ap.add_argument("--page-chars", type=int, default=3500)
    args = ap.parse_args()
    pks = [int(k) for k in args.page_ks.split(",")]

    serp = {json.loads(l)["qid"]: json.loads(l) for l in open(E4 / "serp_deep.jsonl")}
    pages = {json.loads(l)["url"]: json.loads(l) for l in open(E4 / "pages_cache.jsonl")}
    # question set + metadata from the canonical structnt pool (the best live chunker, and
    # the arm the V-family stats pair against) so the baseline n matches the chunk arms.
    meta = [json.loads(l) for l in open(E4 / "pool_structnt.jsonl")]

    out = E4 / f"pool_live_{args.mode}.jsonl"
    with open(out, "w") as f:
        for m in meta:
            qid = m["qid"]
            if qid not in serp:
                continue
            blocks = []
            for r in serp[qid]["results"]:
                if args.mode == "snippet":
                    blk = f"{r.get('title', '')}\n{r.get('snippet', '')}".strip()
                else:  # fullpage: trafilatura text of reachable pages, in SERP rank order
                    pg = pages.get(r["url"])
                    blk = pg["text"][: args.page_chars] if (pg and pg.get("ok")) else ""
                if blk:
                    blocks.append(blk)
            conds = {f"page@{K}": {"chunk_texts": blocks[:K]} for K in pks}
            f.write(json.dumps({"qid": qid, "question": m["question"],
                "golden_answers": m["golden_answers"], "metadata": m["metadata"],
                "n_usable": len(blocks), "conds": conds}) + "\n")
    print(f"live {args.mode} pool -> {out}", flush=True)


if __name__ == "__main__":
    main()
