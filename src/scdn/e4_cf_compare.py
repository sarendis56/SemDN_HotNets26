"""Experiment (a) downstream: does adding the Cloudflare-recovered (previously network/JS-blocked)
pages to the live-web pool improve answer quality? For the recovered questions, build two pools
of coherent chunks, e5-ranked -> top-3:
  orig : reachable pages only (trafilatura text) — the current pipeline.
  aug  : reachable pages + the Cloudflare-recovered blocked pages (their markdown).
Then e4_answer + the CRAG judge score both; paired compare (aug vs orig) shows if recovery helps.
GPU (e5) for ranking only.

Run:  CUDA_VISIBLE_DEVICES=1 uv run python -m scdn.e4_cf_compare
"""
import json

import numpy as np

from scdn import paths
from scdn.e4_granularity import _build_chunks, _make_chunker, _select_conditions

E4 = paths.DATA_ROOT / "e4"


def main():
    from scdn.e4_granularity import _make_encoder
    embed_docs, embed_queries = _make_encoder("e5", "cuda")
    chunker = _make_chunker("coherent")

    serp = {json.loads(l)["qid"]: json.loads(l) for l in open(E4 / "serp_deep.jsonl")}
    pages = {json.loads(l)["url"]: json.loads(l) for l in open(E4 / "pages_cache.jsonl")}
    meta = {json.loads(l)["qid"]: json.loads(l) for l in open(E4 / "pool_coherent.jsonl")}

    # augmented page table = trafilatura pages + recovered Cloudflare markdown (as ok pages)
    aug_pages = dict(pages)
    recovered_qids = set()
    for l in open(E4 / "cf_recovered.jsonl"):
        r = json.loads(l)
        if r["ok"]:
            aug_pages[r["url"]] = {"ok": True, "method": "cloudflare", "text": r["markdown"]}
            recovered_qids.add(r["qid"])
    qids = [q for q in sorted(recovered_qids) if q in serp and q in meta]
    print(f"{len(qids)} recovered questions", flush=True)

    fo = open(E4 / "pool_live_cf_orig.jsonl", "w")
    fa = open(E4 / "pool_live_cf_aug.jsonl", "w")
    for i, qid in enumerate(qids):
        q = meta[qid]
        results = serp[qid]["results"]
        for pagesrc, fout in ((pages, fo), (aug_pages, fa)):
            ctext, crank, n_usable = _build_chunks(results, pagesrc, chunker)
            rec = {"qid": qid, "question": q["question"], "golden_answers": q["golden_answers"],
                   "metadata": q["metadata"], "n_usable": n_usable, "n_chunks": len(ctext),
                   "conds": {}, "deep_topk_max_rank": 0}
            if ctext:
                scores = embed_docs(ctext).astype(np.float32) @ embed_queries([q["question"]]).astype(np.float32)[0]
                conds, deep_max = _select_conditions(ctext, crank, scores)
                rec["conds"] = conds
                rec["deep_topk_max_rank"] = deep_max
            fout.write(json.dumps(rec) + "\n")
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(qids)}]", flush=True)
    fo.close()
    fa.close()
    # report how much the augmented pool grew (usable pages)
    o = [json.loads(l) for l in open(E4 / "pool_live_cf_orig.jsonl")]
    a = [json.loads(l) for l in open(E4 / "pool_live_cf_aug.jsonl")]
    print(f"usable pages/q — orig {np.mean([r['n_usable'] for r in o]):.1f} -> aug "
          f"{np.mean([r['n_usable'] for r in a]):.1f}", flush=True)
    print(f"pools -> {E4}/pool_live_cf_{{orig,aug}}.jsonl", flush=True)


if __name__ == "__main__":
    main()
