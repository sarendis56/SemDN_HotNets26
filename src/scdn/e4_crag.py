"""E4-CRAG: controlled page-vs-chunk granularity on CRAG's PROVIDED pages (Task 1:
5 crawled HTML pages/question, fixed & reproducible — no live-web noise, no fetch failures,
no serper credits).

Two retrieval pipelines, both ending in a chunk-level cross-encoder rerank → top-3:
  pageret@K : PAGE-level retrieval — rank the 5 pages by an e5 page vector (mean of
              the page's chunk embeddings), keep top-K pages, CE-rerank their chunks.
  chunkret  : CHUNK-level retrieval — rank ALL chunks by e5, keep top-N, CE-rerank.
The only difference is the granularity of the FIRST retrieval stage. (The no-RAG
baseline is produced by `e4_answer --closed-book` on the same pool.)

Writes a pool jsonl in the schema e4_answer/e4_judge consume. GENERATION only
(e5 + cross-encoder); reader + Llama-70B judge run separately and are deferred.

Run:  CUDA_VISIBLE_DEVICES=1 uv run python -m scdn.e4_crag --stage build --n 300 --chunker structnt
"""

import argparse
import json
import random

import numpy as np

from scdn import paths
from scdn.crag import SRC, iter_crag
from scdn.e4_granularity import _make_chunker

E4 = paths.DATA_ROOT / "e4"
POOL_CRAG = E4 / "pool_crag.jsonl"
TOPK = 3
PAGE_KS = [1, 3]      # page-retrieval depths for the page-first arm (of 5 pages)
CHUNK_TOPN = 30       # e5 chunk-retrieval candidates before CE rerank (chunk-first arm)


def load_crag_pages(n: int, seed: int) -> list[dict]:
    """Same filtering+sampling as gen_traces/crag, but keeps each question's 5 pages."""
    rows = []
    for i, r in enumerate(iter_crag(SRC)):
        golden = [r["answer"]] + (r.get("alt_ans") or [])
        golden = [a for a in golden if a and a not in ("invalid question", "I don't know")]
        if not golden:
            continue
        rows.append({
            "qid": r.get("interaction_id", f"crag_{i}"), "question": r["query"],
            "golden_answers": golden,
            "metadata": {"dynamism": r.get("static_or_dynamic", "?"),
                         "domain": r.get("domain", "?"),
                         "question_type": r.get("question_type", "?")},
            "pages": [{"title": s.get("page_name", ""), "html": s.get("page_result", "") or ""}
                      for s in (r.get("search_results") or [])],
        })
    random.Random(seed).shuffle(rows)
    return rows[:n] if n else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="build", choices=["build"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--question-seed", type=int, default=0)
    ap.add_argument("--chunker", default="structnt", choices=["fixed", "structure", "structnt"])
    ap.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    E4.mkdir(parents=True, exist_ok=True)

    import torch
    import trafilatura
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from scdn.retrieval_server import E5Encoder
    enc = E5Encoder(str(paths.E5_BASE_V2), device=args.device)
    cetok = AutoTokenizer.from_pretrained(paths.model(args.reranker))
    cemodel = AutoModelForSequenceClassification.from_pretrained(
        paths.model(args.reranker), torch_dtype=torch.float16).to(args.device).eval()
    chunker = _make_chunker(args.chunker)

    @torch.no_grad()
    def ce_scores(q, chunks, batch=64):
        out = []
        for i in range(0, len(chunks), batch):
            pairs = [[q, c] for c in chunks[i:i + batch]]
            inp = cetok(pairs, padding=True, truncation=True, max_length=512,
                        return_tensors="pt").to(args.device)
            out.extend(cemodel(**inp).logits.view(-1).float().cpu().numpy().tolist())
        return np.array(out)

    qs = load_crag_pages(args.n, args.question_seed)
    done = {json.loads(l)["qid"] for l in open(POOL_CRAG)} if POOL_CRAG.exists() else set()
    todo = [q for q in qs if q["qid"] not in done]
    print(f"crag build ({args.chunker}): {len(todo)} questions ({len(done)} cached)", flush=True)
    fout = open(POOL_CRAG, "a")
    for idx, q in enumerate(todo):
        ctext, cpage, npages = [], [], 0
        for pi, pg in enumerate(q["pages"]):
            if not pg["html"]:
                continue
            try:
                txt = trafilatura.extract(pg["html"], include_comments=False, include_tables=True) or ""
            except Exception:
                txt = ""
            if len(txt) < 200:
                continue
            npages += 1
            for ch in chunker(txt, pg["title"]):
                ctext.append(ch); cpage.append(pi)
        rec = {"qid": q["qid"], "question": q["question"], "golden_answers": q["golden_answers"],
               "metadata": q["metadata"], "n_pages": npages, "n_usable": npages,
               "n_chunks": len(ctext), "conds": {}}
        if ctext:
            cp = np.array(cpage)
            cemb = enc.encode_passages(ctext).astype(np.float32)
            qv = enc.encode_queries([q["question"]]).astype(np.float32)[0]
            chunk_e5 = cemb @ qv
            ce = ce_scores(q["question"], ctext)
            pages_present = sorted(set(cpage))
            page_e5 = {p: float(cemb[cp == p].mean(0) @ qv) for p in pages_present}
            page_rank = sorted(pages_present, key=lambda p: -page_e5[p])
            # Arm A — page-level retrieval (top-K pages) then chunk-level CE rerank
            for K in PAGE_KS:
                keep = page_rank[:K]
                mask = np.where(np.isin(cp, keep))[0]
                if mask.size:
                    top = mask[np.argsort(-ce[mask])[:TOPK]]
                    rec["conds"][f"pageret@{K}"] = {"chunk_texts": [ctext[j] for j in top],
                                                    "src_pages": [int(cp[j]) for j in top]}
                else:
                    rec["conds"][f"pageret@{K}"] = {"chunk_texts": [], "src_pages": []}
            # Arm B — chunk-level retrieval (e5 top-N) then chunk-level CE rerank
            cand = np.argsort(-chunk_e5)[:CHUNK_TOPN]
            top = cand[np.argsort(-ce[cand])[:TOPK]]
            rec["conds"]["chunkret"] = {"chunk_texts": [ctext[j] for j in top],
                                        "src_pages": [int(cp[j]) for j in top]}
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{len(todo)}] {q['qid'][:8]}: {npages} pages, {len(ctext)} chunks",
                  flush=True)
    fout.close()
    print(f"crag build done -> {POOL_CRAG}", flush=True)


if __name__ == "__main__":
    main()
