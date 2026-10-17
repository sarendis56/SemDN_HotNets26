"""E2-CRAG: build a SHARED e5 chunk index over pooled CRAG pages, so Search-R1 can
retrieve over it the way E2 retrieved over wiki-18 — and we can measure agentic-stream
locality + cache behaviour on a second, live-web-flavoured domain.

The crux (vs per-question pools): pages are deduplicated by URL across ALL sampled
questions, so one popular page is a single index entry that many questions' sub-queries
can re-touch. That cross-question sharing is exactly the locality a shared SCDN index
would harvest; without the global dedup there is none.

Emits (under data/e2_crag/):
  index.faiss      Flat inner-product e5 index over all unique chunks
  contents.jsonl   chunk id -> "title\\ntext" (Search-R1 passage format; served by the
                   retrieval server)
  sizes.npy        per-chunk text byte size          (cachesim demand/eviction)
  page_ids.npy     per-chunk source-page id          (cachesim page_lru)
  pages.jsonl      page id -> {url, title, n_questions}  (cross-question sharing stats)
  questions.jsonl  FlashRAG-format questions for gen_traces (id = interaction_id, so the
                   driven questions align with the indexed pages by construction)

Run (GPU for e5):  CUDA_VISIBLE_DEVICES=0 uv run python -m scdn.e2_crag_index --task 1 --n 1500
"""

import argparse
import bz2
import glob
import json
import os
import random

import numpy as np

from scdn import paths
from scdn.e1_fetch import chunk_words

E2C = paths.DATA_ROOT / "e2_crag"
TASK1_SRC = paths.DATA_ROOT / "datasets" / "crag" / "crag_task_1_dev_v4_release.jsonl.bz2"
TASK3_SRC = paths.DATA_ROOT / "datasets" / "crag" / "crag_task_3_dev_v4"


def _iter_lines(src):
    src = str(src)
    files = (sorted(glob.glob(os.path.join(src, "*.jsonl")) +
                    glob.glob(os.path.join(src, "*.jsonl.bz2"))) if os.path.isdir(src) else [src])
    for fp in files:
        op = bz2.open if fp.endswith(".bz2") else open
        with op(fp, "rt", encoding="utf-8") as f:
            for line in f:
                yield line


def sample_questions(src, n, seed):
    """Reservoir-sample n questions with a valid golden answer, keeping their pages."""
    rng = random.Random(seed)
    res, seen = [], 0
    for line in _iter_lines(src):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        golden = [r["answer"]] + (r.get("alt_ans") or [])
        golden = [a for a in golden if a and a not in ("invalid question", "I don't know")]
        if not golden:
            continue
        rec = {"qid": r.get("interaction_id", f"crag_{seen}"), "question": r["query"],
               "golden_answers": golden,
               "metadata": {"dynamism": r.get("static_or_dynamic", "?"),
                            "domain": r.get("domain", "?"),
                            "question_type": r.get("question_type", "?")},
               "pages": [{"url": s.get("page_url") or f"_nourl_{i}", "title": s.get("page_name", ""),
                          "html": s.get("page_result", "") or "", "snippet": s.get("page_snippet", "")}
                         for i, s in enumerate(r.get("search_results") or [])]}
        if len(res) < n:
            res.append(rec)
        else:
            j = rng.randint(0, seen)
            if j < n:
                res[j] = rec
        seen += 1
        if seen % 1000 == 0:
            print(f"  ...streamed {seen} valid questions", flush=True)
    print(f"sampled {len(res)} of {seen} valid questions", flush=True)
    return res


def build(args):
    import trafilatura

    from scdn.retrieval_server import E5Encoder
    E2C.mkdir(parents=True, exist_ok=True)
    src = args.src or (str(TASK1_SRC) if args.task == 1 else str(TASK3_SRC))
    qs = sample_questions(src, args.n, args.seed)

    # 1) global page dedup by URL; count how many questions reference each page
    page_of, page_meta = {}, []          # url -> page_id ; page_id -> {url,title,html,nq}
    for q in qs:
        for pg in q["pages"][:args.pages_per_q]:
            url = pg["url"]
            if url not in page_of:
                page_of[url] = len(page_meta)
                page_meta.append({"url": url, "title": pg["title"], "html": pg["html"], "nq": 0})
            page_meta[page_of[url]]["nq"] += 1
    nq_per_page = np.array([p["nq"] for p in page_meta])
    print(f"{len(page_meta)} unique pages from {len(qs)} questions "
          f"(sharing: {(nq_per_page > 1).mean():.1%} of pages used by >1 question, "
          f"max {nq_per_page.max()} questions/page)", flush=True)

    # 2) extract + chunk every unique page once (encoder-independent)
    contents, sizes, page_ids = [], [], []
    usable_pages = 0
    for pid, p in enumerate(page_meta):
        if not p["html"]:
            continue
        try:
            txt = trafilatura.extract(p["html"], include_comments=False, include_tables=True) or ""
        except Exception:
            txt = ""
        if len(txt) < 200:
            continue
        usable_pages += 1
        title = p["title"] or "page"
        for ch in chunk_words(txt, args.chunk_words):
            contents.append(f"{title}\n{ch}")
            sizes.append(len(ch.encode("utf-8")))
            page_ids.append(pid)
        if (pid + 1) % 2000 == 0:
            print(f"  extracted {pid+1}/{len(page_meta)} pages, {len(contents)} chunks", flush=True)
    print(f"{usable_pages}/{len(page_meta)} pages usable -> {len(contents)} chunks", flush=True)

    # 3) e5-embed all chunks, build a Flat (exact) inner-product index
    enc = E5Encoder(str(paths.E5_BASE_V2), device=args.device)
    embs = []
    B = 2048
    for i in range(0, len(contents), B):
        # contents already carry the title line; embed the chunk body the way E2 did
        embs.append(enc.encode_passages([c.split("\n", 1)[-1] for c in contents[i:i + B]]))
        if (i // B) % 10 == 0:
            print(f"  embedded {i+len(embs[-1])}/{len(contents)}", flush=True)
    emb = np.concatenate(embs, axis=0).astype(np.float32)

    import faiss
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(E2C / "index.faiss"))
    np.save(E2C / "sizes.npy", np.array(sizes, dtype=np.int64))
    np.save(E2C / "page_ids.npy", np.array(page_ids, dtype=np.int64))
    with open(E2C / "contents.jsonl", "w") as f:
        for i, c in enumerate(contents):
            f.write(json.dumps({"id": i, "contents": c}) + "\n")
    with open(E2C / "pages.jsonl", "w") as f:
        for pid, p in enumerate(page_meta):
            f.write(json.dumps({"page_id": pid, "url": p["url"], "title": p["title"], "nq": p["nq"]}) + "\n")
    with open(E2C / "questions.jsonl", "w") as f:        # FlashRAG format for gen_traces
        for q in qs:
            f.write(json.dumps({"id": q["qid"], "question": q["question"],
                                "golden_answers": q["golden_answers"], "metadata": q["metadata"]}) + "\n")
    print(f"index built: ntotal={index.ntotal:,}, dim={emb.shape[1]} -> {E2C}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=1, choices=[1, 3], help="CRAG page source (5 vs 50 pages/q)")
    ap.add_argument("--src", default=None, help="override source path")
    ap.add_argument("--n", type=int, default=1500, help="number of questions to pool")
    ap.add_argument("--pages-per-q", type=int, default=50)
    ap.add_argument("--chunk-words", type=int, default=100, help="~wiki-18 passage size, for E2 comparability")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
