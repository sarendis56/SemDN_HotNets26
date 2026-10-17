"""Bytes moved by each wiki-18 matrix client, measured on a sample of its questions.

  SemDN (chunk):        the 5 delivered chunks (UTF-8 bytes of their text)
  page client, e5:      downloads its top-N pages (N = 1, 3, 5, 10)
  page client, rerank:  downloads the top-30 pages (by e5 mean-pool) so the reranker can read
                        them whole, whatever N it then keeps

Pages are re-ranked exactly as in e1_matrix.py (e5 mean-pooled chunk vectors over the pages of
e5's top-100 chunks). Page bytes = live Wikipedia HTML fetched like the paper's lightweight
baseline (one GET, no JS, decompressed; same request as e1_fetch.light_fetch, minus its text
extraction, which is not thread-safe), plus the page's wiki-18 text
as a lower bound. Fetched sizes are cached in data/e1/wiki_html_sizes.jsonl.

Run:  uv run python -m scdn.e1_matrix_bytes --per-ds 50      (CPU + network, ~15 min)
"""
import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import faiss
import numpy as np
import requests

from scdn import paths
from scdn.corpus import Corpus
from scdn.e1_fetch import UA
from scdn.e1_quality import build_page_index
from scdn.retrieval_server import E5Encoder

E1 = paths.DATA_ROOT / "e1"
DATASETS = ["hotpotqa", "2wikimultihopqa", "musique", "bamboogle", "nq"]
NS = [1, 3, 5, 10]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-ds", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cand", type=int, default=100)
    ap.add_argument("--page-cand", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    corpus = Corpus()
    page_index = build_page_index(corpus)
    e5 = E5Encoder(paths.E5_BASE_V2, device="cpu")
    index = faiss.read_index(str(paths.WIKI18_FLAT_INDEX), faiss.IO_FLAG_MMAP)
    nbytes = lambda c: len(corpus.contents(int(c)).encode("utf-8"))

    qs = []   # (dataset, chunk_e5 bytes, chunk_llm bytes, top page-cand pages)
    for ds in DATASETS:
        recs = [json.loads(l) for l in open(E1 / f"matrix_llm_{ds}.jsonl")]
        recs = random.Random(args.seed).sample(recs, min(args.per_ds, len(recs)))
        qv = e5.encode_queries([r["question"] for r in recs]).astype(np.float32)
        _, idxs = index.search(qv, args.cand)
        for qi, r in enumerate(recs):
            cand = [int(c) for c in idxs[qi] if c >= 0]
            cand_pages = list({int(corpus.page_ids[c]) for c in cand})
            means = np.vstack([np.vstack([index.reconstruct(int(m)) for m in page_index[p]]).mean(0)
                               for p in cand_pages])
            means /= np.linalg.norm(means, axis=1, keepdims=True) + 1e-9
            order = [cand_pages[i] for i in np.argsort(-(means @ qv[qi]))][: args.page_cand]
            qs.append((ds, sum(map(nbytes, r["chunk_e5"])), sum(map(nbytes, r["chunk_llm"])), order))
        print(f"{ds}: ranked {len(recs)} questions", flush=True)

    # page sizes: wiki-18 text (lower bound) and live HTML (what a client downloads)
    pages = sorted({p for *_, order in qs for p in order})
    text_b = {p: sum(nbytes(m) for m in page_index[p]) for p in pages}
    title = {p: corpus.contents(int(page_index[p][0])).split("\n", 1)[0].strip('"') for p in pages}
    cache_f = E1 / "wiki_html_sizes.jsonl"
    html_b = {}
    if cache_f.exists():
        for l in open(cache_f):
            r = json.loads(l)
            html_b[r["title"]] = r["html_bytes"]
    todo = [t for t in {title[p] for p in pages} if t not in html_b]
    print(f"{len(pages)} distinct pages; fetching {len(todo)} from Wikipedia", flush=True)

    def fetch(t):   # Wikipedia throttles bursts: back off and retry; 404 = gone since 2018
        for i in range(5):
            try:
                r = requests.get("https://en.wikipedia.org/wiki/" + quote(t.replace(" ", "_")),
                                 headers={"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}, timeout=20)
                st = r.status_code
            except requests.RequestException:
                st = None
            if st in (200, 404):
                break
            time.sleep(2 ** i)
        return t, (len(r.content) if st == 200 else None), st   # r.content is decompressed

    with open(cache_f, "a") as f, ThreadPoolExecutor(args.workers) as ex:
        for i, (t, b, st) in enumerate(ex.map(fetch, todo)):
            html_b[t] = b
            if st in (200, 404):   # transient failures are not cached, so a rerun retries them
                f.write(json.dumps({"title": t, "html_bytes": b}) + "\n")
            if (i + 1) % 500 == 0:
                print(f"  fetched {i + 1}/{len(todo)}", flush=True)
    ok = [html_b[title[p]] for p in pages if html_b.get(title[p])]
    miss = sum(1 for p in pages if not html_b.get(title[p]))
    med_html = float(np.median(ok))
    # a page missing on today's Wikipedia (renamed/deleted since 2018) counts at the median size
    H = lambda p: html_b.get(title[p]) or med_html

    out = {"n_questions": len(qs), "per_dataset": args.per_ds, "pages": len(pages),
           "pages_missing_live": miss, "page_html_median_KB": med_html / 1e3,
           "page_text_median_KB": float(np.median(list(text_b.values()))) / 1e3}
    ce5 = np.array([q[1] for q in qs]); cllm = np.array([q[2] for q in qs])
    out["chunk_KB"] = {"e5": ce5.mean() / 1e3, "llm": cllm.mean() / 1e3}
    rows = [(f"e5@{n}", ce5, [sum(H(p) for p in q[3][:n]) for q in qs],
             [sum(text_b[p] for p in q[3][:n]) for q in qs]) for n in NS]
    rows.append((f"llm (reads {args.page_cand} pages)", cllm, [sum(H(p) for p in q[3]) for q in qs],
                 [sum(text_b[p] for p in q[3]) for q in qs]))
    print(f"\n{len(qs)} questions ({args.per_ds}/dataset), {len(pages)} pages, {miss} not on live "
          f"Wikipedia (counted at the median {med_html / 1e3:.0f} KB)")
    print(f"SemDN delivers {ce5.mean() / 1e3:.2f} KB (e5) / {cllm.mean() / 1e3:.2f} KB (reranker) per question")
    print(f"{'page client':26} {'HTML MB':>8} {'x chunk':>8} {'text KB':>8} {'x chunk':>8}")
    out["clients"] = {}
    for name, cb, hb, tb in rows:
        hb, tb = np.array(hb), np.array(tb)
        rec = {"html_MB": hb.mean() / 1e6, "html_ratio": hb.mean() / cb.mean(),
               "text_KB": tb.mean() / 1e3, "text_ratio": tb.mean() / cb.mean()}
        out["clients"][name] = rec
        print(f"{name:26} {rec['html_MB']:8.2f} {rec['html_ratio']:8.0f} {rec['text_KB']:8.1f} {rec['text_ratio']:8.1f}")
    (paths.REPO_ROOT / "outputs" / "e1_matrix_bytes.json").write_text(json.dumps(out, indent=2))
    print("wrote outputs/e1_matrix_bytes.json")


if __name__ == "__main__":
    main()
