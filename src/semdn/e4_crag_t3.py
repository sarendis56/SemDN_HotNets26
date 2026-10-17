"""E4-CRAG Task-3: pool-DEPTH sweep on the 50-page provided corpus. Task-1 (5 pages)
couldn't show how quality trends as MORE pages enter the pool; Task-3 (≤50 clean,
pre-crawled pages/q) can.

Per question (reservoir-sampled, streaming the tens-of-GB jsonl to avoid OOM): extract
text from the provided pages (trafilatura, single-threaded → safe from the lxml-under-
threads segfault), structnt-chunk, e5-rank chunks; for K in {1,3,5,10,20,50} emit
condition `page@K` = top-3 chunks drawn from the first K usable pages (provided rank).
The trend page@1→page@50 shows whether the deep-tail penalty appears/grows at scale in
a clean corpus. Writes a pool jsonl for e4_answer/e4_judge (GENERATION only).

Run:  CUDA_VISIBLE_DEVICES=1 uv run python -m semdn.e4_crag_t3 --stage build --n 300
"""

import argparse
import bz2
import glob
import json
import math
import os
import random
import re
from collections import Counter

import numpy as np

from semdn import paths
from semdn.e4_granularity import _make_chunker

E4 = paths.DATA_ROOT / "e4"

# ---- shallow-index rankers (Task 1: the SemDN dense index is contrasted against a
# lexical/search-engine index that has no embeddings). Dependency-free so the deep-vs-
# shallow comparison adds no new package and stays CPU-only. ----
_BM25_TOK = re.compile(r"[a-z0-9]+")


def _bm25_tok(s: str) -> list[str]:
    return _BM25_TOK.findall(str(s).lower())


def bm25_scores(question: str, chunks: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """BM25-Okapi score of each chunk against the question, corpus = this question's
    chunk pool (apples-to-apples with how e5 ranks within the same pool). The lexical
    'shallow index' baseline for the dense e5 index."""
    docs = [_bm25_tok(c) for c in chunks]
    n = len(docs)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    dl = np.array([len(d) for d in docs], dtype=np.float32)
    avgdl = float(dl.mean()) or 1.0
    tfs, df = [], Counter()
    for d in docs:
        tf = Counter(d)
        tfs.append(tf)
        df.update(tf.keys())
    qset = set(_bm25_tok(question))
    idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in qset if t in df}
    scores = np.zeros(n, dtype=np.float32)
    for i, tf in enumerate(tfs):
        s = 0.0
        for t in qset:
            f = tf.get(t, 0)
            if f and t in idf:
                s += idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * dl[i] / avgdl))
        scores[i] = s
    return scores


def provided_scores(chunks: list[str]) -> np.ndarray:
    """Shallow search-engine floor: no semantic ranking at all — prefer the earliest
    chunks (top provided/search-rank page, document order). With _depth_conditions this
    yields 'read the top result top-down', the genuine shallow-index baseline."""
    return -np.arange(len(chunks), dtype=np.float32)
POOL_T3 = E4 / "pool_crag_t3.jsonl"
CHUNKS_CACHE = E4 / "chunks_crag_t3.jsonl"   # extracted+chunked pages, encoder-independent (structnt)


def _cache_path(chunker, corpus="t3"):
    """structnt on t3 keeps the original filename (back-compat); everything else is
    keyed by corpus + chunker so a Task-1 run does not clobber Task-3 files."""
    if corpus == "t3" and chunker == "structnt":
        return CHUNKS_CACHE
    return E4 / f"chunks_crag_{corpus}_{chunker}.jsonl"


def _pool_path(chunker, encoder, corpus="t3"):
    if corpus == "t3" and chunker == "structnt":
        return E4 / f"pool_crag_t3_{encoder}.jsonl"
    return E4 / f"pool_crag_{corpus}_{chunker}_{encoder}.jsonl"
# Task-3 extracts to a directory of ~10 jsonl shards (~50GB total), 50 pages/q.
DEFAULT_SRC = paths.DATA_ROOT / "datasets" / "crag" / "crag_task_3_dev_v4"
PAGE_KS = [1, 3, 5, 10, 20, 50]
TOPK = 3


def _iter_lines(src):
    """Stream lines across a single file or all *.jsonl[.bz2] shards in a directory."""
    src = str(src)
    if os.path.isdir(src):
        files = sorted(glob.glob(os.path.join(src, "*.jsonl")) +
                       glob.glob(os.path.join(src, "*.jsonl.bz2")))
    else:
        files = [src]
    for fp in files:
        op = bz2.open if fp.endswith(".bz2") else open
        with op(fp, "rt", encoding="utf-8") as f:
            for line in f:
                yield line


def reservoir_sample(src, n, seed):
    """Uniform n-sample of valid-golden questions in one streaming pass (holds n full
    records ~ a few GB; never the whole corpus)."""
    rng = random.Random(seed)
    res, seen = [], 0
    for line in _iter_lines(src):
        line = line.strip()
        if line:
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
                   "pages": [{"title": s.get("page_name", ""), "html": s.get("page_result", "") or "",
                              "snippet": s.get("page_snippet", "") or ""}
                             for s in (r.get("search_results") or [])]}
            if len(res) < n:
                res.append(rec)
            else:
                j = rng.randint(0, seen)
                if j < n:
                    res[j] = rec
            seen += 1
            if seen % 500 == 0:
                print(f"  ...streamed {seen} valid questions", flush=True)
    print(f"reservoir: sampled {len(res)} of {seen} valid questions", flush=True)
    return res


def build(args):
    import trafilatura

    from semdn.retrieval_server import E5Encoder
    enc = E5Encoder(str(paths.E5_BASE_V2), device=args.device)
    chunker = _make_chunker("structnt")
    src = args.src

    qs = reservoir_sample(src, args.n, args.question_seed)
    done = {json.loads(l)["qid"] for l in open(POOL_T3)} if POOL_T3.exists() else set()
    todo = [q for q in qs if q["qid"] not in done]
    print(f"t3 build: {len(todo)} questions ({len(done)} cached), depths={PAGE_KS}", flush=True)
    fout = open(POOL_T3, "a")
    for idx, q in enumerate(todo):
        ctext, crank, usable = [], [], 0           # chunk text, usable-page rank, n usable
        for pg in q["pages"][:max(PAGE_KS)]:
            if not pg["html"]:
                continue
            try:
                txt = trafilatura.extract(pg["html"], include_comments=False, include_tables=True) or ""
            except Exception:
                txt = ""
            if len(txt) < 200:
                continue
            usable += 1
            for ch in chunker(txt, pg["title"]):
                ctext.append(ch); crank.append(usable)
        rec = {"qid": q["qid"], "question": q["question"], "golden_answers": q["golden_answers"],
               "metadata": q["metadata"], "n_pages": len(q["pages"]), "n_usable": usable,
               "n_chunks": len(ctext), "conds": {}}
        if ctext:
            cr = np.array(crank)
            cemb = enc.encode_passages(ctext).astype(np.float32)
            qv = enc.encode_queries([q["question"]]).astype(np.float32)[0]
            scores = cemb @ qv
            for K in PAGE_KS:
                mask = np.where(cr <= K)[0]
                if mask.size:
                    top = mask[np.argsort(-scores[mask])[:TOPK]]
                    rec["conds"][f"page@{K}"] = {"chunk_texts": [ctext[j] for j in top],
                                                 "src_ranks": [int(cr[j]) for j in top]}
                else:
                    rec["conds"][f"page@{K}"] = {"chunk_texts": [], "src_ranks": []}
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(todo)}] {q['qid'][:8]}: {usable} usable pages, {len(ctext)} chunks",
                  flush=True)
    fout.close()
    print(f"t3 build done -> {POOL_T3}", flush=True)


def _depth_conditions(ctext, crank, scores, page_ks=PAGE_KS):
    """page@K = top-3 chunks from the first K usable pages (provided rank)."""
    cr = np.array(crank)
    conds = {}
    for K in page_ks:
        mask = np.where(cr <= K)[0]
        if mask.size:
            top = mask[np.argsort(-scores[mask])[:TOPK]]
            conds[f"page@{K}"] = {"chunk_texts": [ctext[j] for j in top],
                                  "src_ranks": [int(cr[j]) for j in top]}
        else:
            conds[f"page@{K}"] = {"chunk_texts": [], "src_ranks": []}
    return conds


def stage_cache(args):
    """Extract+chunk the ≤50 pages per question ONCE (encoder-independent), so the
    different encoders can be ranked cheaply afterwards without re-extracting."""
    import trafilatura
    chunker = _make_chunker(args.chunker)
    cache_path = _cache_path(args.chunker, args.corpus)
    qs = reservoir_sample(args.src, args.n, args.question_seed)
    done = {json.loads(l)["qid"] for l in open(cache_path)} if cache_path.exists() else set()
    todo = [q for q in qs if q["qid"] not in done]
    print(f"cache ({args.corpus}/{args.chunker}): {len(todo)} questions to extract ({len(done)} cached)", flush=True)
    fout = open(cache_path, "a")
    for idx, q in enumerate(todo):
        ctext, crank, usable = [], [], 0
        for pg in q["pages"][:max(args.page_ks)]:
            if not pg["html"]:
                continue
            try:
                txt = trafilatura.extract(pg["html"], include_comments=False, include_tables=True) or ""
            except Exception:
                txt = ""
            if len(txt) < 200:
                continue
            usable += 1
            for ch in chunker(txt, pg["title"]):
                ctext.append(ch); crank.append(usable)
        fout.write(json.dumps({"qid": q["qid"], "question": q["question"],
                               "golden_answers": q["golden_answers"], "metadata": q["metadata"],
                               "n_usable": usable, "chunks": ctext, "ranks": crank}) + "\n")
        fout.flush()
        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(todo)}] {usable} usable pages, {len(ctext)} chunks", flush=True)
    fout.close()
    print(f"cache ({args.corpus}/{args.chunker}) done -> {cache_path}", flush=True)


def stage_rank(args):
    """Score the cached chunks with the chosen encoder and write the depth-sweep pool."""
    import numpy as np
    cache = [json.loads(l) for l in open(_cache_path(args.chunker, args.corpus))]
    if args.encoder in ("e5", "qwen3"):
        from semdn.e4_granularity import _make_encoder
        embed_docs, embed_queries = _make_encoder(args.encoder, args.device)

        def score(question, chunks):
            return embed_docs(chunks).astype(np.float32) @ embed_queries([question]).astype(np.float32)[0]
    elif args.encoder == "ce":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(paths.model(args.reranker))
        model = AutoModelForSequenceClassification.from_pretrained(
            paths.model(args.reranker), torch_dtype=torch.float16).to(args.device).eval()

        @torch.no_grad()
        def score(question, chunks, batch=64):
            out = []
            for i in range(0, len(chunks), batch):
                pairs = [[question, c] for c in chunks[i:i + batch]]
                inp = tok(pairs, padding=True, truncation=True, max_length=512,
                          return_tensors="pt").to(args.device)
                out.extend(model(**inp).logits.view(-1).float().cpu().numpy().tolist())
            return np.array(out)
    elif args.encoder == "qwen3rr":       # Qwen3-Reranker-0.6B: P(yes) from the yes/no logit
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        mid = "Qwen/Qwen3-Reranker-0.6B"
        tok = AutoTokenizer.from_pretrained(paths.model(mid), padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(paths.model(mid), torch_dtype=torch.float16).to(args.device).eval()
        yes_id, no_id = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
        pre = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
               'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
        suf = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        instruct = "Given a web search query, retrieve passages that answer the query"

        @torch.no_grad()
        def score(question, chunks, batch=32):
            texts = [f"{pre}<Instruct>: {instruct}\n<Query>: {question}\n<Document>: {c}{suf}" for c in chunks]
            out = []
            for i in range(0, len(texts), batch):
                inp = tok(texts[i:i+batch], padding=True, truncation=True, max_length=640, return_tensors="pt").to(args.device)
                logits = model(**inp).logits[:, -1, :]
                p = torch.softmax(torch.stack([logits[:, no_id], logits[:, yes_id]], dim=-1), dim=-1)[:, 1]
                out.extend(p.float().cpu().numpy().tolist())
            return np.array(out)
    elif args.encoder == "bm25":          # lexical shallow-index baseline (CPU, no GPU)
        def score(question, chunks):
            return bm25_scores(question, chunks)
    elif args.encoder == "provided":      # search-rank order floor (no ranking; CPU)
        def score(question, chunks):
            return provided_scores(chunks)
    else:
        raise ValueError(args.encoder)

    out_path = _pool_path(args.chunker, args.encoder, args.corpus)
    snips = {}
    if args.hybrid:   # inject the CRAG page snippets as rank-0 ranking co-candidates
        out_path = out_path.with_name(out_path.stem + "_hybrid.jsonl")
        spath = E4 / f"pool_crag_{args.corpus}_snippet.jsonl"
        for l in open(spath):
            d = json.loads(l)
            snips[d["qid"]] = d.get("conds", {}).get("page@10", {}).get("chunk_texts", [])
        print(f"hybrid: loaded snippets for {len(snips)} questions from {spath.name}", flush=True)
    done = {json.loads(l)["qid"] for l in open(out_path)} if out_path.exists() else set()
    todo = [r for r in cache if r["qid"] not in done]
    print(f"rank ({args.corpus}/{args.chunker}/{args.encoder}{'/hybrid' if args.hybrid else ''}): "
          f"{len(todo)} questions ({len(done)} cached)", flush=True)
    fout = open(out_path, "a")
    npick = ntot = 0   # snippet-pick rate at the deepest depth
    for idx, r in enumerate(todo):
        ctext, crank = list(r["chunks"]), list(r["ranks"])
        if args.hybrid:
            sn = snips.get(r["qid"], [])
            ctext = ctext + sn; crank = crank + [0] * len(sn)
        rec = {"qid": r["qid"], "question": r["question"], "golden_answers": r["golden_answers"],
               "metadata": r["metadata"], "n_usable": r["n_usable"], "n_chunks": len(ctext), "conds": {}}
        if ctext:
            rec["conds"] = _depth_conditions(ctext, crank, score(r["question"], ctext), args.page_ks)
            if args.hybrid:
                sr = rec["conds"].get(f"page@{max(args.page_ks)}", {}).get("src_ranks", [])
                npick += sum(1 for s in sr if s == 0); ntot += len(sr)
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{len(todo)}] ranked", flush=True)
    fout.close()
    if args.hybrid and ntot:
        print(f"hybrid snippet-pick rate (deepest): {npick}/{ntot} ({100*npick/ntot:.0f}%)", flush=True)
    print(f"rank done -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="build", choices=["build", "cache", "rank"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--question-seed", type=int, default=0)
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--encoder", default="e5", choices=["e5", "qwen3", "ce", "bm25", "provided", "qwen3rr"])
    ap.add_argument("--chunker", default="structnt", choices=["structnt", "structure", "fixed", "coherent"])
    ap.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--corpus", default="t3", help="path tag so a Task-1 run won't clobber Task-3 files")
    ap.add_argument("--page-ks", default="1,3,5,10,20,50", help="depth-sweep page cutoffs (comma list)")
    ap.add_argument("--hybrid", action="store_true",
                    help="rank stage: add CRAG page snippets as co-candidates next to the chunks")
    args = ap.parse_args()
    args.page_ks = [int(k) for k in args.page_ks.split(",")]
    E4.mkdir(parents=True, exist_ok=True)
    if args.stage == "build":
        build(args)
    elif args.stage == "cache":
        stage_cache(args)
    elif args.stage == "rank":
        stage_rank(args)


if __name__ == "__main__":
    main()
