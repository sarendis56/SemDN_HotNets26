"""E1-C matrix: page vs chunk × (e5 | e5 -> LLM reranker), confound-free on wiki-18.

Tests whether the page-vs-chunk delivery gap is an artefact of a weak page
representation. Four retrieval configs from the SAME e5 candidate universe (top-100
chunks), the SAME corpus, time-stable QA:

  chunk_e5   : candidates ranked by e5 chunk score -> global top-k
  chunk_llm  : the 100 candidates reranked by Qwen3-Reranker-4B -> global top-k
  page_e5    : pages ranked by mean-pooled e5 chunk vecs -> top-N -> top-k within
  page_llm   : top-30 pages by e5 mean-pool, reranked by the SAME reranker reading the
               WHOLE page (cut only where its 32K context runs out) -> top-N -> top-k within

The reranker and its scoring code are the CRAG iso-budget ones (e4_isobudget.make_llm_scorer).
"top-k within" = best k candidate chunks on the selected pages, by that arm's chunk score,
so every arm hands the reader k chunks.

Outputs per dataset: data/e1/matrix_<tag><ds>.jsonl (per-question top-k chunk ids
per config + delivery gaps); aggregate outputs/e1_matrix_<tag>stats.json.

Run (one process per dataset is fine: the flat index is mmapped, so RAM is shared):
  uv run python -m semdn.e1_matrix --datasets hotpotqa --n 500
"""

import argparse
import json

import faiss
import numpy as np

from semdn import paths
from semdn.corpus import Corpus
from semdn.e1_quality import build_page_index
from semdn.retrieval_server import E5Encoder

NS = [1, 3, 5, 10]


def topk_within(cand, scores_by_c, pages_of, top_pages, k):
    pool = [c for c in cand if pages_of[c] in top_pages]
    return sorted(pool, key=lambda c: -scores_by_c[c])[:k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="hotpotqa,2wikimultihopqa,musique,bamboogle,nq")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--cand", type=int, default=100, help="e5 candidate chunks per query")
    ap.add_argument("--k", type=int, default=5, help="chunks delivered")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--page-cand", type=int, default=30, help="pages the reranker reads")
    ap.add_argument("--tag", default="llm_")
    ap.add_argument("--llm-reranker", default="Qwen/Qwen3-Reranker-4B")
    ap.add_argument("--batch-tokens", type=int, default=32768)
    args = ap.parse_args()

    import random

    corpus = Corpus()
    page_index = build_page_index(corpus)
    e5 = E5Encoder(paths.E5_BASE_V2, device="cpu")
    print("loading Flat index (~64 GB) ...", flush=True)
    index = faiss.read_index(str(paths.WIKI18_FLAT_INDEX), faiss.IO_FLAG_MMAP)
    from semdn.e4_isobudget import make_llm_scorer

    llm_score = make_llm_scorer(args.llm_reranker, "cuda", args.batch_tokens)
    n_cut = n_pages = 0

    def page_text(p):
        return "\n".join(corpus.contents(int(m)) for m in page_index[p])

    all_stats = {}
    for ds in args.datasets.split(","):
        rows = [json.loads(l) for l in open(paths.FLASHRAG_DATASETS / ds / "dev.jsonl")] \
            if (paths.FLASHRAG_DATASETS / ds / "dev.jsonl").exists() \
            else [json.loads(l) for l in open(paths.FLASHRAG_DATASETS / ds / "test.jsonl")]
        rows = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
        questions = [r["question"] for r in rows]
        qv_e5 = e5.encode_queries(questions).astype(np.float32)   # (Q, 768) CPU
        scores, idxs = index.search(qv_e5, args.cand)             # IP scores = e5 chunk scores

        gap = {("e5", N): [] for N in NS}
        gap.update({("llm", N): [] for N in NS})
        fout = open(paths.DATA_ROOT / "e1" / f"matrix_{args.tag}{ds}.jsonl", "w")
        for qi, r in enumerate(rows):
            cand = [int(c) for c in idxs[qi] if c >= 0]
            e5_cs = {int(c): float(s) for c, s in zip(idxs[qi], scores[qi]) if c >= 0}
            pages_of = {c: int(corpus.page_ids[c]) for c in cand}
            cand_pages = list({pages_of[c] for c in cand})
            # e5 page score = mean-pooled chunk vecs (vectorised: reconstruct is a
            # memcpy, one matmul/query — avoids the per-page BLAS-arena storm)
            member_lists = [page_index[p] for p in cand_pages]
            allids = np.concatenate(member_lists)
            V = np.vstack([index.reconstruct(int(m)) for m in allids])
            ends = np.cumsum([len(ml) for ml in member_lists])
            starts = ends - np.array([len(ml) for ml in member_lists])
            means = np.empty((len(cand_pages), V.shape[1]), dtype=np.float32)
            for i, (s, e) in enumerate(zip(starts, ends)):
                means[i] = V[s:e].mean(0)
            means /= (np.sqrt((means * means).sum(1)) + 1e-9)[:, None]
            e5_ps = {p: float(v) for p, v in zip(cand_pages, means @ qv_e5[qi])}
            # LLM reranker: the 100 candidate chunks, and the top page-cand pages by e5
            # mean-pool read whole (we only need top-N<=10 pages)
            q_cs = dict(zip(cand, map(float, llm_score(r["question"], [corpus.contents(c) for c in cand]))))
            pages_q = sorted(cand_pages, key=lambda p: -e5_ps[p])[: args.page_cand]
            texts = [page_text(p) for p in pages_q]
            q_ps = dict(zip(pages_q, map(float, llm_score(r["question"], texts))))
            n_pages += len(texts)
            n_cut += sum(len(llm_score.tok.encode(t, add_special_tokens=False)) > 32000 for t in texts)

            chunk_e5 = sorted(cand, key=lambda c: -e5_cs[c])[: args.k]
            chunk_q = sorted(cand, key=lambda c: -q_cs[c])[: args.k]
            pe5_rank = sorted(cand_pages, key=lambda p: -e5_ps[p])
            pq_rank = sorted(pages_q, key=lambda p: -q_ps[p])

            rec = {"qid": r["id"], "question": r["question"],
                   "golden_answers": r["golden_answers"],
                   "chunk_e5": chunk_e5, "chunk_llm": chunk_q,
                   "page_e5_byN": {}, "page_llm_byN": {}}
            for N in NS:
                pe = set(pe5_rank[:N])
                pq = set(pq_rank[:N])
                rec["page_e5_byN"][str(N)] = topk_within(cand, e5_cs, pages_of, pe, args.k)
                rec["page_llm_byN"][str(N)] = topk_within(cand, q_cs, pages_of, pq, args.k)
                gap[("e5", N)].append(
                    1 - sum(pages_of[c] in pe for c in chunk_e5) / len(chunk_e5))
                gap[("llm", N)].append(
                    1 - sum(pages_of[c] in pq for c in chunk_q) / len(chunk_q))
            fout.write(json.dumps(rec) + "\n")
            if (qi + 1) % 50 == 0:
                print(f"  {ds}: {qi + 1}/{len(rows)}", flush=True)
        fout.close()
        all_stats[ds] = {
            "n": len(rows),
            "delivery_gap": {f"{e}_N{N}": float(np.mean(gap[(e, N)]))
                             for (e, N) in gap},
        }
        print(f"{ds}: gap e5@5={all_stats[ds]['delivery_gap']['e5_N5']:.3f} "
              f"llm@5={all_stats[ds]['delivery_gap']['llm_N5']:.3f} "
              f"(pages near the 32K reranker limit so far: {n_cut}/{n_pages})", flush=True)

    out = paths.REPO_ROOT / "outputs" / f"e1_matrix_{args.tag}{args.datasets.replace(',', '+')}_stats.json"
    out.write_text(json.dumps(all_stats, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
