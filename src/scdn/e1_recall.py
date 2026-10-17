"""E1-C — the delivery/recall gap: client-side chunking can only rank chunks from
the pages it downloaded; chunks on un-selected pages are structurally invisible.

Confound-free: same corpus (wiki-18), same encoder (e5), time-stable questions
(HotpotQA). Both sides have the full index — only the ranking/delivery granularity
differs. All ranking is done from the Flat index on CPU (exact search +
reconstruct), so no GPU is needed here; answer quality (e1_recall_answer.py) does.

  client (page path): rank PAGES by whole-page relevance (mean-pooled chunk
      vectors, same encoder) -> top-N pages -> the client may chunk+rank only
      WITHIN those pages -> top-k chunks.
  SCDN (chunk path) : rank CHUNKS globally -> top-k.

Outputs:
  outputs/e1_recall_stats.json   delivery gap & bytes vs N
  data/e1/recall_contexts.jsonl  per-question chunk texts for the reader step

Run:  uv run python -m scdn.e1_recall --n 300
"""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np

from scdn import paths
from scdn.corpus import Corpus
from scdn.e1_quality import build_page_index
from scdn.retrieval_server import E5Encoder

PAGE_BYTES = 760_616 // 5   # E1 lightweight HTML per page (~152 KB)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--cand-chunks", type=int, default=200, help="M: global candidate pool")
    ap.add_argument("--k", type=int, default=5, help="chunks delivered to the reader")
    ap.add_argument("--ns", default="1,2,3,5,10,20", help="N pages the client downloads")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    Ns = [int(x) for x in args.ns.split(",")]

    import random

    corpus = Corpus()
    page_index = build_page_index(corpus)
    enc = E5Encoder(paths.E5_BASE_V2, device="cpu")
    print("loading Flat index (~64 GB, CPU) ...", flush=True)
    index = faiss.read_index(str(paths.WIKI18_FLAT_INDEX))
    print(f"flat ntotal={index.ntotal:,}", flush=True)

    rows = [json.loads(l) for l in open(paths.FLASHRAG_DATASETS / args.dataset / "dev.jsonl")]
    rows = random.Random(args.seed).sample(rows, args.n)

    # delivery gap per page-ranking pooling method: frac of SCDN top-k chunks on
    # pages the client did NOT select (structurally invisible to client chunking)
    gap = {pool: {N: [] for N in Ns} for pool in ("mean", "max", "lead")}
    ctx_path = paths.DATA_ROOT / "e1" / "recall_contexts.jsonl"
    fctx = open(ctx_path, "w")

    for qi, r in enumerate(rows):
        q = r["question"]
        qv = enc.encode_queries([q])[0].astype(np.float32)
        scores, idx = index.search(qv[None], args.cand_chunks)
        cand = [int(c) for c in idx[0] if c >= 0]
        cand_score = {c: float(s) for c, s in zip(cand, scores[0])}
        scdn_topk = cand[: args.k]                      # global chunk ranking

        # rank candidate pages three ways (sensitivity band), all from the same
        # reconstructed vectors with <=2 matmuls/query (avoids the BLAS storm):
        #   mean = whole-page relevance (realistic);  max = page's best chunk
        #   (client-optimal bound, ~= chunk ranking);  lead = first passage.
        D = qv.shape[0]
        pages_list = list({int(corpus.page_ids[c]) for c in cand})
        member_lists = [page_index[p] for p in pages_list]
        all_ids = np.concatenate(member_lists)
        V = np.empty((len(all_ids), D), dtype=np.float32)
        for j, m in enumerate(all_ids):
            V[j] = index.reconstruct(int(m))
        member_scores = V @ qv                          # one matmul/query
        ends = np.cumsum([len(ml) for ml in member_lists])
        starts = ends - np.array([len(ml) for ml in member_lists])
        means = np.empty((len(pages_list), D), dtype=np.float32)
        max_s = np.empty(len(pages_list), dtype=np.float32)
        lead_s = np.empty(len(pages_list), dtype=np.float32)
        for i, (s, e) in enumerate(zip(starts, ends)):
            means[i] = V[s:e].mean(0)
            max_s[i] = member_scores[s:e].max()
            lead_s[i] = member_scores[s]
        means /= (np.sqrt((means * means).sum(1)) + 1e-9)[:, None]
        mean_s = means @ qv                              # one matmul/query
        ranked = {
            "mean": [pages_list[i] for i in np.argsort(-mean_s)],
            "max": [pages_list[i] for i in np.argsort(-max_s)],
            "lead": [pages_list[i] for i in np.argsort(-lead_s)],
        }

        client_chunks_byN = {}
        for pool_name, ranked_pages in ranked.items():
            for N in Ns:
                top_pages = set(ranked_pages[:N])
                reach = sum(1 for c in scdn_topk if int(corpus.page_ids[c]) in top_pages)
                gap[pool_name][N].append(1 - reach / len(scdn_topk))
                if pool_name == "mean":  # primary: save client chunk set for the reader
                    poolc = [c for c in cand if int(corpus.page_ids[c]) in top_pages]
                    client_chunks_byN[N] = sorted(poolc, key=lambda c: -cand_score[c])[: args.k]

        fctx.write(json.dumps({
            "qid": r["id"], "question": q, "golden_answers": r["golden_answers"],
            "scdn_topk": scdn_topk,
            "client_topk_byN": {str(N): client_chunks_byN[N] for N in Ns},
        }) + "\n")
        if (qi + 1) % 25 == 0:
            print(f"  {qi + 1}/{len(rows)} done", flush=True)
    fctx.close()

    stats = {
        "n": len(rows), "cand_chunks": args.cand_chunks, "k": args.k, "Ns": Ns,
        "delivery_gap_vs_N": {
            pool: {N: float(np.mean(gap[pool][N])) for N in Ns} for pool in gap
        },
        "client_bytes_vs_N_MB": {N: N * PAGE_BYTES / 1e6 for N in Ns},
        "scdn_delivery_bytes_KB": args.k * 639 / 1e3,
        "contexts": str(ctx_path),
    }
    (paths.REPO_ROOT / "outputs" / "e1_recall_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
