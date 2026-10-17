"""Stronger-pipeline version of the answer-location check (§A's "when the answer is
present, how deep is it?"). Question: e5 and BM25 fail to surface the answers that sit
deep in the pool — do stronger NEURAL rankers (Qwen3-Embedding, Qwen3-Reranker, or the
BGE cross-encoder) do better? Run on the COHERENT chunks.

Bi-encoders (e5, qwen3) score every chunk. Rerankers (ce, qwen3rr) are used the realistic
way — they re-rank the e5 top-`n_rerank` candidates (a chunk e5 didn't recall is counted
as not surfaced); we also print e5 recall@n_rerank so that conditioning is visible.

Per ranker we report, on the questions whose answer is present somewhere in the <=50 pages:
  deep@K surface = of questions whose answer is ONLY beyond the top-K provided pages, the
                   fraction where the ranker puts a gold chunk in the top-3 it would show.
  top3 (all)     = gold chunk reaches the top-3 over all answerable questions.
  med rank       = median rank of the best gold chunk.

Run:  CUDA_VISIBLE_DEVICES=1 uv run python -m scdn.e4_locality_strong --chunker coherent
"""

import argparse
import json

import numpy as np

from scdn import paths
from scdn.e4_answer_locality import gold_in_text
from scdn.e4_crag_t3 import _cache_path, bm25_scores

TOPK = 3
SHALLOW_KS = [1, 3, 5]
QWEN3RR_INSTRUCT = "Given a web search query, retrieve passages that answer the query"


def _bi_scores(name, device, recs):
    from scdn.e4_granularity import _make_encoder
    ed, eq = _make_encoder(name, device)
    out = []
    for r in recs:
        ce = ed(r["chunks"]).astype(np.float32)
        qv = eq([r["question"]]).astype(np.float32)[0]
        out.append(ce @ qv)
    return out


def _ce_scores(device, recs, cand):
    """BGE cross-encoder over the e5-recall candidates (others -> -inf)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    rk = "BAAI/bge-reranker-v2-m3"
    tok = AutoTokenizer.from_pretrained(paths.model(rk))
    model = AutoModelForSequenceClassification.from_pretrained(paths.model(rk), torch_dtype=torch.float16).to(device).eval()
    out = []
    with torch.no_grad():
        for r, idx in zip(recs, cand):
            s = np.full(len(r["chunks"]), -1e9, dtype=np.float32)
            pairs = [[r["question"], r["chunks"][j]] for j in idx]
            for i in range(0, len(pairs), 64):
                inp = tok(pairs[i:i+64], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                sc = model(**inp).logits.view(-1).float().cpu().numpy()
                for k, j in enumerate(idx[i:i+64]):
                    s[j] = sc[k]
            out.append(s)
    return out


def _qwen3rr_scores(device, recs, cand, model_id):
    """Qwen3-Reranker over the e5-recall candidates: P(yes) from the yes/no logit."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(paths.model(model_id), padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(paths.model(model_id), torch_dtype=torch.float16).to(device).eval()
    yes_id = tok.convert_tokens_to_ids("yes")
    no_id = tok.convert_tokens_to_ids("no")
    pre = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
           'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
    suf = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    out = []
    with torch.no_grad():
        for r, idx in zip(recs, cand):
            s = np.full(len(r["chunks"]), -1e9, dtype=np.float32)
            texts = [f"{pre}<Instruct>: {QWEN3RR_INSTRUCT}\n<Query>: {r['question']}\n<Document>: {r['chunks'][j]}{suf}" for j in idx]
            for i in range(0, len(texts), 32):
                inp = tok(texts[i:i+32], padding=True, truncation=True, max_length=640, return_tensors="pt").to(device)
                logits = model(**inp).logits[:, -1, :]      # left-padded -> last col is last real token
                ys, ns = logits[:, yes_id], logits[:, no_id]
                p = torch.softmax(torch.stack([ns, ys], dim=-1), dim=-1)[:, 1].float().cpu().numpy()
                for k, j in enumerate(idx[i:i+32]):
                    s[j] = p[k]
            out.append(s)
    return out


def _best_gold_rank(scores, gmask):
    order = np.argsort(-scores)
    for pos, j in enumerate(order):
        if gmask[j]:
            return pos + 1
    return len(scores) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunker", default="coherent")
    ap.add_argument("--corpus", default="t3", help="t3 (50-page) or t1 (5-page) chunk cache")
    ap.add_argument("--rankers", default="e5,bm25,qwen3,ce,qwen3rr")
    ap.add_argument("--n-rerank", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--qwen3rr", default="Qwen/Qwen3-Reranker-0.6B")
    ap.add_argument("--out", default=str(paths.DATA_ROOT / "e4" / "locality_strong.json"))
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(_cache_path(args.chunker, args.corpus))]
    recs = [r for r in recs if r["chunks"]]
    golds = [np.array([gold_in_text(c, r["golden_answers"]) for c in r["chunks"]]) for r in recs]
    ranks = [np.array(r["ranks"]) for r in recs]
    has = np.array([g.any() for g in golds])
    n = len(recs)
    print(f"{n} questions; {has.mean():.3f} have a gold chunk present (recall ceiling)", flush=True)

    rankers = args.rankers.split(",")
    scores = {}
    e5 = _bi_scores("e5", args.device, recs) if any(r in rankers for r in ("e5", "ce", "qwen3rr")) else None
    if "e5" in rankers:
        scores["e5"] = e5
    cand = [np.argsort(-e5[i])[:args.n_rerank] for i in range(n)] if e5 is not None else None
    if "bm25" in rankers:
        scores["bm25"] = [bm25_scores(r["question"], r["chunks"]) for r in recs]
    if "qwen3" in rankers:
        scores["qwen3"] = _bi_scores("qwen3", args.device, recs)
    if "ce" in rankers:
        scores["ce"] = _ce_scores(args.device, recs, cand)
    if "qwen3rr" in rankers:
        scores["qwen3rr"] = _qwen3rr_scores(args.device, recs, cand, args.qwen3rr)

    # e5 recall@n_rerank among answerable-with-deep golds (conditioning for the rerankers)
    rec_at = np.mean([bool(golds[i][cand[i]].any()) for i in range(n) if has[i]]) if cand is not None else float("nan")
    print(f"e5 recall@{args.n_rerank} (gold in candidate set | answerable): {rec_at:.3f}\n", flush=True)

    def surfaces(sc, g):
        return bool(g[np.argsort(-sc)[:TOPK]].any())

    print(f"{'ranker':9} | {'top3(ans)':>9} | {'medRank':>7} | "
          + " | ".join(f"deep@{K}" for K in SHALLOW_KS))
    summary = {"n": n, "recall_ceiling": float(has.mean()), "e5_recall_at_n": float(rec_at), "rankers": {}}
    for rk in rankers:
        sc = scores[rk]
        ans = [i for i in range(n) if has[i]]
        top3 = np.mean([surfaces(sc[i], golds[i]) for i in ans])
        medr = np.median([_best_gold_rank(sc[i], golds[i]) for i in ans])
        deep = {}
        for K in SHALLOW_KS:
            do = [i for i in range(n) if has[i] and not (golds[i] & (ranks[i] <= K)).any()]
            deep[K] = (np.mean([surfaces(sc[i], golds[i]) for i in do]) if do else float("nan"))
        print(f"{rk:9} | {top3:9.3f} | {medr:7.0f} | "
              + " | ".join(f"{deep[K]:6.3f}" for K in SHALLOW_KS))
        summary["rankers"][rk] = {"top3_answerable": float(top3), "median_gold_rank": float(medr),
                                  "deep_surface": {str(K): (None if np.isnan(deep[K]) else float(deep[K])) for K in SHALLOW_KS}}
    json.dump(summary, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
