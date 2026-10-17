"""Task 3 — utility-aware ranking (the O1 lever). Task 1 showed meaning-similarity (e5)
ties keyword (BM25) on CRAG quality, and the answer-location check showed both fail on
the buried tail: they rank by *similarity*, but the noise is similar-but-useless content.
This ranks candidate chunks by *usefulness for answering* instead — an LLM scores each
"does this passage help answer the question?" — the faithful test of O1 at retrieval time.

Two-stage (the realistic cheap-recall -> utility-rerank design):
  candidates: e5 over the (coherent) chunks -> keep the top-N candidates / question.
  rank      : an LLM scores each candidate 0-3 for answerability; the pool then carries
              two conditions on the SAME N candidates so the comparison is paired and
              isolates the ranking method:
                e5   = top-3 by e5 similarity      (the incumbent)
                util = top-3 by LLM utility score  (tie-break e5)

Run:  CUDA_VISIBLE_DEVICES=1 uv run python -m semdn.e4_utility_rank --stage candidates
      CUDA_VISIBLE_DEVICES=1 uv run python -m semdn.e4_utility_rank --stage rank
"""

import argparse
import json
import re

import numpy as np

from semdn import paths
from semdn.e4_crag_t3 import _cache_path

E4 = paths.DATA_ROOT / "e4"
CAND = E4 / "utility_candidates.jsonl"
POOL = E4 / "pool_crag_t3_utility.jsonl"
TOPK = 3

USYS = "You judge whether a passage helps answer a question."
UUSER = """Question: {q}

Passage: {c}

Does this passage contain information that helps answer the question? Reply with ONE digit:
0 = irrelevant, 1 = same topic but no answer, 2 = partial answer, 3 = directly answers."""


def stage_candidates(args):
    """e5-rank the coherent chunks; keep the top-N candidate chunks per question."""
    from semdn.e4_granularity import _make_encoder
    embed_docs, embed_queries = _make_encoder("e5", args.device)
    recs = [json.loads(l) for l in open(_cache_path(args.chunker))]
    fout = open(CAND, "w")
    for i, r in enumerate(recs):
        chunks = r["chunks"]
        if not chunks:
            continue
        cemb = embed_docs(chunks).astype(np.float32)
        qv = embed_queries([r["question"]]).astype(np.float32)[0]
        sc = cemb @ qv
        top = np.argsort(-sc)[:args.n_cand]
        fout.write(json.dumps({"qid": r["qid"], "question": r["question"],
                               "golden_answers": r["golden_answers"], "metadata": r["metadata"],
                               "n_usable": r["n_usable"],
                               "cand_texts": [chunks[j] for j in top],
                               "e5_scores": [float(sc[j]) for j in top]}) + "\n")
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(recs)}] candidates", flush=True)
    fout.close()
    print(f"candidates -> {CAND}", flush=True)


def _parse_score(text: str) -> int:
    m = re.search(r"[0-3]", text)
    return int(m.group()) if m else 0


def stage_rank(args):
    """LLM-score each candidate's answerability; build a pool with e5 vs util top-3."""
    from vllm import LLM, SamplingParams
    recs = [json.loads(l) for l in open(CAND)]
    llm = LLM(model=paths.model(args.model), dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.90)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=2)

    # one flat batch of (q, candidate) prompts across all questions
    prompts, owner = [], []
    for qi, r in enumerate(recs):
        for c in r["cand_texts"]:
            prompts.append(tok.apply_chat_template(
                [{"role": "system", "content": USYS},
                 {"role": "user", "content": UUSER.format(q=r["question"], c=c)}],
                add_generation_prompt=True, tokenize=False))
            owner.append(qi)
    print(f"scoring {len(prompts)} (question,candidate) pairs", flush=True)
    outs = llm.generate(prompts, sp)
    scores = [_parse_score(o.outputs[0].text) for o in outs]

    util_by_q: dict[int, list[float]] = {}
    for qi, s in zip(owner, scores):
        util_by_q.setdefault(qi, []).append(s)

    fout = open(POOL, "w")
    for qi, r in enumerate(recs):
        u = np.array(util_by_q[qi], dtype=np.float32)
        e5 = np.array(r["e5_scores"], dtype=np.float32)
        cand = r["cand_texts"]
        e5_top = np.argsort(-e5)[:TOPK]
        util_top = np.lexsort((-e5, -u))[:TOPK]   # primary util desc, tie-break e5 desc
        fout.write(json.dumps({"qid": r["qid"], "question": r["question"],
            "golden_answers": r["golden_answers"], "metadata": r["metadata"], "n_usable": r["n_usable"],
            "conds": {"e5":   {"chunk_texts": [cand[j] for j in e5_top]},
                      "util": {"chunk_texts": [cand[j] for j in util_top]}}}) + "\n")
    fout.close()
    print(f"pool -> {POOL}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["candidates", "rank"])
    ap.add_argument("--chunker", default="coherent")
    ap.add_argument("--n-cand", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default=str(paths.QWEN25_7B))
    args = ap.parse_args()
    if args.stage == "candidates":
        stage_candidates(args)
    else:
        stage_rank(args)


if __name__ == "__main__":
    main()
