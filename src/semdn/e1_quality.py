"""E1 quality-parity: controlled page-vs-chunk granularity ablation.

Confound control: BOTH arms are built from the SAME retrieval over the SAME
corpus (wiki-18), so there is zero information disparity — only the delivery
granularity differs. At a fixed context-token budget B:

  PAGE arm  fill B with whole ARTICLES (all passages of each top page), best
            page first. Contains >= the info of the chunk arm (the gold passage
            lives inside its article), so it can never lose on information.
  CHUNK arm fill B with the top-ranked CHUNKS (native wiki-18 passages).

A fixed reader LLM answers from each context; we report EM/F1 vs gold and the
context tokens actually used. The claim is supported iff CHUNK matches/beats
PAGE accuracy at equal or lower budget (it wins on density, not information).

Run (retrieval server up; GPU free):
  CUDA_VISIBLE_DEVICES=0 uv run python -m semdn.e1_quality --n 300
"""

import argparse
import json
from pathlib import Path

import numpy as np
import requests

from semdn import paths
from semdn.corpus import Corpus
from semdn.metrics import cover_em, em, f1

SYS = ("You are a helpful assistant. Answer the question using only the provided "
       "context. Reply with a short, direct answer of a few words.")


def build_page_index(corpus: Corpus) -> dict[int, np.ndarray]:
    """page_id -> sorted chunk ids (one pass via argsort over page_ids)."""
    pid = corpus.page_ids
    order = np.argsort(pid, kind="stable")  # groups chunk ids by page
    sorted_pid = pid[order]
    bounds = np.unique(sorted_pid, return_index=True)[1]
    idx = {}
    starts = list(bounds) + [len(order)]
    pages = sorted_pid[bounds]
    for j, pg in enumerate(pages):
        idx[int(pg)] = order[starts[j]:starts[j + 1]]
    return idx


def retrieve(url: str, queries: list[str], topk: int, chunk: int = 256) -> list[list[dict]]:
    out = []
    for i in range(0, len(queries), chunk):
        r = requests.post(f"{url}/retrieve",
                          json={"queries": queries[i:i + chunk], "topk": topk,
                                "return_scores": True}, timeout=1800)
        r.raise_for_status()
        out.extend(r.json()["result"])
    return out


def tok_trunc(tokenizer, text: str, budget: int) -> tuple[str, int]:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) <= budget:
        return text, len(ids)
    return tokenizer.decode(ids[:budget]), budget


def build_contexts(hit: list[dict], corpus: Corpus, page_index, tokenizer, budget: int):
    """Return (chunk_ctx, chunk_tokens, page_ctx, page_tokens) for one question."""
    # CHUNK: top chunks until budget
    chunk_parts, ctoks = [], 0
    for h in hit:
        t = h["document"]["contents"]
        ids = tokenizer(t, add_special_tokens=False).input_ids
        if ctoks + len(ids) > budget:
            chunk_parts.append(tokenizer.decode(ids[: budget - ctoks]))
            ctoks = budget
            break
        chunk_parts.append(t)
        ctoks += len(ids)
    # PAGE: whole articles of the top pages (dedup, ranked by first appearance)
    seen, page_parts, ptoks = set(), [], 0
    for h in hit:
        pid = int(corpus.page_ids[h["idx"]])
        if pid in seen:
            continue
        seen.add(pid)
        article = "\n".join(corpus.contents(int(c)) for c in page_index[pid])
        ids = tokenizer(article, add_special_tokens=False).input_ids
        if ptoks + len(ids) > budget:
            page_parts.append(tokenizer.decode(ids[: budget - ptoks]))
            ptoks = budget
            break
        page_parts.append(article)
        ptoks += len(ids)
    return "\n\n".join(chunk_parts), ctoks, "\n\n".join(page_parts), ptoks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=paths.QWEN25_7B)
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--retrieval-url", default="http://127.0.0.1:8000")
    ap.add_argument("--topk-retrieve", type=int, default=40)
    ap.add_argument("--budgets", default="256,512,1024,2048")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=paths.DATA_ROOT / "e1" / "quality.jsonl")
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]

    import random

    from vllm import LLM, SamplingParams

    requests.get(f"{args.retrieval_url}/health", timeout=10).raise_for_status()
    corpus = Corpus()
    print("building page index ...", flush=True)
    page_index = build_page_index(corpus)

    rows = [json.loads(l) for l in open(paths.FLASHRAG_DATASETS / args.dataset / "dev.jsonl")]
    rows = random.Random(args.seed).sample(rows, args.n)
    questions = [r["question"] for r in rows]
    golden = [r["golden_answers"] for r in rows]
    print(f"{len(questions)} questions; retrieving top-{args.topk_retrieve} ...", flush=True)
    hits = retrieve(args.retrieval_url, questions, args.topk_retrieve)

    llm = LLM(model=paths.model(str(args.model)), dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.85, seed=args.seed)
    tok = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=64, stop=["\n"])

    def ask(contexts: list[str]) -> list[str]:
        prompts = []
        for ctx, q in zip(contexts, questions):
            msg = [{"role": "system", "content": SYS},
                   {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"}]
            prompts.append(tok.apply_chat_template(msg, add_generation_prompt=True,
                                                   tokenize=False))
        return [o.outputs[0].text.strip() for o in llm.generate(prompts, sampling)]

    results = []
    fout = open(args.out, "w")
    for B in budgets:
        cctx, ctoks, pctx, ptoks = [], [], [], []
        for hit in hits:
            cc, ct, pc, pt = build_contexts(hit, corpus, page_index, tok, B)
            cctx.append(cc); ctoks.append(ct); pctx.append(pc); ptoks.append(pt)
        for arm, ctx, toks in (("chunk", cctx, ctoks), ("page", pctx, ptoks)):
            ans = ask(ctx)
            ems = [em(a, g) for a, g in zip(ans, golden)]
            cems = [cover_em(a, g) for a, g in zip(ans, golden)]
            f1s = [f1(a, g) for a, g in zip(ans, golden)]
            rec = {"budget": B, "arm": arm, "n": len(ans),
                   "em": float(np.mean(ems)), "cover_em": float(np.mean(cems)),
                   "f1": float(np.mean(f1s)), "mean_ctx_tokens": float(np.mean(toks))}
            results.append(rec)
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            print(f"B={B:5d} {arm:5s} EM={rec['em']:.3f} coverEM={rec['cover_em']:.3f} "
                  f"F1={rec['f1']:.3f} tokens={rec['mean_ctx_tokens']:.0f}", flush=True)
    fout.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
