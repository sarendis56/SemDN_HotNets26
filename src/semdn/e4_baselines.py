"""No-chunking RAG baselines from recent agentic-web-search papers, to sit in the same
quality table as chunking and no-RAG (same questions, same reader, same CRAG judge):

  snippet : feed the reader only the TITLE + SERP SNIPPET of the top-K results — no fetch,
            no chunking (ZeroSearch / Tongyi-style: title+snippet is the entire context).
  summary : feed the reader an LLM SUMMARY of each of the top-K pages, concatenated — fetch
            but no chunking (R1-Searcher++ style: GPT-4o-mini page summaries; we use a local
            Qwen2.5-7B summarizer as the stand-in).

Both write a pool with depth conds page@K (top-K results, provided/search rank order) so they
slot straight into the depth tables next to the chunking arms. summary reconstructs each page's
text from the coherent chunk cache (the cleaned page content), grouped by source-page rank.

Run:  uv run python -m semdn.e4_baselines --mode snippet --corpus t3 --page-ks 1,3,5,10
      CUDA_VISIBLE_DEVICES=1 uv run python -m semdn.e4_baselines --mode summary --corpus t3 --page-ks 1,3,5,10
"""

import argparse
import json

from semdn import paths
from semdn.e4_crag_t3 import DEFAULT_SRC, _cache_path, reservoir_sample

E4 = paths.DATA_ROOT / "e4"
TASK1_SRC = paths.DATA_ROOT / "datasets" / "crag" / "crag_task_1_dev_v4_release.jsonl.bz2"
SUM_SYS = ("You faithfully summarize a web page. In 2-4 sentences state ONLY the facts the page "
           "actually contains that relate to the question — names, dates, numbers, entities, quotes. "
           "Do NOT answer the question yourself, do NOT guess or infer, and do NOT add outside "
           "knowledge. If the page does not contain the answer, briefly say what the page is about.")


def _src(args):
    return args.src or (str(TASK1_SRC) if args.corpus == "t1" else str(DEFAULT_SRC))


def build_snippet(args):
    qs = reservoir_sample(_src(args), args.n, args.question_seed)
    out = E4 / f"pool_crag_{args.corpus}_snippet.jsonl"
    with open(out, "w") as f:
        for q in qs:
            pages = q["pages"][:max(args.page_ks)]
            blocks = [f"{p['title']}\n{p['snippet']}".strip() for p in pages]
            blocks = [b for b in blocks if b]
            conds = {f"page@{K}": {"chunk_texts": blocks[:K]} for K in args.page_ks}
            f.write(json.dumps({"qid": q["qid"], "question": q["question"],
                "golden_answers": q["golden_answers"], "metadata": q["metadata"],
                "n_usable": len(blocks), "conds": conds}) + "\n")
    print(f"snippet pool -> {out}", flush=True)


def build_summary(args):
    recs = [json.loads(l) for l in open(_cache_path("coherent", args.corpus))]
    # reconstruct each page's cleaned text from its chunks (grouped by source-page rank)
    page_texts, owners = [], []   # flat list of page texts + (rec_idx, rank) so we can regroup
    for i, r in enumerate(recs):
        by_rank: dict[int, list[str]] = {}
        for c, rk in zip(r["chunks"], r["ranks"]):
            by_rank.setdefault(rk, []).append(c)
        for rk in sorted(by_rank)[:max(args.page_ks)]:
            page_texts.append(" ".join(by_rank[rk])[:4000])   # cap input length (chars)
            owners.append((i, rk))

    from vllm import LLM, SamplingParams
    # max_model_len 8192 with the 4000-char page cap above: even worst-case (1 token/char)
    # a prompt is ~4000 + ~300 overhead + 160 out < 8192, so no page can overflow the batch.
    llm = LLM(model=paths.model(args.summarizer), dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.90)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=200)
    # Qwen3 summarizer: disable its <think> block so it stays a short faithful summary.
    think_kw = {"enable_thinking": False} if "qwen3" in args.summarizer.lower() else {}
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SUM_SYS},
         {"role": "user", "content": f"Question: {recs[i]['question']}\n\nWeb page:\n{t}\n\nFaithful summary:"}],
        add_generation_prompt=True, tokenize=False, **think_kw) for t, (i, _r) in zip(page_texts, owners)]
    print(f"summarizing {len(prompts)} pages", flush=True)
    summaries = [o.outputs[0].text.strip() for o in llm.generate(prompts, sp)]

    by_rec: dict[int, dict[int, str]] = {}
    for (i, rk), s in zip(owners, summaries):
        by_rec.setdefault(i, {})[rk] = s
    out = E4 / f"pool_crag_{args.corpus}_summary.jsonl"
    with open(out, "w") as f:
        for i, r in enumerate(recs):
            ranks = sorted(by_rec.get(i, {}))
            ordered = [by_rec[i][rk] for rk in ranks]
            conds = {f"page@{K}": {"chunk_texts": ordered[:K]} for K in args.page_ks}
            f.write(json.dumps({"qid": r["qid"], "question": r["question"],
                "golden_answers": r["golden_answers"], "metadata": r["metadata"],
                "n_usable": len(ordered), "conds": conds}) + "\n")
    print(f"summary pool -> {out}", flush=True)


def build_fullpage(args):
    """Stuff the top-K WHOLE pages into the reader (no chunking) — the 'just read the pages'
    baseline many agentic-RAG papers use. Page text reconstructed from the coherent chunk
    cache; capped per page so the top-K concatenation fits the reader's 8192 context."""
    recs = [json.loads(l) for l in open(_cache_path("coherent", args.corpus))]
    cap = args.page_chars
    out = E4 / f"pool_crag_{args.corpus}_fullpage.jsonl"
    with open(out, "w") as f:
        for r in recs:
            by_rank: dict[int, list[str]] = {}
            for c, rk in zip(r["chunks"], r["ranks"]):
                by_rank.setdefault(rk, []).append(c)
            page_texts = [" ".join(by_rank[rk])[:cap] for rk in sorted(by_rank)]
            conds = {f"page@{K}": {"chunk_texts": page_texts[:K]} for K in args.page_ks}
            f.write(json.dumps({"qid": r["qid"], "question": r["question"],
                "golden_answers": r["golden_answers"], "metadata": r["metadata"],
                "n_usable": len(page_texts), "conds": conds}) + "\n")
    print(f"fullpage pool -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["snippet", "summary", "fullpage"])
    ap.add_argument("--page-chars", type=int, default=3500, help="char cap per page (fullpage)")
    ap.add_argument("--corpus", default="t3")
    ap.add_argument("--src", default=None)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--question-seed", type=int, default=0)
    ap.add_argument("--page-ks", default="1,3,5,10")
    ap.add_argument("--summarizer", default=str(paths.QWEN25_7B))
    args = ap.parse_args()
    args.page_ks = [int(k) for k in args.page_ks.split(",")]
    if args.mode == "snippet":
        build_snippet(args)
    elif args.mode == "fullpage":
        build_fullpage(args)
    else:
        build_summary(args)


if __name__ == "__main__":
    main()
