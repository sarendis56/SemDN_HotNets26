"""Iso-budget quality: at a FIXED reader context budget (tokens), does delivering the
relevance-selected CHUNKS beat delivering whole PAGES (and snippets)? The fixed-3-chunks-
vs-fixed-K-pages comparisons gave parity because both arms got plenty of context. Constrain
the budget and the packing granularity should matter: chunks fill the budget with the most
answer-bearing tokens; whole pages spend it on document-order text (intro/boilerplate before
the answer).

Same cleaned content for every arm (coherent chunks of CRAG Task-3's provided pages); arms
differ ONLY in the ORDER content fills the budget, then each is truncated to exactly B
tokens (reader tokenizer), so the budget is identical:
  chunk    : chunks in e5-relevance order             (SemDN delivery, bi-encoder rank)
  chunk_ce : chunks in cross-encoder-relevance order  (SemDN delivery, reranker)
  page_rel : whole pages in e5-relevance order        (page delivery, fair: same ranker)
  page_seq : whole pages in provided/search order     (page delivery, realistic CDN/agent)
  snippet  : page snippets (built separately via the snippet_full subcommand)
  chunk_llm / page_llm / page_oracle : LLM-reranker 2x2 arms + page-ranking diagnostic
             (the llm subcommand; see build_llm)

Writes pool_iso_<tag><arm>.jsonl (conds = b128..b6144) for e4_answer + e4_judge.

Run (GPU e5+CE):  CUDA_VISIBLE_DEVICES=0 uv run python -m semdn.e4_isobudget build \
                    --corpus t3b --n 600 --tag t3b_
"""
import argparse
import json
import re

import numpy as np

from semdn import paths

E4 = paths.DATA_ROOT / "e4"
BUDGETS = [128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144]
SEP = "\n\n"


def _packer(tok, budgets):
    def pack(pieces):
        full = SEP.join(p for p in pieces if p)
        ids = tok(full, add_special_tokens=False).input_ids
        return {B: tok.decode(ids[:B]) for B in budgets}
    return pack


def build(args):
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer)

    from semdn.retrieval_server import E5Encoder
    enc = E5Encoder(paths.E5_BASE_V2, device=args.device)
    tok = AutoTokenizer.from_pretrained(paths.model(args.reader))
    pack = _packer(tok, args.budgets)

    ce_tok = AutoTokenizer.from_pretrained(paths.model(args.reranker))
    ce_model = AutoModelForSequenceClassification.from_pretrained(
        paths.model(args.reranker), torch_dtype=torch.float16).to(args.device).eval()

    @torch.no_grad()
    def ce_score(question, chunks, batch=64):
        out = []
        for i in range(0, len(chunks), batch):
            pairs = [[question, c] for c in chunks[i:i + batch]]
            inp = ce_tok(pairs, padding=True, truncation=True, max_length=512,
                         return_tensors="pt").to(args.device)
            out.extend(ce_model(**inp).logits.view(-1).float().cpu().numpy().tolist())
        return np.array(out)

    cache = [json.loads(l) for l in open(E4 / f"chunks_crag_{args.corpus}_coherent.jsonl")]
    cache = cache[: args.n]
    print(f"iso-budget build ({args.corpus}, tag={args.tag!r}): {len(cache)} questions, "
          f"budgets={args.budgets}", flush=True)

    arms = ["chunk", "chunk_ce", "page_rel", "page_seq"]
    fouts = {a: open(E4 / f"pool_iso_{args.tag}{a}.jsonl", "w") for a in arms}

    for idx, r in enumerate(cache):
        chunks, ranks = r["chunks"], r["ranks"]
        meta = {"qid": r["qid"], "question": r["question"],
                "golden_answers": r["golden_answers"], "metadata": r["metadata"],
                "n_usable": r["n_usable"]}
        if not chunks:
            for a in arms:
                fouts[a].write(json.dumps({**meta, "conds": {}}) + "\n")
            continue
        e5s = (enc.encode_passages(chunks).astype(np.float32)
               @ enc.encode_queries([r["question"]]).astype(np.float32)[0])
        ces = ce_score(r["question"], chunks)

        # reconstruct cleaned pages (doc order within page) + page relevance = best chunk (e5)
        pages = {}
        for i, p in enumerate(ranks):
            pages.setdefault(p, []).append(i)
        page_text = {p: SEP.join(chunks[i] for i in idxs) for p, idxs in pages.items()}
        page_best = {p: float(np.max(e5s[idxs])) for p, idxs in pages.items()}

        order = {
            "chunk": [chunks[i] for i in np.argsort(-e5s)],
            "chunk_ce": [chunks[i] for i in np.argsort(-ces)],
            "page_rel": [page_text[p] for p in sorted(pages, key=lambda p: -page_best[p])],
            "page_seq": [page_text[p] for p in sorted(pages)],
        }
        for a in arms:
            ctx_by_b = pack(order[a])
            conds = {f"b{B}": {"chunk_texts": [ctx_by_b[B]]} for B in args.budgets}
            fouts[a].write(json.dumps({**meta, "conds": conds}) + "\n")
        if (idx + 1) % 25 == 0:
            print(f"  [{idx+1}/{len(cache)}] {r['qid'][:8]}: {len(chunks)} chunks, "
                  f"{len(pages)} pages", flush=True)

    for a in arms:
        fouts[a].close()
    print("iso pools written:", ", ".join(f"pool_iso_{args.tag}{a}.jsonl" for a in arms), flush=True)


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[|,]", " ", s.lower())).strip()


def usable_answers(golden):
    """Gold answers that can be string-matched: CRAG stores some as numbers (130.0 must match
    "130"); yes/no and 1-character answers match everywhere, so they are skipped."""
    golden = [str(int(g)) if isinstance(g, float) and g.is_integer() else str(g) for g in golden]
    return [a for a in (norm(g) for g in golden) if len(a) >= 2 and a not in ("yes", "no")]


def has_answer(answers, text):
    return any(re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", text) for a in answers)


RERANK_CTX = 32768   # Qwen3-Reranker context (model card); a doc may fill what the prompt leaves


def make_llm_scorer(model, device, batch_tokens=32768,
                    instruct="Given a web search query, retrieve relevant passages that answer the query"):
    """Qwen3-Reranker usage per its model card: P("yes") at the last position.
    Returns llm_score(question, docs, max_doc_tokens=None) -> np.array of P(yes);
    max_doc_tokens=None cuts a doc only where the reranker's context runs out."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rtok = AutoTokenizer.from_pretrained(paths.model(model), padding_side="left")
    rmodel = AutoModelForCausalLM.from_pretrained(
        paths.model(model), torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    yes_id, no_id = rtok.convert_tokens_to_ids("yes"), rtok.convert_tokens_to_ids("no")
    prefix = rtok.encode("<|im_start|>system\nJudge whether the Document meets the requirements "
                         "based on the Query and the Instruct provided. Note that the answer can "
                         "only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n",
                         add_special_tokens=False)
    suffix = rtok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                         add_special_tokens=False)

    @torch.no_grad()
    def llm_score(question, docs, max_doc_tokens=None):
        head = rtok.encode(f"<Instruct>: {instruct}\n<Query>: {question}\n<Document>: ",
                           add_special_tokens=False)
        room = RERANK_CTX - len(prefix) - len(head) - len(suffix)
        cut = room if max_doc_tokens is None else min(max_doc_tokens, room)
        seqs = [prefix + head + rtok.encode(d, add_special_tokens=False)[:cut] + suffix
                for d in docs]
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        out = np.zeros(len(seqs), dtype=np.float32)
        i = 0
        while i < len(order):   # length-sorted batches capped at batch_tokens (padded size)
            j = i + 1
            while j < len(order) and len(seqs[order[j]]) * (j - i + 1) <= batch_tokens:
                j += 1
            idx = order[i:j]
            inp = rtok.pad({"input_ids": [seqs[k] for k in idx]}, padding=True,
                           return_tensors="pt").to(device)
            # logits_to_keep=1: last position only (full-vocab logits over a 32K-token batch OOM'd)
            logits = rmodel(**inp, logits_to_keep=1).logits[:, -1, :]
            p = torch.log_softmax(torch.stack([logits[:, no_id], logits[:, yes_id]], 1).float(), 1)
            out[idx] = p[:, 1].exp().cpu().numpy()
            i = j
        return out

    llm_score.tok = rtok
    return llm_score


def build_llm(args):
    """2x2 reranker arms + page oracle on the same coherent-chunk cache as build():
      chunk_llm   : chunks ranked by an LLM reranker (Qwen3-Reranker)
      page_llm    : whole pages ranked by the SAME reranker reading the full page (cut at
                    --page-max-tokens), the long-context page ranker that MaxP-e5 is not
      page_oracle : diagnostic -- pages whose text contains a gold answer first, the rest in
                    page_rel (MaxP-e5) order; questions with no string-matchable answer keep
                    page_rel order, so this is a partial upper bound on page ranking
    Pages are rebuilt exactly as in build() (page's chunks in doc order)."""
    from transformers import AutoTokenizer

    from semdn.retrieval_server import E5Encoder
    enc = E5Encoder(paths.E5_BASE_V2, device=args.device)
    tok = AutoTokenizer.from_pretrained(paths.model(args.reader))
    pack = _packer(tok, args.budgets)

    llm_score = make_llm_scorer(args.llm_reranker, args.device, args.batch_tokens)

    cache = [json.loads(l) for l in open(E4 / f"chunks_crag_{args.corpus}_coherent.jsonl")][: args.n]
    arms = ["chunk_llm", "page_llm", "page_oracle"]
    fouts = {a: open(E4 / f"pool_iso_{args.tag}{a}.jsonl", "w") for a in arms}
    n_cut = n_pages = n_oracle = 0
    print(f"llm build ({args.corpus}, {args.llm_reranker}): {len(cache)} questions, "
          f"budgets={args.budgets}", flush=True)
    for idx, r in enumerate(cache):
        chunks, ranks = r["chunks"], r["ranks"]
        meta = {"qid": r["qid"], "question": r["question"],
                "golden_answers": r["golden_answers"], "metadata": r["metadata"],
                "n_usable": r["n_usable"]}
        if not chunks:
            for a in arms:
                fouts[a].write(json.dumps({**meta, "conds": {}}) + "\n")
            continue
        e5s = (enc.encode_passages(chunks).astype(np.float32)
               @ enc.encode_queries([r["question"]]).astype(np.float32)[0])
        pages = {}
        for i, p in enumerate(ranks):
            pages.setdefault(p, []).append(i)
        pids = sorted(pages)
        page_text = {p: SEP.join(chunks[i] for i in pages[p]) for p in pids}
        page_best = {p: float(np.max(e5s[pages[p]])) for p in pids}
        rel = sorted(pids, key=lambda p: -page_best[p])

        cs = llm_score(r["question"], chunks, 2048)
        ps = llm_score(r["question"], [page_text[p] for p in pids], args.page_max_tokens)
        n_pages += len(pids)
        n_cut += sum(len(llm_score.tok.encode(page_text[p], add_special_tokens=False)) > args.page_max_tokens
                     for p in pids)
        ans = usable_answers(r["golden_answers"])
        hit = {p for p in pids if ans and has_answer(ans, norm(page_text[p]))}
        n_oracle += bool(hit)

        order = {
            "chunk_llm": [chunks[i] for i in np.argsort(-cs, kind="stable")],
            "page_llm": [page_text[pids[i]] for i in np.argsort(-ps, kind="stable")],
            "page_oracle": [page_text[p] for p in rel if p in hit] + [page_text[p] for p in rel if p not in hit],
        }
        for a in arms:
            ctx_by_b = pack(order[a])
            conds = {f"b{B}": {"chunk_texts": [ctx_by_b[B]]} for B in args.budgets}
            fouts[a].write(json.dumps({**meta, "conds": conds}) + "\n")
            fouts[a].flush()
        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(cache)}] {len(chunks)} chunks, {len(pids)} pages", flush=True)
    for a in arms:
        fouts[a].close()
    print(f"pages cut at {args.page_max_tokens} tokens: {n_cut}/{n_pages}; "
          f"questions with an answer-bearing page for the oracle: {n_oracle}/{len(cache)}", flush=True)


def build_snippet_full(args):
    """Corrected snippet arm: pack ALL provided page snippets (CRAG gives ~47/q, vs the 10
    the first snippet arm used), so the snippet channel can actually fill the budget. Reuses
    the SAME reservoir sample as the chunk cache (matching qids). No GPU."""
    from transformers import AutoTokenizer

    from semdn.e4_crag_t3 import reservoir_sample
    tok = AutoTokenizer.from_pretrained(paths.model(args.reader))
    pack = _packer(tok, args.budgets)
    qs = reservoir_sample(args.src, args.n, args.question_seed)
    print(f"snippet_full ({args.tag!r}): {len(qs)} questions", flush=True)
    nsnips = []
    out = E4 / f"pool_iso_{args.tag}snippet_full.jsonl"
    with open(out, "w") as f:
        for q in qs:
            snippets = [f"{p['title']}\n{p['snippet']}".strip()
                        for p in q["pages"] if (p.get("snippet") or "").strip()]
            nsnips.append(len(snippets))
            ctx_by_b = pack(snippets)
            conds = {f"b{B}": {"chunk_texts": [ctx_by_b[B]]} for B in args.budgets}
            f.write(json.dumps({"qid": q["qid"], "question": q["question"],
                                "golden_answers": q["golden_answers"], "metadata": q["metadata"],
                                "n_usable": len(snippets), "conds": conds}) + "\n")
    print(f"snippets/question: median {int(np.median(nsnips))}, max {max(nsnips)} -> {out.name}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--n", type=int, default=600)
    b.add_argument("--corpus", default="t3b")
    b.add_argument("--tag", default="t3b_", help="filename tag so corpora don't clobber")
    b.add_argument("--device", default="cuda")
    b.add_argument("--reader", default="Qwen/Qwen3-32B", help="tokenizer for the token budget")
    b.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    sf = sub.add_parser("snippet_full")
    sf.add_argument("--n", type=int, default=600)
    sf.add_argument("--question-seed", type=int, default=0)
    sf.add_argument("--src", default=str(paths.DATA_ROOT / "datasets" / "crag" / "crag_task_3_dev_v4"))
    sf.add_argument("--tag", default="t3b_")
    sf.add_argument("--reader", default="Qwen/Qwen3-32B")
    lm = sub.add_parser("llm")
    lm.add_argument("--n", type=int, default=600)
    lm.add_argument("--corpus", default="t3b")
    lm.add_argument("--tag", default="t3b_")
    lm.add_argument("--device", default="cuda")
    lm.add_argument("--reader", default="Qwen/Qwen3-32B")
    lm.add_argument("--llm-reranker", default="Qwen/Qwen3-Reranker-4B")
    lm.add_argument("--page-max-tokens", type=int, default=16384)
    lm.add_argument("--batch-tokens", type=int, default=32768, help="padded tokens per batch")
    for p in (b, sf, lm):  # long-context extension: --budgets 8192,12288,... --tag t3b_xl_
        p.add_argument("--budgets", type=lambda v: [int(x) for x in v.split(",")], default=BUDGETS)
    args = ap.parse_args()
    if args.cmd == "build":
        build(args)
    elif args.cmd == "llm":
        build_llm(args)
    elif args.cmd == "snippet_full":
        build_snippet_full(args)


if __name__ == "__main__":
    main()
