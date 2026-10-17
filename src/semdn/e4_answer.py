"""E4 reader: answer each CRAG question from the k=3 chunks selected under each
condition, score EM/cover-EM/F1, and SAVE predictions (.preds.jsonl) for deferred
LLM-as-judge evaluation. Conditions are read from the pool (works for the live-web
page@K/chunk@all pools and the CRAG pageret/chunkret pools alike).

--closed-book generates the no-RAG baseline (answer from parametric knowledge, no
context) — used to show how much retrieval helps.

Run:  CUDA_VISIBLE_DEVICES=1 uv run python -m semdn.e4_answer --pool <pool.jsonl> --out <q.jsonl>
      CUDA_VISIBLE_DEVICES=1 uv run python -m semdn.e4_answer --pool <pool.jsonl> --out <q.jsonl> --closed-book
"""

import argparse
import collections
import json

import numpy as np

from semdn import paths
from semdn.e4_granularity import POOL
from semdn.metrics import cover_em, em, f1

SYS = ("You are a helpful assistant. Answer the question using only the provided "
       "context. Reply with a short, direct answer of a few words.")
SYS_CB = ("You are a helpful assistant. Answer the question from your own knowledge. "
          "Reply with a short, direct answer of a few words. If you do not know, "
          "reply exactly: I don't know.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.QWEN25_7B))
    ap.add_argument("--pool", default=str(POOL))
    ap.add_argument("--out", default=str(paths.DATA_ROOT / "e4" / "granularity_quality.jsonl"))
    ap.add_argument("--closed-book", action="store_true",
                    help="no-RAG baseline: answer from parametric knowledge, no context")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size (long-context runs)")
    ap.add_argument("--gpu-mem", type=float, default=0.90, help="vLLM gpu_memory_utilization")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    recs = [json.loads(l) for l in open(args.pool)]
    recs = [r for r in recs if r.get("n_usable", 1) > 0]
    print(f"{len(recs)} questions", flush=True)

    llm = LLM(model=paths.model(args.model), dtype="bfloat16", max_model_len=args.max_model_len,
              tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem)
    ctx_cap = args.max_model_len - 1192   # 7000 at the default 8192
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=64, stop=["\n"])
    # Qwen3 defaults to a <think> reasoning block that would blow past our short
    # answer budget; disable it so it stays a clean short-answer reader.
    think_kw = {"enable_thinking": False} if "qwen3" in args.model.lower() else {}
    golden = [r["golden_answers"] for r in recs]
    dyn = [r["metadata"].get("dynamism", "?") for r in recs]

    def ask_rag(chunk_lists):
        prompts = []
        for chunks, r in zip(chunk_lists, recs):
            ctx = "\n\n".join(chunks) if chunks else "(no documents retrieved)"
            ids = tok(ctx, add_special_tokens=False).input_ids   # truncate over-long contexts
            if len(ids) > ctx_cap:                               # (e.g. nav/URL-heavy markdown) so the
                ctx = tok.decode(ids[:ctx_cap])                  # prompt can't overflow max_model_len
            prompts.append(tok.apply_chat_template(
                [{"role": "system", "content": SYS},
                 {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {r['question']}\nAnswer:"}],
                add_generation_prompt=True, tokenize=False, **think_kw))
        return [o.outputs[0].text.strip() for o in llm.generate(prompts, sp)]

    def ask_closed_book():
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": SYS_CB},
             {"role": "user", "content": f"Question: {r['question']}\nAnswer:"}],
            add_generation_prompt=True, tokenize=False, **think_kw) for r in recs]
        return [o.outputs[0].text.strip() for o in llm.generate(prompts, sp)]

    # which conditions to run
    if args.closed_book:
        conds = ["no_rag"]
    else:
        conds, seen = [], set()
        for r in recs:
            for c in r.get("conds", {}):
                if c not in seen:
                    seen.add(c); conds.append(c)

    fout = open(args.out, "w")
    preds_path = args.out.replace(".jsonl", ".preds.jsonl")
    pfout = open(preds_path, "w")
    summary = {}
    for cfg in conds:
        if cfg == "no_rag":
            ans = ask_closed_book()
        else:
            ans = ask_rag([r["conds"].get(cfg, {}).get("chunk_texts", []) for r in recs])
        for r, a in zip(recs, ans):
            pfout.write(json.dumps({"qid": r["qid"], "config": cfg, "question": r["question"],
                                    "gold": r["golden_answers"], "pred": a}) + "\n")
        per_cem = [cover_em(a, g) for a, g in zip(ans, golden)]
        bydyn = collections.defaultdict(list)
        for d, c in zip(dyn, per_cem):
            bydyn[d].append(c)
        rec = {"config": cfg, "n": len(recs),
               "em": float(np.mean([em(a, g) for a, g in zip(ans, golden)])),
               "cover_em": float(np.mean(per_cem)),
               "f1": float(np.mean([f1(a, g) for a, g in zip(ans, golden)])),
               "cover_em_by_dynamism": {d: round(float(np.mean(v)), 3) for d, v in sorted(bydyn.items())}}
        summary[cfg] = rec
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        print(f"{cfg:12s} EM={rec['em']:.3f} cover_EM={rec['cover_em']:.3f} F1={rec['f1']:.3f}",
              flush=True)
    fout.close()
    pfout.close()
    print(f"wrote predictions -> {preds_path}", flush=True)

    # delivery gap (only for live-web pools that carry deep_topk_max_rank + page@K conds)
    if not args.closed_book and recs and "deep_topk_max_rank" in recs[0]:
        from semdn.e4_granularity import PAGE_KS
        deep_max = [r["deep_topk_max_rank"] for r in recs]
        print("\n--- delivery gap (chunk@all top-3: deepest usable page rank used) ---", flush=True)
        for K in PAGE_KS:
            frac = float(np.mean([d > K for d in deep_max]))
            print(f"  {frac:.0%} of questions used a page ranked > {K}", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
