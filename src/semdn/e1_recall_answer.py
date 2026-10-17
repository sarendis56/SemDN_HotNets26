"""E1-C answer quality (GPU): does the page-bounded client miss useful chunks?

Same reader answers from (a) the SemDN's global top-k chunks vs (b) the client's
top-k chunks bounded to its top-N downloaded pages. Same corpus, same encoder,
same reader, same k -> any gap is the cost of page-bounded delivery.

Run (after e1_recall.py; GPU free):
  CUDA_VISIBLE_DEVICES=0 uv run python -m semdn.e1_recall_answer
"""

import argparse
import json

import numpy as np

from semdn import paths
from semdn.corpus import Corpus
from semdn.metrics import cover_em, em, f1

SYS = ("You are a helpful assistant. Answer the question using only the provided "
       "context. Reply with a short, direct answer of a few words.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.QWEN25_7B))
    ap.add_argument("--contexts", default=str(paths.DATA_ROOT / "e1" / "recall_contexts.jsonl"))
    ap.add_argument("--ns", default="1,3,5,10", help="client N-page settings to score")
    ap.add_argument("--out", default=str(paths.DATA_ROOT / "e1" / "recall_quality.jsonl"))
    args = ap.parse_args()
    Ns = args.ns.split(",")

    from vllm import LLM, SamplingParams

    corpus = Corpus()
    recs = [json.loads(l) for l in open(args.contexts)]
    questions = [r["question"] for r in recs]
    golden = [r["golden_answers"] for r in recs]

    def ctx(ids):
        return "\n\n".join(corpus.contents(int(c)) for c in ids)

    llm = LLM(model=paths.model(args.model), dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=64, stop=["\n"])

    def ask(contexts):
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": SYS},
             {"role": "user", "content": f"Context:\n{c}\n\nQuestion: {q}\nAnswer:"}],
            add_generation_prompt=True, tokenize=False)
            for c, q in zip(contexts, questions)]
        return [o.outputs[0].text.strip() for o in llm.generate(prompts, sp)]

    def score(name, chunk_id_lists):
        ans = ask([ctx(ids) for ids in chunk_id_lists])
        rec = {"arm": name,
               "em": float(np.mean([em(a, g) for a, g in zip(ans, golden)])),
               "cover_em": float(np.mean([cover_em(a, g) for a, g in zip(ans, golden)])),
               "f1": float(np.mean([f1(a, g) for a, g in zip(ans, golden)]))}
        print(f"{name:14s} EM={rec['em']:.3f} coverEM={rec['cover_em']:.3f} F1={rec['f1']:.3f}",
              flush=True)
        return rec

    out = [score("scdn_global", [r["scdn_topk"] for r in recs])]
    for N in Ns:
        out.append(score(f"client_N={N}", [r["client_topk_byN"][N] for r in recs]))

    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
