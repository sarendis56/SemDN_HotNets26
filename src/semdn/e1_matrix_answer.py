"""E1-C matrix answer quality: EM/F1 for each (granularity × ranker) config, per
dataset, plus paired tests. Same reader (Qwen2.5-7B-Instruct), same k chunks delivered.

Writes data/e1/matrix_<tag><reader-tag>quality.jsonl (per-config means) and
matrix_<tag><reader-tag>preds.jsonl (per-question F1, for the paired tests).
Another reader on the same contexts: --model Qwen/Qwen3-32B --reader-tag qwen3_32b_
(uses every visible GPU, tensor parallel).
Paired test: sign-flip permutation on per-question F1 differences, Holm over datasets.

Run:  uv run python -m semdn.e1_matrix_answer --tag llm_          (GPU)
      uv run python -m semdn.e1_matrix_answer --tag llm_ --stats  (CPU, tests only)
"""

import argparse
import json

import numpy as np

from semdn import paths
from semdn.e4_stats import holm
from semdn.metrics import cover_em, em, f1

E1 = paths.DATA_ROOT / "e1"
DATASETS = "hotpotqa,2wikimultihopqa,musique,bamboogle,nq"
SYS = ("You are a helpful assistant. Answer the question using only the provided "
       "context. Reply with a short, direct answer of a few words.")
CONFIGS = ["chunk_e5", "chunk_llm",
           "page_e5@1", "page_e5@3", "page_e5@5", "page_e5@10",
           "page_llm@1", "page_llm@3", "page_llm@5", "page_llm@10"]
# (A, B): Δ = A − B; page arms at their most generous N=10
PAIRS = [("chunk_llm", "page_llm@10"), ("chunk_e5", "page_e5@10"),
         ("chunk_llm", "chunk_e5"), ("page_llm@10", "page_e5@10")]


def chunk_ids(rec, config):
    if config.startswith("chunk_"):
        return rec[config]
    name, N = config.split("@")
    return rec[f"{name}_byN"][N]


def perm_p(d, n=20000, seed=0):
    """Two-sided sign-flip permutation p for mean(d) != 0."""
    d = np.asarray(d, dtype=np.float64)
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n, len(d)))
    return float((np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12).mean())


def stats(tag, datasets, rtag=""):
    by = {}
    for l in open(E1 / f"matrix_{tag}{rtag}preds.jsonl"):
        r = json.loads(l)
        by[(r["dataset"], r["config"])] = r["f1"]
    print(f"\n{'dataset':16} " + " ".join(f"{c:>12}" for c in CONFIGS))
    for ds in datasets:
        print(f"{ds:16} " + " ".join(f"{np.mean(by[(ds, c)]):12.3f}" for c in CONFIGS))
    print(f"{'average':16} " + " ".join(
        f"{np.mean([np.mean(by[(ds, c)]) for ds in datasets]):12.3f}" for c in CONFIGS))
    out = []
    for A, B in PAIRS:
        print(f"\n--- {A} vs {B} (ΔF1 = A − B; permutation p, Holm over {len(datasets)} datasets) ---")
        rows = [(ds, np.array(by[(ds, A)]) - np.array(by[(ds, B)])) for ds in datasets]
        ps = [perm_p(d) for _, d in rows]
        for (ds, d), p, pa in zip(rows, ps, holm(ps)):
            print(f"{ds:16} n={len(d):4d}  ΔF1={d.mean():+.3f}  p={p:.4f}  pHolm={pa:.4f}"
                  f"  {'SIG' if pa < 0.05 else ''}")
            out.append({"A": A, "B": B, "dataset": ds, "n": len(d), "df1": float(d.mean()),
                        "p": p, "p_holm": pa})
        pooled = np.concatenate([d for _, d in rows])
        p = perm_p(pooled)
        print(f"{'pooled':16} n={len(pooled):4d}  ΔF1={pooled.mean():+.3f}  p={p:.4f}")
        out.append({"A": A, "B": B, "dataset": "pooled", "n": len(pooled),
                    "df1": float(pooled.mean()), "p": p})
    json.dump(out, open(E1 / f"matrix_{tag}{rtag}significance.json", "w"), indent=1)
    print(f"\nwrote matrix_{tag}{rtag}significance.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=DATASETS)
    ap.add_argument("--tag", default="llm_")
    ap.add_argument("--model", default=str(paths.QWEN25_7B))
    ap.add_argument("--reader-tag", default="", help="output name suffix for a non-default reader")
    ap.add_argument("--gpu-mem", type=float, default=0.0,
                    help="vLLM memory share per GPU; 0 = what the fullest visible GPU has free, up to 0.92")
    ap.add_argument("--stats", action="store_true", help="paired tests only (no GPU)")
    args = ap.parse_args()
    datasets = args.datasets.split(",")
    if args.stats:
        return stats(args.tag, datasets, args.reader_tag)

    import torch
    from vllm import LLM, SamplingParams

    from semdn.corpus import Corpus

    corpus = Corpus()
    if not args.gpu_mem:   # mixed card sizes / shared cards: same share must fit on every GPU
        args.gpu_mem = min(0.92, min(f / t for f, t in map(torch.cuda.mem_get_info,
                                                             range(torch.cuda.device_count()))) - 0.03)
    llm = LLM(model=paths.model(args.model), dtype="bfloat16", max_model_len=8192,
              tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=args.gpu_mem)
    think_kw = {"enable_thinking": False} if "qwen3" in args.model.lower() else {}
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=64, stop=["\n"])
    fout = open(E1 / f"matrix_{args.tag}{args.reader_tag}quality.jsonl", "w")
    fpred = open(E1 / f"matrix_{args.tag}{args.reader_tag}preds.jsonl", "w")

    for ds in datasets:
        recs = [json.loads(l) for l in open(E1 / f"matrix_{args.tag}{ds}.jsonl")]
        questions = [r["question"] for r in recs]
        golden = [r["golden_answers"] for r in recs]

        def ask(id_lists):
            prompts = []
            for ids, q in zip(id_lists, questions):
                ctx = "\n\n".join(corpus.contents(int(c)) for c in ids)
                prompts.append(tok.apply_chat_template(
                    [{"role": "system", "content": SYS},
                     {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"}],
                    add_generation_prompt=True, tokenize=False, **think_kw))
            return [o.outputs[0].text.strip() for o in llm.generate(prompts, sp)]

        for cfg in CONFIGS:
            ans = ask([chunk_ids(r, cfg) for r in recs])
            f1s = [f1(a, g) for a, g in zip(ans, golden)]
            rec = {"dataset": ds, "config": cfg, "n": len(recs),
                   "em": float(np.mean([em(a, g) for a, g in zip(ans, golden)])),
                   "cover_em": float(np.mean([cover_em(a, g) for a, g in zip(ans, golden)])),
                   "f1": float(np.mean(f1s))}
            fout.write(json.dumps(rec) + "\n")
            fpred.write(json.dumps({"dataset": ds, "config": cfg, "qids": [r["qid"] for r in recs],
                                    "answers": ans, "f1": f1s}) + "\n")
            fout.flush(), fpred.flush()
            print(f"{ds:18s} {cfg:14s} EM={rec['em']:.3f} F1={rec['f1']:.3f}", flush=True)
    fout.close(), fpred.close()
    stats(args.tag, datasets, args.reader_tag)


if __name__ == "__main__":
    assert perm_p([0.0] * 50) == 1.0 and perm_p([1.0] * 50) < 0.001   # sign-flip sanity
    main()
