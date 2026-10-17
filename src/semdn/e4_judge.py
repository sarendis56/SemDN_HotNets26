"""CRAG-style LLM-as-judge metric for E4 predictions. CRAG (Meta KDD Cup 2024)
does NOT use string match — its golden answers are free-form sentences and it
grades with an LLM into perfect/acceptable/missing/incorrect, scoring
1 / 0.5 / 0 / -1 (truthfulness), penalizing hallucination and preferring an
"I don't know" (missing) over a wrong answer.

Default judge = local Qwen2.5-7B-Instruct; pass --model meta-llama/Llama-3.1-70B-Instruct
(CRAG's official critic) for the trustworthy numbers. Multiple pools can be judged
in one model load by comma-separating --preds / --tags / --outs.

Run:  CUDA_VISIBLE_DEVICES=0 uv run python -m semdn.e4_judge \
        --preds a.preds.jsonl,b.preds.jsonl --tags e5,qwen3 --outs ja.jsonl,jb.jsonl \
        --model meta-llama/Llama-3.1-70B-Instruct --quantization fp8 --gpu-memory-utilization 0.96
"""

import argparse
import collections
import json

import numpy as np

from semdn import paths

SCORE = {"perfect": 1.0, "acceptable": 0.5, "missing": 0.0, "incorrect": -1.0}

SYS = ("You are a strict evaluator of question-answering systems. You compare a "
       "candidate answer against the ground-truth answer and judge its correctness.")

USER = """Question: {q}
Ground-truth answer: {gold}
Candidate answer: {pred}

Classify the candidate answer as exactly one of:
- perfect: correct and complete; matches the ground-truth meaning with no wrong information.
- acceptable: essentially correct and useful, but with minor imprecision or extra detail.
- missing: the candidate says it does not know, refuses, is empty, or asks to clarify.
- incorrect: gives wrong, contradictory, or irrelevant information.

Judge by meaning, not wording. Reply with ONLY one word: perfect, acceptable, missing, or incorrect."""


def parse_label(text: str) -> str:
    t = text.strip().lower()
    for lab in ("perfect", "acceptable", "incorrect", "missing"):  # incorrect before missing
        if lab in t:
            return lab
    return "unparseable"  # surfaced separately so a broken judge is visible, not hidden


def judge_pool(llm, tok, sp, preds_path, tag, out_path, crag):
    rows = [json.loads(l) for l in open(preds_path)]
    prompts = []
    for r in rows:
        gold = "; ".join(str(x) for x in r["gold"]) if isinstance(r["gold"], list) else str(r["gold"])
        pred = r["pred"] if r["pred"].strip() else "(empty)"
        prompts.append(tok.apply_chat_template(
            [{"role": "system", "content": SYS},
             {"role": "user", "content": USER.format(q=r["question"], gold=gold, pred=pred)}],
            add_generation_prompt=True, tokenize=False))
    outs = llm.generate(prompts, sp)
    raw = [o.outputs[0].text for o in outs]
    labels = [parse_label(t) for t in raw]

    n_unp = labels.count("unparseable")
    print(f"\n[judge health: {tag}] {n_unp}/{len(labels)} unparseable; sample raw outputs:", flush=True)
    for t in raw[:6]:
        print(f"    {t!r}", flush=True)
    if out_path:
        with open(out_path.replace(".jsonl", ".raw.jsonl"), "w") as rf:
            for r, t, lab in zip(rows, raw, labels):
                rf.write(json.dumps({"qid": r["qid"], "config": r["config"],
                                     "raw": t, "label": lab}) + "\n")

    by_cfg = collections.defaultdict(list)
    for r, lab in zip(rows, labels):
        by_cfg[r["config"]].append(lab)

    fout = open(out_path, "w") if out_path else None
    print(f"\n=== CRAG LLM-judge ({tag}) — truthfulness / accuracy / hallucination ===", flush=True)
    print(f"{'config':10s} {'truthful':>9} {'accuracy':>9} {'halluc':>8} {'missing':>8} {'unparse':>8}   (n)",
          flush=True)
    order = ["page@1", "page@3", "page@5", "page@10", "chunk@all"]
    for cfg in [c for c in order if c in by_cfg] + [c for c in by_cfg if c not in order]:
        labs = by_cfg[cfg]
        n = len(labs)
        cnt = collections.Counter(labs)
        scored = [SCORE[x] for x in labs if x in SCORE]  # truthfulness over parseable only
        truthful = float(np.mean(scored)) if scored else float("nan")
        acc = (cnt["perfect"] + cnt["acceptable"]) / n
        halluc = cnt["incorrect"] / n
        miss = cnt["missing"] / n
        unp = cnt["unparseable"] / n
        print(f"{cfg:10s} {truthful:9.3f} {acc:9.3f} {halluc:8.3f} {miss:8.3f} {unp:8.3f}   ({n})", flush=True)
        if fout:
            fout.write(json.dumps({"tag": tag, "config": cfg, "n": n, "truthfulness": truthful,
                "accuracy": float(acc), "hallucination": float(halluc), "missing": float(miss),
                "unparseable": float(unp), "counts": dict(cnt)}) + "\n")
    if fout:
        fout.close()
        print(f"wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="comma-separated preds files")
    ap.add_argument("--tags", default="", help="comma-separated tags (per preds)")
    ap.add_argument("--outs", default="", help="comma-separated out files (per preds)")
    ap.add_argument("--model", default=str(paths.QWEN25_7B))
    ap.add_argument("--crag", default=str(paths.FLASHRAG_DATASETS / "crag" / "test.jsonl"))
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--quantization", default=None, help="e.g. fp8 (on-the-fly, near-lossless)")
    args = ap.parse_args()

    preds = args.preds.split(",")
    tags = args.tags.split(",") if args.tags else [f"pool{i}" for i in range(len(preds))]
    outs = args.outs.split(",") if args.outs else [""] * len(preds)

    from vllm import LLM, SamplingParams
    crag = {json.loads(l)["id"]: json.loads(l)["metadata"] for l in open(args.crag)}
    llm = LLM(model=paths.model(args.model), dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              tensor_parallel_size=args.tensor_parallel_size,
              quantization=args.quantization)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=16)

    for p, t, o in zip(preds, tags, outs):
        judge_pool(llm, tok, sp, p, t, o, crag)


if __name__ == "__main__":
    main()
