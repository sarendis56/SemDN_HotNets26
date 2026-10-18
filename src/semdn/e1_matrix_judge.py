"""wiki-18 matrix graded by the CRAG LLM judge (Llama-3.1-70B) instead of string F1, so readers
that answer in full sentences (Qwen3-32B) are not penalised for wording.

  prep:  data/e1/matrix_llm_<rtag>preds.jsonl -> data/e1/judge_in_matrix_llm_<rtag>.jsonl
         (one row per question x config; config = "<dataset>|<arm>")
  judge: bash scripts/e4_quality/wiki18_judge.sh   (both readers, one model load)
  (<rtag> "" = the Qwen2.5-7B run; its files are named qwen25_7b_)
  stats: accuracy (perfect|acceptable) per dataset/arm; paired McNemar, Holm over datasets.

Run:  uv run python -m semdn.e1_matrix_judge prep  --reader-tag qwen3_32b_
      uv run python -m semdn.e1_matrix_judge stats --reader-tag qwen3_32b_
"""
import argparse
import json

import numpy as np

from semdn import paths
from semdn.e1_matrix_answer import CONFIGS, DATASETS
from semdn.e4_stats import CORRECT, binom_two_sided, holm

E1 = paths.DATA_ROOT / "e1"
PAIRS = ([(f"chunk_{r}", f"page_{r}@{n}") for r in ("llm", "e5") for n in (1, 3, 5, 10)]
         + [("chunk_llm", "chunk_e5"), ("page_llm@10", "page_e5@10")])


def prep(rtag):
    meta = {}
    for ds in DATASETS.split(","):
        for l in open(E1 / f"matrix_llm_{ds}.jsonl"):
            r = json.loads(l)
            meta[(ds, r["qid"])] = (r["question"], r["golden_answers"])
    out = E1 / f"judge_in_matrix_llm_{rtag or 'qwen25_7b_'}.jsonl"
    with open(out, "w") as f:
        for l in open(E1 / f"matrix_llm_{rtag}preds.jsonl"):
            r = json.loads(l)
            for qid, a in zip(r["qids"], r["answers"]):
                q, g = meta[(r["dataset"], qid)]
                f.write(json.dumps({"qid": qid, "config": f"{r['dataset']}|{r['config']}",
                                    "question": q, "gold": g, "pred": a}) + "\n")
    print(f"wrote {out}")


def stats(rtag):
    ok = {}
    name = rtag or "qwen25_7b_"
    for l in open(E1 / f"judge_matrix_llm_{name}.raw.jsonl"):
        r = json.loads(l)
        ds, cfg = r["config"].split("|")
        ok.setdefault((ds, cfg), {})[r["qid"]] = int(r["label"] in CORRECT)
    dss = DATASETS.split(",")
    print(f"=== judge accuracy, reader tag {rtag!r} ===\n{'dataset':16} " + " ".join(f"{c:>11}" for c in CONFIGS))
    for ds in dss:
        print(f"{ds:16} " + " ".join(f"{np.mean(list(ok[(ds, c)].values())):11.3f}" for c in CONFIGS))
    print(f"{'average':16} " + " ".join(
        f"{np.mean([np.mean(list(ok[(ds, c)].values())) for ds in dss]):11.3f}" for c in CONFIGS))
    out = []
    for A, B in PAIRS:
        print(f"\n--- {A} vs {B} (Δacc = A − B; McNemar, Holm over {len(dss)} datasets) ---")
        rows = []
        for ds in dss:
            a, b = ok[(ds, A)], ok[(ds, B)]
            q = sorted(a.keys() & b.keys())
            x, y = np.array([a[k] for k in q]), np.array([b[k] for k in q])
            bb, cc = int((x > y).sum()), int((y > x).sum())
            rows.append((ds, len(q), x.mean() - y.mean(), bb, cc, binom_two_sided(bb, bb + cc)))
        for (ds, n, d, bb, cc, p), pa in zip(rows, holm([r[-1] for r in rows])):
            print(f"{ds:16} n={n:4d}  Δacc={d:+.3f}  {bb}/{cc}  p={p:.4f}  pHolm={pa:.4f}  {'SIG' if pa < 0.05 else ''}")
            out.append({"A": A, "B": B, "dataset": ds, "n": n, "dacc": float(d), "p": p, "p_holm": pa})
        bb, cc = sum(r[3] for r in rows), sum(r[4] for r in rows)
        n = sum(r[1] for r in rows)
        p = binom_two_sided(bb, bb + cc)
        print(f"{'pooled':16} n={n:4d}  Δacc={(bb - cc) / n:+.3f}  {bb}/{cc}  p={p:.4f}")
        out.append({"A": A, "B": B, "dataset": "pooled", "n": n, "dacc": (bb - cc) / n, "p": p})
    json.dump(out, open(E1 / f"judge_matrix_llm_{name}significance.json", "w"), indent=1)
    print(f"\nwrote judge_matrix_llm_{name}significance.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "stats"])
    ap.add_argument("--reader-tag", default="", help='"" = Qwen2.5-7B run, "qwen3_32b_" = Qwen3-32B')
    a = ap.parse_args()
    (prep if a.stage == "prep" else stats)(a.reader_tag)
