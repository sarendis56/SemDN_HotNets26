"""CRAG (Meta KDD Cup 2024) loader → FlashRAG-format jsonl, so gen_traces.py
loads it unchanged. Keeps the popularity/dynamism/domain labels as metadata.

CRAG task-1 line schema: {interaction_id, query, query_time, answer, alt_ans,
question_type, static_or_dynamic, domain, search_results:[...]}.

Run:  uv run python -m scdn.crag build --n 0   # 0 = all
"""

import argparse
import bz2
import json
from pathlib import Path

from scdn import paths

SRC = paths.DATA_ROOT / "datasets" / "crag" / "crag_task_1_dev_v4_release.jsonl.bz2"
OUT_DIR = paths.FLASHRAG_DATASETS / "crag"


def iter_crag(src: Path):
    op = bz2.open if str(src).endswith(".bz2") else open
    with op(src, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build(n: int = 0, seed: int = 0) -> None:
    import random

    rows = []
    for i, r in enumerate(iter_crag(SRC)):
        golden = [r["answer"]] + (r.get("alt_ans") or [])
        golden = [a for a in golden if a and a not in ("invalid question", "I don't know")]
        if not golden:
            continue
        rows.append({
            "id": r.get("interaction_id", f"crag_{i}"),
            "question": r["query"],
            "golden_answers": golden,
            "metadata": {
                "dynamism": r.get("static_or_dynamic", ""),
                "domain": r.get("domain", ""),
                "question_type": r.get("question_type", ""),
                "query_time": r.get("query_time", ""),
            },
        })
    if n:
        rows = random.Random(seed).sample(rows, min(n, len(rows)))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "test.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    import collections
    dyn = collections.Counter(r["metadata"]["dynamism"] for r in rows)
    dom = collections.Counter(r["metadata"]["domain"] for r in rows)
    print(f"wrote {len(rows)} CRAG questions -> {out}")
    print("dynamism:", dict(dyn))
    print("domain:", dict(dom))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build", "peek"])
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    if args.command == "peek":
        r = next(iter_crag(SRC))
        print("keys:", list(r.keys()))
        print("query:", r.get("query"))
        print("answer:", r.get("answer"), "| dynamism:", r.get("static_or_dynamic"),
              "| domain:", r.get("domain"), "| type:", r.get("question_type"))
        print("n_search_results:", len(r.get("search_results", [])))
    else:
        build(args.n)
