"""Why does trajectory deep-rank prefetch work? Trace the mechanism.

For each task, find "circle-back" events: a chunk c that sat in the DEEP ranks
(4..50) of an earlier turn i and later became a TOP-3 result of turn j>i,
without having been top-3 before j (a genuinely *new* useful chunk that was
foreseeable from an earlier query's tail). For each event we record:
  - turn gap (j - i, using the most recent earlier turn that had c deep)
  - c's rank at turn i
  - whether query_i and query_j are byte-identical (verbatim repeat) or differ
  - whether c, query_i's top-1, and query_j's top-1 share a Wikipedia page
Then dump qualitative examples with chunk titles and both query strings.
"""

import collections
import json

import numpy as np

from semdn import paths
from semdn.corpus import Corpus

VARIANT = paths.TRACES / "full-t07-s0.jsonl"          # examples from one variant
ALL = [paths.TRACES / f"full-t07-s{s}.jsonl" for s in (0, 1, 2)]  # stats from all

corpus = Corpus()


def title(cid: int) -> str:
    return corpus.contents(int(cid)).split("\n", 1)[0].strip().strip('"')


def load(path):
    out = []
    with open(path) as f:
        for line in f:
            t = json.loads(line)
            if t["turns"]:
                out.append(t)
    return out


def circle_back_events(task):
    """Yield dicts for each deep->top3 fresh chunk event in one task."""
    turns = task["turns"]
    shown_before = set()
    # for each chunk, the most recent earlier turn index where it was deep (rank>=4)
    deep_at = {}
    for j, tj in enumerate(turns):
        top3 = tj["ids"][:3]
        for c in top3:
            if c not in shown_before and c in deep_at:
                i = deep_at[c]
                ti = turns[i]
                rank_i = ti["ids"].index(c) + 1
                yield {
                    "task_id": task["task_id"],
                    "question": task["question"],
                    "chunk": c,
                    "chunk_title": title(c),
                    "turn_i": i + 1,
                    "turn_j": j + 1,
                    "gap": j - i,
                    "rank_i": rank_i,
                    "query_i": ti["query"],
                    "query_j": tj["query"],
                    "verbatim": ti["query"].strip().lower() == tj["query"].strip().lower(),
                    "top1_i": title(ti["ids"][0]),
                    "top1_j": title(tj["ids"][0]),
                }
        # update state AFTER scoring this turn
        for c in top3:
            shown_before.add(c)
        for r, c in enumerate(tj["ids"]):
            if r >= 3:
                deep_at[c] = j


# ---- aggregate stats across all variants ----
events = []
for p in ALL:
    for task in load(p):
        events.extend(circle_back_events(task))

n = len(events)
gaps = collections.Counter(e["gap"] for e in events)
ranks = [e["rank_i"] for e in events]
verb = sum(e["verbatim"] for e in events)
same_title_iq = sum(e["chunk_title"] == e["top1_i"] for e in events)
print(f"circle-back events (deep@earlier -> fresh top-3@later): {n}")
print(f"turn-gap distribution: {dict(sorted(gaps.items()))}")
print(f"rank-at-earlier: median={int(np.median(ranks))} mean={np.mean(ranks):.1f} "
      f"frac in 4-10={np.mean([r <= 10 for r in ranks]):.2f}")
print(f"earlier & later query byte-identical: {verb}/{n} = {verb / n:.3f}")
print(f"chunk shares its page with the earlier query's top-1 (sibling chunk): "
      f"{same_title_iq}/{n} = {same_title_iq / n:.3f}")


def words(q):
    return set(q.lower().split())


jacc = [len(words(e["query_i"]) & words(e["query_j"])) /
        len(words(e["query_i"]) | words(e["query_j"])) for e in events]
added = [len(words(e["query_j"]) - words(e["query_i"])) for e in events]
dropped = [len(words(e["query_i"]) - words(e["query_j"])) for e in events]
refine = sum(a >= 1 and d == 0 for a, d in zip(added, dropped))  # strict superset
print(f"query_i vs query_j word-set Jaccard: median={np.median(jacc):.2f} mean={np.mean(jacc):.2f}")
print(f"words added at later turn: median={int(np.median(added))} | "
      f"dropped: median={int(np.median(dropped))}")
print(f"later query is a strict superset of earlier (pure narrowing): "
      f"{refine}/{n} = {refine / n:.3f}")

# ---- qualitative examples from one variant: prefer different-query, small gap ----
ex = []
for task in load(VARIANT):
    ex.extend(circle_back_events(task))
ex_diff = sorted([e for e in ex if not e["verbatim"]], key=lambda e: (e["gap"], e["rank_i"]))
print("\n===== EXAMPLES (different query at the two turns) =====")
for e in ex_diff[:8]:
    print(f"\nQ: {e['question']}")
    print(f"  turn {e['turn_i']} query: {e['query_i']!r}")
    print(f"     -> chunk [{e['chunk']}] '{e['chunk_title']}' appears at rank {e['rank_i']}")
    print(f"  turn {e['turn_j']} query: {e['query_j']!r}   (gap {e['gap']} turns)")
    print(f"     -> same chunk now TOP-3; turn-{e['turn_j']} top-1 = '{e['top1_j']}'")
