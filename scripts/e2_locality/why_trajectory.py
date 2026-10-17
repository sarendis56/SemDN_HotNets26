"""Why does trajectory deep-rank prefetch beat neighbor-of-top-1 prefetch?

For each turn i with a fresh chunk X needed at turn i+1, contrast two predictors
the SCDN could use to have prefetched X:
  TRAJECTORY: deep ranks (4..M) of the SAME query q_i  (the "C/D" the query
              itself surfaced just below the top-3)
  NEIGHBORS : kNN / HNSW-graph neighbors of B_i = q_i's top-1 served chunk
Reports reachability of each, and prints titled examples where X is in q_i's deep
ranks but NOT in B_i's neighbors, to show C/D > B's neighbors with explanation.
"""

import json

import numpy as np

from scdn import paths
from scdn.corpus import Corpus

corpus = Corpus()
ann = {int(k): v for k, v in json.load(open(paths.REPO_ROOT / "outputs" / "ann_neighbors.json")).items()}
hnsw = {int(k): v for k, v in json.load(open(paths.REPO_ROOT / "outputs" / "hnsw_neighbors.json")).items()}


def title(cid):
    return corpus.contents(int(cid)).split("\n", 1)[0].strip().strip('"')


def load(path):
    return [json.loads(l) for l in open(path) if json.loads(l)["turns"]]


def dedup_repeats(turns):
    seen, out = set(), []
    for u in turns:
        q = u["query"].strip().lower()
        if q not in seen:
            seen.add(q)
            out.append(u)
    return out


traces = load(paths.TRACES / "full-t07-s0.jsonl")

DEEP_M = 10        # trajectory: deep ranks 4..(3+DEEP_M) of q_i
NBR_M = 10         # neighbor: top-NBR_M neighbors of B_i
reach = {"traj": 0, "knn": 0, "hnsw": 0}
total = 0
examples = []

for t in traces:
    turns = dedup_repeats(t["turns"])
    shown = set()
    for i in range(len(turns) - 1):
        qi = turns[i]
        Bi = qi["ids"][0]                       # top-1 served chunk of q_i
        deep = set(qi["ids"][3:3 + DEEP_M])     # C/D... : q_i's own deep ranks
        knn = set(ann.get(Bi, [])[:NBR_M])      # true-kNN of B_i
        gnb = set(hnsw.get(Bi, [])[:NBR_M])     # HNSW-graph neighbors of B_i
        shown |= set(qi["ids"][:3])
        nxt = turns[i + 1]
        for X in nxt["ids"][:3]:
            if X in shown:
                continue                        # only genuinely fresh chunks
            total += 1
            in_traj, in_knn, in_hnsw = X in deep, X in knn, X in gnb
            reach["traj"] += in_traj
            reach["knn"] += in_knn
            reach["hnsw"] += in_hnsw
            if in_traj and not in_knn and not in_hnsw and Bi in ann:
                examples.append({
                    "q_i": qi["query"], "q_next": nxt["query"],
                    "B_title": title(Bi), "X_title": title(X),
                    "X_rank_in_qi": qi["ids"].index(X) + 1,
                    "B_knn_titles": [title(c) for c in ann.get(Bi, [])[:5]],
                })

print(f"fresh next-turn chunks: {total}")
for k, v in reach.items():
    print(f"  reachable via {k:5s} (M={DEEP_M}): {v / total:.3f}")

# classify: "comparison/bridge" if X is a different entity than B (the question
# names two), vs "refinement" (same entity, deeper attribute)
def kind(e):
    bt, xt = e["B_title"].lower(), e["X_title"].lower()
    return "refine" if (bt in xt or xt in bt) else "bridge"


for e in examples:
    e["kind"] = kind(e)

seen_b = set()
print("\n===== examples: X in q_i deep ranks but NOT in B_i neighbors =====")
for e in examples:
    if e["B_title"] in seen_b:
        continue
    seen_b.add(e["B_title"])
    print(f"\n[{e['kind']}] q_i: {e['q_i']!r}")
    print(f"  top-1 B='{e['B_title']}'  neighbors={e['B_knn_titles']}")
    print(f"  q_next: {e['q_next']!r}")
    print(f"  -> X='{e['X_title']}' (rank {e['X_rank_in_qi']} of q_i; not a neighbor of B)")
    if len(seen_b) >= 20:
        break

# detailed text excerpt for the single cleanest bridge example
print("\n===== text excerpt (cleanest bridge example) =====")
for e in examples:
    if e["kind"] == "bridge" and "kerber" in e["X_title"].lower():
        print("(handled below)")
        break
