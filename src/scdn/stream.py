"""PoP request-stream synthesis and locality characterization from agent traces.

A trace (one task) is a sequence of turns; each turn carries the chunk ids the
retriever returned (first `k_shown` were shown to the agent; deeper ranks are
logged for prefetch analysis). A stream interleaves many sessions, each
replaying one task, with Zipf task popularity and Poisson session arrivals.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Event:
    time: float
    session: int
    task_id: str
    turn: int            # 1-based turn index within the task
    ids: tuple           # chunk ids requested/served this turn (k_shown)
    deep_ids: tuple      # deeper-ranked candidates from the same query


def load_traces(path: Path, k_shown: int = 3) -> list[dict]:
    traces = []
    with open(path) as f:
        for line in f:
            t = json.loads(line)
            if t["turns"]:
                for turn in t["turns"]:
                    turn["ids_shown"] = turn["ids"][:k_shown]
                    turn["ids_deep"] = turn["ids"][k_shown:]
                traces.append(t)
    return traces


def load_trace_variants(paths_: list, k_shown: int = 3) -> list[list[dict]]:
    """Load N variant trace files (same questions, different sampling seeds).

    Returns one list per task: its trajectory under each variant. Tasks missing
    from some variant (e.g. no search turns) keep whatever variants exist.
    """
    by_task: dict[str, list[dict]] = {}
    for p in paths_:
        for t in load_traces(p, k_shown=k_shown):
            by_task.setdefault(t["task_id"], []).append(t)
    return list(by_task.values())


def synthesize_stream(
    traces: list[dict] | list[list[dict]],
    n_sessions: int | None = None,
    alpha: float = 0.0,
    arrival_rate: float = 1.0,
    turn_spacing: float = 1.0,
    seed: int = 0,
) -> list[Event]:
    """alpha=0 + n_sessions=None -> every task exactly once (no duplicates).

    alpha>0 -> n_sessions draws with replacement, Zipf(alpha) task popularity.

    `traces` may be a flat list (one trajectory per task) or a list of variant
    lists (from load_trace_variants); each session replays a randomly chosen
    variant of its task, so repeated tasks need not repeat one trajectory.
    """
    rng = np.random.default_rng(seed)
    if n_sessions is None:
        chosen = rng.permutation(len(traces))
    else:
        ranks = rng.permutation(len(traces)) + 1
        w = ranks.astype(float) ** -alpha
        w /= w.sum()
        chosen = rng.choice(len(traces), size=n_sessions, replace=True, p=w)

    arrivals = np.cumsum(rng.exponential(1.0 / arrival_rate, size=len(chosen)))
    events = []
    for sess, (ti, t0) in enumerate(zip(chosen, arrivals)):
        trace = traces[ti]
        if isinstance(trace, list):  # variant list: pick one per session
            trace = trace[rng.integers(len(trace))]
        for j, turn in enumerate(trace["turns"]):
            events.append(
                Event(
                    time=t0 + j * turn_spacing,
                    session=sess,
                    task_id=trace["task_id"],
                    turn=turn["turn"],
                    ids=tuple(turn["ids_shown"]),
                    deep_ids=tuple(turn["ids_deep"]),
                )
            )
    events.sort(key=lambda e: e.time)
    return events


def chunk_access_seq(events: list[Event]) -> list[int]:
    return [c for e in events for c in e.ids]


def reuse_distances(seq: list[int]) -> np.ndarray:
    """LRU stack distance per access; np.inf for first access (compulsory miss)."""
    n = len(seq)
    bit = np.zeros(n + 1, dtype=np.int64)

    def bit_add(i: int, v: int) -> None:
        i += 1
        while i <= n:
            bit[i] += v
            i += i & (-i)

    def bit_sum(i: int) -> int:  # prefix sum over [0, i]
        i += 1
        s = 0
        while i > 0:
            s += bit[i]
            i -= i & (-i)
        return s

    last: dict[int, int] = {}
    out = np.empty(n)
    for p, x in enumerate(seq):
        l = last.get(x)
        if l is None:
            out[p] = np.inf
        else:
            out[p] = bit_sum(p) - bit_sum(l)  # distinct items strictly after l
            bit_add(l, -1)
        bit_add(p, 1)
        last[x] = p
    return out


def shuffled_events(events: list[Event], seed: int = 0) -> list[Event]:
    """Destroy task coherence: permute event order, keep global access multiset."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(events))
    return [events[i] for i in order]


def within_task_overlap(traces: list[dict]) -> dict:
    """P(chunk requested at turn t was already returned at an earlier turn of
    the same task), and consecutive-turn Jaccard, aggregated by turn index."""
    seen_frac: dict[int, list[float]] = {}
    jaccard: dict[int, list[float]] = {}
    for t in traces:
        seen: set = set()
        prev: set = set()
        for turn in t["turns"]:
            cur = set(turn["ids_shown"])
            k = turn["turn"]
            if k > 1:
                seen_frac.setdefault(k, []).append(len(cur & seen) / len(cur))
                union = cur | prev
                jaccard.setdefault(k, []).append(len(cur & prev) / len(union) if union else 0.0)
            seen |= cur
            prev = cur
    agg = lambda d: {k: float(np.mean(v)) for k, v in sorted(d.items())}
    return {"seen_before_frac": agg(seen_frac), "consec_jaccard": agg(jaccard)}


def deep_rank_reachability(traces: list[dict], depths: list[int]) -> dict:
    """P(chunk first needed at turn t+1 is within the top-`d` deeper ranks of
    some earlier turn's query in the same task) — trajectory-prefetch headroom.
    Only counts chunks NOT already shown earlier (those are trivial hits)."""
    out = {}
    for d in depths:
        hit, total = 0, 0
        for t in traces:
            shown: set = set()
            deep: set = set()
            for turn in t["turns"]:
                if turn["turn"] > 1:
                    fresh = [c for c in turn["ids_shown"] if c not in shown]
                    hit += sum(c in deep for c in fresh)
                    total += len(fresh)
                shown |= set(turn["ids_shown"])
                deep |= set(turn["ids_deep"][:d])
        out[d] = hit / total if total else float("nan")
    return out
