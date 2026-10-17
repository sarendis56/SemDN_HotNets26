"""Byte-capacity edge cache simulation over a synthesized request stream.

Policies:
  page_lru    today's CDN: cache keyed by source page; a chunk hits only if its
              whole page is cached; a miss fetches the whole page's bytes.
  chunk_lru   semantic cache at chunk granularity.
  chunk_traj  chunk_lru + trajectory prefetch: after each turn, prefetch the
              turn's deeper-ranked candidates (ranks k+1..k+budget).
  chunk_sem   chunk_lru + semantic-neighbor prefetch: after each turn, prefetch
              top-`budget` ANN neighbors of the served chunks (needs a
              precomputed neighbor table).

Origin bytes include prefetch fetches — speculation is not free.
"""

from collections import OrderedDict

import numpy as np

from semdn.stream import Event


class LRU:
    def __init__(self, capacity_bytes: float):
        self.cap = capacity_bytes
        self.used = 0
        self.d: OrderedDict[int, int] = OrderedDict()

    def __contains__(self, key: int) -> bool:
        return key in self.d

    def touch(self, key: int) -> None:
        self.d.move_to_end(key)

    def insert(self, key: int, size: int) -> list[int]:
        evicted = []
        if key in self.d:
            self.touch(key)
            return evicted
        while self.used + size > self.cap and self.d:
            k, s = self.d.popitem(last=False)
            self.used -= s
            evicted.append(k)
        if self.used + size <= self.cap:
            self.d[key] = size
            self.used += size
        return evicted


def simulate(
    events: list[Event],
    policy: str,
    capacity_bytes: float,
    sizes: np.ndarray,
    page_ids: np.ndarray | None = None,
    page_sizes: np.ndarray | None = None,
    prefetch_budget: int = 8,
    neighbors: dict[int, list[int]] | None = None,
) -> dict:
    cache = LRU(capacity_bytes)
    hits = requests = 0
    origin_bytes = 0
    demand_bytes = 0
    prefetched_now: set[int] = set()   # in cache via prefetch, not yet used
    prefetch_count = prefetch_used = 0

    def key_size(chunk: int) -> tuple[int, int]:
        if policy == "page_lru":
            p = int(page_ids[chunk])
            return p, int(page_sizes[p])
        return chunk, int(sizes[chunk])

    for e in events:
        for chunk in e.ids:
            key, size = key_size(chunk)
            requests += 1
            demand_bytes += int(sizes[chunk])
            if key in cache:
                hits += 1
                cache.touch(key)
                if key in prefetched_now:
                    prefetch_used += 1
                    prefetched_now.discard(key)
            else:
                origin_bytes += size
                for ev in cache.insert(key, size):
                    prefetched_now.discard(ev)

        if policy in ("chunk_traj", "chunk_sem") and prefetch_budget > 0:
            if policy == "chunk_traj":
                cands = list(e.deep_ids[:prefetch_budget])
            else:
                cands = []
                for chunk in e.ids:
                    cands.extend((neighbors or {}).get(chunk, []))
                cands = cands[: prefetch_budget * len(e.ids)]
            for c in cands:
                if c in cache:
                    continue
                origin_bytes += int(sizes[c])
                prefetch_count += 1
                for ev in cache.insert(c, int(sizes[c])):
                    prefetched_now.discard(ev)
                prefetched_now.add(c)

    return {
        "policy": policy,
        "capacity_bytes": capacity_bytes,
        "requests": requests,
        "hit_ratio": hits / requests if requests else float("nan"),
        "origin_bytes": origin_bytes,
        "demand_bytes": demand_bytes,
        "origin_to_demand": origin_bytes / demand_bytes if demand_bytes else float("nan"),
        "prefetch_count": prefetch_count,
        "prefetch_precision": prefetch_used / prefetch_count if prefetch_count else float("nan"),
    }
