"""Build the E1 query workload: curated open-web intents (8 categories) +
a sample of real Google queries from Natural Questions (FlashRAG nq/dev).
Local-only (figures/), writes data/e1/queries.jsonl.
"""

import json
import random
from pathlib import Path

from semdn import paths

OUT = paths.DATA_ROOT / "e1" / "queries.jsonl"

# Curated, realistic assistant/agent intents that resolve to diverse, heavy,
# reachable open-web pages (news, commerce, docs, finance, health, howto,
# science, travel, entertainment, sports).
CURATED = {
    "news": [
        "latest developments US-China trade negotiations 2026",
        "what happened in the latest opec meeting",
        "current state of the ukraine ceasefire talks",
        "most recent us inflation report cpi",
        "latest ai regulation eu ai act updates",
    ],
    "commerce": [
        "best noise cancelling headphones 2026 reviews",
        "best robot vacuum for pet hair 2026",
        "iphone 17 pro vs samsung galaxy s26 comparison",
        "best mechanical keyboard for programming 2026",
        "best electric suv under 50000 dollars",
    ],
    "docs": [
        "how to configure nginx reverse proxy with ssl",
        "kubernetes deployment rolling update strategy example",
        "postgresql create index concurrently syntax",
        "rust async await tokio tutorial",
        "docker compose environment variables from file",
    ],
    "finance": [
        "current federal reserve interest rate decision",
        "how does a roth ira conversion ladder work",
        "nvidia stock price forecast 2026",
        "best high yield savings account rates 2026",
        "what is the current 30 year mortgage rate",
    ],
    "health": [
        "symptoms and treatment of long covid",
        "how much protein per day to build muscle",
        "ozempic side effects and weight loss results",
        "best exercises for lower back pain relief",
        "signs of vitamin d deficiency in adults",
    ],
    "howto": [
        "how to make sourdough bread starter from scratch",
        "how to remove a stripped screw",
        "how to negotiate a lower salary offer counteroffer",
        "how to clean a cast iron skillet properly",
        "how to set up a home network with vlans",
    ],
    "science": [
        "james webb telescope recent discoveries 2026",
        "how do mrna vaccines work explained",
        "what causes the aurora borealis northern lights",
        "latest fusion energy breakthrough net gain",
        "how does crispr gene editing work",
    ],
    "travel": [
        "best time to visit japan cherry blossom season",
        "is it safe to travel to thailand right now",
        "schengen visa requirements for us citizens 2026",
        "best all inclusive resorts in cancun 2026",
        "tokyo 5 day itinerary first time visitors",
    ],
    "entertainment": [
        "best movies to stream this month 2026",
        "when does the next season of stranger things release",
        "taylor swift 2026 tour dates and tickets",
        "highest rated video games of 2026 so far",
        "oscar best picture nominees 2026",
    ],
    "sports": [
        "nba trade deadline 2026 biggest moves",
        "premier league top scorers this season",
        "who is favored to win the 2026 world cup",
        "formula 1 2026 driver standings",
        "super bowl 2026 halftime show performer",
    ],
}


def build() -> None:
    rows = []
    for cat, qs in CURATED.items():
        for i, q in enumerate(qs):
            rows.append({"qid": f"{cat}_{i}", "cat": cat, "query": q})

    # real Google queries from Natural Questions dev (reachable-or-not; the SERP
    # is diverse and the byte ratio is a per-page property of whatever resolves).
    nq = paths.FLASHRAG_DATASETS / "nq" / "dev.jsonl"
    if nq.exists():
        pool = [json.loads(l) for l in open(nq)]
        for r in random.Random(0).sample(pool, 50):
            rows.append({"qid": f"nq_{r['id']}", "cat": "nq", "query": r["question"]})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} queries ({len(rows) - 50} curated + 50 NQ) -> {OUT}")


if __name__ == "__main__":
    build()
