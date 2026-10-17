"""Source SERP results (URLs + snippets) via serper.dev — the search step of
the real agent fetch path. Reproducible: fixed query list -> cached results.

Each query costs 1 serper credit (num<=10). Results are cached to disk so a
re-run does not re-spend credits.

Usage:
  uv run python -m scdn.serp --queries data/e1/queries.jsonl \
      --out data/e1/serp.jsonl --num 10
"""

import argparse
import json
import time
from pathlib import Path

import requests

from scdn import paths
from scdn.config import get

ENDPOINT = "https://google.serper.dev/search"


def search(query: str, num: int = 10, gl: str = "us", hl: str = "en",
           page: int | None = None) -> dict:
    body = {"q": query, "num": num, "gl": gl, "hl": hl}
    if page is not None:
        body["page"] = page
    r = requests.post(
        ENDPOINT,
        headers={"X-API-KEY": get("SERP_API_KEY"), "Content-Type": "application/json"},
        json=body,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def search_deep(query: str, depth: int = 50, gl: str = "us", hl: str = "en") -> dict:
    """Paginate to ~`depth` organic URLs. Google killed &num=100 (Sept 2025), so a
    single call returns only ~10; we page through (1 credit/page) and merge,
    deduping by URL and preserving absolute rank. Returns {results, credits}."""
    results, seen, credits = [], set(), 0
    max_pages = depth // 10 + 2
    for pg in range(1, max_pages + 1):
        if len(results) >= depth:
            break
        d = search(query, num=10, gl=gl, hl=hl, page=pg)
        credits += d.get("credits", 1)
        organic = d.get("organic", [])
        if not organic:
            break
        for o in organic:
            u = o.get("link")
            if u and u not in seen:
                seen.add(u)
                results.append({"abs_rank": len(results) + 1, "url": u,
                                "title": o.get("title", ""), "snippet": o.get("snippet", "")})
    return {"results": results[:depth], "credits": credits,
            "answer_box": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, required=True, help="jsonl: {qid,query,cat?}")
    ap.add_argument("--out", type=Path, default=paths.DATA_ROOT / "e1" / "serp.jsonl")
    ap.add_argument("--num", type=int, default=10)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.out.exists():  # resume without re-spending credits
        done = {json.loads(l)["qid"] for l in open(args.out)}
        print(f"resuming; {len(done)} queries already cached")

    queries = [json.loads(l) for l in open(args.queries)]
    spent = 0
    with open(args.out, "a") as f:
        for q in queries:
            if q["qid"] in done:
                continue
            try:
                d = search(q["query"], num=args.num)
            except Exception as e:
                print(f"  serper error on {q['qid']}: {e}", flush=True)
                time.sleep(3)
                continue
            organic = d.get("organic", [])
            rec = {
                "qid": q["qid"],
                "cat": q.get("cat", ""),
                "query": q["query"],
                "answer_box": bool(d.get("answerBox")),
                "results": [
                    {"position": o.get("position"), "url": o.get("link"),
                     "title": o.get("title", ""), "snippet": o.get("snippet", "")}
                    for o in organic
                ],
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            spent += d.get("credits", 1)
            print(f"[{q['qid']}] {len(organic)} results (credits this run: {spent})",
                  flush=True)
    print(f"done; ~{spent} credits spent this run -> {args.out}")


if __name__ == "__main__":
    main()
