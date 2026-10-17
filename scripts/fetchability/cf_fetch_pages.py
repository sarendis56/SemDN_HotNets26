"""Bulk Cloudflare /markdown fetch of live SERP pages, for the fetcher comparison
(our trafilatura GET vs the CF cloud clean-fetch). For a reservoir subset of the live
questions, fetch the top-K SERP URLs via CF /markdown (server-side render: gets past
network failures and bot-403s a plain GET hits), normalise with clean_markdown, and cache.

This gives a UNIFORM CF pool (every reachable URL cleaned the same way), so the chunker
comparison is fair, plus the trafilatura overlap for the cleanliness report and the
trafilatura-blocked set for the recovery/completeness story.

Resumable (skips URLs already in cf_pages_cache or cf_recovered), concurrent, and
HARD-capped on calls to respect the $5 browser-rendering tier.

Run:  set -a; . ./.env; set +a
      uv run python scripts/fetchability/cf_fetch_pages.py --n 120 --topk 12 --max-calls 1500 --conc 8
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from scdn import paths
from scdn.cf_markdown import clean_markdown, url_to_markdown
from scdn.e4_granularity import load_crag

E4 = paths.DATA_ROOT / "e4"
SERP = E4 / "serp_deep.jsonl"
OUT = E4 / "cf_pages_cache.jsonl"
MIN_CHARS = 400  # same usable threshold as the trafilatura pipeline


def _keyed(path, key):
    return {json.loads(l)[key]: json.loads(l) for l in open(path)} if path.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120, help="questions (reservoir order, seed 0)")
    ap.add_argument("--topk", type=int, default=12, help="top-K SERP urls/q to CF-fetch")
    ap.add_argument("--max-calls", type=int, default=1500, help="HARD cap on CF /markdown calls")
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=90)
    args = ap.parse_args()

    qs = load_crag(args.n, 0)
    qids = {q["qid"] for q in qs}
    serp = {qid: r for qid, r in _keyed(SERP, "qid").items() if qid in qids}

    # build the URL work-list: top-K per question, in SERP rank order, deduped across qs
    urls, seen = [], set()
    for qid in (q["qid"] for q in qs):
        if qid not in serp:
            continue
        for r in serp[qid]["results"][: args.topk]:
            u = r["url"]
            if u not in seen:
                seen.add(u)
                urls.append(u)

    # resume: skip what we already have (cf_pages_cache) or can seed from cf_recovered
    have = _keyed(OUT, "url")
    recovered = _keyed(E4 / "cf_recovered.jsonl", "url")

    # seed cf_pages_cache from cf_recovered (clean their markdown) for any in our work-list
    seeded = 0
    if recovered:
        with open(OUT, "a") as f:
            for u in urls:
                if u in have or u not in recovered:
                    continue
                rec = recovered[u]
                if not rec.get("ok"):
                    continue
                txt = clean_markdown(rec.get("markdown", "") or "")
                ok = len(txt) >= MIN_CHARS
                row = {"url": u, "ok": ok, "method": "cf_md", "status": "recovered",
                       "n_chars": len(txt), "raw_chars": rec.get("n_chars", 0), "text": txt if ok else ""}
                f.write(json.dumps(row) + "\n")
                have[u] = row
                seeded += 1
    if seeded:
        print(f"seeded {seeded} pages from cf_recovered.jsonl", flush=True)

    todo = [u for u in urls if u not in have]
    if len(todo) > args.max_calls:
        print(f"work-list {len(todo)} > cap {args.max_calls}; fetching first {args.max_calls} "
              f"(SERP-rank order, so deeper urls of later questions are dropped)", flush=True)
        todo = todo[: args.max_calls]
    print(f"CF /markdown: {len(todo)} urls to fetch ({len(have)} cached), "
          f"conc={args.conc}, cap={args.max_calls}", flush=True)
    if not todo:
        print("nothing to do.", flush=True)
        return

    def fetch(u):
        t = time.time()
        try:
            md = url_to_markdown(u, timeout=args.timeout)
            txt = clean_markdown(md)
            ok = len(txt) >= MIN_CHARS
            return {"url": u, "ok": ok, "method": "cf_md", "status": "ok",
                    "n_chars": len(txt), "raw_chars": len(md), "text": txt if ok else "",
                    "secs": round(time.time() - t, 1)}
        except Exception as e:
            return {"url": u, "ok": False, "method": "cf_md",
                    "status": f"error:{type(e).__name__}", "n_chars": 0, "raw_chars": 0,
                    "text": "", "secs": round(time.time() - t, 1)}

    n_ok = n = 0
    t0 = time.time()
    with open(OUT, "a") as f, ThreadPoolExecutor(max_workers=args.conc) as ex:
        futs = {ex.submit(fetch, u): u for u in todo}
        for fut in as_completed(futs):
            row = fut.result()
            secs = row.pop("secs", 0)
            f.write(json.dumps(row) + "\n")
            f.flush()
            n += 1
            n_ok += int(row["ok"])
            if n % 25 == 0 or n == len(todo):
                rate = n / max(1e-6, time.time() - t0)
                print(f"  [{n}/{len(todo)}] ok={n_ok} ({n_ok/n:.0%}) "
                      f"{rate:.1f}/s last={secs}s", flush=True)
    print(f"done: {n_ok}/{n} usable ({n_ok/max(1,n):.0%}); "
          f"~{n} browser calls this run -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
