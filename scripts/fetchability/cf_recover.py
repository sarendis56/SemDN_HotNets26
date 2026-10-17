"""Bounded Cloudflare recovery (experiment (a)): for a subset of live-web questions, fetch
the pages a plain GET could NOT get (pages_cache ok=False / absent) via the
Cloudflare url-form, and save the markdown. NO GPU. Hard call cap to respect the $5 budget.

Each url-render burns a few seconds of browser-time; we cap total calls and report the count
so browser-hours can be checked against the included tier. Timeouts are not billed.

Run:  set -a; . ./.env; set +a; uv run python scripts/fetchability/cf_recover.py --n-questions 25 --max-per-q 10 --max-calls 250
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from scdn import paths
from scdn.cf_markdown import url_to_markdown

E4 = paths.DATA_ROOT / "e4"
OUT = E4 / "cf_recovered.jsonl"
MIN_OK = 400  # below this (or a challenge page) = not really recovered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-questions", type=int, default=25)
    ap.add_argument("--max-per-q", type=int, default=10)
    ap.add_argument("--max-calls", type=int, default=250, help="HARD cap on Cloudflare url-renders")
    ap.add_argument("--conc", type=int, default=5)
    args = ap.parse_args()

    serp = [json.loads(l) for l in open(E4 / "serp_deep.jsonl")][: args.n_questions]
    pc = {json.loads(l)["url"]: json.loads(l) for l in open(E4 / "pages_cache.jsonl")}
    already = {json.loads(l)["url"] for l in open(OUT)} if OUT.exists() else set()

    # build the task list: per question, its blocked URLs in SERP-rank order, capped
    tasks = []
    for rec in serp:
        qid = rec["qid"]
        blocked = [r["url"] for r in rec["results"]
                   if not (pc.get(r["url"]) or {}).get("ok") and r["url"] not in already]
        for u in blocked[: args.max_per_q]:
            tasks.append((qid, u))
            if len(tasks) >= args.max_calls:
                break
        if len(tasks) >= args.max_calls:
            break
    print(f"{len(tasks)} blocked URLs to recover (<= cap {args.max_calls}, {args.n_questions} questions)", flush=True)

    def recover(qid, url):
        try:
            md = url_to_markdown(url)
            ok = len(md) >= MIN_OK and "blocked by network security" not in md[:400].lower()
            return {"qid": qid, "url": url, "ok": ok, "n_chars": len(md), "markdown": md if ok else ""}
        except Exception as e:
            return {"qid": qid, "url": url, "ok": False, "n_chars": 0, "markdown": "", "err": f"{type(e).__name__}"}

    t0 = time.time()
    n_ok = n_fail = 0
    with open(OUT, "a") as f, ThreadPoolExecutor(max_workers=args.conc) as ex:
        futs = {ex.submit(recover, q, u): u for q, u in tasks}
        for i, fut in enumerate(as_completed(futs)):
            r = fut.result()
            f.write(json.dumps(r) + "\n"); f.flush()
            n_ok += r["ok"]; n_fail += (not r["ok"])
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(tasks)}] recovered={n_ok} failed={n_fail}", flush=True)
    dt = time.time() - t0
    print(f"\ndone: {len(tasks)} calls, recovered {n_ok} ({n_ok/max(1,len(tasks)):.0%}), failed {n_fail}", flush=True)
    print(f"wall {dt:.0f}s; rough browser-time ~{dt*args.conc:.0f}s (~{dt*args.conc/3600:.2f} browser-hours) -> {OUT}", flush=True)
    print("CHECK your Cloudflare dashboard browser-hours against the $5 included tier.", flush=True)


if __name__ == "__main__":
    main()
