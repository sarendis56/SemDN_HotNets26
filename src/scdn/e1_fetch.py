"""E1 — the page tax: bytes along the real LLM-agent web-retrieval path.

The agent fetch path has a spectrum of content-acquisition modes (see the
sourced survey in e1_baseline_research.md). Per query (top-k SERP
results) we measure, for each, the bytes it moves:

  snippet   SERP snippets only (FreshPrompt-style floor; no page fetch).
  lightFetch  DOMINANT BASELINE: one HTTP GET of the HTML document, NO JS, no
              subresources, then readability extraction -> clean text. This is
              what GPT-Researcher, smolagents, WebGPT, Claude web_fetch, and
              default LangChain/LlamaIndex loaders do.
  fullRender  HEAVY UPPER BOUND: headless Chromium loads the page + all
              subresources + JS (ChatGPT Agent/Operator, WebVoyager). HAR gives
              wire transfer + request count + load time.
  scdnChunk   COUNTERFACTUAL: chunk the clean text, e5-rank vs the query, keep
              the global top-k chunks across the query's pages.

Bytes are reported as decompressed content bytes (consistent across arms); the
full-render arm additionally records real compressed wire transfer from the HAR.

Usage:
  uv run python -m scdn.e1_fetch --serp data/e1/serp.jsonl \
      --out data/e1/measure.jsonl --k 5
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np
import requests

from scdn import paths

TAU = 0.5
# render budgets (ms / s), overridable by env for larger-budget re-runs
GOTO_MS = int(os.environ.get("E1_GOTO_MS", "20000"))
IDLE_MS = int(os.environ.get("E1_IDLE_MS", "4000"))
INNER_S = float(os.environ.get("E1_INNER_S", "8"))
BUDGET_S = float(os.environ.get("E1_BUDGET_S", "45"))
UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def chunk_words(text: str, chunk_tokens: int = 256) -> list[str]:
    w = text.split()
    return [" ".join(w[i : i + chunk_tokens]) for i in range(0, max(1, len(w)), chunk_tokens)
            if w[i : i + chunk_tokens]]


# ---- fullRender: headless Chromium + HAR (hang-proof, concurrent) ----
async def render_page(browser, url: str, har_path: Path, sem) -> dict:
    """One page in its own context. Every await is bounded; the page is closed
    before the context so HAR flush never blocks on live (streaming) connections.
    The whole coroutine is also wrapped in a hard budget by the caller."""
    async with sem:
        ctx = await browser.new_context(record_har_path=str(har_path), user_agent=UA)
        page = await ctx.new_page()
        t0 = time.time()
        status, rendered = None, ""
        try:
            resp = await page.goto(url, wait_until="load", timeout=GOTO_MS)
            status = resp.status if resp else None
            try:
                await page.wait_for_load_state("networkidle", timeout=IDLE_MS)
            except Exception:
                pass
            rendered = await asyncio.wait_for(page.inner_text("body"), timeout=INNER_S)
        except Exception as e:
            status = status or f"error:{type(e).__name__}"
        load_s = time.time() - t0
        for closer in (page.close, ctx.close):  # page first => stops streams
            try:
                await asyncio.wait_for(closer(), timeout=6)
            except Exception:
                pass
        rec = {"render_status": status, "load_s": round(load_s, 2),
               "rendered_text": rendered}
        if har_path.exists():
            try:
                rec.update(har_totals(har_path))
            except Exception:
                pass
            har_path.unlink(missing_ok=True)
        return rec


async def render_budgeted(browser, url: str, har_path: Path, sem, budget: float = BUDGET_S) -> dict:
    try:
        return await asyncio.wait_for(render_page(browser, url, har_path, sem), timeout=budget)
    except (asyncio.TimeoutError, Exception) as e:
        har_path.unlink(missing_ok=True)
        return {"render_status": f"timeout:{type(e).__name__}", "rendered_text": ""}


def har_totals(har_path: Path) -> dict:
    har = json.loads(har_path.read_text())
    entries = har.get("log", {}).get("entries", [])
    transfer = content = 0
    for e in entries:
        resp = e.get("response", {})
        ts = resp.get("_transferSize", 0) or 0
        if ts <= 0:  # -1 = not available (e.g. blocked/cached); fall back
            ts = max(0, resp.get("bodySize", 0) or 0)
        transfer += ts
        content += (resp.get("content", {}) or {}).get("size", 0) or 0
    return {"render_transfer_bytes": transfer, "render_content_bytes": content,
            "n_requests": len(entries)}


# ---- lightFetch: single HTTP GET + readability ----
def light_fetch(url: str, timeout: int = 20) -> dict:
    import trafilatura

    try:
        r = requests.get(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip, deflate"},
                         timeout=timeout)
        html = r.content  # decompressed
        clean = trafilatura.extract(r.text, include_comments=False, include_tables=True) or ""
        return {"http_status": r.status_code, "html_bytes": len(html),
                "clean_text": clean, "clean_bytes": len(clean.encode("utf-8"))}
    except Exception as e:
        return {"fetch_status": f"error:{type(e).__name__}", "html_bytes": 0,
                "clean_text": "", "clean_bytes": 0}


# ---- scdnChunk ----
def useful_chunks(encoder, query: str, pages_text: list[str], k: int) -> int:
    chunks = [c for txt in pages_text for c in chunk_words(txt)]
    if not chunks:
        return 0
    cemb = encoder.encode_passages(chunks)
    qemb = encoder.encode_queries([query])[0]
    top = np.argsort(-(cemb @ qemb))[:k]
    return int(sum(len(chunks[i].encode("utf-8")) for i in top))


_ROW_KEYS = ("render_status", "load_s", "render_transfer_bytes", "render_content_bytes",
             "n_requests", "http_status", "html_bytes", "clean_bytes", "fetch_status",
             "js_required")


async def measure_query(browser, encoder, rec: dict, k: int, har_dir: Path, sem) -> dict:
    qid, query = rec["qid"], rec["query"]
    results = rec["results"][:k]
    snippet_bytes = sum(len((r.get("snippet") or "").encode("utf-8")) for r in results)

    async def one(ui: int, res: dict):
        url = res["url"]
        r, b = await asyncio.gather(
            render_budgeted(browser, url, har_dir / f"{qid}_{ui}.har", sem),
            asyncio.to_thread(light_fetch, url),
        )
        rendered, clean = r.get("rendered_text", ""), b.get("clean_text", "")
        merged = {**r, **b}
        row = {"url": url, "position": res.get("position")}
        row.update({key: merged[key] for key in _ROW_KEYS if key in merged})
        row["js_required"] = bool(len(rendered) > 0 and len(clean) < TAU * len(rendered))
        page_text = clean if len(clean) >= 0.5 * len(rendered) else (rendered or clean)
        return row, page_text

    pairs = await asyncio.gather(*(one(ui, res) for ui, res in enumerate(results)))
    rows = [p[0] for p in pairs]
    pages = [p[1] for p in pairs]
    useful = await asyncio.to_thread(useful_chunks, encoder, query, pages, k)
    return {"qid": qid, "cat": rec.get("cat", ""), "query": query,
            "n_results": len(results), "snippet_bytes": snippet_bytes,
            "scdn_useful_bytes": useful, "urls": rows}


async def run(args) -> None:
    from playwright.async_api import async_playwright

    from scdn.retrieval_server import E5Encoder

    har_dir = args.out.parent / "har"
    har_dir.mkdir(parents=True, exist_ok=True)
    encoder = E5Encoder(paths.E5_BASE_V2, device="cpu")
    recs = [json.loads(l) for l in open(args.serp)]
    if args.limit:
        recs = recs[: args.limit]
    done = {json.loads(l)["qid"] for l in open(args.out)} if args.out.exists() else set()
    todo = [r for r in recs if r["qid"] not in done]
    print(f"{len(todo)} queries to measure ({len(done)} cached)", flush=True)

    sem = asyncio.Semaphore(args.render_concurrency)
    qsem = asyncio.Semaphore(args.query_concurrency)
    wlock = asyncio.Lock()
    n_done = [0]
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox"])
        f = open(args.out, "a")

        async def do(rec):
            async with qsem:
                out = await measure_query(browser, encoder, rec, args.k, har_dir, sem)
            async with wlock:
                f.write(json.dumps(out) + "\n")
                f.flush()
                n_done[0] += 1
                ok = sum(1 for u in out["urls"] if isinstance(u.get("render_status"), int))
                print(f"[{n_done[0]}/{len(todo)}] {out['qid']}: {ok}/{out['n_results']} "
                      f"rendered, snippet={out['snippet_bytes']}B "
                      f"useful={out['scdn_useful_bytes']}B", flush=True)

        await asyncio.gather(*(do(r) for r in todo))
        f.close()
        await browser.close()
    print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serp", type=Path, default=paths.DATA_ROOT / "e1" / "serp.jsonl")
    ap.add_argument("--out", type=Path, default=paths.DATA_ROOT / "e1" / "measure.jsonl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--render-concurrency", type=int, default=4)
    ap.add_argument("--query-concurrency", type=int, default=3)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
