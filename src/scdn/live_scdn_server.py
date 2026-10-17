"""Live SCDN: a build-as-you-go semantic edge cache over the real web, exposing
the same /retrieve interface Search-R1 uses (so gen_traces.py drives it unchanged
— just point --retrieval-url here instead of at the wiki-18 server).

Per (sub-)query:
  1. embed q (e5); ANN over the incremental chunk cache.
  2. HIT if top cached score >= tau -> serve cached chunks, 0 origin bytes.
  3. MISS -> serper(q, num=100) -> fetch top-F URLs (lightweight; render JS-required
     if --render) -> skip blocked -> chunk (256-tok) -> e5-embed -> add to cache ->
     serve top-k from the updated cache.

No LEANN (embeddings stored directly in a faiss flat index; low storage overhead
at this scale). Per-query metrics are logged to --metrics-log for E4 analysis.

Run:  CUDA_VISIBLE_DEVICES=1 uv run python -m scdn.live_scdn_server --device cuda --render
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from scdn import paths
from scdn.e1_fetch import chunk_words, light_fetch
from scdn.retrieval_server import E5Encoder
from scdn.serp import search as serper_search

MIN_CLEAN_CHARS = 400  # below this, treat the lightweight fetch as JS-required


class RetrieveReq(BaseModel):
    queries: list[str]
    topk: int = 3
    return_scores: bool = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=str(paths.E5_BASE_V2))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--serper-num", type=int, default=100)
    ap.add_argument("--fetch-n", type=int, default=100, help="URLs to fetch per miss")
    ap.add_argument("--tau", type=float, default=0.86, help="e5 cosine cache-hit threshold")
    ap.add_argument("--chunk-tokens", type=int, default=256)
    ap.add_argument("--render", action="store_true", help="render JS-required pages")
    ap.add_argument("--fetch-conc", type=int, default=16)
    ap.add_argument("--fetch-timeout", type=int, default=8,
                    help="per-URL HTTP timeout; blocked/slow URLs fail fast")
    ap.add_argument("--metrics-log", default=str(paths.DATA_ROOT / "e1" / "e4_metrics.jsonl"))
    args = ap.parse_args()

    enc = E5Encoder(args.encoder, device=args.device)
    dim = enc.encode_queries(["dim probe"]).shape[1]
    index = faiss.IndexFlatIP(dim)
    chunk_texts: list[str] = []
    url_seen: dict[str, int] = {}     # url -> html bytes (0 if blocked); presence = cached
    pool = ThreadPoolExecutor(max_workers=args.fetch_conc)
    mlog = open(args.metrics_log, "w")
    render_browser = {"b": None}      # lazy Playwright (only if --render)

    def fetch_one(url: str) -> dict:
        try:
            b = light_fetch(url, timeout=args.fetch_timeout)
            clean = b.get("clean_text", "")
            if args.render and len(clean) < MIN_CLEAN_CHARS:
                rt = _render_text(url, render_browser)
                if len(rt) > len(clean):
                    clean = rt
            ok = b.get("http_status", 0) and b["http_status"] < 400 and len(clean) >= 50
            return {"url": url, "ok": bool(ok), "bytes": b.get("html_bytes", 0),
                    "clean": clean if ok else ""}
        except Exception:
            return {"url": url, "ok": False, "bytes": 0, "clean": ""}

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok", "cached_chunks": index.ntotal, "cached_urls": len(url_seen)}

    @app.post("/retrieve")
    def retrieve(req: RetrieveReq):
        result = []
        for q in req.queries:
            t0 = time.time()
            qv = enc.encode_queries([q]).astype(np.float32)
            top_score = float(index.search(qv, 1)[0][0][0]) if index.ntotal > 0 else 0.0
            hit = top_score >= args.tau
            urls_fetched = urls_blocked = origin_bytes = chunks_added = 0
            t_fetch = 0.0
            err = ""
            if not hit:
                # A miss must never crash the server: a bad/empty sub-query (serper
                # 400), a fetch failure, or an embed error degrades to "serve what
                # the cache has" rather than a 500 that kills the agent.
                try:
                    serp = serper_search(q, num=args.serper_num) if q.strip() else {}
                    urls = [o.get("link") for o in serp.get("organic", [])][: args.fetch_n]
                    new_urls = [u for u in urls if u and u not in url_seen]
                    tf = time.time()
                    fetched = list(pool.map(fetch_one, new_urls)) if new_urls else []
                    t_fetch = time.time() - tf
                    new_chunks = []
                    for fr in fetched:
                        url_seen[fr["url"]] = fr["bytes"]
                        if fr["ok"]:
                            urls_fetched += 1
                            origin_bytes += fr["bytes"]
                            new_chunks += chunk_words(fr["clean"], args.chunk_tokens)
                        else:
                            urls_blocked += 1
                    if new_chunks:
                        index.add(enc.encode_passages(new_chunks).astype(np.float32))
                        chunk_texts.extend(new_chunks)
                        chunks_added = len(new_chunks)
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    print(f"miss-path error for {q!r}: {err}", flush=True)
            # single final retrieval over the (possibly updated) cache
            served = []
            if index.ntotal > 0:
                sc, ids = index.search(qv, min(req.topk, index.ntotal))
                for i, s in zip(ids[0], sc[0]):
                    if i >= 0:
                        served.append({"document": {"id": int(i), "contents": chunk_texts[int(i)]},
                                       "score": float(s), "idx": int(i)})
            mlog.write(json.dumps({
                "query": q, "hit": hit, "pre_fetch_top_score": top_score,
                "urls_requested": args.serper_num, "urls_fetched": urls_fetched,
                "urls_blocked": urls_blocked, "origin_bytes": origin_bytes,
                "chunks_added": chunks_added, "cache_chunks": index.ntotal,
                "latency_s": round(time.time() - t0, 3), "fetch_s": round(t_fetch, 3),
                "err": err,
            }) + "\n")
            mlog.flush()
            result.append(served)
        return {"result": result}

    print(f"live SCDN on :{args.port} (serper={args.serper_num}, fetch={args.fetch_n}, "
          f"tau={args.tau}, render={args.render}) -> {args.metrics_log}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


def _render_text(url: str, holder: dict) -> str:
    """Synchronous one-page render fallback (lazy Playwright)."""
    try:
        from playwright.sync_api import sync_playwright
        if holder["b"] is None:
            holder["pw"] = sync_playwright().start()
            holder["b"] = holder["pw"].chromium.launch(args=["--no-sandbox"])
        ctx = holder["b"].new_context()
        page = ctx.new_page()
        page.goto(url, wait_until="load", timeout=20000)
        try:
            page.wait_for_load_state("networkidle", timeout=4000)
        except Exception:
            pass
        txt = page.inner_text("body")
        ctx.close()
        return txt
    except Exception:
        return ""


if __name__ == "__main__":
    main()
