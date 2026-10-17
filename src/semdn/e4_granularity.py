"""E4 — page-vs-chunk granularity on the LIVE web (single-shot RAG over CRAG).

E2 already shows agentic locality over a complete, deeply-ranked index. On the
open web we have neither a complete index nor deep dense ranking: Serper returns
a shallow, *lexically* ranked ~10-page window per page (Google killed &num=100 in
Sept 2025; we paginate to ~50 at 1 credit/page). So E4 does NOT measure caching —
it asks the granularity question of E1 on real SERPs:

  Given the same reader and the same k=3 chunks fed to it, does dense CHUNK
  retrieval over a DEEP pool (all ~50 pages) beat chunk retrieval restricted to
  the top-K Serper pages (the page-granularity baseline an agent/CDN uses)?

Pipeline per CRAG question (the question text IS the query — single-shot):
  serp  : paginate Serper to `depth` URLs (cached, resumable; costs credits).
  fetch : lightweight HTTP + readability; render fallback for JS/bot-blocked
          pages (cached by URL, resumable; no credits). Honestly skip the
          unreachable (network/timeout/bot) — they shrink the usable pool.
  select: chunk usable pages (256-tok), e5-embed, and for each condition
          (page@1/3/5/10 = chunks from the top-K usable pages; chunk@all = the
          full pool) keep the top-3 chunks. Writes the selected chunk TEXTS so
          the reader (e4_answer.py) needs no corpus.

Stages share on-disk caches under data/e4/, run independently, and resume.

Run (network only, no GPU):  uv run python -m semdn.e4_granularity --stage serp
                             uv run python -m semdn.e4_granularity --stage fetch --render
     (GPU e5):  CUDA_VISIBLE_DEVICES=1 uv run python -m semdn.e4_granularity --stage select
"""

import argparse
import asyncio
import json
import random

import numpy as np

from semdn import paths
from semdn.e1_fetch import chunk_words, light_fetch, render_budgeted

E4 = paths.DATA_ROOT / "e4"
SERP = E4 / "serp_deep.jsonl"
PAGES = E4 / "pages_cache.jsonl"
POOL = E4 / "pool.jsonl"
POOL_QWEN3 = E4 / "pool_qwen3.jsonl"
POOL_CE = E4 / "pool_ce.jsonl"
CRAG = paths.FLASHRAG_DATASETS / "crag" / "test.jsonl"

MIN_CLEAN_CHARS = 400        # below this a lightweight fetch is treated as blocked/JS-required
PAGE_KS = [1, 3, 5, 10]      # page-granularity baselines: chunks from top-K usable pages
TOPK = 3                     # chunks fed to the reader (Search-R1 / E1 default)

# ---------------- structure-aware chunking + boilerplate removal ----------------
# trafilatura's clean text keeps paragraph breaks (\n) and tables (| rows). Naive
# fixed 256-token windows ignore both — splitting answer tables, packing boilerplate
# next to content, and decontextualizing. This chunker respects block boundaries,
# keeps tables atomic, drops boilerplate, and prepends the page title for context.
import re  # noqa: E402

_BOILER = re.compile(
    r"\b(followers?|\d+\s*posts|subscribe|sign\s?in|sign\s?up|log\s?in|cookies?|"
    r"privacy policy|terms of (use|service)|newsletter|advertisement|advertis|"
    r"all rights reserved|share this|read more|click here|skip to (content|main)|"
    r"accept all|manage (consent|cookies))\b", re.I)


def _is_table_row(line: str) -> bool:
    return line.count("|") >= 2


def _is_boilerplate(block: str) -> bool:
    b = block.strip()
    if _is_table_row(b):
        return False  # never drop table content
    if len(b) < 40:
        return True   # nav/menu fragment
    letters = sum(c.isalpha() for c in b)
    if letters / max(1, len(b)) < 0.5:
        return True   # mostly symbols/numbers (menus, counters)
    if _BOILER.search(b[:160]) and len(b) < 220:
        return True   # short boilerplate line
    return False


def chunk_structure(text: str, title: str = "", target_tokens: int = 256) -> list[str]:
    lines = text.split("\n")
    # group contiguous table rows into atomic blocks; other non-empty lines are paragraphs
    blocks, i = [], 0
    while i < len(lines):
        if _is_table_row(lines[i]):
            j = i
            while j < len(lines) and _is_table_row(lines[j]):
                j += 1
            blocks.append(("table", "\n".join(lines[i:j])))
            i = j
        else:
            if lines[i].strip():
                blocks.append(("para", lines[i].strip()))
            i += 1
    blocks = [(t, b) for t, b in blocks if not _is_boilerplate(b)]

    chunks, buf, buf_tok = [], [], 0

    def flush():
        nonlocal buf, buf_tok
        if buf:
            chunks.append(" ".join(buf)); buf, buf_tok = [], 0

    for t, b in blocks:
        ntok = len(b.split())
        if t == "table":
            flush()
            rows = b.split("\n")
            if ntok <= target_tokens * 2:
                chunks.append(b)                       # keep whole table atomic
            else:                                      # split huge table, repeat header row
                header, cur, curtok = rows[0], [rows[0]], len(rows[0].split())
                for r in rows[1:]:
                    rtok = len(r.split())
                    if curtok + rtok > target_tokens * 2 and len(cur) > 1:
                        chunks.append("\n".join(cur)); cur, curtok = [header], len(header.split())
                    cur.append(r); curtok += rtok
                if len(cur) > 1:
                    chunks.append("\n".join(cur))
        elif ntok > target_tokens:                     # oversized paragraph (no breaks): word-split
            flush()
            w = b.split()
            chunks += [" ".join(w[k:k + target_tokens]) for k in range(0, len(w), target_tokens)]
        else:                                          # pack small paragraphs up to target
            if buf and buf_tok + ntok > target_tokens:
                flush()
            buf.append(b); buf_tok += ntok
    flush()

    t = title.strip()
    if t:
        chunks = [c if c.lower().startswith(t.lower()[:30]) else f"{t}. {c}" for c in chunks]
    return [c for c in chunks if c.strip()]


# ---------------- COHERENT chunking (a 3rd strategy) ----------------
# Naive `fixed` cuts blindly mid-sentence; `structure`/`structnt` respects blocks but
# leaks pipe-formatted boilerplate, emits ~14% tiny orphan fragments, and keeps sparse
# mega-tables (rows of empty cells). `coherent` fixes all three: split only at sentence
# ends, merge orphans so no fragment is < min_words, drop nav/boilerplate even in table
# form, compact sparse tables (drop empty cells), and size to ~256 tokens (~190 words).
_NAV = re.compile(
    r"(view this page on wikipedia|from wikipedia, the free encyclopedia|main page\s*:|"
    r"jump to|edit source|retrieved from|this page uses creative commons|view authors|"
    r"^\s*categories\s*:|^\s*references\b|^\s*external links\b|^\s*see also\b)", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def _coherent_boiler(block: str) -> bool:
    b = block.strip()
    if _is_table_row(b):
        return False                       # tables handled separately (and compacted)
    if len(b) < 25 or _NAV.search(b[:80]):
        return True
    letters = sum(c.isalpha() for c in b)
    return letters / max(1, len(b)) < 0.5  # mostly symbols/numbers -> menu/counter junk


def _compact_row(line: str) -> str:
    cells = [c.strip() for c in line.split("|")]
    return " | ".join(c for c in cells if c)  # drop empty cells from sparse spec tables


def _pack_sentences(text: str, target: int) -> list[str]:
    """Greedily pack whole sentences up to `target` words; never break inside one
    (a lone monster sentence is word-split only as a last resort)."""
    out, buf, n = [], [], 0
    for s in _SENT_SPLIT.split(text):
        w = len(s.split())
        if buf and n + w > target:
            out.append(" ".join(buf)); buf, n = [], 0
        if w > target:
            words = s.split()
            out += [" ".join(words[k:k + target]) for k in range(0, len(words), target)]
        else:
            buf.append(s); n += w
    if buf:
        out.append(" ".join(buf))
    return out


def chunk_coherent(text: str, title: str = "", target_words: int = 190,
                   min_words: int = 35, max_words: int = 320) -> list[str]:
    lines = text.split("\n")
    blocks, i = [], 0
    while i < len(lines):
        if _is_table_row(lines[i]):
            j, rows = i, []
            while j < len(lines) and _is_table_row(lines[j]):
                cr = _compact_row(lines[j])
                if len(cr.split()) >= 2:        # drop all-empty / 1-cell table noise
                    rows.append(cr)
                j += 1
            if rows:
                blocks.append(("table", rows))
            i = j
        else:
            ln = lines[i].strip()
            if ln and not _coherent_boiler(ln):
                blocks.append(("para", ln))
            i += 1

    chunks, buf, n = [], [], 0

    def flush():
        nonlocal buf, n
        if buf:
            chunks.append(" ".join(buf)); buf, n = [], 0

    for t, b in blocks:
        if t == "table":
            flush()
            total = sum(len(r.split()) for r in b)
            if total <= max_words:
                chunks.append("\n".join(b))
            else:                                # split big table, repeat the header row
                header = b[0]
                cur, curn = [header], len(header.split())
                for r in b[1:]:
                    rw = len(r.split())
                    if curn + rw > target_words and len(cur) > 1:
                        chunks.append("\n".join(cur)); cur, curn = [header], len(header.split())
                    cur.append(r); curn += rw
                if len(cur) > 1:
                    chunks.append("\n".join(cur))
        else:
            w = len(b.split())
            if w > target_words:
                flush()
                chunks += _pack_sentences(b, target_words)
            else:
                if buf and n + w > target_words:
                    flush()
                buf.append(b); n += w
    flush()

    # merge orphans: a short prose line (e.g. a table caption / stray label) is prepended
    # to the FOLLOWING chunk so it leads its content; a trailing short line folds into the
    # last chunk. Short standalone tables stay atomic. Result: no prose fragment < min_words.
    out: list[str] = []
    pending = ""
    for c in chunks:
        if pending:
            c, pending = f"{pending} {c}", ""
        if len(c.split()) < min_words and "|" not in c:
            pending = c                      # hold to prepend to the next chunk
        else:
            out.append(c)
    if pending:
        if out:
            out[-1] = f"{out[-1]} {pending}"
        else:
            out.append(pending)
    return [c for c in out if c.strip()]


def load_crag(n: int, seed: int) -> list[dict]:
    """Same sampling as gen_traces.load_questions (shuffle(seed)[:n]) but keeps metadata."""
    rows = []
    with open(CRAG) as fh:
        for line in fh:
            r = json.loads(line)
            rows.append({"qid": str(r["id"]), "question": r["question"],
                         "golden_answers": r["golden_answers"], "metadata": r.get("metadata", {})})
    random.Random(seed).shuffle(rows)
    return rows[:n] if n else rows


def _load_jsonl_keyed(path, key):
    if not path.exists():
        return {}
    return {json.loads(l)[key]: json.loads(l) for l in open(path)}


# ---------------- stage: serp (deep pagination, cached) ----------------
def stage_serp(qs, depth, credit_cap=0):
    from semdn.serp import search_deep
    done = _load_jsonl_keyed(SERP, "qid")
    todo = [q for q in qs if q["qid"] not in done]
    print(f"serp: {len(todo)} questions to fetch ({len(done)} cached), depth={depth}, "
          f"credit_cap={credit_cap or 'none'}", flush=True)
    spent = 0
    with open(SERP, "a") as f:
        for i, q in enumerate(todo):
            if credit_cap and spent + depth // 10 + 2 > credit_cap:   # stop before risking overspend
                print(f"  credit cap reached ({spent}/{credit_cap}); stopping at {i} new questions",
                      flush=True)
                break
            try:
                d = search_deep(q["question"], depth=depth)
            except Exception as e:
                print(f"  serp error {q['qid']}: {type(e).__name__}: {e}", flush=True)
                continue
            spent += d["credits"]
            f.write(json.dumps({"qid": q["qid"], "question": q["question"],
                                "results": d["results"], "credits": d["credits"]}) + "\n")
            f.flush()
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(todo)}] {len(d['results'])} urls, "
                      f"{spent} credits this run", flush=True)
    print(f"serp done; ~{spent} credits spent this run -> {SERP}", flush=True)


# ---------------- stage: fetch (light + render fallback, cached by url) ----------------
async def _fetch_all(urls, render, conc, timeout):
    cached = _load_jsonl_keyed(PAGES, "url")
    todo = [u for u in urls if u not in cached]
    print(f"fetch: {len(todo)} new urls ({len(cached)} cached), render={render}, "
          f"timeout={timeout}s", flush=True)
    if not todo:
        return
    sem = asyncio.Semaphore(conc)
    rsem = asyncio.Semaphore(max(2, conc // 2))   # SEPARATE gate for render: render_page
    wlock = asyncio.Lock()                         # acquires it internally, so it must not be `sem`
    fout = open(PAGES, "a")
    n = [0]
    browser = None
    if render:
        from playwright.async_api import async_playwright
        pw = await async_playwright().start()
        browser = await pw.chromium.launch(args=["--no-sandbox"])
    har_dir = E4 / "har"; har_dir.mkdir(parents=True, exist_ok=True)

    async def one(url):
        async with sem:                            # light fetch holds `sem` only for itself
            b = await asyncio.to_thread(light_fetch, url, timeout)
        text = b.get("clean_text", "") or ""
        status = b.get("http_status", b.get("fetch_status", 0))
        method = "light"
        # Render ONLY where it can help: light got a real HTTP response but thin
        # text (JS-required / bot-403). If light hit a network error (no
        # http_status — network/DNS/timeout), render uses the same host network and
        # cannot reach it either, so skip it instead of burning a render timeout.
        # Gate render on `rsem` (NOT `sem`): render_page acquires its arg internally,
        # so passing the already-held `sem` would self-deadlock until the 40s budget.
        # Render ONLY 2xx-but-thin (JS shells, e.g. Reddit). 4xx (bot/auth/rate) and
        # network errors fail in headless Chromium too, so rendering them only burns the
        # 40s budget — skip them (verified: a Cloudflare 403 times out in the browser).
        hs = b.get("http_status")
        soft = isinstance(hs, int) and 200 <= hs < 300
        if render and soft and len(text) < MIN_CLEAN_CHARS and browser is not None:
            r = await render_budgeted(browser, url, har_dir / f"{abs(hash(url))}.har", rsem)
            rt = r.get("rendered_text", "") or ""
            if len(rt) > len(text):
                text, method = rt, "render"
        ok = len(text) >= MIN_CLEAN_CHARS
        rec = {"url": url, "ok": ok, "method": method if ok else "blocked",
               "status": status, "n_chars": len(text), "text": text if ok else ""}
        async with wlock:
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            n[0] += 1
            if n[0] % 50 == 0:
                print(f"  [{n[0]}/{len(todo)}] fetched", flush=True)
        return ok

    res = await asyncio.gather(*(one(u) for u in todo))
    fout.close()
    if browser is not None:
        await browser.close()
    print(f"fetch done; usable {sum(res)}/{len(res)} new ({sum(res)/max(1,len(res)):.0%}) -> {PAGES}",
          flush=True)


def stage_fetch(render, conc, timeout):
    serp = _load_jsonl_keyed(SERP, "qid")
    urls = []
    seen = set()
    for rec in serp.values():
        for r in rec["results"]:
            if r["url"] not in seen:
                seen.add(r["url"]); urls.append(r["url"])
    asyncio.run(_fetch_all(urls, render, conc, timeout))


# ---------------- stage: select (chunk retrieval per condition) ----------------
def _make_encoder(name, device):
    """Return (embed_docs, embed_queries) for the chosen encoder. No e5 pre-filter
    for qwen3 — it must score ALL pool chunks directly or the deep-pool comparison
    is contaminated by e5's recall."""
    if name == "e5":
        from semdn.retrieval_server import E5Encoder
        enc = E5Encoder(str(paths.E5_BASE_V2), device=device)
        return enc.encode_passages, enc.encode_queries
    if name == "qwen3":
        from semdn.qwen3_encoder import Qwen3Encoder
        enc = Qwen3Encoder(paths.QWEN3_EMB_06B, device=device, max_length=512)
        web_task = "Given a question, retrieve web passages that answer the question"
        return enc.encode_docs, (lambda qs: enc.encode_queries(qs, task=web_task))
    raise ValueError(name)


def _make_chunker(name):
    if name == "fixed":
        return lambda text, title: chunk_words(text, 256)
    if name == "structure":
        return lambda text, title: chunk_structure(text, title, 256)
    if name == "structnt":  # structure-aware WITHOUT the title prefix (ablation)
        return lambda text, title: chunk_structure(text, "", 256)
    if name == "coherent":  # 3rd strategy: sentence-aware, orphan-merged, table-compacted
        return lambda text, title: chunk_coherent(text, "")
    raise ValueError(name)


def _build_chunks(results, pages, chunker):
    """Usable pages in Serper rank order -> (chunk_texts, usable_rank_per_chunk, n_usable)."""
    ctext, crank, usable_rank = [], [], 0
    for r in results:
        pg = pages.get(r["url"])
        if not pg or not pg["ok"]:
            continue
        usable_rank += 1
        for ch in chunker(pg["text"], r.get("title", "")):
            ctext.append(ch); crank.append(usable_rank)
    return ctext, crank, usable_rank


def _snippet_candidates(results):
    """SERP snippets (title + snippet) as extra ranking candidates so the snippet
    competes head-to-head with the deep chunks. src_rank 0 marks a snippet origin."""
    txt, rk = [], []
    for r in results:
        s = r.get("snippet", "")
        if s:
            txt.append(f"{r.get('title', '')}\n{s}".strip())
            rk.append(0)
    return txt, rk


def _select_conditions(ctext, crank, scores):
    """Given per-chunk scores, pick top-3 for each pool depth. Identical selection
    logic for bi-encoder and cross-encoder — only `scores` differs."""
    cr = np.array(crank)
    conds = {}
    for K in PAGE_KS:
        mask = np.where(cr <= K)[0]
        if mask.size:
            top = mask[np.argsort(-scores[mask])[:TOPK]]
            conds[f"page@{K}"] = {"chunk_texts": [ctext[j] for j in top],
                                 "src_ranks": [int(cr[j]) for j in top]}
        else:
            conds[f"page@{K}"] = {"chunk_texts": [], "src_ranks": []}
    top = np.argsort(-scores)[:TOPK]
    conds["chunk@all"] = {"chunk_texts": [ctext[j] for j in top],
                          "src_ranks": [int(cr[j]) for j in top]}
    deep_max = max((int(cr[j]) for j in top), default=0)
    return conds, deep_max


def _run_select(qs, out_path, score_fn, label, chunker, hybrid=False):
    """Shared driver: build chunks, score with `score_fn(question, chunks)`, select, write.
    hybrid=True adds the SERP snippets as ranking co-candidates (n_usable stays the
    fetched-page count, so the question set matches the chunk-only arm exactly)."""
    serp = _load_jsonl_keyed(SERP, "qid")
    pages = _load_jsonl_keyed(PAGES, "url")
    done = _load_jsonl_keyed(out_path, "qid")
    todo = [q for q in qs if q["qid"] not in done and q["qid"] in serp]
    print(f"{label}: {len(todo)} questions ({len(done)} cached)", flush=True)
    fout = open(out_path, "a")
    for i, q in enumerate(todo):
        results = serp[q["qid"]]["results"]
        ctext, crank, n_usable = _build_chunks(results, pages, chunker)
        if hybrid:
            stext, srank = _snippet_candidates(results)
            ctext = ctext + stext; crank = crank + srank
        if not ctext:
            fout.write(json.dumps({"qid": q["qid"], "question": q["question"],
                "golden_answers": q["golden_answers"], "metadata": q["metadata"],
                "n_pages_serp": len(results), "n_usable": 0, "n_chunks": 0,
                "conds": {}, "deep_topk_max_rank": 0}) + "\n")
            fout.flush(); continue
        scores = score_fn(q["question"], ctext)
        conds, deep_max = _select_conditions(ctext, crank, scores)
        fout.write(json.dumps({"qid": q["qid"], "question": q["question"],
            "golden_answers": q["golden_answers"], "metadata": q["metadata"],
            "n_pages_serp": len(results), "n_usable": n_usable, "n_chunks": len(ctext),
            "conds": conds, "deep_topk_max_rank": deep_max}) + "\n")
        fout.flush()
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(todo)}] {q['qid']}: {n_usable} usable pages, "
                  f"{len(ctext)} chunks", flush=True)
    fout.close()
    print(f"{label} done -> {out_path}", flush=True)


def stage_select(qs, device, encoder, out_path, chunker_name, hybrid=False):
    """Build a live-web pool by ranking each question's chunks with `encoder`. Supports
    the bi-encoders (e5/qwen3), the lexical/shallow baselines (bm25/provided), and the
    Qwen3-reranker (qwen3rr) — the same ranker set as the CRAG depth sweep."""
    if encoder in ("e5", "qwen3"):
        embed_docs, embed_queries = _make_encoder(encoder, device)

        def score_fn(question, chunks):
            cemb = embed_docs(chunks).astype(np.float32)
            qv = embed_queries([question]).astype(np.float32)[0]
            return cemb @ qv
    elif encoder == "bm25":
        from semdn.e4_crag_t3 import bm25_scores
        score_fn = bm25_scores
    elif encoder == "provided":
        from semdn.e4_crag_t3 import provided_scores

        def score_fn(question, chunks):
            return provided_scores(chunks)
    elif encoder == "qwen3rr":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        mid = "Qwen/Qwen3-Reranker-0.6B"
        tok = AutoTokenizer.from_pretrained(paths.model(mid), padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(paths.model(mid), torch_dtype=torch.float16).to(device).eval()
        yes_id, no_id = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
        pre = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
               'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
        suf = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        instruct = "Given a web search query, retrieve passages that answer the query"

        @torch.no_grad()
        def score_fn(question, chunks, batch=32):
            texts = [f"{pre}<Instruct>: {instruct}\n<Query>: {question}\n<Document>: {c}{suf}" for c in chunks]
            out = []
            for i in range(0, len(texts), batch):
                inp = tok(texts[i:i+batch], padding=True, truncation=True, max_length=640, return_tensors="pt").to(device)
                logits = model(**inp).logits[:, -1, :]
                p = torch.softmax(torch.stack([logits[:, no_id], logits[:, yes_id]], dim=-1), dim=-1)[:, 1]
                out.extend(p.float().cpu().numpy().tolist())
            return np.array(out)
    else:
        raise ValueError(encoder)

    _run_select(qs, out_path, score_fn, f"select ({encoder}/{chunker_name}{'/hybrid' if hybrid else ''})",
                _make_chunker(chunker_name), hybrid)


def stage_rerank(qs, device, reranker_path, out_path, chunker_name):
    """Score chunks with a CROSS-ENCODER (true (query,chunk) relevance) instead of
    bi-encoder cosine — tests whether utility-aware ranking rescues the deep pool."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(paths.model(reranker_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        paths.model(reranker_path), torch_dtype=torch.float16).to(device).eval()

    @torch.no_grad()
    def score_fn(question, chunks, batch=64, max_length=512):
        out = []
        for i in range(0, len(chunks), batch):
            pairs = [[question, c] for c in chunks[i:i + batch]]
            inp = tok(pairs, padding=True, truncation=True, max_length=max_length,
                      return_tensors="pt").to(device)
            out.extend(model(**inp).logits.view(-1).float().cpu().numpy().tolist())
        return np.array(out)

    _run_select(qs, out_path, score_fn, f"rerank (cross-encoder/{chunker_name})", _make_chunker(chunker_name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["serp", "fetch", "select", "rerank"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--question-seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=50)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--fetch-conc", type=int, default=12)
    ap.add_argument("--fetch-timeout", type=int, default=8,
                    help="per-URL HTTP timeout; unreachable/dead URLs fail fast")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--encoder", default="e5", choices=["e5", "qwen3", "bm25", "provided", "qwen3rr"])
    ap.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3",
                    help="cross-encoder for --stage rerank")
    ap.add_argument("--chunker", default="fixed", choices=["fixed", "structure", "structnt", "coherent"])
    ap.add_argument("--hybrid", action="store_true",
                    help="add SERP snippets as ranking co-candidates (select stage)")
    ap.add_argument("--credit-cap", type=int, default=0,
                    help="serp stage: stop before this many credits are spent this run (0 = no cap)")
    args = ap.parse_args()
    E4.mkdir(parents=True, exist_ok=True)
    qs = load_crag(args.n, args.question_seed)
    suffix = {"fixed": "", "structure": "_struct", "structnt": "_structnt",
              "coherent": "_coherent"}[args.chunker]
    if args.stage == "serp":
        stage_serp(qs, args.depth, args.credit_cap)
    elif args.stage == "fetch":
        stage_fetch(args.render, args.fetch_conc, args.fetch_timeout)
    elif args.stage == "select":
        base = "pool" if args.encoder == "e5" else f"pool_{args.encoder}"
        hyb = "_hybrid" if args.hybrid else ""
        stage_select(qs, args.device, args.encoder, E4 / f"{base}{suffix}{hyb}.jsonl", args.chunker, args.hybrid)
    elif args.stage == "rerank":
        stage_rerank(qs, args.device, args.reranker, E4 / f"pool_ce{suffix}.jsonl", args.chunker)


if __name__ == "__main__":
    main()
