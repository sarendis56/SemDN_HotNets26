"""Where does the gold answer disappear on the way from CRAG's raw HTML to the chunks the
iso-budget arms read? For the same 600 t3b questions, test whether any golden answer
string occurs in (1) the raw HTML with tags stripped, (2) trafilatura's extracted text,
(3) the cached coherent chunks (what every chunk AND page arm actually reads).

Matching: lowercase, drop '|' and commas, collapse whitespace, word-boundary substring.
Answers under 2 characters and yes/no answers are skipped (they match everywhere).

Run:  uv run python scripts/e4_quality/answer_survival.py      (CPU only, ~16 processes)
"""
import html as htmllib
import json
import pickle
import re
import sys
from multiprocessing import Pool

from semdn import paths
from semdn.e4_crag_t3 import reservoir_sample
from semdn.e4_isobudget import has_answer as found, norm, usable_answers

E4 = paths.DATA_ROOT / "e4"
SRC = paths.DATA_ROOT / "datasets" / "crag" / "crag_task_3_dev_v4"
TAG = re.compile(r"<(script|style)\b.*?</\1>|<[^>]+>", re.S | re.I)




def per_question(q):
    import trafilatura
    raw, ext = [], []
    for pg in q["pages"][:50]:
        h = pg["html"]
        if not h:
            continue
        raw.append(norm(htmllib.unescape(TAG.sub(" ", h))))
        try:
            t = trafilatura.extract(h, include_comments=False, include_tables=True) or ""
        except Exception:
            t = ""
        if len(t) >= 200:        # same usable-page rule as the chunk cache
            ext.append(norm(t))
    return q["qid"], " \n ".join(raw), " \n ".join(ext)


def main():
    chunks = {}
    for l in open(E4 / "chunks_crag_t3b_coherent.jsonl"):
        r = json.loads(l)
        chunks[r["qid"]] = r
    qs = reservoir_sample(SRC, 600, 0)
    assert all(q["qid"] in chunks for q in qs), "sample does not match the chunk cache"
    cache = E4 / "answer_survival_texts_t3b.pkl"   # the slow part; keep it across reruns
    if cache.exists():
        texts = pickle.load(open(cache, "rb"))
    else:
        with Pool(16) as pool:
            texts = {qid: (raw, ext) for qid, raw, ext in pool.imap_unordered(per_question, qs, chunksize=4)}
        pickle.dump(texts, open(cache, "wb"))

    rows = []
    for q in qs:
        ans = usable_answers(q["golden_answers"])
        if not ans:
            continue
        raw, ext = texts[q["qid"]]
        chk = norm(" \n ".join(chunks[q["qid"]]["chunks"]))
        rows.append({"qid": q["qid"], "domain": q["metadata"]["domain"], "answers": ans,
                     "raw": found(ans, raw), "extracted": found(ans, ext), "chunks": found(ans, chk)})
    json.dump(rows, open(E4 / "answer_survival_t3b.json", "w"), indent=1)

    n = len(rows)
    c = lambda k: sum(r[k] for r in rows)
    print(f"\n{n} of {len(qs)} questions have a matchable answer")
    print(f"answer in raw HTML text   : {c('raw'):4d} ({c('raw')/n:.1%})")
    print(f"answer in trafilatura text: {c('extracted'):4d} ({c('extracted')/n:.1%})")
    print(f"answer in cached chunks   : {c('chunks'):4d} ({c('chunks')/n:.1%})")
    lost_ext = [r for r in rows if r["raw"] and not r["extracted"]]
    lost_chk = [r for r in rows if r["extracted"] and not r["chunks"]]
    print(f"lost by trafilatura (raw yes, extracted no): {len(lost_ext)}")
    print(f"lost by the chunker (extracted yes, chunks no): {len(lost_chk)}")
    by = {}
    for r in lost_chk:
        by[r["domain"]] = by.get(r["domain"], 0) + 1
    print("chunker losses by domain:", by)
    for r in lost_chk[:15]:
        print("  ", r["domain"], r["answers"][:2])
    sys.stdout.flush()


if __name__ == "__main__":
    main()
