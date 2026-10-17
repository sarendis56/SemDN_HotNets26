"""Cleanliness + completeness report: Cloudflare cloud clean-fetch vs our trafilatura GET,
on the SAME live SERP urls (the n/topk set actually fetched). Quantifies (1) recovery of
pages a plain GET could not fetch (network/bot-403/JS), and (2) text cleanliness on the overlap.

Run:  uv run python scripts/fetchability/cf_cleanliness_report.py
"""
import json
import re
import statistics as st

from scdn import paths

E4 = paths.DATA_ROOT / "e4"
_NAV = re.compile(r"(cookie|subscribe|sign ?in|sign ?up|newsletter|advertis|privacy policy|"
                  r"terms of|all rights reserved|skip to|accept all|^menu$|follow us|share this|©)", re.I)


def keyed(p, k="url"):
    return {json.loads(l)[k]: json.loads(l) for l in open(p)} if p.exists() else {}


def clean_stats(text):
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return None
    link = sum(1 for l in lines if l.count("](") >= 1 or l.count("http") >= 2)
    boiler = sum(1 for l in lines if _NAV.search(l))
    short = sum(1 for l in lines if len(l.strip()) < 30)
    return dict(chars=len(text), lines=len(lines),
                link=link / len(lines), boiler=boiler / len(lines), short=short / len(lines))


def main():
    traf = keyed(E4 / "pages_cache.jsonl")
    cf = keyed(E4 / "cf_pages_cache.jsonl")
    urls = list(cf)  # the set we CF-fetched
    print(f"CF-fetched urls: {len(urls)}")

    # completeness / recovery: of the urls CF fetched, what did each fetcher get?
    both = [u for u in urls if u in traf]
    traf_ok_in_both = sum(1 for u in both if traf[u].get("ok"))
    cf_ok_in_both = sum(1 for u in both if cf[u].get("ok"))
    # recovery: trafilatura-blocked but CF got it
    blocked = [u for u in both if not traf[u].get("ok")]
    recovered = [u for u in blocked if cf[u].get("ok")]
    print("\n=== COMPLETENESS (on the same urls) ===")
    print(f"  overlap (both attempted): {len(both)}")
    print(f"  trafilatura usable: {traf_ok_in_both}/{len(both)} ({traf_ok_in_both/max(1,len(both)):.0%})")
    print(f"  cloudflare  usable: {cf_ok_in_both}/{len(both)} ({cf_ok_in_both/max(1,len(both)):.0%})")
    print(f"  trafilatura-BLOCKED pages recovered by CF: {len(recovered)}/{len(blocked)} "
          f"({len(recovered)/max(1,len(blocked)):.0%})")

    # cleanliness on pages BOTH got (fair comparison of the extracted text)
    pairs = [u for u in both if traf[u].get("ok") and cf[u].get("ok")]
    print(f"\n=== CLEANLINESS (on {len(pairs)} pages both fetched; CF text already clean_markdown-normalised) ===")

    # per-page stats for the overlap (both texts are non-empty here, so clean_stats != None)
    paired = [(clean_stats(traf[u]["text"]), clean_stats(cf[u]["text"])) for u in pairs]
    paired = [(t, c) for t, c in paired if t and c]
    tr = [t for t, _ in paired]
    cc = [c for _, c in paired]
    for name, A in (("trafilatura", tr), ("cloudflare ", cc)):
        print(f"  {name}: chars med {int(st.median([x['chars'] for x in A])):6d}  "
              f"link/line {st.mean([x['link'] for x in A]):.2f}  "
              f"boiler/line {st.mean([x['boiler'] for x in A]):.2f}  "
              f"short/line {st.mean([x['short'] for x in A]):.2f}")
    ratios = [c["chars"] / max(1, t["chars"]) for t, c in paired]
    print(f"  CF/traf char ratio: median {st.median(ratios):.2f} (CF retains more page text)")
    json.dump({"cf_fetched": len(urls), "overlap": len(both),
               "traf_ok": traf_ok_in_both, "cf_ok": cf_ok_in_both,
               "recovered": len(recovered), "blocked": len(blocked),
               "n_clean_pairs": len(pairs)},
              open(E4 / "cf_cleanliness.json", "w"), indent=1)
    print(f"\nwrote {E4/'cf_cleanliness.json'}")


if __name__ == "__main__":
    main()
