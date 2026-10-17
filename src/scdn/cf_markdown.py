"""Cloudflare Browser Rendering `/markdown` — turn HTML or a URL into clean, LLM-friendly
markdown. Creds from env (CF_ACCOUNT_ID, CF_API_TOKEN; kept in the gitignored .env).

Billing note: this is the Browser Rendering endpoint, billed by *browser-hours* against the
$5 plan's included tier — NOT by request frequency. Keep runs small; for pages where we
already have clean HTML, prefer local trafilatura and reserve this for the JS-required
minority. Paid-plan rate limit ~10 req/s.
"""

import os
import re

import requests

# ---- markdown -> plain-text normaliser ------------------------------------------------
# CF /markdown returns LLM-friendly markdown, but it keeps link/image SYNTAX and page
# chrome that trafilatura's text output never has (link_frac ~0.4-0.9 in pilots). To
# compare the CF fetcher against our trafilatura fetcher fairly, normalise CF markdown to
# trafilatura-like plain text (strip md syntax, keep anchor text + table pipes), then feed
# BOTH through the SAME chunkers — so boilerplate/short-fragment handling is identical and
# only the FETCH (reachability + render) differs.
_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_REFDEF = re.compile(r"^\s*\[[^\]]+\]:\s*\S+.*$")
_BARE = re.compile(r"https?://\S+")
_HRULE = re.compile(r"^\s*([-*_]\s*){3,}$")
_HEAD = re.compile(r"^[#>\s]*#+\s*")


def clean_markdown(md: str) -> str:
    """Normalise CF markdown to trafilatura-like plain text. Keeps anchor text and table
    pipe rows (the chunkers treat `|`-rows as atomic); drops link/image URLs, reference
    defs, horizontal rules, heading markers, and symbol-only short fragments."""
    out = []
    for ln in md.split("\n"):
        if _REFDEF.match(ln) or _HRULE.match(ln):
            continue
        s = _IMG.sub("", ln)
        s = _LINK.sub(r"\1", s)            # [text](url) -> text
        s = _BARE.sub("", s)              # leftover bare urls
        s = _HEAD.sub("", s)              # keep heading text, drop the ###
        s = s.strip(" \t")
        if not s:
            out.append("")                # preserve paragraph breaks for the chunkers
            continue
        letters = sum(c.isalpha() for c in s)
        if len(s) < 30 and letters / max(1, len(s)) < 0.6:
            continue                      # symbol/number/menu fragment (chunkers drop more)
        out.append(s)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def _endpoint() -> str:
    return (f"https://api.cloudflare.com/client/v4/accounts/{os.environ['CF_ACCOUNT_ID']}"
            "/browser-rendering/markdown")


def _post(payload: dict, timeout: int) -> str:
    r = requests.post(_endpoint(),
                      headers={"Authorization": f"Bearer {os.environ['CF_API_TOKEN']}"},
                      json=payload, timeout=timeout)
    r.raise_for_status()
    d = r.json()
    if not d.get("success"):
        raise RuntimeError(f"cloudflare markdown error: {d.get('errors')}")
    return d["result"]


def html_to_markdown(html: str, timeout: int = 90) -> str:
    """Convert raw HTML we already have (no fetch/JS render)."""
    return _post({"html": html}, timeout)


def fetch_via_accept_markdown(url: str, timeout: int = 10):
    """'Markdown for Agents' content-negotiation: a plain GET with `Accept: text/markdown`.
    Cloudflare-hosted sites that opted in return CLEANED markdown (nav/ads stripped — the
    agent feature, unlike the /markdown converter); everyone else silently returns text/html.
    No API token, no browser-hours. Returns (markdown, x_markdown_tokens) if served as
    markdown, else (None, None) so the caller falls back to normal fetch + trafilatura."""
    r = requests.get(url, headers={"Accept": "text/markdown"}, timeout=timeout)
    ct = r.headers.get("content-type", "").strip().lower()
    if ct.startswith("text/markdown"):
        return r.text, r.headers.get("x-markdown-tokens")
    return None, None


def url_to_markdown(url: str, wait: str = "networkidle0", reject_css: bool = True,
                    timeout: int = 120) -> str:
    """Let Cloudflare fetch+render the URL first (for JS-required pages). Trims CSS to cut
    browser-time; waits for network idle so JS-rendered content is present."""
    payload: dict = {"url": url, "gotoOptions": {"waitUntil": wait, "timeout": 30000}}
    if reject_css:
        payload["rejectRequestPattern"] = [r"/^.*\.(css)/"]
    return _post(payload, timeout)
