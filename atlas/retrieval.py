"""Stage 4a: retrieve source text for a concept.

Design doc section 9: a derivation must be located in a real source and
restated, never generated from memory, and a note may only cite sources
actually retrieved in that run. This module is the "actually retrieved" half;
`cited_but_not_retrieved` is the guard that makes the rule enforceable.

Two free, keyless sources:
  arxiv      papers, for algorithms
  wikipedia  full plaintext extracts, which carry the standard derivations for
             most canon mathematics

Both are best-effort. A concept with no grounded source is reported as such
rather than quietly falling back to recall.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from typing import Callable

ALIASES = None  # lazily loaded from data/source_aliases.json

USER_AGENT = "DerivationAtlas/0.1 (research tool; contact via repository)"
ARXIV = "http://export.arxiv.org/api/query"
WIKIPEDIA = "https://en.wikipedia.org/w/api.php"
ATOM = "{http://www.w3.org/2005/Atom}"

Transport = Callable[[str], bytes]

MIN_INTERVAL = 0.5  # polite floor between live requests
_last_request = 0.0


class RetrievalError(RuntimeError):
    """A source could not be reached. Distinct from a source not existing."""


@dataclass
class Source:
    """One retrieved document. `id` is what a derivation step may cite."""

    id: str
    title: str
    url: str
    kind: str  # "paper" | "encyclopedia" | "docs"
    text: str
    retrieved_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    @property
    def is_substantive(self) -> bool:
        """An abstract alone cannot ground a derivation."""
        return len(self.text) >= 1200

    def excerpt(self, limit: int = 400) -> str:
        return self.text[:limit].replace("\n", " ").strip()


def _fetch(url: str, transport: Transport | None = None, retries: int = 4) -> bytes:
    """Fetch with throttling and backoff.

    Batch retrieval trips rate limits, and a throttled request must never be
    mistaken for a missing article -- that would look like "this concept cannot
    be grounded" and licence a fallback to recall.
    """
    global _last_request
    if transport is not None:
        return transport(url)

    delay = 2.0
    for attempt in range(retries):
        gap = time.monotonic() - _last_request
        if gap < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - gap)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                _last_request = time.monotonic()
                return response.read()
        except urllib.error.HTTPError as exc:
            _last_request = time.monotonic()
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise RetrievalError(f"HTTP {exc.code} for {url}") from exc
        except urllib.error.URLError as exc:
            _last_request = time.monotonic()
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise RetrievalError(f"unreachable: {exc.reason}") from exc
    raise RetrievalError(f"retries exhausted for {url}")


def title_variants(name: str) -> list[str]:
    """Wikipedia titles are sentence case, not title case.

    "Covariance Matrix" is a miss; "Covariance matrix" is a hit. Node names are
    generated in title case, so the sentence-cased form must be tried too.
    """
    variants = [name]
    words = name.split()
    if len(words) > 1:
        sentence = " ".join([words[0]] + [w.lower() for w in words[1:]])
        if sentence != name:
            variants.append(sentence)
    return variants


def search_arxiv(query: str, limit: int = 3, transport: Transport | None = None) -> list[Source]:
    params = urllib.parse.urlencode(
        {"search_query": f"all:{query}", "start": 0, "max_results": limit}
    )
    try:
        raw = _fetch(f"{ARXIV}?{params}", transport)
    except RetrievalError:
        return []

    try:
        feed = ET.fromstring(raw)
    except ET.ParseError:
        return []

    sources: list[Source] = []
    for entry in feed.findall(f"{ATOM}entry"):
        title = (entry.findtext(f"{ATOM}title") or "").strip().replace("\n", " ")
        summary = (entry.findtext(f"{ATOM}summary") or "").strip()
        url = (entry.findtext(f"{ATOM}id") or "").strip()
        if not title or not url:
            continue
        sources.append(
            Source(
                id=url.rsplit("/", 1)[-1],
                title=title,
                url=url,
                kind="paper",
                text=summary,
            )
        )
    return sources


def fetch_wikipedia(title: str, transport: Transport | None = None) -> Source | None:
    """Full plaintext extract, not the summary.

    The summary is one paragraph and never contains a derivation; `explaintext`
    with no `exintro` returns the whole article including the derivation
    sections that matter here.
    """
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
            "format": "json",
            "titles": title,
        }
    )
    # RetrievalError deliberately propagates: a failed request is not a miss.
    try:
        payload = json.loads(_fetch(f"{WIKIPEDIA}?{params}", transport))
    except json.JSONDecodeError as exc:
        raise RetrievalError(f"malformed wikipedia response for {title!r}") from exc

    pages = payload.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if page_id == "-1" or "extract" not in page:
            continue
        extract = page["extract"].strip()
        if not extract:
            continue
        return Source(
            id=f"wikipedia:{page['title'].replace(' ', '_')}",
            title=page["title"],
            url="https://en.wikipedia.org/wiki/"
            + urllib.parse.quote(page["title"].replace(" ", "_")),
            kind="encyclopedia",
            text=extract,
        )
    return None


def load_aliases() -> dict[str, list[str]]:
    """Search titles per node id, loaded once."""
    global ALIASES
    if ALIASES is None:
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "data" / "source_aliases.json"
        ALIASES = json.loads(path.read_text())["aliases"] if path.exists() else {}
    return ALIASES


def aliases_for(node_id: str) -> list[str] | None:
    """Search titles for a node, or None if it has no entry.

    An explicit empty list means "known to have no source" -- a convention or a
    manipulation move -- and is distinct from having no entry at all.
    """
    table = load_aliases()
    return table.get(node_id)


def retrieve(name: str, aliases: tuple[str, ...] = (), transport: Transport | None = None,
             include_papers: bool = True) -> list[Source]:
    """Gather grounded sources for one concept, best first."""
    sources: list[Source] = []
    for base in (name, *aliases):
        for candidate in title_variants(base):
            page = fetch_wikipedia(candidate, transport)
            if page is not None:
                if not any(s.id == page.id for s in sources):
                    sources.append(page)
                break  # a variant hit; do not try the rest
    if include_papers:
        sources += search_arxiv(name, transport=transport)
    return sources


# ---------- the grounding guard ----------

def cited_but_not_retrieved(citations: list[str], sources: list[Source]) -> set[str]:
    """Citations that name no retrieved source.

    Must be empty. A non-empty result means the model cited from memory, which
    is the failure mode section 9 exists to prevent -- plausible references are
    worse than none, because the reader cannot tell.
    """
    available = {s.id for s in sources}
    return {c for c in citations if c not in available}


def retrieve_node(node_id: str, name: str, transport: Transport | None = None,
                  include_papers: bool = False) -> list[Source]:
    """Retrieve for a canon node, preferring its alias titles over its name.

    A node whose alias list is explicitly empty is skipped without a request.
    """
    listed = aliases_for(node_id)
    if listed == []:
        return []
    if listed:
        return retrieve(listed[0], tuple(listed[1:]), transport, include_papers)
    return retrieve(name, (), transport, include_papers)


def grounding_report(name: str, sources: list[Source]) -> dict:
    """Whether this concept can be grounded at all."""
    substantive = [s for s in sources if s.is_substantive]
    return {
        "concept": name,
        "sources": len(sources),
        "substantive": len(substantive),
        "grounded": bool(substantive),
        "best": substantive[0].id if substantive else None,
    }
