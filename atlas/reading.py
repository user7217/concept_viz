"""Where to go when the sheet is not enough.

A sheet restates one article. That is a starting point, not an education, and
the honest response to "this is not enough to understand" is to say where the
real treatments are rather than to generate more prose about them.

Wikipedia publishes the external links of every article, so the further
reading is already curated by the people who wrote it -- Welch and Bishop's
Kalman paper, Maybeck chapter 1, the CMU tutorial. Pulling those is mechanical:
no model, no scraping service, and nothing that can invent a citation.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import dataclass

from .retrieval import Transport, _fetch

API = "https://en.wikipedia.org/w/api.php"

# Domains that host primary sources and teaching material, roughly best first.
# Everything else on a Wikipedia page is news, archives and dead commercial
# links, which is most of the list.
# Domains that host primary sources and teaching material. Course notes come
# first on purpose: this is a tool for learning to derive something, and a
# lecture handout does that better than the paper that introduced it.
TRUSTED = (
    (".edu", "course notes"), ("ac.uk", "course notes"),
    ("mathworld.wolfram.com", "reference"),
    ("cambridge.org", "book"), ("springer", "book"),
    ("arxiv.org", "preprint"),
    ("doi.org", "paper"), ("ieee.org", "paper"), ("ieeecss.org", "paper"),
    ("acm.org", "paper"), ("jstor.org", "paper"),
    ("sciencedirect", "paper"), ("nature.com", "paper"),
    ("ncbi.nlm.nih.gov", "paper"),
)
JUNK = re.compile(r"(?i)(web\.archive\.org|books\.google|amazon\.|youtube\.com|"
                  r"twitter\.com|facebook\.com|/rss|\.xml$|archive\.today)")


@dataclass
class Reading:
    url: str
    kind: str
    title: str

    def to_dict(self) -> dict:
        return {"url": self.url, "kind": self.kind, "title": self.title}


def _title_from(url: str) -> str:
    """A readable label from a bare URL, since the API gives no titles."""
    path = urllib.parse.urlparse(url)
    stem = path.path.rstrip("/").rsplit("/", 1)[-1] or path.netloc
    stem = re.sub(r"\.(pdf|html?|php|aspx)$", "", stem, flags=re.I)
    stem = re.sub(r"[_\-+]+", " ", stem).strip()
    return f"{stem} — {path.netloc}" if stem else path.netloc


def external_links(title: str, transport: Transport | None = None,
                   limit: int = 6) -> list[Reading]:
    """Primary and teaching links an article itself points at."""
    query = urllib.parse.urlencode({
        "action": "query", "format": "json", "prop": "extlinks",
        "titles": title, "ellimit": "300",
    })
    payload = _fetch(f"{API}?{query}", transport)
    if not payload:
        return []
    try:
        pages = json.loads(payload)["query"]["pages"]
    except (KeyError, ValueError):
        return []

    # Round-robin by kind. Taking the best domain first filled every slot
    # with arXiv ids and never reached the lecture notes, which are the more
    # useful half for someone trying to derive the thing.
    seen: set[str] = set()
    by_kind: dict[str, list[Reading]] = {}
    for needle, kind in TRUSTED:
        for page in pages.values():
            for link in page.get("extlinks", []):
                url = link.get("*", "")
                if needle not in url or JUNK.search(url):
                    continue
                parts = urllib.parse.urlparse(url)
                key = parts.netloc + parts.path
                if key in seen:
                    continue
                seen.add(key)
                by_kind.setdefault(kind, []).append(
                    Reading(url=url, kind=kind, title=_title_from(url)))

    found: list[Reading] = []
    order = list(by_kind)
    while len(found) < limit and any(by_kind[k] for k in order):
        for kind in order:
            if by_kind[kind] and len(found) < limit:
                found.append(by_kind[kind].pop(0))
    return found[:limit]
