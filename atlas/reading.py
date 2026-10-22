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

import html
import http.client
import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from .retrieval import ARXIV, ATOM, RetrievalError, Transport, _fetch

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


# ---------------------------------------------------------------------------
# Real titles. Wikipedia's extlinks API returns bare URLs, so the panel showed
# "node7", "978 3 642 01688 2" and "10.2307%2F2371328". The registries that
# issue these identifiers publish the metadata for free; a page's own <title>
# covers the rest. Nothing here is generated: a link that resolves nowhere
# keeps its URL label and says so.

ARXIV_ID = re.compile(r"arxiv\.org/(?:abs|pdf)/([\w.\-/]+?)(?:v\d+)?(?:\.pdf)?/?$", re.I)
DOI = re.compile(r"doi\.org/(10\.[^?#\s]+)", re.I)
ISBN = re.compile(r"(97[89](?:[-\s]?\d){10})")
TITLE_TAG = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
TAG = re.compile(r"<[^>]+>")
# Titles that name the server's state, not the document.
NOT_A_TITLE = re.compile(r"(?i)^(home|index of|404|page not found|not found|"
                         r"access denied|just a moment|redirect|error|"
                         r"attention required|untitled|document|client challenge|"
                         r"are you a robot|captcha|wyszukiwanie)\b"
                         # a whole title that is only a hostname or a section name
                         r"|^[\w.-]+\.[a-z]{2,}$|^(search|computer science|mathematics|"
                         r"statistics|physics)$")
# A JSTOR stable id is the suffix of a 10.2307 DOI, and JSTOR itself answers
# robots with a "Client Challenge" page.
JSTOR = re.compile(r"jstor\.org/stable/(\d+)", re.I)
# "Lecture 3: Inverse matrices | Linear Algebra | MIT OpenCourseWare" names
# the document first and the site after; the host is shown beside it anyway.
SITE_SUFFIX = re.compile(r"\s+(?:\||--)\s+.*$")
CROSSREF = "https://api.crossref.org/works/"
OPEN_LIBRARY = "https://openlibrary.org/isbn/"


@dataclass
class Citation:
    title: str
    byline: str  # "Welch, Bishop · 1995", or "" when nobody says
    via: str     # which registry or page supplied it


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(TAG.sub("", text))).strip()


def _byline(names: list[str], year: str | int | None) -> str:
    who = ", ".join(names[:3]) + (" et al." if len(names) > 3 else "")
    return " · ".join(str(x) for x in (who, year) if x)


def _from_arxiv(paper: str, transport: Transport | None) -> Citation | None:
    payload = _fetch(f"{ARXIV}?id_list={urllib.parse.quote(paper)}", transport)
    entry = ET.fromstring(payload).find(f"{ATOM}entry")
    title = entry.findtext(f"{ATOM}title") if entry is not None else None
    if not title or title.strip() == "Error":
        return None
    names = [a.findtext(f"{ATOM}name", "").split()[-1]
             for a in entry.findall(f"{ATOM}author") if a.findtext(f"{ATOM}name")]
    year = (entry.findtext(f"{ATOM}published") or "")[:4]
    return Citation(_clean(title), _byline(names, year), "arxiv")


def _from_crossref(doi: str, transport: Transport | None) -> Citation | None:
    payload = _fetch(CROSSREF + urllib.parse.quote(doi, safe="/"), transport)
    work = json.loads(payload).get("message", {})
    titles = work.get("title") or []
    if not titles:
        return None
    names = [a["family"] for a in work.get("author", []) if a.get("family")]
    parts = (work.get("issued") or {}).get("date-parts") or [[None]]
    return Citation(_clean(titles[0]), _byline(names, parts[0][0]), "crossref")


def _from_open_library(isbn: str, transport: Transport | None) -> Citation | None:
    payload = _fetch(OPEN_LIBRARY + isbn + ".json", transport)
    book = json.loads(payload)
    if not book.get("title"):
        return None
    year = re.search(r"\d{4}", book.get("publish_date", ""))
    return Citation(_clean(book["title"]), _byline([], year and year.group()),
                    "openlibrary")


def tidy_title(title: str) -> str:
    """Drop a trailing site name, keeping the title if that leaves too little."""
    trimmed = SITE_SUFFIX.sub("", title).strip()
    return trimmed if len(trimmed) >= 8 else title


def _from_page(url: str, transport: Transport | None) -> Citation | None:
    # Two tries, not four: a dead lecturer's page is common and costs minutes.
    payload = _fetch(url, transport, retries=2)
    found = TITLE_TAG.search(payload[:200_000].decode("utf-8", "replace"))
    title = _clean(found.group(1)) if found else ""
    if len(title) < 4 or NOT_A_TITLE.match(title):
        return None
    return Citation(tidy_title(title)[:160], "", "page")


def cite(url: str, transport: Transport | None = None) -> Citation | None:
    """The document's real title, from whoever issued its identifier.

    None means nobody could say -- a PDF, a dead link, a page titled "Home".
    A network failure is also None here, because a label is not worth failing
    a run over; the caller counts both and reports them.
    """
    try:
        if found := ARXIV_ID.search(url):
            return _from_arxiv(found.group(1), transport)
        if found := DOI.search(url):
            return _from_crossref(urllib.parse.unquote(found.group(1)), transport)
        if found := JSTOR.search(url):
            return _from_crossref("10.2307/" + found.group(1), transport)
        if found := ISBN.search(urllib.parse.unquote(url)):
            return _from_open_library(re.sub(r"\D", "", found.group(1)), transport)
        if urllib.parse.urlparse(url).path.lower().endswith(".pdf"):
            return None
        return _from_page(url, transport)
    except (RetrievalError, OSError, http.client.HTTPException,
            ValueError, KeyError, IndexError, ET.ParseError):
        return None
