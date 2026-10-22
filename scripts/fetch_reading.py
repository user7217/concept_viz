"""Further reading for every node that has a sheet. Writes data/reading.json.

    python3 scripts/fetch_reading.py

No model: the links come from each article's own external links, so they are
whatever the people who wrote that article thought worth citing. Resumable --
nodes already fetched are skipped.

A second pass names each link from whoever issued it: arXiv, Crossref, Open
Library, or the page's own <title>. A link nobody names keeps its URL label
and is marked `via: "url"`, so the map can show it as unnamed rather than
pass a filename off as a title. Also resumable: named links are skipped.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.bootstrap import build_graph, load
from atlas.reading import _title_from, cite, external_links
from atlas.retrieval import RetrievalError, aliases_for

ROOT = Path(__file__).resolve().parent.parent
SHEETS = ROOT / "data" / "derivations"
OUT = ROOT / "data" / "reading.json"

graph = build_graph(load())
stored = json.loads(OUT.read_text())["reading"] if OUT.exists() else {}
wanted = [f.stem for f in sorted(SHEETS.glob("*.json")) if f.stem not in stored]
print(f"{len(wanted)} nodes to fetch ({len(stored)} already done)\n", flush=True)

for index, node_id in enumerate(wanted, 1):
    # The alias is what actually resolved during retrieval, so it is the title
    # whose links belong to this node.
    titles = aliases_for(node_id) or [graph.nodes[node_id].name
                                      if node_id in graph.nodes else node_id]
    found, failed = [], False
    for title in titles[:2]:
        try:
            found += [r.to_dict() for r in external_links(title)]
        except RetrievalError as exc:
            failed = True
            print(f"  {index}/{len(wanted)} {node_id}: {str(exc)[:60]}", flush=True)
        time.sleep(0.4)
    # A rate limit is not an article with no links. Recording [] here made
    # the node look done, and resuming skipped it for good.
    if failed:
        print(f"  {index}/{len(wanted)} {node_id}: not recorded, will retry", flush=True)
        continue
    seen, unique = set(), []
    for item in found:
        if item["url"] in seen:
            continue
        seen.add(item["url"])
        unique.append(item)
    stored[node_id] = unique[:6]
    print(f"  {index}/{len(wanted)} {node_id}: {len(unique[:6])} links", flush=True)
    OUT.write_text(json.dumps({"reading": stored}, indent=2) + "\n")

unnamed = [item for items in stored.values() for item in items
           if "via" not in item]
print(f"\nnaming {len(unnamed)} links", flush=True)
for index, item in enumerate(unnamed, 1):
    found = cite(item["url"])
    if found:
        item.update(title=found.title, byline=found.byline, via=found.via)
    else:
        item.update(title=_title_from(item["url"]), byline="", via="url")
    if index % 10 == 0 or index == len(unnamed):
        OUT.write_text(json.dumps({"reading": stored}, indent=2) + "\n")
        print(f"  {index}/{len(unnamed)} named", flush=True)

kept = sum(1 for items in stored.values() for item in items if item.get("via") == "url")
print(f"\n{kept} links kept their URL label (PDFs, dead links, untitled pages)")
print(f"\nwrote {OUT.name}: "
      f"{sum(len(v) for v in stored.values())} links across {len(stored)} nodes")
