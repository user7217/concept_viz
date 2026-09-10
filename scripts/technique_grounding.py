"""Measure technique grounding via the derivations that use them.

Techniques have no encyclopedia articles -- nobody writes one on "moving a
transpose through a product" -- so they are grounded by extracting the passages
where their consumers perform the move. This reports whether that works.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.bootstrap import build_graph, load
from atlas.retrieval import (
    RetrievalError,
    aliases_for,
    consumers_of,
    ground_from_consumers,
    retrieve_node,
)
from atlas.schema import NodeType, Tier

graph = build_graph(load())
techniques = sorted(
    (n for n in graph.nodes.values()
     if n.type is NodeType.TECHNIQUE and n.tier is not Tier.ASSUMED),
    key=lambda n: n.id,
)

rows = []
for index, node in enumerate(techniques, 1):
    direct = []
    if aliases_for(node.id) != []:
        try:
            direct = retrieve_node(node.id, node.name)
        except RetrievalError:
            direct = []
    direct_ok = any(s.is_substantive for s in direct)

    derived = []
    if not direct_ok:
        try:
            derived = ground_from_consumers(graph, node.id, node.name)
        except RetrievalError:
            derived = []
    derived_ok = any(s.is_substantive for s in derived)

    rows.append({
        "id": node.id,
        "direct": direct_ok,
        "derived": derived_ok,
        "consumers": consumers_of(graph, node.id)[:3],
        "via": [s.via for s in derived if s.is_substantive][:3],
        "chars": max((len(s.text) for s in derived), default=0),
    })
    print(f"  {index}/{len(techniques)} {node.id}", flush=True)

out = Path(__file__).resolve().parent.parent / "data" / "technique_grounding.json"
out.write_text(json.dumps({
    "_note": "Technique grounding. 'direct' means an article exists; 'derived' "
             "means passages were extracted from a consumer's source. A technique "
             "with neither cannot have a derivation written under section 9.",
    "rows": rows,
}, indent=2) + "\n")

direct = sum(r["direct"] for r in rows)
derived = sum(r["derived"] for r in rows)
neither = [r["id"] for r in rows if not r["direct"] and not r["derived"]]
print(f"\ntechniques: {len(rows)}")
print(f"  direct article      {direct:>3}")
print(f"  derived from users  {derived:>3}")
print(f"  grounded total      {direct + derived:>3}  ({100*(direct+derived)/len(rows):.0f}%)")
print(f"  neither             {len(neither):>3}  {', '.join(neither)}")
