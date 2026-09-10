"""Measure how much of the canon can be grounded in a real source.

Design doc section 9 forbids generating a derivation from memory, so a node
with no substantive source cannot have one written at all. This reports which
nodes those are.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.bootstrap import build_graph, load
from atlas.retrieval import RetrievalError, aliases_for, grounding_report, retrieve_node
from atlas.schema import NodeType, Tier

graph = build_graph(load())
nodes = [
    n for n in graph.nodes.values()
    if n.type in (NodeType.ALGORITHM, NodeType.CONCEPT, NodeType.TECHNIQUE)
    and n.tier is not Tier.ASSUMED
]

rows, errors = [], []
for i, node in enumerate(sorted(nodes, key=lambda n: n.id), 1):
    if aliases_for(node.id) == []:
        rows.append({"id": node.id, "type": node.type.value, "state": "known_absent"})
        continue
    try:
        sources = retrieve_node(node.id, node.name)
    except RetrievalError as exc:
        errors.append(node.id)
        rows.append({"id": node.id, "type": node.type.value, "state": "error",
                     "detail": str(exc)[:80]})
        continue
    report = grounding_report(node.id, sources)
    rows.append({
        "id": node.id, "type": node.type.value,
        "state": "grounded" if report["grounded"] else "ungrounded",
        "best": report["best"], "chars": max((len(s.text) for s in sources), default=0),
    })
    if i % 25 == 0:
        print(f"  ... {i}/{len(nodes)}", flush=True)

out = Path(__file__).resolve().parent.parent / "data" / "grounding.json"
out.write_text(json.dumps({
    "_note": "Which canon nodes can be grounded in a retrieved source. Nodes in "
             "state known_absent are conventions or manipulation moves with no "
             "reference article; they need a different grounding strategy, not a "
             "better search.",
    "rows": rows,
}, indent=2) + "\n")

from collections import Counter
states = Counter(r["state"] for r in rows)
print(f"\ntotal {len(rows)}")
for state, count in states.most_common():
    print(f"  {state:<14} {count:>4}  ({100*count/len(rows):.0f}%)")
by_type = {}
for r in rows:
    d = by_type.setdefault(r["type"], Counter())
    d[r["state"]] += 1
print()
for t, counts in sorted(by_type.items()):
    total = sum(counts.values())
    print(f"  {t:<11} grounded {counts['grounded']}/{total}")
print(f"\nwrote {out.relative_to(out.parent.parent)}")
