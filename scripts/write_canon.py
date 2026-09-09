"""Consolidate cluster proposals into the canon. Sole writer to data/."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.merge import merge, read_proposals
from atlas.schema import Relation

ROOT = Path(__file__).resolve().parent.parent
ROUND = int(sys.argv[2]) if len(sys.argv) > 2 else 1
proposals = read_proposals(Path(sys.argv[1]))
graph, report = merge(proposals)

why = {}
cluster_of = {}
for proposal in proposals:
    for edge in proposal.edges:
        why[(edge["from"], edge["to"])] = edge.get("why", "")
        cluster_of[(edge["from"], edge["to"])] = proposal.cluster

demoted = set(report.demoted)
edges = []
for edge in graph.edges.values():
    pair = (edge.src, edge.dst)
    if pair not in why:
        continue  # seed -> canon edge, already in seeds.json
    edges.append({
        "from": edge.src,
        "to": edge.dst,
        "rel": edge.rel.value,
        "why": why[pair],
        "cluster": cluster_of[pair],
        "demoted": pair in demoted,
    })
edges.sort(key=lambda e: (e["cluster"], e["from"], e["to"]))

(ROOT / "data" / "canon_edges.json").write_text(
    json.dumps({
        "_note": "Internal prerequisite edges within the canon. Authored per cluster "
                 "by independent passes, merged under a single writer. 'demoted' marks "
                 "an edge that would have closed a REQUIRES cycle and was kept as "
                 "related_to instead.",
        "edges": edges,
    }, indent=2) + "\n"
)

# Flags accumulate across rounds. Regenerating from the current proposal set
# alone would silently discard earlier rounds' adjudication record.
flags_path = ROOT / "data" / "canon_flags.json"
existing = []
if flags_path.exists():
    existing = [
        f for f in json.loads(flags_path.read_text())["flags"]
        if f.get("round", 1) < ROUND
    ]
flags = existing + [{"cluster": c, "round": ROUND, **f} for c, f in report.flags]
flags_path.write_text(
    json.dumps({
        "_note": "Issues raised while authoring canon edges. Adjudicated flags are "
                 "resolved in canon_amendments.json; unadjudicated ones are input to "
                 "the next curation pass.",
        "flags": flags,
    }, indent=2) + "\n"
)

requires = sum(1 for e in edges if e["rel"] == "requires")
print(f"canon_edges.json  {len(edges)} edges ({requires} requires, {len(edges)-requires} related_to)")
print(f"canon_flags.json  {len(flags)} flags")
