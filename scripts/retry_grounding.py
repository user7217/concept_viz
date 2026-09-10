"""Re-run grounding for nodes that errored or came back ungrounded.

Errors are rate limiting, not absence, so they must be retried before any
coverage figure is trustworthy. Ungrounded nodes are retried too, since most
were alias misses rather than genuinely sourceless concepts.

Runs slower than the first pass: throughput is not the constraint here,
finishing without throttling is.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import atlas.retrieval as retrieval
from atlas.bootstrap import build_graph, load
from atlas.retrieval import (
    RetrievalError,
    aliases_for,
    grounding_report,
    ground_from_consumers,
    retrieve_node,
)

retrieval.MIN_INTERVAL = 2.5  # slow enough to stop tripping the limit

ROOT = Path(__file__).resolve().parent.parent
path = ROOT / "data" / "grounding.json"
payload = json.loads(path.read_text())
rows = {r["id"]: r for r in payload["rows"]}
graph = build_graph(load())

retry = [r for r in payload["rows"] if r["state"] in ("error", "ungrounded")]
print(f"retrying {len(retry)} nodes at {retrieval.MIN_INTERVAL}s intervals\n", flush=True)

fixed_direct = fixed_derived = 0
for index, row in enumerate(retry, 1):
    node = graph.nodes[row["id"]]
    try:
        sources = retrieve_node(node.id, node.name)
    except RetrievalError as exc:
        row.update(state="error", detail=str(exc)[:80])
        print(f"  {index}/{len(retry)} {node.id}: still erroring", flush=True)
        continue

    report = grounding_report(node.id, sources)
    if report["grounded"]:
        row.update(state="grounded", best=report["best"], route="direct")
        fixed_direct += 1
        print(f"  {index}/{len(retry)} {node.id}: grounded via {report['best']}", flush=True)
        continue

    # No article. Fall back to extracting the passages where consumers use it.
    try:
        derived = ground_from_consumers(graph, node.id, node.name)
    except RetrievalError:
        derived = []
    if any(s.is_substantive for s in derived):
        best = next(s for s in derived if s.is_substantive)
        row.update(state="grounded", best=best.id, route="derived", via=best.via)
        fixed_derived += 1
        print(f"  {index}/{len(retry)} {node.id}: derived via {best.via}", flush=True)
    else:
        row.update(state="ungrounded", route=None)
        print(f"  {index}/{len(retry)} {node.id}: still ungrounded", flush=True)

path.write_text(json.dumps(payload, indent=2) + "\n")

from collections import Counter
states = Counter(r["state"] for r in payload["rows"])
routes = Counter(r.get("route") for r in payload["rows"] if r["state"] == "grounded")
total = len(payload["rows"])
print(f"\nfixed this pass: {fixed_direct} direct, {fixed_derived} derived")
print(f"\ntotal {total}")
for state, count in states.most_common():
    print(f"  {state:<14} {count:>4}  ({100*count/total:.0f}%)")
still = [r["id"] for r in payload["rows"] if r["state"] in ("error", "ungrounded")]
if still:
    print(f"\nstill unresolved ({len(still)}): {', '.join(still)}")
