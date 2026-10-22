"""Join a repo's Graphify graph to what Atlas ingests from it, and report.

    python3 scripts/graphify_join.py <repo>

Reads <repo>/graphify-out/graph.json; run graphify on the repo first. Prints
which components the graph knows, with their code locations, descriptions and
data flow, and both sides of what failed to match.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.brief import resolve
from atlas.codegraph import GraphMissing, check_flows, join, load_graph, report
from atlas.topics import endpoints, links

if len(sys.argv) < 2:
    print(__doc__)
    raise SystemExit(1)

repo = Path(sys.argv[1]).resolve()
try:
    graph = load_graph(repo)
except GraphMissing as exc:
    print(exc)
    raise SystemExit(1)

project, _, _ = resolve(repo)
result = join(project, graph)
print(report(result, graph))

confirmed, unconfirmed = check_flows(result, graph, links(endpoints(project, repo)))
print(f"\nGraphify's flows between components, checked against stated topics:")
for a, b, relation, confidence, topics in confirmed:
    print(f"  confirmed    {a} -- {b}  ({confidence.lower()})  via {', '.join(topics)}")
for a, b, relation, confidence, topics in unconfirmed:
    print(f"  unconfirmed  {a} -- {b}  ({confidence.lower()} {relation})")
