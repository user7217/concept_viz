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
from atlas.codegraph import GraphMissing, join, load_graph, report

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
print(report(join(project, graph), graph))
