"""Build or update a repo's Graphify graph, with whatever model is available.

    python3 scripts/build_codegraph.py <repo> [--force]

Writes <repo>/graphify-out/. Incremental by default: unchanged files are not
sent to a model again. See atlas/graphify_build.py for how the model is chosen.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.codegraph import load_graph
from atlas.graphify_build import GraphifyUnavailable, build

args = [a for a in sys.argv[1:] if not a.startswith("--")]
if not args:
    print(__doc__)
    raise SystemExit(1)

repo = Path(args[0]).resolve()
try:
    choice, path = build(repo, force="--force" in sys.argv)
except GraphifyUnavailable as exc:
    print(f"error: {exc}")
    raise SystemExit(1)

graph = load_graph(repo)
semantic = sum(1 for n in graph.nodes.values() if not n.deterministic)
print(f"\n{path}: {len(graph.nodes)} nodes, {len(graph.edges)} edges "
      f"({semantic} described by {choice.backend or 'no model'})")
