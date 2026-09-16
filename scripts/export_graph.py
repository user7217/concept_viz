"""Export the canon as one JSON payload for the 3D view.

    python3 scripts/export_graph.py [repo] [subsystem] > graph.json

Nodes carry the depth a layered layout needs, plus whatever each one has
accumulated: a sheet, a project note, exercises, a profile state. Edges keep
their relation so REQUIRES can be drawn differently from RELATED_TO.

Deliberately not a renderer. This writes data; the page draws it.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.bootstrap import build_graph, load, load_canon_edges
from atlas.code import evidence_for, index_repo
from atlas.exercise import available_count
from atlas.profile import Profile
from atlas.schema import NodeType

ROOT = Path(__file__).resolve().parent.parent
SHEETS = ROOT / "data" / "derivations"


def depths(graph) -> dict[str, int]:
    """Longest REQUIRES chain below each node.

    Longest, not shortest: a node sits above everything it needs, so a layer
    number that any prerequisite could exceed would draw edges pointing back up.
    """
    memo: dict[str, int] = {}

    def below(node_id: str, seen: frozenset) -> int:
        if node_id in memo:
            return memo[node_id]
        if node_id in seen:
            return 0
        outs = graph._requires_out(node_id)
        value = 1 + max((below(n, seen | {node_id}) for n in outs), default=-1)
        memo[node_id] = value
        return value

    return {n: below(n, frozenset()) for n in graph.nodes}


def _trim(text: str, limit: int) -> str:
    """Cut on a sentence end, never mid-equation.

    A hard slice ended one statement at "and variance $\\mathbf{B}..., i.e., $",
    leaving an unclosed math span and a dangling clause on the page.
    """
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    stop = max(cut.rfind(". "), cut.rfind(".\n"))
    return (cut[:stop + 1] if stop > limit * 0.4 else cut.rstrip()) + " \u2026"


TIER_RANK = {"assumed": 0, "statement": 1, "derivation": 2, "claim": 3}
STATE_RANK = {"": 0, "declared": 1, "read": 2, "demonstrated": 3}


def reuse_counts(graph) -> dict[str, int]:
    """How many of the canon's algorithms transitively need each node.

    The load-bearing axis. summation and algebraic_rearrangement are needed by
    17 of 20; 71 nodes are needed by exactly one, which is what "specialist"
    means here.
    """
    counts = {n: 0 for n in graph.nodes}
    for node in graph.nodes.values():
        if node.type is not NodeType.ALGORITHM:
            continue
        for dep in graph.requires_closure(node.id):
            counts[dep] = counts.get(dep, 0) + 1
    return counts


def main() -> None:
    graph = build_graph(load())
    profile = Profile.load()
    level = depths(graph)
    reuse = reuse_counts(graph)

    sources = index_repo(Path(sys.argv[1])) if len(sys.argv) > 1 else []

    path: set[str] = set()
    if len(sys.argv) > 2:
        from atlas.architecture import apply, propose
        from atlas.ingest import ingest_repo

        project = ingest_repo(Path(sys.argv[1]))
        proposal = propose(project)
        proposal.unclassified, proposal.confirmed = [], True
        pgraph = apply(proposal)
        pid = "".join(c if c.isalnum() else "_"
                      for c in project.name.lower()).strip("_")
        target = f"{pid}__{sys.argv[2].lower()}"
        if target in pgraph.nodes:
            path = set(pgraph.learning_path(
                target, known=profile.known(), not_known=profile.not_known))

    states = {n: e.state.value for n, e in profile.entries.items()}
    nodes = []
    for node in graph.nodes.values():
        sheet_path = SHEETS / f"{node.id}.json"
        sheet = json.loads(sheet_path.read_text()) if sheet_path.exists() else None
        note = (sheet or {}).get("instantiation") or {}
        nodes.append({
            "id": node.id,
            "name": node.name,
            "type": node.type.value,
            "tier": node.tier.value,
            "depth": level.get(node.id, 0),
            "measures": {
                "depth": level.get(node.id, 0),
                "reuse": reuse.get(node.id, 0),
                "cost": len(graph.requires_closure(node.id)),
                "tier": TIER_RANK.get(node.tier.value, 0),
                "known": STATE_RANK.get(states.get(node.id, ""), 0),
                "steps": len((sheet or {}).get("steps", [])),
                "exercises": available_count(sheet) if sheet else 0,
            },
            "state": states.get(node.id, ""),
            "onPath": node.id in path,
            "steps": len((sheet or {}).get("steps", [])),
            "exercises": available_count(sheet) if sheet else 0,
            "statement": _trim((sheet or {}).get("statement") or "", 1400),
            "whyHere": _trim(str(note.get("why_here", "")), 900),
            "code": [span.to_dict() for span in
                     evidence_for(node.name, sources, note)] if sources else [],
            "knobs": [{"parameter": k.get("parameter"), "file": k.get("file"),
                       "means": str(k.get("means", ""))[:240]}
                      for k in note.get("knobs", [])[:6]],
        })

    # Every curated edge carries the reason it exists. A line labelled
    # "requires" says nothing the arrow did not; the rationale is the part
    # worth reading.
    why = {(e["from"], e["to"]): e.get("why", "") for e in load_canon_edges()}
    edges = [{"from": e.src, "to": e.dst, "rel": e.rel.value,
              "why": why.get((e.src, e.dst), "")}
             for e in graph.edges.values()]
    json.dump({"nodes": nodes, "edges": edges,
               "maxDepth": max(level.values(), default=0)},
              sys.stdout)


if __name__ == "__main__":
    main()
