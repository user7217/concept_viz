"""What the map adds about a project's own components, beyond the mathematics.

Two things, from the two readers that sit beside ingest:

    topics     who sends data to whom, stated in code, config and launch
               files (atlas/topics.py). Drawn as links between components.
    graphify   where each component's code lives and what the docs say about
               it (atlas/codegraph.py). Shown in the component's panel.

The architecture layer only places components that run a canon algorithm.
The relay, the RTK simulator, navsat_transform and the Gazebo bridge run none,
so the map never showed them -- and they are exactly the components the
localization chain passes through. A component that takes part in a topic
link is added as a supporting component: part of the program, with no
mathematics of its own behind it.

Topic links are lateral. They are kept out of the walkthrough's descent --
program, subsystem, component, algorithm, mathematics -- so the hierarchy
reads the same as before, and the links are drawn between whatever is shown.

Graphify is optional. A repo without a graph gets no Graphify panels and the
map still builds; nothing here decides anything on a model's reading.
"""

from __future__ import annotations

from pathlib import Path

from .architecture import _slug
from .codegraph import GraphMissing, join, load_graph
from .ingest import Project
from .topics import endpoints, links

SUPPORTING_NOTE = ("Runs as part of the program, but no algorithm in the canon "
                   "sits behind it. It is here because data passes through it.")


def component_id(root: str, name: str) -> str:
    """The id the architecture layer gives a component."""
    return f"{root}__{_slug(name)}"


def topic_edges(project: Project, repo: Path, root: str) -> list[dict]:
    """One edge per sender/reader pair, carrying every topic between them."""
    by_pair: dict[tuple[str, str], dict] = {}
    for link in links(endpoints(project, repo)):
        key = (link.publisher, link.subscriber)
        edge = by_pair.setdefault(key, {
            "from": component_id(root, link.publisher),
            "to": component_id(root, link.subscriber),
            "rel": "topic", "topics": [], "evidence": [],
        })
        edge["topics"].append(link.topic)
        edge["evidence"].append({"topic": link.topic,
                                 "published": link.publisher_evidence,
                                 "read": link.subscriber_evidence})
    for (pub, sub), edge in by_pair.items():
        edge["why"] = (f"{pub} publishes {', '.join(edge['topics'])} "
                       f"and {sub} reads it.")
    return list(by_pair.values())


def supporting_nodes(edges: list[dict], known: set[str], root: str,
                     names: dict[str, str]) -> tuple[list[dict], list[dict]]:
    """Components a topic link needs that the map does not already hold.

    Returns the nodes and the edges that place them under the program.
    """
    nodes, placed = [], []
    for node_id in sorted({e["from"] for e in edges} | {e["to"] for e in edges}):
        if node_id in known:
            continue
        nodes.append({
            "id": node_id, "name": names.get(node_id, node_id.split("__", 1)[-1]),
            "type": "component", "role": "supporting", "tier": "assumed",
            "depth": 0, "state": "", "onPath": False, "steps": 0, "exercises": 0,
            "statement": SUPPORTING_NOTE, "whyHere": "",
            "code": [], "knobs": [], "reading": [],
            "measures": {k: 0 for k in ("depth", "reuse", "cost", "tier",
                                        "known", "steps", "exercises")},
        })
        placed.append({"from": root, "to": node_id, "rel": "contains", "why": ""})
    return nodes, placed


def graphify_panels(project: Project, repo: Path, root: str) -> dict[str, dict]:
    """Per component id: its code locations and the docs' description of it.

    Code locations come from Graphify's parser and are facts. Descriptions
    come from its model, and the map labels them as such.
    """
    try:
        graph = load_graph(repo)
    except GraphMissing:
        return {}
    panels: dict[str, dict] = {}
    for match in join(project, graph).matches:
        if not match.found:
            continue
        panels[component_id(root, match.component)] = {
            "codeRefs": [{"file": n.source_file, "line": n.location or "",
                          "label": n.label} for n in match.code],
            "described": [{"label": n.label, "rationale": n.rationale}
                          for n in match.described_by],
        }
    return panels
