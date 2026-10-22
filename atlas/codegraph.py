"""Join a Graphify code graph to the project Atlas ingested.

Graphify builds the structural picture of a repo better than ingest.py does:
a deterministic AST pass over the code, plus a model pass over docs and
configs that recovers rationale and data flow. What it cannot produce is the
part the mathematics hangs on -- which nodes a launch file actually starts,
which libraries package.xml declares, and the exact values of the parameters
those nodes set. So neither side replaces the other. They are joined.

The graph is read as a file, never imported. Atlas stays stdlib-only, and
Graphify stays a tool that happens to leave a JSON file behind.

Two kinds of evidence come out of the join, and they are kept apart on
purpose:

    code          AST nodes. Deterministic, re-derived identically every run.
    described_by  Semantic nodes. A model chose the label and the rationale,
                  so they are context for a reader, never input to a guard.

The same split applies to edges: Graphify marks each one EXTRACTED, INFERRED
or AMBIGUOUS, and only EXTRACTED may be treated as fact. This codebase has
repeatedly built guards that compared a model's output with itself; a
model-inferred edge used as evidence would be the same mistake at one remove.

Nothing is dropped silently. Components the graph does not mention and graph
nodes no component claims are both reported, because a join that hides its
misses looks exactly like a join that has none.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .ingest import Project

GRAPH_FILE = Path("graphify-out") / "graph.json"

# Graphify's own audit labels. Only the first is a fact about the source.
EXTRACTED = "EXTRACTED"

# Edges that say how data or intent moves between parts of the system, as
# opposed to imports and containment, which the AST already implies.
FLOW_RELATIONS = frozenset({"shares_data_with", "rationale_for", "calls"})

# Files that start nodes. Only these say "this process runs here"; a config
# file such as nav2_params.yaml describes many nodes at once.
LAUNCH_FILE = re.compile(r"\.launch\.(py|xml|ya?ml)$")

# Lint and packaging scaffolding. Real files, but no component runs them.
SCAFFOLDING = re.compile(r"(^|/)(test|tests)/|(^|/)setup\.py$|__init__\.py$")


class GraphMissing(FileNotFoundError):
    """The repo has no Graphify output to join against."""


@dataclass(frozen=True)
class GraphNode:
    id: str
    label: str
    origin: str            # "ast" or "semantic"
    file_type: str
    source_file: str       # repo-relative
    location: str | None   # e.g. "L19"
    community: str | None
    rationale: str
    external: bool         # an import target with no file in this repo

    @property
    def deterministic(self) -> bool:
        return self.origin == "ast"


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    relation: str
    confidence: str

    @property
    def is_fact(self) -> bool:
        return self.confidence == EXTRACTED


@dataclass
class CodeGraph:
    nodes: dict[str, GraphNode]
    edges: list[GraphEdge]

    def touching(self, ids: set[str]) -> list[GraphEdge]:
        return [e for e in self.edges if e.source in ids or e.target in ids]


@dataclass
class Match:
    component: str
    code: list[GraphNode] = field(default_factory=list)
    described_by: list[GraphNode] = field(default_factory=list)
    sources_seen: list[str] = field(default_factory=list)
    sources_missing: list[str] = field(default_factory=list)
    flows: list[GraphEdge] = field(default_factory=list)
    # Only for a component nothing names: the launch files that start it.
    launched_by: list[GraphNode] = field(default_factory=list)

    @property
    def found(self) -> bool:
        """Named in the graph, by its own code or a description of it."""
        return bool(self.code or self.described_by)

    @property
    def located(self) -> bool:
        """Found, or at least placed by the launch file that starts it."""
        return self.found or bool(self.launched_by)


@dataclass
class Join:
    matches: list[Match]
    unclaimed: list[GraphNode]

    @property
    def unmatched_components(self) -> list[str]:
        return [m.component for m in self.matches if not m.located]

    @property
    def launch_file_only(self) -> list[str]:
        return [m.component for m in self.matches if m.located and not m.found]


# ---------- loading ----------

def _relative(path: str, root: Path) -> str:
    """Graphify relativizes on build, but a semantic node can keep the
    absolute path its extractor was handed. Both must compare equal."""
    if path and Path(path).is_absolute():
        try:
            return str(Path(path).resolve().relative_to(root.resolve()))
        except ValueError:
            return path
    return path


def load_graph(repo: Path) -> CodeGraph:
    path = repo / GRAPH_FILE
    if not path.exists():
        raise GraphMissing(
            f"no {GRAPH_FILE} in {repo}. Build it with: "
            f"python3 scripts/build_codegraph.py {repo}")
    payload = json.loads(path.read_text(encoding="utf-8"))

    nodes: dict[str, GraphNode] = {}
    for raw in payload.get("nodes", []):
        nodes[raw["id"]] = GraphNode(
            id=raw["id"],
            label=str(raw.get("label") or raw["id"]),
            origin=str(raw.get("_origin") or "ast"),
            file_type=str(raw.get("file_type") or ""),
            source_file=_relative(str(raw.get("source_file") or ""), repo),
            location=raw.get("source_location"),
            community=raw.get("community_name"),
            rationale=str(raw.get("rationale") or ""),
            external=raw.get("type") == "external",
        )

    # networkx node-link JSON names them "links"; older exports say "edges".
    edges = [
        GraphEdge(source=e["source"], target=e["target"],
                  relation=str(e.get("relation") or ""),
                  confidence=str(e.get("confidence") or ""))
        for e in payload.get("links", payload.get("edges", []))
    ]
    return CodeGraph(nodes=nodes, edges=edges)


# ---------- matching ----------

def tokens(text: str) -> set[str]:
    """Words in a name, however it was cased or punctuated.

    "RegulatedPurePursuitController", "regulated_pure_pursuit_controller"
    and "Regulated Pure Pursuit controller (FollowPath)" must meet.
    """
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return {t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t}


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _names(component, shared: frozenset[str] = frozenset()) -> list[str]:
    """What this component can be recognised by.

    An executable two components share identifies neither. ekf_local and
    ekf_global both run robot_localization's ekf_node, so matching on it
    described each filter with the other's configuration.
    """
    names = [component.name]
    exe = component.executable
    if exe and exe != component.name and exe not in shared:
        names.append(exe)
    return names


def _code_for(component, graph: CodeGraph,
              shared: frozenset[str] = frozenset()) -> list[GraphNode]:
    """AST nodes that are this component's own source.

    A node whose executable is odom_cov_relay is implemented in a module named
    odom_cov_relay.py. That is a file-name fact, so it is deterministic. Nodes
    from someone else's package -- ekf_node lives in robot_localization, not in
    this repo -- correctly match nothing here.
    """
    wanted = {_compact(n) for n in _names(component, shared)}
    return sorted(
        (n for n in graph.nodes.values()
         if n.deterministic and not n.external and n.source_file
         and _compact(Path(n.source_file).stem) in wanted),
        key=lambda n: (n.source_file, _line(n.location)))


def _line(location: str | None) -> int:
    """'L19' -> 19. Sorted as text, L11 and L19 landed before L5."""
    digits = "".join(c for c in (location or "") if c.isdigit())
    return int(digits) if digits else 0


def _described_by(component, graph: CodeGraph,
                  shared: frozenset[str] = frozenset()) -> list[GraphNode]:
    """Semantic nodes whose label names this component.

    Every word of the name must appear in the label, or the name must appear
    as one word: "BackUp" is written "backup" in the label that mentions it.
    """
    found = []
    for node in graph.nodes.values():
        if node.deterministic:
            continue
        have = tokens(node.label)
        for name in _names(component, shared):
            if tokens(name) <= have or _compact(name) in have:
                found.append(node)
                break
    return sorted(found, key=lambda n: n.label)


def _launched_by(component, graph: CodeGraph) -> list[GraphNode]:
    """Graph nodes for the launch files that start a component.

    The names often do not meet. Graphify read the README, which calls
    robot_state_publisher "RSP" and rviz2 "RViz", and calls ros_gz_sim's
    `create` "spawn". Matching names by similarity would fix those and break
    the ekf_local / ekf_global case, which similarity rates as one thing.
    The launch file is a fact both sides hold: ingest knows which file starts
    the component, and the graph has that file as a node, or a description
    naming it.

    It is weaker than a match by name, since every node a file starts shares
    the file. So it is a fallback, and kept apart from `described_by`.
    """
    files = [f for f in component.sources if LAUNCH_FILE.search(f)]
    found = []
    for node in graph.nodes.values():
        if node.external:
            continue
        for path in files:
            name = Path(path).name
            if (node.source_file == path and node.label == name) if node.deterministic \
                    else name in node.label:
                found.append(node)
                break
    return sorted(found, key=lambda n: (n.origin, n.label))


def _flows(ids: set[str], graph: CodeGraph) -> list[GraphEdge]:
    """Edges that leave the component for another part of this system.

    main() calling OdomCovRelay is the component talking to itself, and a call
    into rclpy is a call into a library, not a link between parts of the
    robot. Neither says how this component connects to its neighbours.
    """
    seen: set[tuple[str, str, str]] = set()
    out: list[GraphEdge] = []
    for edge in graph.touching(ids):
        if edge.relation not in FLOW_RELATIONS:
            continue
        other = edge.target if edge.source in ids else edge.source
        if other in ids:
            continue
        node = graph.nodes.get(other)
        if node is None or node.external:
            continue
        key = (min(edge.source, edge.target), max(edge.source, edge.target),
               edge.relation)
        if key in seen:
            continue
        seen.add(key)
        out.append(edge)
    return out


def join(project: Project, graph: CodeGraph) -> Join:
    files_in_graph = {n.source_file for n in graph.nodes.values()}
    counts: dict[str, int] = {}
    for component in project.components:
        if component.executable:
            counts[component.executable] = counts.get(component.executable, 0) + 1
    shared = frozenset(exe for exe, n in counts.items() if n > 1)

    matches: list[Match] = []
    claimed: set[str] = set()

    for component in project.components:
        match = Match(component=component.name)
        match.code = _code_for(component, graph, shared)
        match.described_by = _described_by(component, graph, shared)
        for source in component.sources:
            (match.sources_seen if source in files_in_graph
             else match.sources_missing).append(source)
        ids = {n.id for n in match.code + match.described_by}
        match.flows = _flows(ids, graph)
        if not match.found:
            match.launched_by = _launched_by(component, graph)
        claimed |= ids | {n.id for n in match.launched_by}
        matches.append(match)

    unclaimed = sorted(
        (n for n in graph.nodes.values()
         if n.id not in claimed and not n.external
         and not SCAFFOLDING.search(n.source_file)),
        key=lambda n: (n.origin, n.source_file, n.label))
    return Join(matches=matches, unclaimed=unclaimed)


# ---------- report ----------

def report(result: Join, graph: CodeGraph) -> str:
    lines: list[str] = []
    found = [m for m in result.matches if m.found]
    lines.append(f"components: {len(found)}/{len(result.matches)} found in the graph"
                 + (f", {len(result.launch_file_only)} more only through the launch "
                    f"file that starts them" if result.launch_file_only else ""))
    lines.append("")

    for m in result.matches:
        if not m.found:
            continue
        lines.append(f"  {m.component}")
        for node in m.code:
            lines.append(f"    code       {node.source_file}:{node.location or '?'}"
                         f"  {node.label}")
        for node in m.described_by:
            lines.append(f"    described  {node.label}")
        for edge in m.flows:
            other = edge.target if edge.source in {n.id for n in m.code + m.described_by} \
                else edge.source
            label = graph.nodes[other].label if other in graph.nodes else other
            tag = "" if edge.is_fact else f"  [{edge.confidence.lower()}]"
            lines.append(f"    {edge.relation:<18} {label}{tag}")

    for m in result.matches:
        if not m.launched_by:
            continue
        lines.append(f"  {m.component}  (not named in the graph)")
        for node in m.launched_by:
            where = node.source_file if node.deterministic else node.label
            lines.append(f"    launched by  {where}")

    lines.append("")
    lines.append(f"not in the graph ({len(result.unmatched_components)}): "
                 + (", ".join(result.unmatched_components) or "none"))

    missing = sorted({s for m in result.matches for s in m.sources_missing})
    if missing:
        lines.append(f"source files Atlas read that Graphify did not ({len(missing)}): "
                     + ", ".join(missing))

    semantic = [n for n in result.unclaimed if not n.deterministic]
    structural = [n for n in result.unclaimed if n.deterministic]
    lines.append("")
    lines.append(f"graph nodes no component claims: {len(semantic)} described, "
                 f"{len(structural)} code")
    for node in semantic:
        lines.append(f"    {node.label}")
    return "\n".join(lines)


# ---------- checking the model's flows against stated topics ----------

def check_flows(result: Join, graph: CodeGraph,
                topic_links: list) -> tuple[list[tuple], list[tuple]]:
    """Each inferred flow between two components, against the topic links.

    Graphify's data-flow edges are mostly a model reading the README. The
    topic links are what the code, parameters and launch files state. A flow
    the topics also show is confirmed; one they do not is still only a model's
    guess -- possibly right, since a plugin and its costmap share memory, not
    a topic, but not something to build on.

    Returns (confirmed, unconfirmed), each a list of
    (component, other_component, relation, confidence, topics).
    """
    owners: dict[str, set[str]] = {}
    for m in result.matches:
        for node in m.code + m.described_by:
            owners.setdefault(node.id, set()).add(m.component)

    wired: dict[frozenset, set[str]] = {}
    for link in topic_links:
        wired.setdefault(frozenset((link.publisher, link.subscriber)),
                         set()).add(link.topic)

    confirmed, unconfirmed, seen = [], [], set()
    for m in result.matches:
        mine = {n.id for n in m.code + m.described_by}
        for edge in m.flows:
            other = edge.target if edge.source in mine else edge.source
            for component in sorted(owners.get(other, set()) - {m.component}):
                pair = frozenset((m.component, component))
                key = (pair, edge.relation)
                if key in seen:
                    continue
                seen.add(key)
                topics = sorted(wired.get(pair, set()))
                row = (m.component, component, edge.relation,
                       edge.confidence, topics)
                (confirmed if topics else unconfirmed).append(row)
    return confirmed, unconfirmed

