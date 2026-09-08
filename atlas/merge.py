"""Merge cluster edge proposals into the canon.

Proposals are authored independently per cluster, so this is where the
cross-cluster problems surface: unknown ids, edges out of floor nodes,
duplicates, branching-cap violations, and cycles that only exist once two
clusters' proposals are combined.

Nothing here writes the canon. It reports; a human decides.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .bootstrap import build_graph, load
from .graph import Graph
from .schema import NodeType, Relation

BRANCHING_CAP = 5


@dataclass
class Proposal:
    cluster: str
    edges: list[dict]
    flags: list[dict]
    source: Path


@dataclass
class MergeReport:
    accepted: list[tuple[str, str, str]] = field(default_factory=list)
    rejected: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    demoted: list[tuple[str, str]] = field(default_factory=list)
    flags: list[tuple[str, dict]] = field(default_factory=list)

    def reject(self, reason: str, detail: str) -> None:
        self.rejected[reason].append(detail)

    @property
    def rejected_count(self) -> int:
        return sum(len(v) for v in self.rejected.values())


def read_proposals(directory: Path) -> list[Proposal]:
    proposals = []
    for path in sorted(directory.glob("*.json")):
        with path.open() as fh:
            payload = json.load(fh)
        proposals.append(
            Proposal(
                cluster=payload.get("cluster", path.stem),
                edges=payload.get("edges", []),
                flags=payload.get("flags", []),
                source=path,
            )
        )
    return proposals


def merge(proposals: list[Proposal]) -> tuple[Graph, MergeReport]:
    """Apply proposals to the seed graph in a deterministic order.

    Order matters only for which edge of a conflicting pair gets demoted, so
    clusters are processed alphabetically and edges in file order.
    """
    payload = load()
    # Start from the seed graph only. Merging must not see its own prior output,
    # or cycle detection changes behaviour between runs.
    graph = build_graph(payload, with_canon_edges=False)
    floor = set(payload["base"]["concepts"])
    report = MergeReport()

    seen: set[tuple[str, str]] = set()
    out_degree: dict[str, int] = defaultdict(int)

    for proposal in sorted(proposals, key=lambda p: p.cluster):
        for flag in proposal.flags:
            report.flags.append((proposal.cluster, flag))

        for edge in proposal.edges:
            src, dst = edge.get("from"), edge.get("to")
            label = f"{src} -> {dst} [{proposal.cluster}]"

            if src not in graph.nodes:
                report.reject("unknown source id", label)
                continue
            if dst not in graph.nodes:
                report.reject("unknown target id", label)
                continue
            if src == dst:
                report.reject("self-edge", label)
                continue
            if src in floor:
                report.reject("edge out of a floor node", label)
                continue
            if graph.nodes[src].type is NodeType.ALGORITHM:
                report.reject("edge out of a seed algorithm", label)
                continue
            if (src, dst) in seen:
                report.reject("duplicate", label)
                continue
            if out_degree[src] >= BRANCHING_CAP:
                report.reject(f"over branching cap of {BRANCHING_CAP}", label)
                continue

            result = graph.add_edge(src, dst, Relation.REQUIRES)
            seen.add((src, dst))
            out_degree[src] += 1

            if result.demoted:
                report.demoted.append((src, dst))
            else:
                report.accepted.append((src, dst, proposal.cluster))

    return graph, report


def depth_profile(graph: Graph) -> dict[str, int]:
    """Longest REQUIRES path below each seed algorithm.

    Before internal edges exist every algorithm has depth 1. Anything above 1
    means the canon now has real structure to sort a syllabus by.
    """
    memo: dict[str, int] = {}

    def depth(node_id: str, stack: frozenset[str] = frozenset()) -> int:
        if node_id in memo:
            return memo[node_id]
        if node_id in stack:
            return 0
        children = [
            e.dst
            for e in graph.edges.values()
            if e.rel is Relation.REQUIRES and e.src == node_id
        ]
        value = 0 if not children else 1 + max(
            depth(c, stack | {node_id}) for c in children
        )
        memo[node_id] = value
        return value

    return {n.id: depth(n.id) for n in graph.by_type(NodeType.ALGORITHM)}
