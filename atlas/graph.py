"""Graph store with the invariants the design doc requires.

The load-bearing rule: REQUIRES is strictly acyclic. An edge that would close
a cycle is demoted to RELATED_TO rather than dropped, so the information
survives, the DAG survives, and topological sort stays valid.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .schema import (
    GLOBAL_TYPES,
    PROJECT_TYPES,
    Edge,
    Node,
    NodeType,
    Relation,
    Tier,
)


@dataclass
class AddEdgeResult:
    edge: Edge
    demoted: bool = False  # REQUIRES downgraded to RELATED_TO to preserve the DAG


@dataclass
class Graph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[tuple[str, str, Relation], Edge] = field(default_factory=dict)

    # ---------- construction ----------

    def add_node(self, node: Node) -> Node:
        existing = self.nodes.get(node.id)
        if existing is not None:
            return existing
        self.nodes[node.id] = node
        return node

    def add_edge(
        self,
        src: str,
        dst: str,
        rel: Relation,
        invoked_at: tuple[str, ...] = (),
    ) -> AddEdgeResult:
        if src not in self.nodes:
            raise KeyError(f"unknown src node: {src}")
        if dst not in self.nodes:
            raise KeyError(f"unknown dst node: {dst}")
        if src == dst:
            raise ValueError(f"self-edge on {src}")

        demoted = False
        if rel is Relation.REQUIRES and self._would_cycle(src, dst):
            rel = Relation.RELATED_TO
            demoted = True

        edge = Edge(src, dst, rel, invoked_at)
        self.edges.setdefault(edge.key(), edge)
        return AddEdgeResult(edge=edge, demoted=demoted)

    def _would_cycle(self, src: str, dst: str) -> bool:
        """True if src already sits in dst's REQUIRES closure."""
        return src == dst or src in self.requires_closure(dst)

    # ---------- traversal ----------

    def _requires_out(self, node_id: str) -> list[str]:
        return [
            e.dst
            for e in self.edges.values()
            if e.rel is Relation.REQUIRES and e.src == node_id
        ]

    def requires_closure(self, node_id: str) -> set[str]:
        """Every prerequisite reachable from node_id, transitively."""
        seen: set[str] = set()
        queue = deque(self._requires_out(node_id))
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            queue.extend(self._requires_out(current))
        return seen

    def syllabus(self, target: str, known: set[str] | None = None) -> list[str]:
        """Prerequisites of target in dependency order, cut at the reader's floor.

        Returns deepest-first: what to learn, in the order to learn it.
        """
        known = known or set()
        included = {n for n in self.requires_closure(target) if n not in known}

        # Kahn's algorithm over the induced subgraph, deepest-first.
        deps = {n: {d for d in self._requires_out(n) if d in included} for n in included}
        ordered: list[str] = []
        ready = sorted(n for n, d in deps.items() if not d)
        while ready:
            current = ready.pop(0)
            ordered.append(current)
            newly_ready = []
            for node_id, remaining in deps.items():
                if current in remaining:
                    remaining.discard(current)
                    if not remaining and node_id not in ordered:
                        newly_ready.append(node_id)
            ready = sorted(set(ready) | set(newly_ready))

        if len(ordered) != len(included):
            missing = included - set(ordered)
            raise ValueError(f"cycle detected in REQUIRES subgraph: {sorted(missing)}")
        return ordered

    # ---------- validation ----------

    def validate(self) -> list[str]:
        """Deterministic checks, enforced at write time rather than by an agent."""
        problems: list[str] = []

        for edge in self.edges.values():
            src, dst = self.nodes[edge.src], self.nodes[edge.dst]

            if edge.rel is Relation.CONTAINS and (
                src.type not in PROJECT_TYPES or dst.type in {NodeType.CONCEPT, NodeType.TECHNIQUE}
            ):
                problems.append(
                    f"CONTAINS must stay in the architecture layer: {src.id} -> {dst.id}"
                )

            if edge.rel is Relation.USES and (
                src.type not in PROJECT_TYPES or dst.type is not NodeType.ALGORITHM
            ):
                problems.append(
                    f"USES must cross project -> algorithm: {src.id} -> {dst.id}"
                )

            if edge.rel is Relation.REQUIRES and dst.type not in GLOBAL_TYPES:
                problems.append(
                    f"REQUIRES must point at a global node: {src.id} -> {dst.id}"
                )

        for node_id in self.nodes:
            try:
                self.syllabus(node_id)
            except ValueError as exc:
                problems.append(str(exc))
                break

        orphans = self._orphans()
        if orphans:
            problems.append(f"unreachable nodes: {sorted(orphans)}")

        return problems

    def _orphans(self) -> set[str]:
        """Nodes nothing references.

        Assumed (floor) nodes are exempt: they are materialised so that edges
        have something to point at, and a partial graph legitimately leaves
        some of them unreferenced.
        """
        connected = {e.src for e in self.edges.values()} | {
            e.dst for e in self.edges.values()
        }
        candidates = {
            node_id
            for node_id, node in self.nodes.items()
            if node.tier is not Tier.ASSUMED
        }
        return candidates - connected

    # ---------- closure assertion (design doc section 4) ----------

    def closure_holes(
        self, node_id: str, invoked: set[str], known: set[str] | None = None
    ) -> set[str]:
        """Anything the derivation used that is not reachable and not assumed.

        This is the completeness criterion: it must return the empty set.
        """
        known = known or set()
        return invoked - self.requires_closure(node_id) - known - {node_id}

    # ---------- reporting ----------

    def by_type(self, node_type: NodeType) -> list[Node]:
        return sorted(
            (n for n in self.nodes.values() if n.type is node_type), key=lambda n: n.id
        )

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for node in self.nodes.values():
            out[node.type.value] = out.get(node.type.value, 0) + 1
        return out
