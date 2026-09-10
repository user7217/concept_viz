"""Graph store with the invariants the design doc requires.

The load-bearing rule: REQUIRES is strictly acyclic. An edge that would close
a cycle is demoted to RELATED_TO rather than dropped, so the information
survives, the DAG survives, and topological sort stays valid.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace

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
    aliases: dict[str, str] = field(default_factory=dict)

    # ---------- identity ----------

    def resolve(self, node_id: str) -> str:
        """Map a merged or historical id onto its surviving canonical node.

        Node ids are never reassigned, so an id that loses a merge stays
        resolvable forever rather than dangling.
        """
        seen: set[str] = set()
        while node_id in self.aliases and node_id not in seen:
            seen.add(node_id)
            node_id = self.aliases[node_id]
        return node_id

    def merge_node(self, losing_id: str, surviving_id: str) -> None:
        """Fold one node into another, rewriting its edges onto the survivor."""
        if surviving_id not in self.nodes:
            raise KeyError(f"unknown surviving node: {surviving_id}")
        if losing_id == surviving_id:
            raise ValueError(f"cannot merge {losing_id} into itself")

        self.aliases[losing_id] = surviving_id
        self.nodes.pop(losing_id, None)

        for key, edge in list(self.edges.items()):
            if losing_id not in (edge.src, edge.dst):
                continue
            del self.edges[key]
            src = surviving_id if edge.src == losing_id else edge.src
            dst = surviving_id if edge.dst == losing_id else edge.dst
            if src == dst:
                continue  # the merge collapsed this edge into a self-edge
            self.add_edge(src, dst, edge.rel, edge.invoked_at)

    # ---------- construction ----------

    def add_node(self, node: Node) -> Node:
        existing = self.nodes.get(node.id)
        if existing is not None:
            return existing
        self.nodes[node.id] = node
        for alias in node.aliases:
            self.aliases[alias] = node.id
        return node

    def retier(self, node_id: str, tier: Tier) -> None:
        node = self.nodes[self.resolve(node_id)]
        self.nodes[node.id] = replace(node, tier=tier)

    def drop_edge(self, src: str, dst: str) -> bool:
        """Remove an edge regardless of relation. Used when retargeting a split."""
        removed = False
        for rel in Relation:
            if (src, dst, rel) in self.edges:
                del self.edges[(src, dst, rel)]
                removed = True
        return removed

    def add_edge(
        self,
        src: str,
        dst: str,
        rel: Relation,
        invoked_at: tuple[str, ...] = (),
    ) -> AddEdgeResult:
        src, dst = self.resolve(src), self.resolve(dst)
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

    def transitive_reduction(self) -> list[tuple[str, str]]:
        """Drop REQUIRES edges implied by a longer path.

        Several clusters trimmed these by hand ("reachable via X"), which is
        judgment applied inconsistently. Reachability is what syllabus() uses,
        so removing an implied edge cannot change any learning path -- it only
        stops one node claiming a prerequisite another node already supplies.

        Edges out of ALGORITHM nodes are exempt. Those record what a
        derivation actually invokes, which is extraction data: an EKF
        derivation really does use a Jacobian, whether or not something else
        it requires happens to reach one.
        """
        removed: list[tuple[str, str]] = []
        candidates = [
            e
            for e in self.edges.values()
            if e.rel is Relation.REQUIRES
            and self.nodes[e.src].type is not NodeType.ALGORITHM
        ]
        for edge in sorted(candidates, key=lambda e: (e.src, e.dst)):
            others = [
                d for d in self._requires_out(edge.src) if d != edge.dst
            ]
            reachable: set[str] = set()
            for start in others:
                reachable.add(start)
                reachable |= self.requires_closure(start)
            if edge.dst in reachable:
                del self.edges[edge.key()]
                removed.append((edge.src, edge.dst))
        return removed

    def algorithms_under(self, node_id: str) -> set[str]:
        """Every algorithm reachable from an architecture node.

        Walks CONTAINS down the project layer and crosses USES at the seam.
        """
        found: set[str] = set()
        queue = deque([node_id])
        seen: set[str] = set()
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            if self.nodes[current].type is NodeType.ALGORITHM:
                found.add(current)
                continue
            for edge in self.edges.values():
                if edge.src == current and edge.rel in (
                    Relation.CONTAINS,
                    Relation.USES,
                ):
                    queue.append(edge.dst)
        return found

    def learning_path(self, node_id: str, known: set[str] | None = None) -> list[str]:
        """What to learn, in order, to understand an architecture node.

        The product query: given a subsystem or the whole system, collect every
        algorithm under it and return the union of their prerequisites in
        dependency order, cut at the reader's floor.
        """
        algorithms = self.algorithms_under(node_id)
        # Assumed-tier nodes are the canon's own floor. A caller asking what to
        # learn should never be handed them, whatever their personal profile.
        floor = {n.id for n in self.nodes.values() if n.tier is Tier.ASSUMED}
        known = (known or set()) | floor
        if not algorithms:
            return self.syllabus(node_id, known)

        wanted: set[str] = set()
        for algorithm in algorithms:
            wanted |= self.requires_closure(algorithm)
        wanted -= known

        ordered: list[str] = []
        for algorithm in sorted(algorithms):
            for prerequisite in self.syllabus(algorithm, known):
                if prerequisite in wanted and prerequisite not in ordered:
                    ordered.append(prerequisite)
        return ordered

    # ---------- validation ----------

    def validate(self, check_orphans: bool = True) -> list[str]:
        """Deterministic checks, enforced at write time rather than by an agent.

        `check_orphans` is a whole-graph property and is meaningless on a
        partially merged graph, where most nodes legitimately have no edges yet.
        """
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

        if check_orphans:
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
