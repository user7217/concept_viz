"""Canon bootstrap: grow the foundations ontology from real derivations.

Design doc section 5. The canon is the union of what the seed algorithms'
derivations actually invoke. Saturation tells us when to stop adding seeds.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .graph import Graph
from .schema import Node, NodeType, Relation, Tier

DATA = Path(__file__).resolve().parent.parent / "data" / "seeds.json"
CANON_EDGES = Path(__file__).resolve().parent.parent / "data" / "canon_edges.json"


@dataclass
class SaturationPoint:
    n_seeds: int
    canon_size: int
    added: int

    @property
    def added_pct(self) -> float:
        prior = self.canon_size - self.added
        return 100.0 * self.added / prior if prior else 100.0


def load(path: Path = DATA) -> dict:
    with path.open() as fh:
        return json.load(fh)


def load_canon_edges(path: Path = CANON_EDGES) -> list[dict]:
    """Internal prerequisite edges within the canon. Absent before curation."""
    if not path.exists():
        return []
    with path.open() as fh:
        return json.load(fh)["edges"]


def build_graph(payload: dict, with_canon_edges: bool = True) -> Graph:
    """One algorithm node per seed, REQUIRES edges to everything it invokes.

    With `with_canon_edges`, also applies the curated internal edges so the
    canon has the depth a syllabus can be ordered by.
    """
    graph = Graph()
    base = set(payload["base"]["concepts"])

    # Materialise the whole floor up front. A floor concept that no seed happens
    # to invoke must still exist as a node, or edges pointing at it are rejected
    # as unknown ids.
    for concept_id in sorted(base):
        graph.add_node(Node(concept_id, NodeType.CONCEPT, _title(concept_id), Tier.ASSUMED))

    for seed in payload["seeds"]:
        graph.add_node(
            Node(seed["id"], NodeType.ALGORITHM, seed["name"], Tier.DERIVATION)
        )
        for kind, node_type in (
            ("concepts", NodeType.CONCEPT),
            ("techniques", NodeType.TECHNIQUE),
        ):
            for item in seed[kind]:
                tier = Tier.ASSUMED if item in base else Tier.STATEMENT
                graph.add_node(Node(item, node_type, _title(item), tier))
                graph.add_edge(seed["id"], item, Relation.REQUIRES)

    if with_canon_edges:
        for edge in load_canon_edges():
            if edge["from"] in graph.nodes and edge["to"] in graph.nodes:
                graph.add_edge(edge["from"], edge["to"], Relation(edge["rel"]))
    return graph


def _title(node_id: str) -> str:
    small = {"of", "the", "to", "and"}
    acronyms = {"pca", "icp", "pid", "lqr", "iou", "so3", "se3", "svd"}
    words = []
    for word in node_id.split("_"):
        if word in acronyms:
            words.append(word.upper())
        elif words and word in small:
            words.append(word)
        else:
            words.append(word.capitalize())
    return " ".join(words)


def saturation(payload: dict, order: list[int] | None = None) -> list[SaturationPoint]:
    seeds = payload["seeds"]
    order = order if order is not None else list(range(len(seeds)))

    canon: set[str] = set()
    curve: list[SaturationPoint] = []
    for position, index in enumerate(order, start=1):
        seed = seeds[index]
        invoked = set(seed["concepts"]) | set(seed["techniques"])
        added = len(invoked - canon)
        canon |= invoked
        curve.append(SaturationPoint(position, len(canon), added))
    return curve


def mean_curve(payload: dict, trials: int = 500, seed: int = 7) -> list[float]:
    """Average canon size after k seeds, over random seed orderings.

    The listed order is arbitrary; averaging over permutations removes the
    artifact of which algorithms happen to come first.
    """
    rng = random.Random(seed)
    n = len(payload["seeds"])
    totals = [0.0] * n
    for _ in range(trials):
        order = list(range(n))
        rng.shuffle(order)
        for point in saturation(payload, order):
            totals[point.n_seeds - 1] += point.canon_size
    return [total / trials for total in totals]


def reuse(payload: dict) -> dict[str, int]:
    """How many seed algorithms invoke each canon node."""
    counts: dict[str, int] = {}
    for seed in payload["seeds"]:
        for item in set(seed["concepts"]) | set(seed["techniques"]):
            counts[item] = counts.get(item, 0) + 1
    return counts
