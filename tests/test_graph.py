"""Tests for the graph invariants the design depends on."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.bootstrap import build_graph, load, saturation
from atlas.graph import Graph
from atlas.schema import Node, NodeType, Relation, Tier


def concept(node_id: str) -> Node:
    return Node(node_id, NodeType.CONCEPT, node_id.replace("_", " ").title())


class TestCycleDemotion(unittest.TestCase):
    def setUp(self) -> None:
        self.g = Graph()
        for n in ("ekf", "gaussian", "matrix"):
            self.g.add_node(concept(n))

    def test_back_edge_is_demoted_not_dropped(self) -> None:
        self.g.add_edge("ekf", "gaussian", Relation.REQUIRES)
        self.g.add_edge("gaussian", "matrix", Relation.REQUIRES)
        result = self.g.add_edge("matrix", "ekf", Relation.REQUIRES)

        self.assertTrue(result.demoted)
        self.assertIs(result.edge.rel, Relation.RELATED_TO)
        # information survives
        self.assertIn(("matrix", "ekf", Relation.RELATED_TO), self.g.edges)
        # DAG survives
        self.assertEqual(self.g.validate(), [])

    def test_direct_two_cycle_is_demoted(self) -> None:
        self.g.add_edge("ekf", "gaussian", Relation.REQUIRES)
        result = self.g.add_edge("gaussian", "ekf", Relation.REQUIRES)
        self.assertTrue(result.demoted)

    def test_self_edge_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.g.add_edge("ekf", "ekf", Relation.REQUIRES)

    def test_diamond_is_not_a_cycle(self) -> None:
        self.g.add_node(concept("probability"))
        self.g.add_edge("ekf", "gaussian", Relation.REQUIRES)
        self.g.add_edge("ekf", "probability", Relation.REQUIRES)
        r1 = self.g.add_edge("gaussian", "matrix", Relation.REQUIRES)
        r2 = self.g.add_edge("probability", "matrix", Relation.REQUIRES)
        self.assertFalse(r1.demoted)
        self.assertFalse(r2.demoted)  # shared prerequisite, reused not duplicated


class TestSyllabus(unittest.TestCase):
    def setUp(self) -> None:
        self.g = Graph()
        for n in ("ekf", "gaussian_conditioning", "completing_the_square",
                  "quadratic_form", "matrix"):
            self.g.add_node(concept(n))
        self.g.add_edge("ekf", "gaussian_conditioning", Relation.REQUIRES)
        self.g.add_edge("gaussian_conditioning", "completing_the_square", Relation.REQUIRES)
        self.g.add_edge("completing_the_square", "quadratic_form", Relation.REQUIRES)
        self.g.add_edge("quadratic_form", "matrix", Relation.REQUIRES)

    def test_order_is_deepest_first(self) -> None:
        self.assertEqual(
            self.g.syllabus("ekf"),
            ["matrix", "quadratic_form", "completing_the_square", "gaussian_conditioning"],
        )

    def test_floor_truncates_the_path(self) -> None:
        self.assertEqual(
            self.g.syllabus("ekf", known={"matrix", "quadratic_form"}),
            ["completing_the_square", "gaussian_conditioning"],
        )

    def test_knowing_everything_yields_nothing_to_learn(self) -> None:
        everything = self.g.requires_closure("ekf")
        self.assertEqual(self.g.syllabus("ekf", known=everything), [])


class TestClosureAssertion(unittest.TestCase):
    """The completeness criterion from design doc section 4."""

    def setUp(self) -> None:
        self.g = Graph()
        for n in ("ekf", "jacobian", "taylor"):
            self.g.add_node(concept(n))
        self.g.add_edge("ekf", "jacobian", Relation.REQUIRES)

    def test_unreachable_invocation_is_a_hole(self) -> None:
        holes = self.g.closure_holes("ekf", invoked={"jacobian", "taylor"})
        self.assertEqual(holes, {"taylor"})

    def test_known_invocation_is_not_a_hole(self) -> None:
        holes = self.g.closure_holes("ekf", invoked={"jacobian", "taylor"}, known={"taylor"})
        self.assertEqual(holes, set())


class TestLayerRules(unittest.TestCase):
    def test_uses_must_target_an_algorithm(self) -> None:
        g = Graph()
        g.add_node(Node("localization", NodeType.SUBSYSTEM, "Localization"))
        g.add_node(Node("covariance", NodeType.CONCEPT, "Covariance"))
        g.add_edge("localization", "covariance", Relation.USES)
        self.assertTrue(any("USES must cross" in p for p in g.validate()))

    def test_requires_must_target_a_global_node(self) -> None:
        g = Graph()
        g.add_node(Node("ekf", NodeType.ALGORITHM, "EKF", Tier.DERIVATION))
        g.add_node(Node("localization", NodeType.SUBSYSTEM, "Localization"))
        g.add_edge("ekf", "localization", Relation.REQUIRES)
        self.assertTrue(any("REQUIRES must point" in p for p in g.validate()))


class TestBootstrap(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = load()

    def test_seed_graph_validates_clean(self) -> None:
        self.assertEqual(build_graph(self.payload).validate(), [])

    def test_canon_is_deduplicated_across_seeds(self) -> None:
        g = build_graph(self.payload)
        total = sum(len(s["concepts"]) + len(s["techniques"]) for s in self.payload["seeds"])
        distinct = len(g.by_type(NodeType.CONCEPT)) + len(g.by_type(NodeType.TECHNIQUE))
        self.assertLess(distinct, total)  # reuse actually happened

    def test_saturation_is_monotonic_and_decaying(self) -> None:
        curve = saturation(self.payload)
        sizes = [p.canon_size for p in curve]
        self.assertEqual(sizes, sorted(sizes))
        first_half = sum(p.added for p in curve[:10])
        second_half = sum(p.added for p in curve[10:])
        self.assertLess(second_half, first_half)

    def test_base_concepts_are_marked_assumed(self) -> None:
        g = build_graph(self.payload)
        self.assertIs(g.nodes["matrix"].tier, Tier.ASSUMED)
        self.assertIs(g.nodes["gaussian_conditioning"].tier, Tier.STATEMENT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
