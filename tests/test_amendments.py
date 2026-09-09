"""Tests for flag adjudication: added nodes, merges, tier corrections."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.bootstrap import build_graph, load, load_amendments
from atlas.graph import Graph
from atlas.schema import Node, NodeType, Relation, Tier


class TestAliasResolution(unittest.TestCase):
    def setUp(self) -> None:
        self.g = Graph()
        for n in ("log_odds", "logit", "occupancy", "bayes_rule"):
            self.g.add_node(Node(n, NodeType.CONCEPT, n))
        self.g.add_edge("occupancy", "logit", Relation.REQUIRES)
        self.g.add_edge("logit", "bayes_rule", Relation.REQUIRES)

    def test_merged_id_stays_resolvable(self) -> None:
        """Ids are never reassigned: a losing id must not dangle."""
        self.g.merge_node("logit", "log_odds")
        self.assertEqual(self.g.resolve("logit"), "log_odds")
        self.assertNotIn("logit", self.g.nodes)

    def test_merge_rewrites_edges_onto_survivor(self) -> None:
        self.g.merge_node("logit", "log_odds")
        self.assertIn(("occupancy", "log_odds", Relation.REQUIRES), self.g.edges)
        self.assertIn(("log_odds", "bayes_rule", Relation.REQUIRES), self.g.edges)
        self.assertFalse(any("logit" in (e.src, e.dst) for e in self.g.edges.values()))

    def test_merge_collapses_edges_that_become_self_edges(self) -> None:
        self.g.add_edge("log_odds", "logit", Relation.REQUIRES)
        self.g.merge_node("logit", "log_odds")
        self.assertFalse(
            any(e.src == e.dst for e in self.g.edges.values()), "self-edge survived"
        )

    def test_add_edge_resolves_aliases(self) -> None:
        self.g.merge_node("logit", "log_odds")
        result = self.g.add_edge("occupancy", "logit", Relation.REQUIRES)
        self.assertEqual(result.edge.dst, "log_odds")

    def test_alias_chain_resolves_to_the_end(self) -> None:
        self.g.add_node(Node("final", NodeType.CONCEPT, "Final"))
        self.g.merge_node("logit", "log_odds")
        self.g.merge_node("log_odds", "final")
        self.assertEqual(self.g.resolve("logit"), "final")


class TestAmendmentsApplied(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = build_graph(load())
        self.amendments = load_amendments()

    def test_every_added_node_exists(self) -> None:
        for key in ("add_floor", "add_concepts", "add_techniques"):
            for entry in self.amendments[key]:
                self.assertIn(entry["id"], self.graph.nodes, entry["id"])

    def test_added_techniques_have_the_right_type(self) -> None:
        for entry in self.amendments["add_techniques"]:
            self.assertIs(self.graph.nodes[entry["id"]].type, NodeType.TECHNIQUE)

    def test_floor_additions_are_assumed(self) -> None:
        for entry in self.amendments["add_floor"]:
            self.assertIs(self.graph.nodes[entry["id"]].tier, Tier.ASSUMED)

    def test_retiers_took_effect(self) -> None:
        for entry in self.amendments["retier"]:
            self.assertIs(
                self.graph.nodes[entry["id"]].tier, Tier(entry["tier"]), entry["id"]
            )

    def test_merged_nodes_are_gone_but_resolvable(self) -> None:
        for entry in self.amendments["merge"]:
            self.assertNotIn(entry["from"], self.graph.nodes)
            self.assertEqual(self.graph.resolve(entry["from"]), entry["into"])

    def test_every_rejected_flag_carries_a_reason(self) -> None:
        for entry in self.amendments["rejected_flags"]:
            self.assertTrue(entry.get("reason"), entry["flag"])


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestAmendmentsFailLoudly(unittest.TestCase):
    """A curation amendment that cannot be applied must not silently no-op."""

    def test_merge_with_missing_survivor_raises(self) -> None:
        import atlas.bootstrap as bootstrap

        real = bootstrap.load_amendments
        bootstrap.load_amendments = lambda *a, **k: {
            "merge": [{"from": "bayes_rule", "into": "node_that_does_not_exist"}]
        }
        try:
            with self.assertRaises(ValueError) as ctx:
                build_graph(load())
            self.assertIn("survivor missing", str(ctx.exception))
        finally:
            bootstrap.load_amendments = real

    def test_retier_of_missing_node_raises(self) -> None:
        import atlas.bootstrap as bootstrap

        real = bootstrap.load_amendments
        bootstrap.load_amendments = lambda *a, **k: {
            "retier": [{"id": "node_that_does_not_exist", "tier": "claim"}]
        }
        try:
            with self.assertRaises(ValueError):
                build_graph(load())
        finally:
            bootstrap.load_amendments = real


class TestSplitsAndRetargets(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = build_graph(load())

    def test_split_nodes_both_exist_and_are_ordered(self) -> None:
        """The multivariate expansion builds on the univariate one."""
        self.assertIn("taylor_expansion", self.graph.nodes)
        self.assertIn("multivariate_taylor_expansion", self.graph.nodes)
        self.assertIn(
            "taylor_expansion",
            self.graph.requires_closure("multivariate_taylor_expansion"),
        )

    def test_retargeted_consumers_point_at_the_multivariate_form(self) -> None:
        for consumer in ("linearization", "local_minimum", "descent_direction"):
            closure = self.graph.requires_closure(consumer)
            self.assertIn("multivariate_taylor_expansion", closure, consumer)

    def test_univariate_consumers_were_left_alone(self) -> None:
        for consumer in ("step_size", "exponential_map"):
            out = [
                e.dst for e in self.graph.edges.values() if e.src == consumer
            ]
            self.assertIn("taylor_expansion", out, consumer)
            self.assertNotIn("multivariate_taylor_expansion", out, consumer)

    def test_dropped_edges_are_gone_but_targets_stay_reachable(self) -> None:
        """A drop must not remove a prerequisite from the closure entirely."""
        cases = [
            ("riccati_equation", "quadratic_form"),
            ("eigenvalue", "determinant"),
            ("classification_loss", "entropy"),
        ]
        for src, dst in cases:
            direct = [e.dst for e in self.graph.edges.values() if e.src == src]
            self.assertNotIn(dst, direct, f"{src} -> {dst} should be dropped")
            self.assertIn(dst, self.graph.requires_closure(src),
                          f"{dst} no longer reachable from {src}")

    def test_branching_cap_holds_across_the_whole_canon(self) -> None:
        from collections import Counter
        from atlas.merge import BRANCHING_CAP
        from atlas.schema import NodeType, Relation

        counts = Counter(
            e.src for e in self.graph.edges.values()
            if e.rel is Relation.REQUIRES
            and self.graph.nodes[e.src].type is not NodeType.ALGORITHM
        )
        over = {n: c for n, c in counts.items() if c > BRANCHING_CAP}
        self.assertEqual(over, {}, f"nodes over the branching cap: {over}")
