"""Tests for the architecture layer and its confirmation gate."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.architecture import ArchitectureProposal, NotConfirmed, apply, propose
from atlas.ingest import Component, Project
from atlas.schema import NodeType, Relation


def sample_project() -> Project:
    project = Project(name="Agri Robot", origin="repo")
    project.components = [
        Component("ekf_filter_node", library="robot_localization"),
        Component("nav2", library="nav2_bringup"),
        Component("detector", library="ultralytics"),
        Component("relay", library="my_custom_pkg"),
    ]
    return project


class TestPropose(unittest.TestCase):
    def setUp(self) -> None:
        self.proposal = propose(sample_project())

    def test_components_group_by_their_algorithm_domain(self) -> None:
        self.assertEqual(self.proposal.subsystems["Localization"], ["ekf_filter_node"])
        self.assertEqual(self.proposal.subsystems["Navigation"], ["nav2"])
        self.assertEqual(self.proposal.subsystems["Perception"], ["detector"])

    def test_unknown_library_is_unclassified_not_guessed(self) -> None:
        self.assertEqual(self.proposal.unclassified, ["relay"])
        self.assertNotIn("relay", self.proposal.algorithms)

    def test_round_trips_through_json(self) -> None:
        restored = ArchitectureProposal.from_json(self.proposal.to_json())
        self.assertEqual(restored.subsystems, self.proposal.subsystems)
        self.assertFalse(restored.confirmed)


class TestGate(unittest.TestCase):
    def test_unconfirmed_proposal_is_refused(self) -> None:
        with self.assertRaises(NotConfirmed):
            apply(propose(sample_project()))

    def test_unclassified_components_block_apply_even_when_confirmed(self) -> None:
        proposal = propose(sample_project())
        proposal.confirmed = True
        with self.assertRaises(NotConfirmed) as ctx:
            apply(proposal)
        self.assertIn("relay", str(ctx.exception))

    def test_unknown_algorithm_id_raises(self) -> None:
        proposal = propose(sample_project())
        proposal.unclassified = []
        proposal.confirmed = True
        proposal.algorithms["nav2"] = ["not_a_real_algorithm"]
        with self.assertRaises(KeyError):
            apply(proposal)


class TestApply(unittest.TestCase):
    def setUp(self) -> None:
        proposal = propose(sample_project())
        proposal.unclassified = []
        proposal.confirmed = True
        self.graph = apply(proposal)

    def test_architecture_nodes_are_created(self) -> None:
        self.assertIs(self.graph.nodes["agri_robot"].type, NodeType.SYSTEM)
        self.assertIs(
            self.graph.nodes["agri_robot__localization"].type, NodeType.SUBSYSTEM
        )
        self.assertIs(
            self.graph.nodes["agri_robot__ekf_filter_node"].type, NodeType.COMPONENT
        )

    def test_seam_edge_crosses_component_to_algorithm(self) -> None:
        self.assertIn(
            ("agri_robot__ekf_filter_node", "extended_kalman_filter", Relation.USES),
            self.graph.edges,
        )

    def test_graph_still_validates(self) -> None:
        self.assertEqual(self.graph.validate(check_orphans=False), [])

    def test_algorithms_under_the_system(self) -> None:
        found = self.graph.algorithms_under("agri_robot")
        self.assertIn("extended_kalman_filter", found)
        self.assertIn("yolo_detection_loss", found)

    def test_learning_path_crosses_the_seam(self) -> None:
        """A subsystem must yield prerequisites, not an empty list."""
        path = self.graph.learning_path("agri_robot__localization")
        self.assertIn("gaussian_conditioning", path)
        self.assertIn("jacobian_matrix", path)

    def test_learning_path_respects_the_floor(self) -> None:
        full = self.graph.learning_path("agri_robot__localization")
        cut = self.graph.learning_path(
            "agri_robot__localization", known={"gaussian_conditioning"}
        )
        self.assertNotIn("gaussian_conditioning", cut)
        self.assertLess(len(cut), len(full))

    def test_system_path_is_larger_than_one_subsystem(self) -> None:
        whole = self.graph.learning_path("agri_robot")
        part = self.graph.learning_path("agri_robot__localization")
        self.assertGreater(len(whole), len(part))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestFloorExclusion(unittest.TestCase):
    def setUp(self) -> None:
        proposal = propose(sample_project())
        proposal.unclassified = []
        proposal.confirmed = True
        self.graph = apply(proposal)

    def test_assumed_tier_never_appears_in_a_learning_path(self) -> None:
        """The canon's own floor drops out without the caller listing it."""
        from atlas.schema import Tier

        path = self.graph.learning_path("agri_robot")
        assumed = [n for n in path if self.graph.nodes[n].tier is Tier.ASSUMED]
        self.assertEqual(assumed, [])

    def test_retiered_nodes_are_excluded(self) -> None:
        path = self.graph.learning_path("agri_robot")
        for node_id in ("algebraic_rearrangement", "matrix", "graph"):
            self.assertNotIn(node_id, path)
