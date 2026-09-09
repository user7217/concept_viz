"""Tests for merging independently-authored cluster proposals."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.merge import BRANCHING_CAP, merge, read_proposals
from atlas.schema import Relation


def write_proposals(directory: Path, clusters: dict[str, list[tuple[str, str]]]) -> None:
    for cluster, edges in clusters.items():
        payload = {
            "cluster": cluster,
            "edges": [{"from": a, "to": b, "why": "test"} for a, b in edges],
            "flags": [],
        }
        (directory / f"{cluster}.json").write_text(json.dumps(payload))


class TestMerge(unittest.TestCase):
    def merge_with(self, clusters: dict[str, list[tuple[str, str]]]):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_proposals(directory, clusters)
            return merge(read_proposals(directory))

    def test_valid_edges_are_accepted(self) -> None:
        _, report = self.merge_with(
            {"a": [("gaussian_conditioning", "joint_gaussian"),
                   ("jacobian_matrix", "partial_derivative")]}
        )
        self.assertEqual(len(report.accepted), 2)
        self.assertEqual(report.rejected_count, 0)

    def test_unknown_ids_are_rejected_not_created(self) -> None:
        graph, report = self.merge_with(
            {"a": [("gaussian_conditioning", "sheaf_cohomology")]}
        )
        self.assertIn("unknown target id", report.rejected)
        self.assertNotIn("sheaf_cohomology", graph.nodes)

    def test_edges_out_of_floor_nodes_are_rejected(self) -> None:
        _, report = self.merge_with({"a": [("matrix", "gaussian_distribution")]})
        self.assertIn("edge out of a floor node", report.rejected)

    def test_edges_out_of_seed_algorithms_are_rejected(self) -> None:
        _, report = self.merge_with({"a": [("extended_kalman_filter", "trace")]})
        self.assertIn("edge out of a seed algorithm", report.rejected)

    def test_duplicate_across_clusters_is_dropped_once(self) -> None:
        _, report = self.merge_with(
            {"aaa": [("gaussian_conditioning", "joint_gaussian")],
             "bbb": [("gaussian_conditioning", "joint_gaussian")]}
        )
        self.assertEqual(len(report.accepted), 1)
        self.assertIn("duplicate", report.rejected)

    def test_branching_cap_is_enforced(self) -> None:
        targets = ["trace", "quadratic_form", "eigenvalue", "eigenvector",
                   "orthogonal_matrix", "gram_matrix", "pseudoinverse"]
        _, report = self.merge_with(
            {"a": [("covariance_matrix", t) for t in targets]}
        )
        self.assertEqual(len(report.accepted), BRANCHING_CAP)
        self.assertTrue(
            any("branching cap" in reason for reason in report.rejected)
        )

    def test_cross_cluster_cycle_is_demoted_not_dropped(self) -> None:
        """The failure mode independent authoring actually produces."""
        graph, report = self.merge_with(
            {"aaa": [("gaussian_conditioning", "covariance_matrix")],
             "bbb": [("covariance_matrix", "gaussian_conditioning")]}
        )
        self.assertEqual(len(report.demoted), 1)
        self.assertIn(
            ("covariance_matrix", "gaussian_conditioning", Relation.RELATED_TO),
            graph.edges,
        )
        # orphans are expected here: this is a two-edge synthetic merge
        self.assertEqual(graph.validate(check_orphans=False), [])

    def test_merge_is_order_independent_in_accepted_set(self) -> None:
        edges = {"aaa": [("gaussian_conditioning", "joint_gaussian")],
                 "bbb": [("jacobian_matrix", "partial_derivative")]}
        first, _ = self.merge_with(edges)
        second, _ = self.merge_with({k: edges[k] for k in reversed(list(edges))})
        self.assertEqual(sorted(first.edges), sorted(second.edges))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestCanonEdgesLoaded(unittest.TestCase):
    """Once curated edges exist, the canon must have real depth."""

    def test_canon_gives_algorithms_depth_beyond_one(self) -> None:
        from atlas.bootstrap import build_graph, load
        from atlas.merge import depth_profile

        graph = build_graph(load())
        depths = depth_profile(graph)
        self.assertTrue(depths, "no algorithm nodes found")
        self.assertGreater(min(depths.values()), 1, "syllabus ordering still vacuous")
        self.assertEqual(graph.validate(), [])

    def test_merge_is_idempotent(self) -> None:
        """Re-merging the same proposals must not change the result."""
        import json
        import tempfile
        from atlas.bootstrap import CANON_EDGES
        from atlas.merge import merge, read_proposals

        with CANON_EDGES.open() as fh:
            existing = json.load(fh)["edges"]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "all.json").write_text(json.dumps({
                "cluster": "all",
                "edges": [{"from": e["from"], "to": e["to"], "why": e["why"]}
                          for e in existing],
                "flags": [],
            }))
            _, report = merge(read_proposals(directory))
        self.assertEqual(report.rejected_count, 0)
        self.assertEqual(len(report.demoted), 2)
