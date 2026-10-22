"""Joining a Graphify graph to an ingested project."""

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atlas.codegraph import GraphMissing, join, load_graph, report, tokens
from atlas.ingest import Component, Project


def node(id, label, origin="ast", source_file="", location=None, external=False,
         file_type="code"):
    raw = {"id": id, "label": label, "_origin": origin, "file_type": file_type,
           "source_file": source_file, "source_location": location}
    if external:
        raw["type"] = "external"
    return raw


def edge(source, target, relation, confidence="EXTRACTED"):
    return {"source": source, "target": target, "relation": relation,
            "confidence": confidence}


RELAY = "src/pkg/pkg/odom_cov_relay.py"

GRAPH = {
    "nodes": [
        node("relay_file", "odom_cov_relay.py", source_file=RELAY, location="L1"),
        node("relay_cls", "OdomCovRelay", source_file=RELAY, location="L5"),
        node("relay_main", "main()", source_file=RELAY, location="L19"),
        node("rclpy", "rclpy", external=True),
        node("relay_doc", "odom_cov_relay.py (odometry covariance injection)",
             origin="semantic", file_type="concept", source_file="README.md"),
        node("ekf_local_cfg", "ekf_local node config", origin="semantic",
             file_type="concept", source_file="config/ekf_local.yaml"),
        node("ekf_global_cfg", "ekf_global node config", origin="semantic",
             file_type="concept", source_file="config/ekf_global.yaml"),
        node("rpp", "Regulated Pure Pursuit controller (FollowPath)",
             origin="semantic", file_type="concept", source_file="config/nav2.yaml"),
        node("behaviors", "behavior_server (backup, drive_on_heading, wait)",
             origin="semantic", file_type="concept", source_file="config/nav2.yaml"),
        node("tf_tree", "TF tree map -> odom -> base_footprint", origin="semantic",
             file_type="concept", source_file="README.md"),
        node("lint", "test_flake8.py", source_file="src/pkg/test/test_flake8.py"),
    ],
    "links": [
        edge("relay_main", "relay_cls", "calls"),
        edge("relay_cls", "rclpy", "calls"),
        edge("relay_doc", "ekf_local_cfg", "shares_data_with", "INFERRED"),
        edge("ekf_local_cfg", "relay_doc", "shares_data_with", "INFERRED"),
        edge("relay_doc", "ekf_global_cfg", "shares_data_with"),
    ],
}


def project() -> Project:
    return Project(name="demo", origin="repo", components=[
        Component(name="odom_cov_relay", library="pkg", executable="odom_cov_relay",
                  sources=["launch/localization.launch.py"]),
        Component(name="ekf_local", library="robot_localization",
                  executable="ekf_node", sources=["launch/localization.launch.py"]),
        Component(name="ekf_global", library="robot_localization",
                  executable="ekf_node", sources=["launch/localization.launch.py"]),
        Component(name="RegulatedPurePursuitController",
                  library="nav2_regulated_pure_pursuit_controller"),
        Component(name="BackUp", library="nav2_behaviors"),
        Component(name="rviz2", library="rviz2", executable="rviz2"),
    ])


class CodeGraphTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        out = self.repo / "graphify-out"
        out.mkdir()
        (out / "graph.json").write_text(json.dumps(GRAPH))
        self.graph = load_graph(self.repo)
        self.result = join(project(), self.graph)
        self.by_name = {m.component: m for m in self.result.matches}

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_a_missing_graph_says_what_to_run(self) -> None:
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(GraphMissing) as caught:
                load_graph(Path(empty))
        self.assertIn("graphify", str(caught.exception))

    def test_own_code_is_found_by_its_module_name(self) -> None:
        labels = {n.label for n in self.by_name["odom_cov_relay"].code}
        self.assertEqual(labels, {"odom_cov_relay.py", "OdomCovRelay", "main()"})

    def test_code_is_listed_in_line_order(self) -> None:
        """Sorted as text, L19 came before L5."""
        lines = [n.location for n in self.by_name["odom_cov_relay"].code]
        self.assertEqual(lines, ["L1", "L5", "L19"])

    def test_a_library_component_has_no_code_in_this_repo(self) -> None:
        """ekf_node lives in robot_localization, not here."""
        self.assertEqual(self.by_name["ekf_local"].code, [])

    def test_a_shared_executable_identifies_neither_component(self) -> None:
        """Both filters run ekf_node. Matching on it described each with the
        other's configuration."""
        local = {n.label for n in self.by_name["ekf_local"].described_by}
        glob = {n.label for n in self.by_name["ekf_global"].described_by}
        self.assertEqual(local, {"ekf_local node config"})
        self.assertEqual(glob, {"ekf_global node config"})

    def test_names_meet_however_they_are_cased(self) -> None:
        self.assertEqual(
            [n.label for n in self.by_name["RegulatedPurePursuitController"].described_by],
            ["Regulated Pure Pursuit controller (FollowPath)"])
        self.assertEqual(tokens("RegulatedPurePursuitController"),
                         {"regulated", "pure", "pursuit", "controller"})

    def test_a_name_written_as_one_word_still_matches(self) -> None:
        """"BackUp" appears in the label as "backup"."""
        self.assertTrue(self.by_name["BackUp"].described_by)

    def test_flows_leave_the_component(self) -> None:
        """main() calling OdomCovRelay is internal, and rclpy is a library."""
        flows = self.by_name["odom_cov_relay"].flows
        others = {e.target if e.source in {"relay_doc", "relay_cls", "relay_main",
                                           "relay_file"} else e.source
                  for e in flows}
        self.assertEqual(others, {"ekf_local_cfg", "ekf_global_cfg"})

    def test_a_flow_stated_twice_is_listed_once(self) -> None:
        local = [e for e in self.by_name["odom_cov_relay"].flows
                 if "ekf_local_cfg" in (e.source, e.target)]
        self.assertEqual(len(local), 1)

    def test_only_extracted_edges_are_facts(self) -> None:
        facts = {e.is_fact for e in self.by_name["odom_cov_relay"].flows}
        self.assertEqual(facts, {True, False})

    def test_misses_are_reported_on_both_sides(self) -> None:
        self.assertEqual(self.result.unmatched_components, ["rviz2"])
        unclaimed = {n.label for n in self.result.unclaimed}
        self.assertIn("TF tree map -> odom -> base_footprint", unclaimed)
        text = report(self.result, self.graph)
        self.assertIn("rviz2", text)
        self.assertIn("TF tree", text)

    def test_lint_scaffolding_is_not_reported_as_unclaimed(self) -> None:
        self.assertNotIn("test_flake8.py", {n.label for n in self.result.unclaimed})

    def test_an_absolute_source_path_compares_equal(self) -> None:
        raw = dict(GRAPH)
        raw["nodes"] = GRAPH["nodes"] + [node(
            "abs", "abs node", origin="semantic",
            source_file=str(self.repo / "README.md"))]
        (self.repo / "graphify-out" / "graph.json").write_text(json.dumps(raw))
        self.assertEqual(load_graph(self.repo).nodes["abs"].source_file, "README.md")


class AtlasStaysStdlibOnly(unittest.TestCase):
    """Graphify is a file Atlas reads, never a package it imports."""

    def test_nothing_imports_graphify(self) -> None:
        pattern = re.compile(r"^\s*(import|from)\s+graphify\b", re.M)
        offenders = [str(p.relative_to(ROOT))
                     for folder in ("atlas", "scripts")
                     for p in (ROOT / folder).glob("*.py")
                     if pattern.search(p.read_text())]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
