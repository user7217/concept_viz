"""What the map adds about a project's components: topic links and Graphify."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atlas.ingest import Component, Parameter, Project
from atlas.wiring import (component_id, graphify_panels, supporting_nodes,
                          topic_edges)

RELAY = ("class R:\n"
         "    def __init__(self):\n"
         "        self.create_publisher(Odometry, '/odom_cov', 10)\n"
         "        self.create_publisher(Odometry, '/odom_raw', 10)\n")


def project() -> Project:
    return Project(name="demo_ws", origin="repo", components=[
        Component(name="odom_cov_relay", executable="odom_cov_relay"),
        Component(name="ekf_local", library="robot_localization", executable="ekf_node",
                  parameters=[Parameter("odom0", "/odom_cov", "ekf_local.yaml"),
                              Parameter("odom1", "/odom_raw", "ekf_local.yaml")]),
    ])


class TopicEdges(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        (self.repo / "odom_cov_relay.py").write_text(RELAY)
        self.edges = topic_edges(project(), self.repo, "demo_ws")

    def tearDown(self):
        self.tmp.cleanup()

    def test_one_edge_per_pair_carries_every_topic(self):
        self.assertEqual(len(self.edges), 1)
        self.assertEqual(sorted(self.edges[0]["topics"]), ["/odom_cov", "/odom_raw"])

    def test_edges_use_the_architecture_layers_ids(self):
        edge = self.edges[0]
        self.assertEqual((edge["from"], edge["to"]),
                         ("demo_ws__odom_cov_relay", "demo_ws__ekf_local"))
        self.assertEqual(edge["rel"], "topic")

    def test_each_topic_keeps_both_ends_evidence(self):
        ev = {e["topic"]: e for e in self.edges[0]["evidence"]}
        self.assertEqual(ev["/odom_cov"]["published"],
                         "odom_cov_relay.py:3 create_publisher")
        self.assertEqual(ev["/odom_cov"]["read"], "ekf_local.yaml odom0")


class SupportingNodes(unittest.TestCase):
    EDGES = [{"from": "w__relay", "to": "w__ekf", "rel": "topic"}]

    def test_a_component_the_map_lacks_is_added_under_the_program(self):
        nodes, placed = supporting_nodes(self.EDGES, {"w__ekf"}, "w",
                                         {"w__relay": "relay"})
        self.assertEqual([n["id"] for n in nodes], ["w__relay"])
        self.assertEqual(nodes[0]["role"], "supporting")
        self.assertEqual(placed, [{"from": "w", "to": "w__relay",
                                   "rel": "contains", "why": ""}])

    def test_a_component_already_shown_is_not_duplicated(self):
        nodes, placed = supporting_nodes(self.EDGES, {"w__ekf", "w__relay"}, "w", {})
        self.assertEqual((nodes, placed), ([], []))


class GraphifyPanels(unittest.TestCase):
    def test_no_graph_means_no_panels_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(graphify_panels(project(), Path(tmp), "demo_ws"), {})

    def test_code_and_descriptions_are_kept_apart(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "graphify-out").mkdir()
            (repo / "graphify-out" / "graph.json").write_text(json.dumps({
                "nodes": [
                    {"id": "a", "label": "OdomCovRelay", "_origin": "ast",
                     "source_file": "pkg/odom_cov_relay.py", "source_location": "L5"},
                    {"id": "b", "label": "ekf_local node config", "_origin": "semantic",
                     "source_file": "README.md", "rationale": "50 Hz, odom frame"},
                ], "links": []}))
            panels = graphify_panels(project(), repo, "demo_ws")
        relay = panels[component_id("demo_ws", "odom_cov_relay")]
        self.assertEqual(relay["codeRefs"][0]["file"], "pkg/odom_cov_relay.py")
        ekf = panels[component_id("demo_ws", "ekf_local")]
        self.assertEqual(ekf["codeRefs"], [])
        self.assertEqual(ekf["described"][0]["rationale"], "50 Hz, odom frame")


class TheMapTreatsTopicsAsLateral(unittest.TestCase):
    TEMPLATE = (ROOT / "viz" / "canon_map_template.html").read_text()

    def test_topics_never_enter_the_walkthrough_descent(self):
        """A topic between two components is not a step down into either.
        Letting it into `under` would make clicking a relay reveal the EKF
        as if the relay were made of it."""
        block = self.TEMPLATE.split("/* ---------- adjacency ---------- */")[1][:1200]
        self.assertLess(block.index("if (e.rel === 'topic')"),
                        block.index("under.get(e.from).push(e.to)"))

    def test_the_panel_shows_both_ends_of_a_link(self):
        self.assertIn("renderTalk(n)", self.TEMPLATE)
        self.assertIn("'sent: ' + ev.published", self.TEMPLATE)


if __name__ == "__main__":
    unittest.main()
