"""What the map adds about a project's components: topic links and Graphify."""

import json
import shutil
import subprocess
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

    def test_a_component_the_map_lacks_is_added(self):
        nodes = supporting_nodes(self.EDGES, {"w__ekf"}, {"w__relay": "relay"})
        self.assertEqual([n["id"] for n in nodes], ["w__relay"])
        self.assertEqual(nodes[0]["role"], "supporting")

    def test_a_component_already_shown_is_not_duplicated(self):
        self.assertEqual(supporting_nodes(self.EDGES, {"w__ekf", "w__relay"}, {}), [])


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

    def test_a_supporting_component_hangs_off_nothing(self):
        """Hung off the program, all seven appeared at level 0, before any
        subsystem was opened. They come in beside what they talk to."""
        self.assertIn("for (const p of partners(id)) if (isSupporting(p)) reveal(p);",
                      self.TEMPLATE)

    def test_a_bright_line_never_ends_at_a_faded_node(self):
        """The selected node's lines are drawn bright, so its parts and
        parents must count as related too."""
        self.assertIn("for (const n of under.get(selected) || []) related.add(n);",
                      self.TEMPLATE)

    def test_the_panel_shows_both_ends_of_a_link(self):
        self.assertIn("renderTalk(n)", self.TEMPLATE)
        self.assertIn("'sent: ' + ev.published", self.TEMPLATE)


@unittest.skipUnless(shutil.which("node"), "needs node to run the template's JS")
class TheMapFormatsMaths(unittest.TestCase):
    """Runs the template's own tex() rather than a copy of it."""

    def tex(self, *inputs: str) -> list[str]:
        page = (ROOT / "viz" / "canon_map_template.html").read_text()
        start = page.index("const GREEK")
        end = page.index("/* ---------- adjacency ---------- */")
        script = (page[start:end] + "\nconsole.log(JSON.stringify("
                  + json.dumps(list(inputs)) + ".map(tex)));")
        out = subprocess.run(["node", "-e", script], capture_output=True,
                             text=True, check=True)
        return json.loads(out.stdout)

    def test_an_identifier_keeps_its_underscore(self):
        """The drawer printed "ekfglobal" and "robotlocalization"."""
        self.assertEqual(self.tex("ekf_global from robot_localization"),
                         ["ekf_global from robot_localization"])

    def test_a_symbol_subscript_still_renders(self):
        self.assertEqual(self.tex("x_k", "\\Sigma_12", "P_\\alpha", "x_{k-1}"),
                         ["x<sub>k</sub>", "Σ<sub>12</sub>", "P<sub>α</sub>",
                          "x<sub>k-1</sub>"])


@unittest.skipUnless(shutil.which("node"), "needs node to run the template's JS")
class ThePromptNamesItsSources(unittest.TestCase):
    """Runs the template's own sourceLines() on a node."""

    DOCS = {"arxiv:2406.06427": {"title": "Notes on Kalman Filter", "authors": "Gyubeom Im",
                                 "year": 2024, "section": "4 Extended Kalman Filter (EKF)",
                                 "page": "https://arxiv.org/abs/2406.06427"}}

    def lines(self, node: dict) -> list[str]:
        page = (ROOT / "viz" / "canon_map_template.html").read_text()
        start = page.index("const hostOf")
        end = page.index("function studyPrompt")
        script = ("const DATA = " + json.dumps({"documents": self.DOCS}) + ";\n"
                  + page[start:end] + "\nconsole.log(JSON.stringify(sourceLines("
                  + json.dumps(node) + ")));")
        out = subprocess.run(["node", "-e", script], capture_output=True,
                             text=True, check=True)
        return json.loads(out.stdout)

    def test_the_restated_article_and_the_reading_are_both_listed(self):
        got = self.lines({
            "statement": "x", "sources": ["wikipedia:Multivariate_normal_distribution"],
            "reading": [{"url": "https://arxiv.org/abs/1", "kind": "preprint",
                         "title": "Kalman Filtering", "byline": "Hartley · 2016",
                         "via": "arxiv"},
                        {"url": "http://x.edu/n.pdf", "kind": "course notes",
                         "title": "n — x.edu", "byline": "", "via": "url"}]})
        self.assertIn("Multivariate normal distribution", got[0])
        self.assertIn('"Kalman Filtering", Hartley · 2016 [preprint]', got[1])
        self.assertTrue(got[2].startswith("- (untitled link) [course notes]"))

    def test_a_pinned_paper_is_named_with_its_section(self):
        got = self.lines({"statement": "x", "sources": ["arxiv:2406.06427"], "reading": []})
        self.assertEqual(len(got), 1)
        self.assertIn('"Notes on Kalman Filter", Gyubeom Im 2024', got[0])
        self.assertIn('section "4 Extended Kalman Filter (EKF)"', got[0])

    def test_no_sheet_means_no_claimed_source(self):
        """Without a statement nothing was restated, so Wikipedia is not
        named as the source of it."""
        got = self.lines({"statement": "", "sources": ["wikipedia:X"], "reading": []})
        self.assertEqual(got, [])


class TheMapFitsTheScreen(unittest.TestCase):
    TEMPLATE = (ROOT / "viz" / "canon_map_template.html").read_text()

    def test_the_camera_centres_in_what_the_panel_leaves(self):
        """Centred on the whole canvas, the clicked node sat under the panel."""
        block = self.TEMPLATE.split("function project(n) {")[1][:200]
        self.assertIn("cx()", block)
        self.assertNotIn("cw / 2", block)

    def test_a_phone_gets_a_sheet_and_slide_in_controls(self):
        self.assertIn('id="sheetgrip"', self.TEMPLATE)
        self.assertIn("body.menu-open aside.rail{transform:none}", self.TEMPLATE)

    def test_two_fingers_zoom(self):
        self.assertIn("zoomAt(now.d / pinch.d, now.mx, now.my)", self.TEMPLATE)

    def test_code_rows_are_not_double_spaced(self):
        self.assertNotIn("codeEl.appendChild(document.createTextNode('\\n'))",
                         self.TEMPLATE)


if __name__ == "__main__":
    unittest.main()
