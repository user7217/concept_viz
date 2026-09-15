import unittest

from atlas.bootstrap import build_graph, load
from atlas.graph import Graph
from atlas.schema import Node, NodeType, Relation, Tier
from atlas.worksheet import build


def _graph() -> Graph:
    g = Graph()
    for nid, ntype in [("ekf", NodeType.ALGORITHM), ("bayes", NodeType.CONCEPT),
                       ("marg", NodeType.CONCEPT), ("loose", NodeType.CONCEPT)]:
        g.add_node(Node(id=nid, name=nid, type=ntype, tier=Tier.STATEMENT))
    g.add_edge("ekf", "bayes", Relation.REQUIRES)
    g.add_edge("bayes", "marg", Relation.REQUIRES)
    return g


SHEET = {
    "node_id": "marg", "statement": "Marginals come from joints.",
    "steps": [
        {"n": 1, "text": "Start from the joint density of X and Y.",
         "invokes": []},
        {"n": 2, "text": "Integrate over every value Y takes, discarding its"
                         " identity, to leave the marginal.",
         "invokes": ["algebraic_rearrangement"]},
    ],
    "symbols": [{"symbol": "p", "meaning": "a probability density function"}],
    "assumptions": [], "failure_modes": [],
    "instantiation": {
        "why_here": "The EKF prediction step marginalises the previous state.",
        "knobs": [{"parameter": "process_noise_covariance",
                   "file": "config/ekf.yaml", "means": "sets Q"}],
    },
}


class TestRequiresChain(unittest.TestCase):
    def test_the_chain_explains_why_a_node_is_on_the_path(self) -> None:
        self.assertEqual(_graph().requires_chain("ekf", "marg"),
                         ["ekf", "bayes", "marg"])

    def test_an_unreachable_node_has_no_chain(self) -> None:
        self.assertEqual(_graph().requires_chain("ekf", "loose"), [])

    def test_a_node_is_its_own_chain(self) -> None:
        self.assertEqual(_graph().requires_chain("marg", "marg"), ["marg"])

    def test_the_real_canon_chains_to_the_algorithm(self) -> None:
        g = build_graph(load())
        chain = g.requires_chain("extended_kalman_filter", "marginalization")
        self.assertEqual(chain[0], "extended_kalman_filter")
        self.assertEqual(chain[-1], "marginalization")


class TestWorksheet(unittest.TestCase):
    def render(self, **kw) -> str:
        return build(_graph(), ["marg"], {"marg": SHEET}, algorithm="ekf",
                     project_name="rig", subsystem="Localization", **kw)

    def test_every_node_says_why_it_is_on_the_path(self) -> None:
        # the complaint this exists to answer: an ordered list reads as a pile
        # of unrelated mathematics
        out = self.render()
        self.assertIn("Why it is here", out)
        self.assertIn("ekf", out)
        self.assertIn("bayes", out)

    def test_the_project_note_and_knobs_are_carried_through(self) -> None:
        out = self.render()
        self.assertIn("EKF prediction step marginalises", out)
        self.assertIn("process_noise_covariance", out)

    def test_symbols_are_off_by_default(self) -> None:
        out = self.render()
        self.assertNotIn("What is p?", out)
        self.assertIn("What is p?", self.render(include_symbols=True))

    def test_answers_come_after_every_question(self) -> None:
        out = self.render()
        self.assertLess(out.index("**1."), out.index("# Answers"))
        # the answer text must not sit in the question half
        self.assertGreater(out.rindex("Integrate over every value"),
                           out.index("# Answers"))

    def test_a_stepless_sheet_says_why_it_is_still_listed(self) -> None:
        stepless = dict(SHEET, steps=[])
        out = build(_graph(), ["marg"], {"marg": stepless}, algorithm="ekf",
                    project_name="rig", subsystem="Localization")
        self.assertIn("here as a prerequisite", out)


if __name__ == "__main__":
    unittest.main()
