import json
import tempfile
import unittest
from pathlib import Path

from atlas.bootstrap import build_graph, load
from atlas.brief import Brief, propose_from_brief, to_project, unknown

GOOD = {
    "project": "a second repo",
    "subsystems": {"Information Geometry": ["whitening"]},
    "components": {"whitening": {
        "algorithms": ["natural_gradient_descent"],
        "files": ["geometry/whitening.py"],
        "purpose": "regularised F^-1/2",
    }},
}


class TestBrief(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = build_graph(load())

    def _brief(self, **over):
        return Brief.from_json(json.dumps({**GOOD, **over}))

    def test_it_produces_the_same_project_shape_as_the_repo_path(self) -> None:
        # stage 2 must not be able to tell the two inputs apart
        project = to_project(self._brief())
        self.assertEqual(project.origin, "brief")
        self.assertEqual([c.name for c in project.components], ["whitening"])
        self.assertEqual(project.components[0].sources, ["geometry/whitening.py"])

    def test_subsystems_are_stated_not_looked_up(self) -> None:
        proposal = propose_from_brief(self._brief())
        self.assertEqual(proposal.subsystems, {"Information Geometry": ["whitening"]})
        self.assertEqual(proposal.algorithms,
                         {"whitening": ["natural_gradient_descent"]})
        self.assertEqual(proposal.unclassified, [])

    def test_a_component_with_no_algorithm_is_unclassified(self) -> None:
        brief = self._brief(components={"whitening": {"files": []}})
        self.assertEqual(propose_from_brief(brief).unclassified, ["whitening"])

    def test_an_algorithm_the_canon_lacks_is_caught(self) -> None:
        brief = self._brief(components={"whitening": {
            "algorithms": ["quantum_gradient_flow"]}})
        problems = unknown(brief, self.graph)
        self.assertIn("algorithms not in the canon", problems)

    def test_a_file_that_is_not_there_is_caught(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            problems = unknown(self._brief(), self.graph, Path(tmp))
        self.assertIn("files not found in the repo", problems)

    def test_a_component_in_no_subsystem_is_caught(self) -> None:
        brief = self._brief(subsystems={"Information Geometry": []})
        self.assertIn("components described but in no subsystem",
                      unknown(brief, self.graph))

    def test_a_subsystem_naming_an_undescribed_component_is_caught(self) -> None:
        brief = self._brief(
            subsystems={"Information Geometry": ["whitening", "ghost"]})
        self.assertIn("components in a subsystem but never described",
                      unknown(brief, self.graph))

    def test_the_real_v1_brief_is_clean(self) -> None:
        path = Path(__file__).resolve().parent.parent / "data/briefs/v1.json"
        if not path.exists():
            self.skipTest("no v1 brief")
        from atlas.brief import load_brief
        # algorithms must resolve even without the repo present
        problems = unknown(load_brief(path), self.graph)
        self.assertEqual(problems, {})


if __name__ == "__main__":
    unittest.main()
