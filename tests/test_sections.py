"""Pointers to sections another sheet derives."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.bootstrap import build_graph, load
from atlas.sections import dangling, owners, prompt_block

DOCUMENTS = {
    "arxiv:2406.06427": {"section": "4 Extended Kalman Filter (EKF)"},
    "arxiv:2406.06427#8": {"sections": ["8.1 Derivation of KF prediction step",
                                        "8.2 Derivation of KF update step (ver. 1)"]},
}
ALIASES = {
    "extended_kalman_filter": ["arxiv:2406.06427", "Extended Kalman filter"],
    "linear_kalman_filter": ["arxiv:2406.06427#8"],
}


class Owners(unittest.TestCase):
    graph = build_graph(load())

    def found(self, node_id):
        return owners(self.graph, node_id, ALIASES, DOCUMENTS)

    def test_the_ekf_hands_section_8_to_the_linear_filter(self):
        found = self.found("extended_kalman_filter")
        self.assertEqual({(o.section, o.node_id) for o in found},
                         {("8.1", "linear_kalman_filter"), ("8.2", "linear_kalman_filter")})
        self.assertEqual(found[0].document, "arxiv:2406.06427")

    def test_a_sheet_is_never_its_own_owner(self):
        """The linear KF's section 8 is its own; section 4 sits above it."""
        self.assertEqual(self.found("linear_kalman_filter"), [])

    def test_the_prompt_names_the_sheet_only_for_the_cited_document(self):
        found = self.found("extended_kalman_filter")
        self.assertIn("linear_kalman_filter", prompt_block(found, {"arxiv:2406.06427"}))
        self.assertEqual(prompt_block(found, {"wikipedia:Extended_Kalman_filter"}), "")


class Dangling(unittest.TestCase):
    found = owners(build_graph(load()), "extended_kalman_filter", ALIASES, DOCUMENTS)
    cited = {"arxiv:2406.06427"}

    def test_the_stored_ekf_pointer_is_caught(self):
        text = ("the detailed Gaussian integral algebra is cross-referenced to "
                "Section 8 in [arxiv:2406.06427]")
        problems = dangling([text], self.cited, self.found)
        self.assertEqual(len(problems), 1)
        self.assertIn("linear_kalman_filter", problems[0])

    def test_naming_the_sheet_is_the_hand_off(self):
        for text in ("Section 8 of the paper; derived on linear_kalman_filter",
                     "derived on the Linear Kalman Filter sheet (Section 8.1)"):
            self.assertEqual(dangling([text], self.cited, self.found), [], text)

    def test_a_section_nobody_derives_is_left_alone(self):
        """8.3 is not pinned anywhere; there is no sheet to send the reader to."""
        self.assertEqual(dangling(["see Section 8.3"], self.cited, self.found), [])

    def test_another_document_s_section_8_is_not_this_one(self):
        self.assertEqual(dangling(["see Section 8"], {"wikipedia:Kalman_filter"}, self.found), [])

    def test_the_symbol_forms_are_read(self):
        self.assertEqual(len(dangling(["as in §8.2"], self.cited, self.found)), 1)
        self.assertEqual(len(dangling(["Sec. 8"], self.cited, self.found)), 1)


if __name__ == "__main__":
    unittest.main()
