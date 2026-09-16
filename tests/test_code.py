import tempfile
import unittest
from pathlib import Path

from atlas.code import CodeFile, evidence_for, index_repo, terms_for


class TestTerms(unittest.TestCase):
    def test_parameters_outrank_name_words(self) -> None:
        out = terms_for("Covariance Matrix",
                        {"knobs": [{"parameter": "process_noise_covariance"}]})
        self.assertEqual(out[0], ("process_noise_covariance", "parameter"))

    def test_generic_words_are_dropped(self) -> None:
        # searching a robotics repo for "matrix" returns everything
        words = [t for t, _ in terms_for("Covariance Matrix", None)]
        self.assertIn("covariance", words)
        self.assertNotIn("matrix", words)

    def test_short_words_are_dropped(self) -> None:
        # nothing here is long enough to discriminate anything
        self.assertEqual(terms_for("Row Of A Set", None), [])

    def test_a_term_is_not_repeated(self) -> None:
        out = terms_for("Covariance", {"knobs": [{"parameter": "covariance"}]})
        self.assertEqual(len(out), 1)


class TestEvidence(unittest.TestCase):
    def setUp(self) -> None:
        self.files = [
            CodeFile("src/relay.py", "python", [
                "def cb(self, msg):", "    c = [0.0] * 36", "    c[0] = 0.05",
                "    c[35] = 0.02", "    msg.twist.covariance = c",
            ]),
            CodeFile("config/ekf.yaml", "yaml", [
                "ekf:", "  process_noise_covariance: [0.05, 0.0]",
            ]),
        ]

    def test_executable_code_outranks_configuration(self) -> None:
        # the YAML is already shown as a knob, so leading with it says nothing
        spans = evidence_for("Covariance Matrix", self.files,
                             {"components": ["ekf_local"],
                              "knobs": [{"parameter": "process_noise_covariance"}]})
        self.assertEqual(spans[0].language, "python")

    def test_a_span_reports_a_real_line_number(self) -> None:
        spans = evidence_for("Covariance Matrix", self.files,
                             {"components": ["ekf_local"]})
        span = spans[0]
        line = self.files[0].lines[span.hit - 1]
        self.assertIn("covariance", line.lower())

    def test_context_surrounds_the_hit(self) -> None:
        span = evidence_for("Covariance Matrix", self.files,
                            {"components": ["ekf_local"]})[0]
        self.assertIn("c = [0.0] * 36", "\n".join(span.lines))

    def test_no_terms_means_no_evidence(self) -> None:
        self.assertEqual(
            evidence_for("Row", self.files, {"components": ["ekf_local"]}), [])

    def test_nearby_hits_collapse_to_one_span(self) -> None:
        files = [CodeFile("a.py", "python",
                          ["covariance = 1", "covariance = 2", "covariance = 3"])]
        self.assertEqual(
            len(evidence_for("Covariance", files, {"components": ["x"]})), 1)


class TestIndex(unittest.TestCase):
    def test_build_output_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "build").mkdir()
            (root / "src" / "keep.py").write_text("x = 1\n")
            (root / "build" / "drop.py").write_text("x = 2\n")
            paths = [f.path for f in index_repo(root)]
            self.assertIn("src/keep.py", paths)
            self.assertNotIn("build/drop.py", paths)

    def test_unknown_suffixes_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "notes.md").write_text("# hi\n")
            self.assertEqual(index_repo(root), [])


if __name__ == "__main__":
    unittest.main()


class TestFalseFriends(unittest.TestCase):
    """Matches that look like evidence and are about something else."""

    def test_an_import_line_is_never_evidence(self) -> None:
        files = [CodeFile("src/sim.py", "python",
                          ["import random", "x = 1", "y = 2"])]
        note = {"components": ["rtk_sim"]}
        self.assertEqual(evidence_for("Random Walk", files, note), [])

    def test_a_licence_header_is_never_evidence(self) -> None:
        files = [CodeFile("src/a.py", "python",
                          ["# distributed on an AS IS BASIS", "x = 1"])]
        self.assertEqual(
            evidence_for("Basis Vector", files, {"components": ["x"]}), [])

    def test_launch_wiring_is_not_searched_by_name(self) -> None:
        # joint_state_publisher is a robot joint, not a joint distribution
        files = [CodeFile("launch/display.launch.py", "python",
                          ["node = 'joint_state_publisher_gui'"])]
        self.assertEqual(
            evidence_for("Joint Density", files, {"components": ["x"]}), [])

    def test_a_node_nothing_reaches_gets_no_name_matches(self) -> None:
        files = [CodeFile("src/a.py", "python", ["basis = compute()"])]
        # no components reach it, so a bare word match is not evidence
        self.assertEqual(evidence_for("Basis Vector", files, None), [])
        self.assertTrue(evidence_for("Basis Vector", files,
                                     {"components": ["ekf_local"]}))

    def test_common_config_words_are_not_search_terms(self) -> None:
        files = [CodeFile("config/nav2.yaml", "yaml", ["change_penalty: 0.15"])]
        self.assertEqual(
            evidence_for("Change Of Variables", files,
                         {"components": ["controller_server"]}), [])
