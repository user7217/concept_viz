"""Tests for project-specific notes on canon sheets."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.architecture import apply, propose
from atlas.ingest import Component, Parameter, Project
from atlas.instantiate import Instantiation, build_prompt, components_reaching, invented


def sample_project() -> Project:
    project = Project(name="rig", origin="repo")
    ekf = Component("ekf_local", library="robot_localization")
    ekf.executable = "ekf_node"
    ekf.parameters = [
        Parameter("frequency", "30.0", "config/ekf.yaml"),
        *[Parameter(f"filler_{i}", "0", "config/ekf.yaml") for i in range(20)],
        Parameter("process_noise_covariance", "[" + ", ".join(["0.05"] * 12) + "]",
                  "config/ekf.yaml"),
    ]
    project.components = [ekf]
    return project


def built():
    proposal = propose(sample_project())
    proposal.unclassified = []
    proposal.confirmed = True
    return apply(proposal)


SHEET = {"statement": "K = E[(X-EX)(X-EX)^T]",
         "symbols": [{"symbol": "K", "meaning": "covariance", "dimensions": "n x n"}]}


class TestComponentsReaching(unittest.TestCase):
    def test_component_whose_algorithm_requires_the_node(self) -> None:
        found = components_reaching(built(), "rig", "covariance_matrix")
        self.assertEqual(found, ["rig__ekf_local"])

    def test_unrelated_node_reaches_nothing(self) -> None:
        self.assertEqual(components_reaching(built(), "rig", "path_cost"), [])


class TestPrompt(unittest.TestCase):
    def test_matrix_parameters_survive_truncation(self) -> None:
        """The leak signal must not fall outside the shown window.

        Showing parameters in file order hid process_noise_covariance behind
        twenty fillers and produced a note claiming nothing tunes covariance.
        """
        prompt = build_prompt(sample_project(), "rig", "covariance_matrix",
                              "Covariance Matrix", SHEET, ["rig__ekf_local"])
        self.assertIn("process_noise_covariance", prompt)
        self.assertIn("[matrix-valued]", prompt)

    def test_sheet_symbols_reach_the_prompt(self) -> None:
        prompt = build_prompt(sample_project(), "rig", "covariance_matrix",
                              "Covariance Matrix", SHEET, ["rig__ekf_local"])
        self.assertIn("K = covariance", prompt)
        self.assertIn("n x n", prompt)


class TestInvented(unittest.TestCase):
    """The independent comparison: does every concrete claim point at something real."""

    def setUp(self) -> None:
        self.project = sample_project()

    def test_real_claims_pass(self) -> None:
        note = Instantiation("covariance_matrix", "rig",
                             concrete=[{"symbol": "K", "in_your_project": "15x15"}],
                             knobs=[{"parameter": "process_noise_covariance",
                                     "file": "config/ekf.yaml", "means": "..."}])
        self.assertEqual(invented(note, self.project, SHEET), {})

    def test_a_symbol_not_in_the_sheet_is_caught(self) -> None:
        note = Instantiation("covariance_matrix", "rig",
                             concrete=[{"symbol": "Q", "in_your_project": "..."}])
        self.assertIn("symbols not in the sheet", invented(note, self.project, SHEET))

    def test_a_parameter_this_project_does_not_set_is_caught(self) -> None:
        note = Instantiation("covariance_matrix", "rig",
                             knobs=[{"parameter": "alpha_filter",
                                     "file": "config/ekf.yaml", "means": "..."}])
        self.assertIn("parameters not set by this project",
                      invented(note, self.project, SHEET))

    def test_a_file_not_in_the_repo_is_caught(self) -> None:
        note = Instantiation("covariance_matrix", "rig",
                             knobs=[{"parameter": "process_noise_covariance",
                                     "file": "config/imagined.yaml", "means": "..."}])
        self.assertIn("files not in this repo", invented(note, self.project, SHEET))

    def test_a_symbol_from_a_combined_entry_is_not_invented(self) -> None:
        # Sheets group symbols: positive_semidefinite_matrix lists "x, v, z"
        # as one row. Matching the whole string only, the guard called a
        # correct reference to x an invention.
        sheet = {"symbols": [{"symbol": "M", "meaning": "the matrix"},
                             {"symbol": "x, v, z", "meaning": "test vectors"}]}
        note = Instantiation("positive_semidefinite_matrix", "rig",
                             concrete=[{"symbol": "x", "in_your_project": "..."},
                                       {"symbol": "z", "in_your_project": "..."}])
        self.assertEqual(invented(note, self.project, sheet), {})

    def test_a_combined_entry_still_catches_a_real_invention(self) -> None:
        sheet = {"symbols": [{"symbol": "x, v, z", "meaning": "test vectors"}]}
        note = Instantiation("positive_semidefinite_matrix", "rig",
                             concrete=[{"symbol": "Q", "in_your_project": "..."}])
        self.assertIn("symbols not in the sheet",
                      invented(note, self.project, sheet))

    def test_a_note_that_says_nothing_is_thin(self) -> None:
        from atlas.instantiate import thin
        # invented() passes this trivially: it makes no claims to be false.
        note = Instantiation("schur_complement", "rig", components=["ekf_local"],
                             why_here="The EKF requires it.")
        self.assertEqual(invented(note, self.project, SHEET), {})
        self.assertIn("maps none", thin(note, SHEET))

    def test_a_note_with_a_mapped_symbol_is_not_thin(self) -> None:
        from atlas.instantiate import thin
        note = Instantiation("covariance_matrix", "rig", components=["ekf_local"],
                             concrete=[{"symbol": "K_XX", "in_your_project": "15x15"}])
        self.assertEqual(thin(note, SHEET), "")

    def test_a_node_nothing_reaches_is_not_thin(self) -> None:
        from atlas.instantiate import thin
        # No component depends on it, so an empty note is the honest answer.
        self.assertEqual(thin(Instantiation("x", "rig"), SHEET), "")


class WhatTheNoteWriterIsShown(unittest.TestCase):
    """Each parameter was cut at 60 characters and no library source was
    shown, so a note could say "process_noise_covariance sets Q" and no more."""

    ENUM = {"path": "include/filter_common.hpp", "start": "41",
            "lines": ["enum StateMembers", "{", "  StateMemberX = 0,",
                      "  StateMemberY,", "  StateMemberYaw"]}
    PREDICT = {"path": "src/ekf.cpp", "start": "430",
               "lines": ["  // (3) Project the error forward", "  P = J * P * J';",
                         "  estimate_error_covariance_.noalias() +=",
                         "    delta_sec * (*process_noise_covariance);"]}

    def test_the_state_order_comes_from_the_librarys_enum(self) -> None:
        from atlas.instantiate import state_order
        self.assertEqual(state_order([self.ENUM]), ["X", "Y", "Yaw"])

    def test_a_matrix_is_shown_as_its_labelled_diagonal(self) -> None:
        from atlas.instantiate import matrix_view
        view = matrix_view("[0.05, 0, 0,  0, 0.05, 0,  0, 0, 0.02]", ["X", "Y", "Yaw"])
        self.assertEqual(view, "3x3, diagonal X=0.05, Y=0.05, Yaw=0.02; off-diagonal all zero")

    def test_a_mask_is_shown_as_what_it_fuses(self) -> None:
        from atlas.instantiate import flags_view
        self.assertEqual(flags_view("[true, false, true]", ["X", "Y", "Yaw"]),
                         "fuses X, Yaw (of 3 state entries)")
        self.assertIsNone(flags_view("[true, false]", ["X", "Y", "Yaw"]))

    def test_the_library_line_that_applies_a_parameter_is_shown(self) -> None:
        from atlas.instantiate import library_excerpts
        text = library_excerpts([self.ENUM, self.PREDICT], ["process_noise_covariance"])
        self.assertIn("delta_sec * (*process_noise_covariance)", text)
        self.assertIn("src/ekf.cpp:433", text)
        self.assertNotIn("StateMember", text)

    def test_a_senders_covariance_is_shown_with_what_fills_it(self) -> None:
        import tempfile
        from pathlib import Path
        from atlas.instantiate import upstream
        from atlas.topics import TopicLink
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "relay.py").write_text(
                "def cb(self, msg):\n    c = [0.0] * 36\n    c[0] = 0.05  # vx\n"
                "    msg.twist.covariance = c\n    self.pub.publish(msg)\n")
            link = TopicLink("relay", "ekf_local", "/odom_cov",
                             "relay.py:1 create_publisher", "ekf.yaml odom0")
            text = upstream([link], repo, {"ekf_local"})
        self.assertIn("relay publishes /odom_cov; ekf_local reads it", text)
        self.assertIn("c[0] = 0.05  # vx", text)
        self.assertNotIn("publish(msg)", text)

    def test_a_link_into_another_component_is_left_out(self) -> None:
        from atlas.instantiate import upstream
        from atlas.topics import TopicLink
        link = TopicLink("a", "someone_else", "/t", "x", "y")
        self.assertEqual(upstream([link], None, {"ekf_local"}), "(nothing upstream is stated)")
