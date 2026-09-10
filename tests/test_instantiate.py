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
