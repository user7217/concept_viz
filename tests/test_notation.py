"""One notation per family of sheets."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.bootstrap import build_graph, load
from atlas.notation import Convention, conflicts, convention_for, prompt_block

ESTIMATION = Convention("state estimation", ("linear_kalman_filter",), (
    {"symbol": "Q", "means": "process noise covariance",
     "recognise": r"(?=.*covariance)(?=.*noise)(?=.*\b(process|motion)\b)"},
    {"symbol": "R", "means": "measurement noise covariance",
     "recognise": r"(?=.*covariance)(?=.*noise)(?=.*\b(measurement|observation)\b)"},
    {"symbol": "F", "means": "state transition matrix", "recognise": ""},
))


class Conflicts(unittest.TestCase):
    def test_the_probabilistic_robotics_swap_is_caught(self):
        """The linear KF sheet wrote R for process noise, Q for measurement."""
        found = conflicts([{"symbol": "\\mathbf{R}_t", "meaning": "process noise covariance matrix"},
                           {"symbol": "\\mathbf{Q}_t", "meaning": "measurement noise covariance"}],
                          ESTIMATION)
        self.assertEqual(len(found), 2)
        self.assertIn("this family writes that Q", found[0])

    def test_the_convention_itself_passes(self):
        self.assertEqual(conflicts([{"symbol": "Q_k", "meaning": "covariance of the motion noise"},
                                    {"symbol": "R", "meaning": "observation noise covariance"}],
                                   ESTIMATION), [])

    def test_a_jacobian_described_by_the_motion_model_is_not_noise(self):
        """F's meaning says 'motion model'; that is not process noise."""
        self.assertEqual(conflicts([{"symbol": "\\mathbf{F}_t",
                                     "meaning": "Jacobian of the motion model w.r.t. the state"},
                                    {"symbol": "F_w", "meaning": "Jacobian of the motion model w.r.t. the noise"}],
                                   ESTIMATION), [])

    def test_the_same_letter_for_another_object_is_not_a_clash(self):
        """Taylor's R is a remainder, not a swapped covariance."""
        self.assertEqual(conflicts([{"symbol": "R_k(x)", "meaning": "remainder term of order k"}],
                                   ESTIMATION), [])


class Scope(unittest.TestCase):
    graph = build_graph(load())

    def test_it_applies_to_the_filters_and_what_they_require(self):
        convs = [ESTIMATION]
        self.assertIsNotNone(convention_for(self.graph, "linear_kalman_filter", convs))
        self.assertIsNotNone(convention_for(self.graph, "gaussian_conditioning", convs))

    def test_it_does_not_reach_unrelated_mathematics(self):
        self.assertIsNone(convention_for(self.graph, "a_star", [ESTIMATION]))

    def test_the_prompt_names_every_symbol(self):
        block = prompt_block(ESTIMATION)
        self.assertIn("Q = process noise covariance", block)
        self.assertIn("say so once in the statement", block)
        self.assertEqual(prompt_block(None), "")


if __name__ == "__main__":
    unittest.main()
