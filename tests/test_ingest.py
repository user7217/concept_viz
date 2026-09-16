"""Tests for project ingest (design doc stage 1)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.ingest import Component, parse_package_xml, parse_params, parse_requirements

EKF_YAML = """
ekf_filter_node:
  ros__parameters:
    frequency: 30.0
    two_d_mode: true
    # a commented-out knob is not a knob
    # publish_tf: false
    process_noise_covariance: [0.05, 0.0,  0.0, 0.0,
                               0.0,  0.05, 0.0, 0.0,
                               0.0,  0.0,  0.06, 0.0]
    odom0: /odometry/wheel
"""


class TestParams(unittest.TestCase):
    def setUp(self) -> None:
        self.params = {p.name: p for p in parse_params(EKF_YAML, "config/ekf.yaml")}

    def test_scalar_parameters_are_read(self) -> None:
        self.assertEqual(self.params["frequency"].value, "30.0")
        self.assertEqual(self.params["two_d_mode"].value, "true")

    def test_multiline_matrix_folds_onto_one_value(self) -> None:
        covariance = self.params["process_noise_covariance"]
        self.assertTrue(covariance.value.endswith("]"), covariance.value)
        self.assertTrue(covariance.is_matrix)

    def test_scalars_are_not_matrices(self) -> None:
        self.assertFalse(self.params["frequency"].is_matrix)
        self.assertFalse(self.params["odom0"].is_matrix)

    def test_comments_are_not_parameters(self) -> None:
        self.assertNotIn("publish_tf", self.params)

    def test_block_openers_are_not_parameters(self) -> None:
        """`ros__parameters:` opens a block; it is not a knob."""
        self.assertNotIn("ros__parameters", self.params)
        self.assertNotIn("ekf_filter_node", self.params)

    def test_source_is_recorded(self) -> None:
        self.assertEqual(self.params["frequency"].source, "config/ekf.yaml")


class TestManifests(unittest.TestCase):
    def test_package_xml(self) -> None:
        name, deps = parse_package_xml(
            "<package><name>agri_bringup</name>"
            "<depend>robot_localization</depend>"
            "<exec_depend>nav2_bringup</exec_depend></package>"
        )
        self.assertEqual(name, "agri_bringup")
        self.assertIn("robot_localization", deps)
        self.assertIn("nav2_bringup", deps)

    def test_requirements_pins_versions(self) -> None:
        deps = parse_requirements("ultralytics==8.0.1\nnumpy>=1.24\n# comment\n-r other.txt")
        self.assertEqual(deps["ultralytics"], "8.0.1")
        self.assertIn("numpy", deps)
        self.assertNotIn("-r", deps)


class TestComponent(unittest.TestCase):
    def test_component_with_no_parameters_is_glue(self) -> None:
        self.assertTrue(Component("topic_relay").is_glue)

    def test_component_that_sets_a_knob_is_not_glue(self) -> None:
        component = Component("ekf")
        component.parameters = parse_params(EKF_YAML, "config/ekf.yaml")
        self.assertFalse(component.is_glue)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestEncoding(unittest.TestCase):
    def test_a_utf16_manifest_does_not_kill_stage_one(self) -> None:
        import tempfile
        from atlas.ingest import ingest_repo

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # A real requirements.txt turned up UTF-16LE with a BOM; read_text() dies on
            # it and stage 1 never returned, so nothing downstream could run.
            (root / "requirements.txt").write_bytes(
                "numpy==2.3.5\nscipy==1.14.0\n".encode("utf-16"))
            project = ingest_repo(root)
        self.assertIn("numpy", project.dependencies)
        self.assertIn("scipy", project.dependencies)

    def test_a_latin1_file_is_read_not_crashed_on(self) -> None:
        import tempfile
        from atlas.ingest import read_source

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notes.txt"
            path.write_bytes("caf\xe9 au lait".encode("latin-1"))
            self.assertIn("caf", read_source(path))
