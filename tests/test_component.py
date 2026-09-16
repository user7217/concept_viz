import unittest

from atlas.component import ComponentNote, build_prompt, invented
from atlas.ingest import Component, Parameter, Project


def _project() -> tuple[Project, Component]:
    project = Project(name="rig", origin="repo")
    local = project.component("ekf_local")
    local.library, local.executable = "robot_localization", "ekf_node"
    local.sources = ["launch/localization.launch.py"]
    local.parameters = [
        Parameter("world_frame", "odom", "config/ekf_local.yaml"),
        Parameter("frequency", "50.0", "config/ekf_local.yaml"),
        Parameter("process_noise_covariance", "0.05, " * 40,
                  "config/ekf_local.yaml"),
    ]
    other = project.component("ekf_global")
    other.library, other.executable = "robot_localization", "ekf_node"
    other.parameters = [Parameter("world_frame", "map", "config/ekf_global.yaml")]
    return project, local


class TestPrompt(unittest.TestCase):
    def test_telling_parameters_come_before_tuning_ones(self) -> None:
        project, local = _project()
        prompt = build_prompt(project, local, ["extended_kalman_filter"])
        # world_frame is what separates ekf_local from ekf_global; a covariance
        # is how it is tuned and says nothing about what it is for
        self.assertLess(prompt.index("world_frame"),
                        prompt.index("process_noise_covariance"))

    def test_the_facts_reach_the_prompt(self) -> None:
        project, local = _project()
        prompt = build_prompt(project, local, ["extended_kalman_filter"])
        for fact in ("robot_localization", "ekf_node",
                     "launch/localization.launch.py", "extended_kalman_filter"):
            self.assertIn(fact, prompt)


class TestGuard(unittest.TestCase):
    def setUp(self) -> None:
        self.project, self.local = _project()
        self.algorithms = ["extended_kalman_filter"]

    def _check(self, note):
        return invented(note, self.project, self.local, self.algorithms)

    def test_a_true_note_passes(self) -> None:
        note = ComponentNote("ekf_local", "rig",
                             purpose="Fuses wheel odometry and IMU into a "
                                     "continuous estimate in the odom frame.",
                             distinguishes="Its world_frame is odom where "
                                           "ekf_global uses map.",
                             evidence=["world_frame", "frequency"])
        self.assertEqual(self._check(note), {})

    def test_a_parameter_the_project_never_sets_is_caught(self) -> None:
        note = ComponentNote("ekf_local", "rig", evidence=["alpha_filter"])
        self.assertIn("parameters this project does not set", self._check(note))

    def test_an_invented_identifier_in_the_prose_is_caught(self) -> None:
        # fluent, plausible, and about a node that does not exist
        note = ComponentNote("ekf_local", "rig",
                             purpose="It subscribes to /wheel_encoder_raw and "
                                     "feeds imu_preintegration_node.")
        problems = self._check(note)
        self.assertIn("identifiers not found in the project", problems)
        self.assertIn("imu_preintegration_node",
                      problems["identifiers not found in the project"])

    def test_ordinary_prose_is_not_flagged(self) -> None:
        note = ComponentNote("ekf_local", "rig",
                             purpose="It runs a filter at a fixed rate and "
                                     "publishes a smooth, continuous estimate.")
        self.assertEqual(self._check(note), {})

    def test_a_real_sibling_and_file_are_allowed(self) -> None:
        note = ComponentNote("ekf_local", "rig",
                             purpose="Unlike ekf_global it stays in odom; see "
                                     "config/ekf_local.yaml.",
                             evidence=["world_frame"])
        self.assertEqual(self._check(note), {})


if __name__ == "__main__":
    unittest.main()

    def test_a_parameter_value_is_not_an_invention(self) -> None:
        # base_link_frame: base_footprint -- the value is in the repo too
        self.local.parameters.append(
            Parameter("base_link_frame", "base_footprint", "config/ekf_local.yaml"))
        note = ComponentNote("ekf_local", "rig",
                             purpose="It publishes the transform for "
                                     "base_footprint.")
        self.assertEqual(self._check(note), {})

    def test_a_topic_set_as_a_value_is_not_an_invention(self) -> None:
        self.local.parameters.append(
            Parameter("odom0", "/odom_cov", "config/ekf_local.yaml"))
        note = ComponentNote("ekf_local", "rig",
                             purpose="It fuses odom_cov as its first input.")
        self.assertEqual(self._check(note), {})

    def test_a_multi_part_extension_survives(self) -> None:
        note = ComponentNote("ekf_local", "rig",
                             purpose="Started by launch/localization.launch.py.")
        self.assertEqual(self._check(note), {})
