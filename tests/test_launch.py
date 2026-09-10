import sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, "/Users/harsha1/Documents/concept_viz")
from atlas.ingest import dataflow, ingest_repo, parse_launch_py, parse_launch_xml

LAUNCH_PY = '''
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
def generate_launch_description():
    return [
        Node(package="robot_localization", executable="ekf_node", name="ekf_filter_node",
             parameters=["config/ekf.yaml"],
             remappings=[("odometry/filtered", "/odom")]),
        Node(package="nav2_bringup", executable="nav2", name="nav2",
             parameters=[PathJoinSubstitution(["x"]), LaunchConfiguration("params")],
             remappings=[("odom", "/odom")]),
        Node(package="relay", executable="relay"),
    ]
'''

class TestLaunchPy(unittest.TestCase):
    def setUp(self): self.nodes = parse_launch_py(LAUNCH_PY, "b.launch.py")
    def test_all_nodes_found(self): self.assertEqual(len(self.nodes), 3)
    def test_literal_param_file_recovered(self):
        self.assertEqual(self.nodes[0].param_files, ["config/ekf.yaml"])
    def test_unresolved_substitutions_are_recorded_not_guessed(self):
        self.assertEqual(self.nodes[1].param_files, [])
        self.assertIn("LaunchConfiguration", self.nodes[1].unresolved)
        self.assertIn("PathJoinSubstitution", self.nodes[1].unresolved)
    def test_missing_kwargs_do_not_crash(self):
        self.assertIsNone(self.nodes[2].name)
        self.assertEqual(self.nodes[2].remappings, [])
    def test_syntax_error_returns_empty_not_raises(self):
        self.assertEqual(parse_launch_py("def ("), [])
    def test_launch_file_is_never_executed(self):
        """Parsing must not run the file: it would need the ROS environment."""
        self.assertEqual(parse_launch_py("import does_not_exist\nNode(package='p')"), 
                         parse_launch_py("import does_not_exist\nNode(package='p')"))
        self.assertEqual(parse_launch_py("import does_not_exist\nNode(package='p')")[0].package, "p")

class TestLaunchXml(unittest.TestCase):
    def test_node_and_remap(self):
        nodes = parse_launch_xml(
            '<launch><node pkg="robot_localization" exec="ekf_node" name="ekf">'
            '<param from="config/ekf.yaml"/><remap from="a" to="/odom"/></node></launch>')
        self.assertEqual(nodes[0].package, "robot_localization")
        self.assertEqual(nodes[0].param_files, ["config/ekf.yaml"])
        self.assertEqual(nodes[0].topics, {"/odom"})
    def test_malformed_xml_returns_empty(self):
        self.assertEqual(parse_launch_xml("<launch>"), [])

class TestDataflow(unittest.TestCase):
    def test_shared_topic_links_two_nodes(self):
        links = dataflow(parse_launch_py(LAUNCH_PY))
        self.assertEqual(links, [("ekf_filter_node", "nav2", "/odom")])
    def test_node_with_no_topics_is_unlinked(self):
        names = {n for link in dataflow(parse_launch_py(LAUNCH_PY)) for n in link[:2]}
        self.assertNotIn("relay", names)

class TestIngestRepo(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "config").mkdir()
        (root / "launch").mkdir()
        (root / "package.xml").write_text(
            "<package><name>agri</name><depend>robot_localization</depend></package>")
        (root / "config" / "ekf.yaml").write_text(
            "ekf_filter_node:\n  ros__parameters:\n    frequency: 30.0\n"
            "    process_noise_covariance: [1,2,3,4,\n5,6,7,8,9]\n")
        (root / "config" / "orphan.yaml").write_text("standalone:\n  gain: 2.0\n")
        (root / "launch" / "b.launch.py").write_text(LAUNCH_PY)
        self.project = ingest_repo(root)
    def tearDown(self): self.tmp.cleanup()
    def test_dependencies_collected(self):
        self.assertIn("robot_localization", self.project.dependencies)
    def test_component_gets_its_params_and_library(self):
        ekf = self.project.component("ekf_filter_node")
        self.assertEqual(ekf.library, "robot_localization")
        self.assertIn("process_noise_covariance", {p.name for p in ekf.parameters})
    def test_matrix_param_is_the_leak_signal(self):
        matrices = [p.name for p in self.project.parameters if p.is_matrix]
        self.assertIn("process_noise_covariance", matrices)
    def test_component_setting_nothing_is_glue(self):
        self.assertTrue(self.project.component("relay").is_glue)
    def test_unclaimed_param_file_still_counts(self):
        """A knob set in the repo counts even if no launch file references it."""
        self.assertIn("orphan", {c.name for c in self.project.components})


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestNestedConfig(unittest.TestCase):
    """A ROS2 workspace nests config under src/<package>/config/."""

    def test_config_below_the_root_is_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            deep = root / "src" / "pkg" / "config"
            deep.mkdir(parents=True)
            (deep / "ekf.yaml").write_text(
                "ekf_node:\n  ros__parameters:\n    frequency: 30.0\n")
            (root / "src" / "pkg" / "launch").mkdir()
            (root / "src" / "pkg" / "launch" / "b.launch.py").write_text(
                'from launch_ros.actions import Node\n'
                'Node(package="robot_localization", executable="ekf_node",\n'
                '     name="ekf", parameters=["ekf.yaml"])\n')
            project = ingest_repo(root)
        self.assertIn("frequency", {p.name for p in project.parameters})
