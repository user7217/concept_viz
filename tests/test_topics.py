"""Topic links between components, from where ROS states them."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.codegraph import CodeGraph, GraphEdge, GraphNode, Join, Match, check_flows
from atlas.ingest import Component, Parameter, Project, parse_launch_py
from atlas.topics import (PUBLISHES, SUBSCRIBES, Endpoint, from_bridge, from_code,
                          from_library, from_parameters, links, one_sided, resolve)

RELAY = (
    "class OdomCovRelay(Node):\n"
    "    def __init__(self):\n"
    "        self.pub = self.create_publisher(Odometry, '/odom_cov', 10)\n"
    "        self.create_subscription(Odometry, 'odom', self.cb, 10)\n"
    "        self.create_subscription(Imu, topic='/imu', callback=self.cb,\n"
    "                                 qos_profile=10)\n"
)

TABLE = {
    "robot_localization/ekf_node": {"publishes": ["odometry/filtered"],
                                    "subscribes": []},
    "robot_localization/navsat_transform_node": {
        "publishes": ["odometry/gps"],
        "subscribes": ["imu", "gps/fix", "odometry/filtered"]},
}


def param(name, value, source="config/ekf.yaml"):
    return Parameter(name=name, value=value, source=source)


class TestStatedEndpoints(unittest.TestCase):
    def test_code_states_direction_and_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "pkg").mkdir()
            (repo / "pkg" / "odom_cov_relay.py").write_text(RELAY)
            relay = Component(name="odom_cov_relay", executable="odom_cov_relay")
            ends = {(e.topic, e.role, e.evidence) for e in from_code(repo, relay)}
        self.assertIn(("/odom_cov", PUBLISHES,
                       "pkg/odom_cov_relay.py:3 create_publisher"), ends)

    def test_a_relative_name_resolves_to_the_root_namespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "odom_cov_relay.py").write_text(RELAY)
            relay = Component(name="odom_cov_relay", executable="odom_cov_relay")
            topics = {e.topic for e in from_code(repo, relay) if e.role == SUBSCRIBES}
        self.assertEqual(topics, {"/odom", "/imu"})

    def test_a_built_copy_is_not_read_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "src").mkdir()
            (repo / "install").mkdir()
            (repo / "src" / "odom_cov_relay.py").write_text(RELAY)
            (repo / "install" / "odom_cov_relay.py").write_text(RELAY)
            relay = Component(name="odom_cov_relay", executable="odom_cov_relay")
            self.assertEqual(len([e for e in from_code(repo, relay)
                                  if e.role == PUBLISHES]), 1)

    def test_a_topic_valued_parameter_is_an_input(self):
        ekf = Component(name="ekf_local", parameters=[
            param("odom0", "/odom_cov"), param("imu0", '"/imu"'),
            param("world_frame", "odom"), param("frequency", "50.0")])
        ends = {(e.topic, e.role) for e in from_parameters(ekf)}
        self.assertEqual(ends, {("/odom_cov", SUBSCRIBES), ("/imu", SUBSCRIBES)})

    def test_an_output_named_parameter_publishes(self):
        node = Component(name="n", parameters=[param("output_topic", "/fused")])
        self.assertEqual(from_parameters(node)[0].role, PUBLISHES)

    def test_the_bridge_states_which_way_each_topic_crosses(self):
        bridge = Component(name="parameter_bridge", executable="parameter_bridge",
                           arguments=[
                               "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
                               "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
                               "/chatter@std_msgs/msg/String@gz.msgs.StringMsg",
                           ])
        ends = {(e.topic, e.role) for e in from_bridge(bridge)}
        self.assertEqual(ends, {("/odom", PUBLISHES), ("/cmd_vel", SUBSCRIBES),
                                ("/chatter", PUBLISHES), ("/chatter", SUBSCRIBES)})

    def test_a_library_default_follows_the_launch_remapping(self):
        ekf_global = Component(name="ekf_global", library="robot_localization",
                               executable="ekf_node",
                               remappings=[("odometry/filtered", "/odometry/global")])
        self.assertEqual([e.topic for e in from_library(ekf_global, TABLE)],
                         ["/odometry/global"])

    def test_an_unlisted_library_contributes_nothing(self):
        node = Component(name="x", library="nav2_planner", executable="planner_server")
        self.assertEqual(from_library(node, TABLE), [])

    def test_remapping_matches_with_or_without_the_slash(self):
        node = Component(name="n", remappings=[("/imu", "/rtk/heading")])
        self.assertEqual(resolve("imu", node), "/rtk/heading")


class TestLinks(unittest.TestCase):
    ENDS = [
        Endpoint("relay", "/odom_cov", PUBLISHES, "relay.py:8"),
        Endpoint("ekf_local", "/odom_cov", SUBSCRIBES, "ekf_local.yaml odom0"),
        Endpoint("ekf_global", "/odom_cov", SUBSCRIBES, "ekf_global.yaml odom0"),
        Endpoint("ekf_global", "/odom_cov", SUBSCRIBES, "ekf_global.yaml odom0"),
        Endpoint("relay", "/odom_cov", SUBSCRIBES, "relay.py:9"),
        Endpoint("ekf_local", "/odometry/filtered", PUBLISHES, "default"),
        Endpoint("bridge", "/cmd_vel", SUBSCRIBES, "bridge argument"),
    ]

    def test_every_publisher_meets_every_reader(self):
        pairs = {(l.publisher, l.subscriber) for l in links(self.ENDS)}
        self.assertEqual(pairs, {("relay", "ekf_local"), ("relay", "ekf_global")})

    def test_a_node_reading_its_own_topic_is_not_a_link(self):
        self.assertFalse(any(l.publisher == l.subscriber for l in links(self.ENDS)))

    def test_each_link_carries_both_ends_evidence(self):
        link = next(l for l in links(self.ENDS) if l.subscriber == "ekf_local")
        self.assertEqual((link.publisher_evidence, link.subscriber_evidence),
                         ("relay.py:8", "ekf_local.yaml odom0"))

    def test_one_sided_topics_are_reported_both_ways(self):
        unread, unfed = one_sided(self.ENDS)
        self.assertEqual([e.topic for e in unread], ["/odometry/filtered"])
        self.assertEqual([e.topic for e in unfed], ["/cmd_vel"])


class TestLaunchArguments(unittest.TestCase):
    def test_literal_arguments_are_kept(self):
        launch = ("Node(package='ros_gz_bridge', executable='parameter_bridge',\n"
                  "     arguments=['/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry'])\n")
        self.assertEqual(parse_launch_py(launch)[0].arguments,
                         ["/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry"])


class TestCheckingGraphifyFlows(unittest.TestCase):
    """A model-inferred flow is confirmed only by a topic actually stated."""

    def node(self, id, label):
        return GraphNode(id=id, label=label, origin="semantic", file_type="concept",
                         source_file="README.md", location=None, community=None,
                         rationale="", external=False)

    def test_a_flow_with_a_topic_is_confirmed_and_one_without_is_not(self):
        relay = self.node("relay", "odom_cov_relay.py")
        ekf = self.node("ekf", "ekf_local node config")
        rpp = self.node("rpp", "Regulated Pure Pursuit controller")
        costmap = self.node("costmap", "Nav2 local costmap")
        guessed = GraphEdge("relay", "ekf", "shares_data_with", "INFERRED")
        memory = GraphEdge("rpp", "costmap", "shares_data_with", "INFERRED")
        graph = CodeGraph(nodes={n.id: n for n in (relay, ekf, rpp, costmap)},
                          edges=[guessed, memory])
        result = Join(matches=[
            Match("odom_cov_relay", described_by=[relay], flows=[guessed]),
            Match("ekf_local", described_by=[ekf], flows=[guessed]),
            Match("rpp", described_by=[rpp], flows=[memory]),
            Match("local_costmap", described_by=[costmap], flows=[memory]),
        ], unclaimed=[])
        topic = links([Endpoint("odom_cov_relay", "/odom_cov", PUBLISHES, "a"),
                       Endpoint("ekf_local", "/odom_cov", SUBSCRIBES, "b")])
        confirmed, unconfirmed = check_flows(result, graph, topic)
        self.assertEqual([(r[0], r[1], r[4]) for r in confirmed],
                         [("odom_cov_relay", "ekf_local", ["/odom_cov"])])
        self.assertEqual([(r[0], r[1]) for r in unconfirmed],
                         [("rpp", "local_costmap")])


if __name__ == "__main__":
    unittest.main()
