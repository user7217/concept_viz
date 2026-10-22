"""Which components talk to which, and over what, stated rather than guessed.

The map's links between project components were vague because nothing said
what actually flows along them. Graphify offers data-flow edges, but on
agrios_ws all but one are INFERRED -- a model's reading of the README. The
links are knowable exactly: a ROS system declares every connection somewhere,
and this module reads each place it can be declared.

    code        create_publisher / create_subscription in the repo's own nodes
    parameters  a parameter whose value is a topic: odom0: /odom_cov
    bridge      a ros_gz_bridge argument: /odom@nav_msgs/msg/Odometry[gz.msgs...
    library     a library node's own default names (data/library_topics.json),
                which the repo never states because it never has to

Every endpoint carries the file and line or the parameter it came from, so a
link can be checked by opening one file. Nothing here asks a model anything.

Two limits, deliberately not papered over:

  - Namespaces are not modelled. A relative name resolves to the root
    namespace, which is right for agrios_ws and wrong for a namespaced robot.
  - TF is not a topic in this sense. The local EKF reaches Nav2 through the
    odom -> base_footprint transform, not through /odometry/filtered, so a
    topic graph shows that output as read by nothing. That is a fact about
    the channel, not a missing link.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .ingest import Component, Project, read_source

LIBRARY_TOPICS = Path(__file__).resolve().parent.parent / "data" / "library_topics.json"

PUBLISHES, SUBSCRIBES = "publishes", "subscribes"

SKIP_DIRS = {"build", "install", "log", ".git", "__pycache__", "graphify-out"}

# A topic is an absolute ROS name. Frames are written without the slash, so
# requiring it keeps `global_frame: odom` out.
_TOPIC = re.compile(r"^/[A-Za-z_][\w/]*$")

# The one convention parameters break: most name an input, a few an output.
_OUTPUT = re.compile(r"(publish|output|(^|_)out($|_))", re.I)

# ros_gz_bridge: TOPIC@ROS_TYPE then [ (Gazebo to ROS), ] (ROS to Gazebo) or
# @ (both), then the Gazebo type.
_BRIDGE = re.compile(r"^(?P<topic>/?[\w/]+)@(?P<ros>[\w/]+)(?P<dir>[\[\]@])(?P<gz>[\w.]+)$")


@dataclass(frozen=True)
class Endpoint:
    component: str
    topic: str       # absolute
    role: str        # PUBLISHES or SUBSCRIBES
    evidence: str    # where it is stated


@dataclass(frozen=True)
class TopicLink:
    publisher: str
    subscriber: str
    topic: str
    publisher_evidence: str
    subscriber_evidence: str


def resolve(name: str, component: Component) -> str:
    """A node's own name for a topic, after the launch file's remappings."""
    remaps = {}
    for inside, outside in component.remappings:
        remaps[inside] = outside
        remaps[inside.lstrip("/")] = outside
    name = remaps.get(name, remaps.get(name.lstrip("/"), name))
    return name if name.startswith("/") else "/" + name


# ---------- the four places a connection is stated ----------

def _modules(repo: Path, executable: str) -> list[Path]:
    return sorted(p for p in repo.rglob(f"{executable}.py")
                  if not SKIP_DIRS & set(p.relative_to(repo).parts))


def from_code(repo: Path, component: Component) -> list[Endpoint]:
    """create_publisher(Type, 'topic', ...) in the component's own module."""
    if not component.executable:
        return []
    found: list[Endpoint] = []
    for path in _modules(repo, component.executable):
        try:
            tree = ast.parse(read_source(path))
        except SyntaxError:
            continue
        rel = path.relative_to(repo)
        for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
            method = getattr(call.func, "attr", None)
            role = {"create_publisher": PUBLISHES,
                    "create_subscription": SUBSCRIBES}.get(method)
            if role is None:
                continue
            topic = None
            if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant):
                topic = call.args[1].value
            for kw in call.keywords:
                if kw.arg == "topic" and isinstance(kw.value, ast.Constant):
                    topic = kw.value.value
            if isinstance(topic, str):
                found.append(Endpoint(component.name, resolve(topic, component),
                                      role, f"{rel}:{call.lineno} {method}"))
    return found


def from_parameters(component: Component) -> list[Endpoint]:
    """A parameter whose value is a topic: odom0: /odom_cov is an input."""
    found = []
    for param in component.parameters:
        value = param.value.strip().strip("\"'")
        if not _TOPIC.match(value):
            continue
        role = PUBLISHES if _OUTPUT.search(param.name) else SUBSCRIBES
        found.append(Endpoint(component.name, resolve(value, component), role,
                              f"{param.source} {param.name}"))
    return found


def from_bridge(component: Component) -> list[Endpoint]:
    """Each bridge argument states a topic and which way it crosses."""
    if component.executable != "parameter_bridge":
        return []
    found = []
    for argument in component.arguments:
        match = _BRIDGE.match(argument)
        if not match:
            continue
        topic = resolve(match.group("topic"), component)
        evidence = f"bridge argument {argument}"
        direction = match.group("dir")
        if direction in "[@":
            found.append(Endpoint(component.name, topic, PUBLISHES, evidence))
        if direction in "]@":
            found.append(Endpoint(component.name, topic, SUBSCRIBES, evidence))
    return found


def load_library_topics(path: Path = LIBRARY_TOPICS) -> dict[str, dict]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())["nodes"]


def from_library(component: Component, table: dict[str, dict]) -> list[Endpoint]:
    """Default names a library node opens, after the launch file's remappings."""
    key = f"{component.library}/{component.executable}"
    entry = table.get(key)
    if not entry:
        return []
    found = []
    for role in (PUBLISHES, SUBSCRIBES):
        for name in entry.get(role, []):
            found.append(Endpoint(component.name, resolve(name, component), role,
                                  f"{key} default '{name}'"))
    return found


# ---------- assembling ----------

def endpoints(project: Project, repo: Path,
              table: dict[str, dict] | None = None) -> list[Endpoint]:
    table = load_library_topics() if table is None else table
    found: list[Endpoint] = []
    for component in project.components:
        found += from_code(repo, component)
        found += from_parameters(component)
        found += from_bridge(component)
        found += from_library(component, table)
    return found


def links(ends: list[Endpoint]) -> list[TopicLink]:
    """Every publisher joined to every subscriber of the same topic."""
    by_topic: dict[str, list[Endpoint]] = {}
    for end in ends:
        by_topic.setdefault(end.topic, []).append(end)
    found: list[TopicLink] = []
    seen: set[tuple[str, str, str]] = set()
    for topic in sorted(by_topic):
        pubs = [e for e in by_topic[topic] if e.role == PUBLISHES]
        subs = [e for e in by_topic[topic] if e.role == SUBSCRIBES]
        for pub in pubs:
            for sub in subs:
                key = (pub.component, sub.component, topic)
                if pub.component == sub.component or key in seen:
                    continue
                seen.add(key)
                found.append(TopicLink(pub.component, sub.component, topic,
                                       pub.evidence, sub.evidence))
    return found


def one_sided(ends: list[Endpoint]) -> tuple[list[Endpoint], list[Endpoint]]:
    """Topics with a publisher and no reader here, and the reverse.

    Neither is an error on its own. A topic read by nothing may be consumed
    through another channel, or by a tool outside the repo; a topic published
    by nothing may come from hardware. But a sensor input nobody publishes is
    exactly what a misspelled topic looks like, so both are reported.
    """
    published = {e.topic for e in ends if e.role == PUBLISHES}
    read = {e.topic for e in ends if e.role == SUBSCRIBES}
    unread = [e for e in ends if e.role == PUBLISHES and e.topic not in read]
    unfed = [e for e in ends if e.role == SUBSCRIBES and e.topic not in published]
    return unread, unfed


def report(ends: list[Endpoint]) -> str:
    found = links(ends)
    lines = [f"{len(found)} links over "
             f"{len({l.topic for l in found})} topics, from {len(ends)} stated endpoints", ""]
    for link in found:
        lines.append(f"  {link.publisher} -> {link.subscriber}   {link.topic}")
        lines.append(f"      published: {link.publisher_evidence}")
        lines.append(f"      read:      {link.subscriber_evidence}")
    unread, unfed = one_sided(ends)
    lines.append("")
    lines.append(f"published here, read by nothing in this repo ({len(unread)}):")
    lines += [f"  {e.topic:<22} {e.component}   ({e.evidence})" for e in unread]
    lines.append("")
    lines.append(f"read here, published by nothing in this repo ({len(unfed)}):")
    lines += [f"  {e.topic:<22} {e.component}   ({e.evidence})" for e in unfed]
    return "\n".join(lines)
