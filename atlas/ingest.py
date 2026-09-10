"""Stage 1: turn a project into an architecture layer.

Two input paths that must converge on one representation before stage 2:

  repo      -> parse manifests, params and launch files
  brief     -> a structured description, for a project with no code yet

The repo path is preferred because config files are a *leak detector*: a
parameter a project actually sets is proof that the abstraction failed to hide
the mathematics behind it. A parameter left at its default is not.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Parameter:
    """A knob this project actually sets."""

    name: str
    value: str
    source: str  # file it was read from

    @property
    def is_matrix(self) -> bool:
        """Matrix-valued parameters are the strongest leak signal.

        A 15x15 covariance written out by hand cannot be set correctly without
        understanding the object behind it.
        """
        return self.value.count(",") >= 8


@dataclass
class Component:
    """A running unit: a package, node, or module."""

    name: str
    library: str | None = None
    executable: str | None = None
    plugin: str | None = None   # pluginlib class, e.g. "ns::ClassName"
    version: str | None = None
    parameters: list[Parameter] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    @property
    def is_glue(self) -> bool:
        """A component that sets nothing is wiring, and needs little depth."""
        return not self.parameters


@dataclass
class Project:
    """The single representation both input paths produce."""

    name: str
    components: list[Component] = field(default_factory=list)
    dependencies: dict[str, str | None] = field(default_factory=dict)
    origin: str = "brief"  # "repo" or "brief"

    def component(self, name: str) -> Component:
        for existing in self.components:
            if existing.name == name:
                return existing
        created = Component(name=name)
        self.components.append(created)
        return created

    @property
    def parameters(self) -> list[Parameter]:
        return [p for c in self.components for p in c.parameters]


# ---------- repo path ----------

_SCALAR = re.compile(r"^\s*([A-Za-z_][\w/]*)\s*:\s*(.+?)\s*$")


def _unbalanced(value: str) -> bool:
    """True while a bracketed value is still open across lines."""
    return value.count("[") > value.count("]")


def parse_params(text: str, source: str) -> list[Parameter]:
    """Read a ROS2-style params file without a YAML dependency.

    Deliberately shallow: it recovers `name: value` pairs and folds multi-line
    list values onto one line. That is all the leak analysis needs -- which
    knobs are set, and whether a value is matrix-shaped.
    """
    parameters: list[Parameter] = []
    pending: Parameter | None = None

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        # A bracketed value continues until its bracket closes. Matrix-valued
        # parameters are written across many lines and must fold onto one.
        if pending is not None and _unbalanced(pending.value):
            pending.value += " " + line.strip()
            if not _unbalanced(pending.value):
                pending = None
            continue

        match = _SCALAR.match(line)
        if not match:
            pending = None
            continue

        name, value = match.group(1), match.group(2).strip()
        if not value or value in {"|", ">"}:
            # a bare `name:` opens a block; not a parameter itself
            pending = None
            continue

        parameter = Parameter(name=name, value=value, source=source)
        parameters.append(parameter)
        pending = parameter if _unbalanced(value) else None

    return parameters


def parse_package_xml(text: str) -> tuple[str | None, dict[str, str | None]]:
    """Package name and its declared dependencies."""
    root = ET.fromstring(text)
    name_el = root.find("name")
    name = name_el.text.strip() if name_el is not None and name_el.text else None

    deps: dict[str, str | None] = {}
    for tag in ("depend", "build_depend", "exec_depend", "run_depend"):
        for el in root.findall(tag):
            if el.text:
                deps[el.text.strip()] = el.get("version_eq")
    return name, deps


def parse_requirements(text: str) -> dict[str, str | None]:
    deps: dict[str, str | None] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = re.match(r"^([A-Za-z0-9_.\-\[\]]+)\s*(?:==|>=|~=)?\s*([\w.]+)?", line)
        if match:
            deps[match.group(1).lower()] = match.group(2)
    return deps


# ---------- launch files: runtime topology ----------

@dataclass
class LaunchNode:
    """A node a launch file actually starts."""

    package: str
    executable: str
    name: str | None = None
    param_files: list[str] = field(default_factory=list)
    remappings: list[tuple[str, str]] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    source: str = ""

    @property
    def topics(self) -> set[str]:
        """Every topic this node is wired to, in either direction."""
        return {t for pair in self.remappings for t in pair if t.startswith("/")}


def _const(node) -> str | None:
    import ast

    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def parse_launch_py(text: str, source: str = "") -> list[LaunchNode]:
    """Recover Node(...) declarations from a ROS2 Python launch file.

    Parsed as an AST rather than executed: a launch file is arbitrary code and
    running it would need the whole ROS environment present.

    Substitutions (LaunchConfiguration, PathJoinSubstitution) are not resolved
    -- only literal strings are recovered. That is a real limit, recorded in
    `unresolved` on the returned node rather than guessed at.
    """
    import ast

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    nodes: list[LaunchNode] = []
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        func = call.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name != "Node":
            continue

        kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg}
        package = _const(kwargs.get("package")) or "?"
        executable = _const(kwargs.get("executable")) or "?"

        param_files: list[str] = []
        unresolved: list[str] = []
        params = kwargs.get("parameters")
        if isinstance(params, ast.List):
            for element in ast.walk(params):
                literal = _const(element)
                if literal and literal.endswith((".yaml", ".yml")):
                    param_files.append(literal)
                elif isinstance(element, ast.Call):
                    fn = element.func
                    label = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                    if label in {"LaunchConfiguration", "PathJoinSubstitution",
                                 "FindPackageShare", "ParameterFile"}:
                        unresolved.append(label)

        remappings: list[tuple[str, str]] = []
        remaps = kwargs.get("remappings")
        if isinstance(remaps, ast.List):
            for element in remaps.elts:
                if isinstance(element, ast.Tuple) and len(element.elts) == 2:
                    src, dst = (_const(e) for e in element.elts)
                    if src and dst:
                        remappings.append((src, dst))

        nodes.append(
            LaunchNode(
                package=package,
                executable=executable,
                name=_const(kwargs.get("name")),
                param_files=param_files,
                remappings=remappings,
                unresolved=unresolved,
                source=source,
            )
        )
    return nodes


def parse_launch_xml(text: str, source: str = "") -> list[LaunchNode]:
    """Recover <node> declarations from an XML launch file."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []

    nodes: list[LaunchNode] = []
    for el in root.iter("node"):
        package = el.get("pkg") or el.get("package") or "?"
        executable = el.get("exec") or el.get("executable") or el.get("type") or "?"

        param_files = [
            p.get("from")
            for p in el.iter("param")
            if p.get("from", "").endswith((".yaml", ".yml"))
        ]
        remappings = [
            (r.get("from"), r.get("to"))
            for r in el.iter("remap")
            if r.get("from") and r.get("to")
        ]

        nodes.append(
            LaunchNode(
                package=package,
                executable=executable,
                name=el.get("name"),
                param_files=[p for p in param_files if p],
                remappings=remappings,
                source=source,
            )
        )
    return nodes


def dataflow(nodes: list[LaunchNode]) -> list[tuple[str, str, str]]:
    """Infer which components talk to each other, via shared topics.

    Two nodes remapped onto the same topic are wired together. This is the
    dataflow layer the design doc defers as a separate toggle -- recovered here
    because launch files are where it is stated, not inferred.
    """
    links: list[tuple[str, str, str]] = []
    for i, first in enumerate(nodes):
        for second in nodes[i + 1:]:
            for topic in sorted(first.topics & second.topics):
                links.append((first.name or first.executable,
                              second.name or second.executable, topic))
    return links


# ---------- repo path: assemble a Project ----------

PARAM_DIRS = ("config", "params", "param")


def ingest_repo(root: Path) -> Project:
    """Walk a repo and build the Project both input paths converge on."""
    root = Path(root)
    project = Project(name=root.name, origin="repo")

    for manifest in root.rglob("package.xml"):
        name, deps = parse_package_xml(manifest.read_text())
        project.dependencies.update(deps)
    for reqs in root.rglob("requirements.txt"):
        project.dependencies.update(parse_requirements(reqs.read_text()))

    launch_nodes: list[LaunchNode] = []
    for path in root.rglob("*.launch.py"):
        launch_nodes += parse_launch_py(path.read_text(), str(path.relative_to(root)))
    for pattern in ("*.launch.xml", "*.launch"):
        for path in root.rglob(pattern):
            launch_nodes += parse_launch_xml(path.read_text(), str(path.relative_to(root)))

    # Parameter directories are found anywhere in the tree, not just at the
    # root: a ROS2 workspace nests them under src/<package>/config/.
    params_by_file: dict[str, list[Parameter]] = {}
    for directory in root.rglob("*"):
        if not directory.is_dir() or directory.name not in PARAM_DIRS:
            continue
        for path in sorted(directory.rglob("*.y*ml")):
            rel = str(path.relative_to(root))
            params_by_file[Path(rel).name] = parse_params(path.read_text(), rel)

    for node in launch_nodes:
        component = project.component(node.name or node.executable)
        component.library = node.package
        component.executable = node.executable
        component.version = project.dependencies.get(node.package)
        component.sources.append(node.source)
        for param_file in node.param_files:
            component.parameters += params_by_file.get(Path(param_file).name, [])

    # A pluginlib class declaration names the algorithm that actually runs.
    # Attributing at the package layer instead is how nav2_controller -- a
    # lifecycle node that loads whatever you configure -- got read as PID.
    for name, params in params_by_file.items():
        for param in params:
            if param.name != "plugin":
                continue
            klass = param.value.strip().strip('"').strip("'")
            if "::" not in klass:
                continue
            namespace, _, short = klass.partition("::")
            component = project.component(short)
            component.plugin = klass
            component.library = namespace
            if param.source not in component.sources:
                component.sources.append(param.source)

    # Params in the repo that no launch file claims still count as knobs set.
    claimed = {p.source for c in project.components for p in c.parameters}
    for name, params in params_by_file.items():
        if params and params[0].source not in claimed:
            project.component(Path(name).stem).parameters += params

    return project
