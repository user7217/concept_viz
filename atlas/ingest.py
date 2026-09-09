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
