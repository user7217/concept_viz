"""What one running component is for, in its own system.

The map descends program -> subsystem -> component -> algorithm, and the
component was the only level with nothing to read. It is also the level a
reader most wants explained: `ekf_local` and `ekf_global` run the same
executable from the same package and do different jobs, and nothing in the
ingest says which is which.

Everything here is grounded in what stage 1 already extracted -- package,
executable, the parameters actually set and the files they came from, the
launch file that starts it, and the algorithm it runs. The model gets those
facts and nothing else, and `invented()` checks every identifier it names back
against them. That check matters more here than anywhere else in the pipeline:
a derivation is restated from a source that can be diffed against it, while
"what is this component for" has no source at all, and a fluent wrong answer
would be indistinguishable from a right one.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from .ingest import Component, Project
from .llm import Provider, complete_json

SYSTEM = """You explain what one running component of a robot system is for.

You are given only facts extracted from the project's own files. Use nothing
else. Never name a parameter, file, package or topic that is not in the facts
below, and never guess at behaviour the facts do not show.

Where two components look alike, say what distinguishes them using their
parameters -- that difference is usually the whole answer.

Reply with JSON only."""

TEMPLATE = """Project: {project}
Component: {name}

Facts extracted from the repository:
  package:    {library}
  executable: {executable}
  plugin:     {plugin}
  started by: {sources}
  runs the algorithm: {algorithms}

Parameters it sets ({count} total, the distinguishing ones first):
{parameters}

Return JSON:
{{
  "purpose": "2-3 sentences: what this component does in this system",
  "distinguishes": "one sentence: what makes it different from its siblings, \
citing a parameter that differs. Empty string if it has no siblings.",
  "evidence": ["the parameter names you relied on"]
}}"""

# Parameters that say what a node is FOR rather than how it is tuned. A frame
# or a topic distinguishes two copies of the same executable; a covariance
# does not.
TELLING = re.compile(
    r"(?i)(frame|topic|_config$|mode|enable|publish|broadcast|source|child|"
    r"world|map|odom|base_link|datum|magnetic|yaw_offset|two_d|frequency)")


@dataclass
class ComponentNote:
    component: str
    project: str
    purpose: str = ""
    distinguishes: str = ""
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _render_parameters(component: Component, limit: int = 22) -> str:
    """Telling parameters first: they are what separates two copies of a node."""
    ordered = sorted(
        component.parameters,
        key=lambda p: (not TELLING.search(p.name), p.is_matrix, p.name),
    )
    rows = [f"    {p.name} = {p.value[:70]}  [{p.source}]" for p in ordered[:limit]]
    return "\n".join(rows) or "    (none)"


def build_prompt(project: Project, component: Component,
                 algorithms: list[str]) -> str:
    return TEMPLATE.format(
        project=project.name, name=component.name,
        library=component.library or "unknown",
        executable=component.executable or "(a plugin, not a node)",
        plugin=component.plugin or "none",
        sources=", ".join(component.sources) or "not started by any launch file",
        algorithms=", ".join(algorithms) or "none recorded",
        count=len(component.parameters),
        parameters=_render_parameters(component),
    )


def describe(provider: Provider, project: Project, component: Component,
             algorithms: list[str]) -> ComponentNote:
    payload = complete_json(
        provider, build_prompt(project, component, algorithms), system=SYSTEM)
    return ComponentNote(
        component=component.name, project=project.name,
        purpose=str(payload.get("purpose", "")),
        distinguishes=str(payload.get("distinguishes", "")),
        evidence=[str(e) for e in payload.get("evidence", [])],
    )


def invented(note: ComponentNote, project: Project, component: Component,
             algorithms: list[str]) -> dict[str, list[str]]:
    """Identifiers the note names that the project does not contain.

    The independent comparison. Unlike a derivation there is no source to diff
    the prose against, so the check is on every concrete thing it names: a
    parameter, a file, a package or an algorithm either appears in the ingest
    or it was invented.
    """
    known_params = {p.name for p in project.parameters}
    known_files = {p.source for p in project.parameters} | set(component.sources)
    # Parameter VALUES count as known, not just names. The first run flagged
    # base_footprint, /odom_cov and /odometry/gps as inventions when every one
    # is sitting in the YAML as a value -- base_link_frame: base_footprint.
    # Whitelisting names only made the guard reject the model for reading the
    # configuration it was given.
    known_values: set[str] = set()
    for parameter in project.parameters:
        known_values.update(re.findall(r"[A-Za-z][\w./]*",
                                       parameter.value.lstrip("/")))
    known_words = (
        known_params | known_files | known_values | set(algorithms)
        | {c.name for c in project.components}
        | {c.library for c in project.components if c.library}
        | {c.executable for c in project.components if c.executable}
        | {project.name}
    )

    problems: dict[str, list[str]] = {}
    bad_evidence = [e for e in note.evidence if e not in known_params]
    if bad_evidence:
        problems["parameters this project does not set"] = sorted(set(bad_evidence))

    # Identifier-shaped tokens in the prose: snake_case names, paths, anything
    # that reads like it was copied from the repo rather than written about it.
    prose = f"{note.purpose} {note.distinguishes}"
    # `(?:\.[a-z]+)?` stopped at the first dot, so localization.launch.py came
    # back as localization.launch and was reported missing from its own repo.
    claimed = set(re.findall(
        r"\b[a-z][a-z0-9]*(?:[_/][a-z0-9]+)+(?:\.[a-z0-9]+)*\b", prose))
    unknown = sorted(t for t in claimed
                     if t not in known_words and t.lstrip("/") not in known_words)
    if unknown:
        problems["identifiers not found in the project"] = unknown
    return problems
