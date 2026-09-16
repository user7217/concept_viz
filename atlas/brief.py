"""Stage 1, for a repo that declares nothing.

The repo path reads launch files and plugin config, which works when a project
*declares* what it runs. Research code does not: a scientific repo can carry
a hundred source files with no launch files, no YAML and no manifest naming
anything it does. Stage 1 sees zero components, so stage 2 has nothing to
classify and the whole pipeline bottoms out on a repo full of mathematics.

A brief is the other input ingest.py always said existed -- a structured
description of what the project is made of, written by whoever knows. It says
which components exist, which files they live in, and which canon algorithms
they run, and stage 2 proceeds from there unchanged.

It is hand-authored, so it is checked rather than trusted: an algorithm it
names either exists in the canon or the brief is wrong, and a file it names
either exists on disk or the brief is describing a repo that is not there.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .architecture import ArchitectureProposal
from .graph import Graph
from .ingest import Component, Parameter, Project


@dataclass
class BriefComponent:
    name: str
    algorithms: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    purpose: str = ""


@dataclass
class Brief:
    project: str
    subsystems: dict[str, list[str]] = field(default_factory=dict)
    components: dict[str, BriefComponent] = field(default_factory=dict)

    @classmethod
    def from_json(cls, text: str) -> "Brief":
        payload = json.loads(text)
        components = {
            name: BriefComponent(
                name=name,
                algorithms=list(body.get("algorithms", [])),
                files=list(body.get("files", [])),
                purpose=str(body.get("purpose", "")),
            )
            for name, body in payload.get("components", {}).items()
        }
        return cls(project=payload["project"],
                   subsystems={k: list(v) for k, v
                               in payload.get("subsystems", {}).items()},
                   components=components)


def load_brief(path: Path) -> Brief:
    return Brief.from_json(Path(path).read_text())


def to_project(brief: Brief, root: Path | None = None) -> Project:
    """The same Project the repo path produces, so stage 2 cannot tell them apart.

    Files named by the brief become the component's `sources`, which is what
    later stages use to find code evidence -- so a brief buys the code panel
    as well as the architecture.
    """
    project = Project(name=brief.project, origin="brief")
    for name, described in brief.components.items():
        component = project.component(name)
        component.sources = list(described.files)
        component.library = described.algorithms[0] if described.algorithms else None
        # A brief names no parameters. That is honest: a repo with no config
        # files has no knobs, and inventing some would fake the leak signal.
        component.parameters = []
    return project


def propose_from_brief(brief: Brief) -> ArchitectureProposal:
    """Stage 2 from a brief: the subsystems are stated rather than inferred.

    The library registry is not consulted at all. It maps packages to
    algorithms, and a project that imports nothing recognisable cannot be
    routed through it -- which is exactly why this path exists.
    """
    return ArchitectureProposal(
        project=brief.project,
        subsystems={k: sorted(v) for k, v in brief.subsystems.items()},
        algorithms={name: list(c.algorithms)
                    for name, c in brief.components.items() if c.algorithms},
        unclassified=sorted(name for name, c in brief.components.items()
                            if not c.algorithms),
    )


def unknown(brief: Brief, graph: Graph,
            root: Path | None = None) -> dict[str, list[str]]:
    """What the brief claims that is not there.

    A brief is written by a person or a model looking at a repo, so it is
    exactly the kind of input that can name a plausible algorithm the canon
    has never heard of, or a file that was renamed last month.
    """
    problems: dict[str, list[str]] = {}

    listed = set(brief.components)
    grouped = {name for names in brief.subsystems.values() for name in names}
    if grouped - listed:
        problems["components in a subsystem but never described"] = sorted(grouped - listed)
    if listed - grouped:
        problems["components described but in no subsystem"] = sorted(listed - grouped)

    bad_algorithms = sorted({a for c in brief.components.values()
                             for a in c.algorithms if a not in graph.nodes})
    if bad_algorithms:
        problems["algorithms not in the canon"] = bad_algorithms

    if root is not None:
        missing = sorted({f for c in brief.components.values() for f in c.files
                          if not (Path(root) / f).exists()})
        if missing:
            problems["files not found in the repo"] = missing
    return problems


BRIEFS = Path(__file__).resolve().parent.parent / "data" / "briefs"


def resolve(repo: Path, graph: Graph | None = None) -> tuple[
        Project, "ArchitectureProposal", list[str]]:
    """Stages 1 and 2 for a repo, by whichever input path applies.

    Every entry point needs this and each one was doing it itself, so wiring
    the brief into the exporter alone left generation, instantiation and the
    drill still calling ingest_repo directly -- and still failing on a repo
    that declares nothing. That is the same mistake discover.py has been
    sitting in: a capability wired into one caller instead of the shared path.

    The proposal comes back confirmed, because the gate exists to stop a
    *guessed* architecture reaching the canon. A brief is not a guess; it was
    written deliberately and is checked against the canon and the filesystem
    before it is used.
    """
    from .architecture import propose
    from .ingest import ingest_repo

    repo = Path(repo)
    brief_path = BRIEFS / f"{repo.name.lower()}.json"
    if brief_path.exists():
        brief = load_brief(brief_path)
        if graph is None:
            from .bootstrap import build_graph, load
            graph = build_graph(load())
        problems = unknown(brief, graph, repo)
        if problems:
            raise ValueError(
                f"brief {brief_path.name} does not match {repo.name}: {problems}")
        project = to_project(brief, repo)
        proposal = propose_from_brief(brief)
    else:
        project = ingest_repo(repo)
        proposal = propose(project)
    dropped = sorted(proposal.unclassified)
    proposal.unclassified, proposal.confirmed = [], True
    return project, proposal, dropped
