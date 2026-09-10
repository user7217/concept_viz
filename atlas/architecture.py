"""Stage 2: the architecture layer, behind a confirmation gate.

Everything downstream inherits from this structure, and errors here are cheap
to fix now and expensive to inherit. So the proposal is never applied to the
graph until a human confirms it -- this is the pipeline's only human gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .bootstrap import DATA, build_graph, load
from .graph import Graph
from .ingest import Project
from .schema import Node, NodeType, Relation, Tier

LIBRARY_MAP = Path(__file__).resolve().parent.parent / "data" / "library_map.json"


class NotConfirmed(RuntimeError):
    """Raised when an unconfirmed proposal is applied to the graph."""


@dataclass
class ArchitectureProposal:
    project: str
    subsystems: dict[str, list[str]] = field(default_factory=dict)
    algorithms: dict[str, list[str]] = field(default_factory=dict)
    unclassified: list[str] = field(default_factory=list)
    confirmed: bool = False

    def to_json(self) -> str:
        return json.dumps(
            {
                "_note": "Review and edit, then set confirmed to true. Nothing "
                         "downstream runs until you do. Move any unclassified "
                         "component into a subsystem and give it algorithms.",
                "project": self.project,
                "confirmed": self.confirmed,
                "subsystems": self.subsystems,
                "algorithms": self.algorithms,
                "unclassified": self.unclassified,
            },
            indent=2,
        ) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "ArchitectureProposal":
        payload = json.loads(text)
        return cls(
            project=payload["project"],
            subsystems=payload.get("subsystems", {}),
            algorithms=payload.get("algorithms", {}),
            unclassified=payload.get("unclassified", []),
            confirmed=bool(payload.get("confirmed", False)),
        )


def _library_map() -> tuple[dict[str, list[str]], dict[str, str]]:
    payload = json.loads(LIBRARY_MAP.read_text())
    return payload["libraries"], payload["domain_subsystem"]


def _domains() -> dict[str, str]:
    return {seed["id"]: seed["domain"] for seed in load()["seeds"]}


def propose(project: Project) -> ArchitectureProposal:
    """Map components onto canon algorithms and group them into subsystems.

    Conservative by construction: a component whose library is not in the
    registry is listed as unclassified rather than assigned a plausible-looking
    subsystem. A wrong guess here is inherited by every derivation downstream.
    """
    libraries, domain_subsystem = _library_map()
    domains = _domains()

    proposal = ArchitectureProposal(project=project.name)
    for component in project.components:
        # Most specific key wins. A pluginlib class names the algorithm that
        # actually runs; a package/executable pair narrows a package that ships
        # several unrelated nodes; the bare package is the last resort and the
        # one that produces confidently wrong answers.
        algorithms = None
        for key in (component.plugin,
                    f"{component.library}/{component.executable}",
                    component.library or ""):
            if key and key in libraries:
                algorithms = libraries[key]
                break
        if not algorithms:
            proposal.unclassified.append(component.name)
            continue

        proposal.algorithms[component.name] = algorithms
        # A component's subsystem follows its first algorithm's domain.
        subsystem = domain_subsystem.get(domains.get(algorithms[0], ""), "Other")
        proposal.subsystems.setdefault(subsystem, []).append(component.name)

    for names in proposal.subsystems.values():
        names.sort()
    return proposal


def apply(proposal: ArchitectureProposal, graph: Graph | None = None) -> Graph:
    """Add the project-local architecture layer to the graph.

    Raises unless the proposal has been confirmed.
    """
    if not proposal.confirmed:
        raise NotConfirmed(
            f"architecture for {proposal.project!r} is not confirmed; "
            "review the proposal and set confirmed to true"
        )
    if proposal.unclassified:
        raise NotConfirmed(
            "unclassified components must be assigned before applying: "
            f"{sorted(proposal.unclassified)}"
        )

    graph = graph if graph is not None else build_graph(load())
    system_id = _slug(proposal.project)
    graph.add_node(Node(system_id, NodeType.SYSTEM, proposal.project, Tier.ASSUMED))

    for subsystem, components in proposal.subsystems.items():
        subsystem_id = f"{system_id}__{_slug(subsystem)}"
        graph.add_node(
            Node(subsystem_id, NodeType.SUBSYSTEM, subsystem, Tier.ASSUMED)
        )
        graph.add_edge(system_id, subsystem_id, Relation.CONTAINS)

        for component in components:
            component_id = f"{system_id}__{_slug(component)}"
            graph.add_node(
                Node(component_id, NodeType.COMPONENT, component, Tier.ASSUMED)
            )
            graph.add_edge(subsystem_id, component_id, Relation.CONTAINS)

            for algorithm in proposal.algorithms.get(component, []):
                if algorithm not in graph.nodes:
                    raise KeyError(f"unknown algorithm id: {algorithm}")
                graph.add_edge(component_id, algorithm, Relation.USES)

    return graph


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")
