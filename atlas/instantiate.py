"""Layer 2: what a canon node means in one specific project.

A sheet on covariance_matrix is otherwise indistinguishable from the article it
was restated from. What makes it worth having is the other half: this is the
15x15 process_noise_covariance in your ekf_local.yaml, these rows are the
velocities you fuse, and this is what happens when you get it wrong.

Needs no retrieval. Everything comes from the cached sheet plus the ingested
project, so it is cheap and regenerates freely when the repo changes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from .graph import Graph
from .ingest import Component, Project
from .llm import Provider, complete_json
from .schema import Relation

SYSTEM = """You explain where a piece of mathematics sits in a specific codebase.

You are given a result the reader is learning, and the components of their
project that depend on it, with the parameters those components actually set.

Ground everything in what you are given. Never invent a parameter, a file or a
value. If the connection is only that the component's algorithm requires this
result, say that plainly rather than manufacturing detail.

Reply with JSON only."""

TEMPLATE = """Project: {project}
Concept: {name}

What the reader is learning:
{statement}

Symbols in the sheet:
{symbols}

Components in this project that depend on it:
{components}

Explain where this sits in their system.

Return JSON:
{{
  "why_here": "2-3 sentences: what this concept is doing in THIS project",
  "concrete": [
    {{"symbol": "a symbol from the sheet above",
      "in_your_project": "what it actually is here, with dimensions if known"}}
  ],
  "knobs": [
    {{"parameter": "a parameter name from the list above",
      "file": "the file it is set in",
      "means": "what setting it does, in terms of this concept"}}
  ]
}}

Only use symbols listed above. Only use parameters and files listed above.
Leave "knobs" empty if none of the listed parameters relate to this concept."""


@dataclass
class Instantiation:
    node_id: str
    project: str
    components: list[str] = field(default_factory=list)
    why_here: str = ""
    concrete: list[dict] = field(default_factory=list)
    knobs: list[dict] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2) + "\n"


def components_reaching(graph: Graph, project_id: str, node_id: str) -> list[str]:
    """Project components whose algorithm requires this node.

    Walks the architecture layer down to algorithms, then asks which of those
    have the node in their prerequisite closure.
    """
    found: list[str] = []
    for edge in graph.edges.values():
        if edge.rel is not Relation.USES:
            continue
        if not edge.src.startswith(project_id):
            continue
        closure = graph.requires_closure(edge.dst)
        if node_id in closure or node_id == edge.dst:
            found.append(edge.src)
    return sorted(set(found))


def _render_components(project: Project, component_ids: list[str],
                       project_id: str) -> str:
    """Component names with the parameters they actually set."""
    by_name = {c.name: c for c in project.components}
    blocks: list[str] = []
    for component_id in component_ids:
        name = component_id[len(project_id) + 2:]
        component = by_name.get(name)
        if component is None:
            blocks.append(f"- {name}")
            continue
        line = f"- {name} (library: {component.library or 'unknown'})"
        if component.parameters:
            # Matrix-valued parameters first: they are the leak signal, the
            # knobs you cannot set without understanding the object behind
            # them. Taking the first N in file order hid
            # process_noise_covariance and produced a note claiming nothing in
            # the project tunes covariance.
            ordered = sorted(component.parameters,
                             key=lambda x: (not x.is_matrix, x.name))
            shown = []
            for param in ordered[:16]:
                mark = "  [matrix-valued]" if param.is_matrix else ""
                shown.append(f"    {param.name} = {param.value[:60]}"
                             f"  in {param.source}{mark}")
            line += "\n  parameters it sets:\n" + "\n".join(shown)
        blocks.append(line)
    return "\n".join(blocks) or "(none)"


def build_prompt(project: Project, project_id: str, node_id: str, name: str,
                 sheet: dict, component_ids: list[str]) -> str:
    symbols = "\n".join(
        f"  {s.get('symbol')} = {s.get('meaning', '')}"
        f"{'  [' + s.get('dimensions', '') + ']' if s.get('dimensions') else ''}"
        for s in sheet.get("symbols", [])
    ) or "(none listed)"
    return TEMPLATE.format(
        project=project.name, name=name,
        statement=sheet.get("statement", "")[:800],
        symbols=symbols,
        components=_render_components(project, component_ids, project_id),
    )


def instantiate(provider: Provider, project: Project, project_id: str,
                graph: Graph, node_id: str, sheet: dict) -> Instantiation:
    components = components_reaching(graph, project_id, node_id)
    if not components:
        return Instantiation(node_id=node_id, project=project.name)

    name = graph.nodes[node_id].name if node_id in graph.nodes else node_id
    payload = complete_json(
        provider,
        build_prompt(project, project_id, node_id, name, sheet, components),
        system=SYSTEM,
    )
    return Instantiation(
        node_id=node_id,
        project=project.name,
        components=[c[len(project_id) + 2:] for c in components],
        why_here=str(payload.get("why_here", "")),
        concrete=list(payload.get("concrete", [])),
        knobs=list(payload.get("knobs", [])),
    )


def _symbol_set(sheet: dict) -> set[str]:
    """Every symbol a sheet declares, including combined entries.

    Sheets group related symbols into one row -- "x, v, z" is a single entry
    covering three. Matching the whole string only, the guard flagged a
    correct reference to `x` as invented, so it has to split them too.
    """
    found: set[str] = set()
    for entry in sheet.get("symbols", []):
        raw = str(entry.get("symbol", "")).strip()
        if not raw:
            continue
        found.add(raw)
        found.update(part.strip() for part in raw.split(",") if part.strip())
    return found


def invented(instantiation: Instantiation, project: Project,
             sheet: dict) -> dict[str, list[str]]:
    """Symbols, parameters and files it names that do not exist.

    The independent comparison: not "is this internally consistent" but "does
    every concrete claim point at something really in the sheet or the repo".
    """
    symbols = _symbol_set(sheet)
    parameters = {p.name for p in project.parameters}
    files = {p.source for p in project.parameters}

    problems: dict[str, list[str]] = {}
    bad_symbols = [str(c.get("symbol")) for c in instantiation.concrete
                   if str(c.get("symbol")) not in symbols]
    bad_params = [str(k.get("parameter")) for k in instantiation.knobs
                  if str(k.get("parameter")) not in parameters]
    bad_files = [str(k.get("file")) for k in instantiation.knobs
                 if k.get("file") and str(k.get("file")) not in files]
    if bad_symbols:
        problems["symbols not in the sheet"] = bad_symbols
    if bad_params:
        problems["parameters not set by this project"] = bad_params
    if bad_files:
        problems["files not in this repo"] = bad_files
    return problems
