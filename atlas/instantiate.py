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
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .graph import Graph
from .ingest import Component, Project
from .llm import Provider, complete_json
from .schema import Relation

SYSTEM = """You explain where a piece of mathematics sits in a specific codebase.

You are given a result the reader is learning, and the components of their
project that depend on it, with the parameters those components actually set.

Ground everything in what you are given. Never invent a parameter, a file or a
value. You may be shown the library's own source: use it to say how a
parameter enters the computation, and never claim more than those lines show.

Describe what the shown lines DO, never what they are FOR. Do not say a
parameter compensates for, accounts for, absorbs or corrects an error unless
a shown line says so. Do not relate parameters to each other or to variables
in the code unless a shown line connects them. If nothing in the project
computes this concept itself, say so, then say only what the shown lines and
the sheet support about where it sits. If the connection is only that the component's algorithm requires this
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

What the library's source does with those parameters:
{library}

What feeds these components (topics they read, and what the sender sets):
{upstream}

Explain where this sits in their system. Where a matrix-valued parameter is
shown entry by entry, say which entries matter here by their labels. Where
the library source shows how a parameter is applied -- scaled, added,
inverted -- say so in that knob. Where a sender sets a quantity the sheet
names, such as a measurement covariance, map it in "concrete".

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
  ],
  "notation": "one sentence if the sheet's symbol for a quantity differs from the parameter's or the library's name for it (e.g. the sheet calls process noise R where the parameter sets Q); otherwise empty"
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
    notation: str = ""

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


# ---------- what the note writer is shown ----------
# A note can only map what it is given, and it was given too little: each
# parameter cut at 60 characters -- six entries of a 225-entry covariance --
# and no library source. So "process_noise_covariance sets Q" was the most it
# could say, while the readers needed which entry is yaw rate, that the
# library adds Q * dt per step, and that R arrives from odom_cov_relay. All
# three facts were already in the project's data; none reached the prompt.

LIBRARY_CODE = Path(__file__).resolve().parent.parent / "data" / "library_code.json"
LIBRARY_BUDGET = 2400
ENUM = re.compile(r"enum\s+\w+\s*\{([^}]*)")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def state_order(spans: list[dict]) -> list[str]:
    """The library's names for its state entries, from its own enum.

    Read leniently: the stored excerpt of robot_localization's StateMembers
    ends before its closing brace.
    """
    for span in spans:
        found = ENUM.search("\n".join(span.get("lines", [])))
        if not found:
            continue
        names = re.findall(r"^\s*([A-Za-z_]\w*)\s*(?:=\s*\d+)?\s*,?\s*$",
                           found.group(1), re.M)
        if len(names) < 2:
            continue
        prefix = names[0]
        for name in names[1:]:
            while not name.startswith(prefix):
                prefix = prefix[:-1]
        return [n[len(prefix):] or n for n in names]
    return []


def matrix_view(value: str, labels: list[str]) -> str | None:
    """A square matrix as its diagonal, labelled, instead of a cut string.

    Labelled in code from the library's enum rather than left to the model:
    asked to count to entry 11 of a flattened 15x15, it should not have to.
    """
    numbers = NUMBER.findall(value)
    n = math.isqrt(len(numbers))
    if n < 2 or n * n != len(numbers):
        return None
    names = labels if len(labels) == n else [str(i) for i in range(n)]
    diagonal = ", ".join(f"{names[i]}={numbers[i * n + i]}" for i in range(n))
    off = any(float(numbers[r * n + c]) != 0 for r in range(n) for c in range(n) if r != c)
    return (f"{n}x{n}, diagonal {diagonal}; off-diagonal "
            + ("entries set" if off else "all zero"))


def flags_view(value: str, labels: list[str]) -> str | None:
    """A boolean mask over the state as the entries it switches on.

    odom0_config is fifteen true/false flags in the library's state order;
    cut at sixty characters it showed eight of them.
    """
    flags = re.findall(r"\b(true|false)\b", value.lower())
    if not labels or len(flags) != len(labels):
        return None
    on = [labels[i] for i, f in enumerate(flags) if f == "true"]
    return "fuses " + (", ".join(on) if on else "nothing") + f" (of {len(labels)} state entries)"


def _covariance_lines(code: list[str], path: str) -> list[str]:
    """Lines that set a covariance, and the lines that fill what they assign.

    `msg.twist.covariance = c` says nothing until `c[0] = 0.05  # vx` is
    beside it.
    """
    keep: set[int] = set()
    for i, line in enumerate(code):
        if not re.search(r"(?i)covariance", line):
            continue
        keep.add(i)
        source = re.search(r"covariance\w*\s*=\s*([A-Za-z_]\w*)\s*$", line.strip())
        if source:
            name = re.escape(source.group(1))
            keep |= {j for j in range(max(0, i - 15), i)
                     if re.match(rf"\s*{name}\b", code[j])}
    return [f"    {path}:{i + 1}  {code[i].strip()}" for i in sorted(keep)]


def library_excerpts(spans: list[dict], parameters: list[str],
                     budget: int = LIBRARY_BUDGET, own: list[dict] = ()) -> str:
    """Spans filed under this node, whole; elsewhere, lines that read a parameter.

    Filtering every span by parameter name hid the gate. ekf.cpp passes
    measurement.mahalanobis_thresh_, never the YAML name, so the gating note
    saw no gating code and said the library has none. A span filed under a
    node was chosen for that node; the name filter is for the rest.
    """
    out, used = [], 0
    for span in own:
        lines = span.get("lines", [])
        start = int(span.get("start", 1))
        block = (f"{span['path']}:{start}-{start + len(lines) - 1}"
                 f"  ({span.get('term', '')})\n" + "\n".join(lines))
        if used + len(block) > budget:
            return "\n\n".join(out)
        out.append(block)
        used += len(block)
    names = [re.compile(rf"\b{re.escape(p)}_?\b") for p in parameters]
    for span in spans:
        if span in own:
            continue
        lines = span.get("lines", [])
        start = int(span.get("start", 1))
        hits = [i for i, line in enumerate(lines) if any(n.search(line) for n in names)]
        for i in hits:
            window = lines[max(0, i - 4):i + 2]
            block = f"{span['path']}:{start + i}\n" + "\n".join(window)
            if used + len(block) > budget:
                return "\n\n".join(out)
            out.append(block)
            used += len(block)
    return "\n\n".join(out)


def upstream(links: list, repo: Path | None, names: set[str]) -> str:
    """Who sends each topic these components read, and what the sender sets.

    The sender's own covariance lines are included because that is where a
    measurement covariance comes from: odom_cov_relay writes the R that
    ekf_local fuses, and no parameter file says so.
    """
    out = []
    for link in links:
        if link.subscriber not in names:
            continue
        out.append(f"- {link.publisher} publishes {link.topic}; {link.subscriber} "
                   f"reads it ({link.subscriber_evidence})")
        where = re.match(r"(\S+\.py):\d+", link.publisher_evidence)
        if repo is None or not where or not (repo / where.group(1)).exists():
            continue
        code = (repo / where.group(1)).read_text().splitlines()
        out.extend(_covariance_lines(code, where.group(1))[:10])
    return "\n".join(dict.fromkeys(out)) or "(nothing upstream is stated)"


def _render_components(project: Project, component_ids: list[str],
                       project_id: str, labels: list[str] | None = None) -> str:
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
            # Every matrix-valued parameter, always. They are the leak signal,
            # the knobs you cannot set without understanding the object behind
            # them, and a blind cut at N in file order once hid
            # process_noise_covariance and produced a note claiming nothing in
            # the project tunes covariance.
            #
            # Plain parameters too, all of them. A budget of four, taken in
            # alphabetical order, kept base_link_frame and cut
            # imu0_twist_rejection_threshold, and the gating note said the
            # project sets no gate. The budget existed because `agy` dropped
            # long prompts; that was a denied tool call, not the length.
            matrix = [x for x in component.parameters if x.is_matrix]
            plain = sorted((x for x in component.parameters if not x.is_matrix),
                           key=lambda x: x.name)
            shown = []
            for param in matrix + plain:
                view = (matrix_view(param.value, labels or [])
                        or flags_view(param.value, labels or [])) if param.is_matrix else None
                if view:
                    shown.append(f"    {param.name} = {view}  in {param.source}"
                                 f"  [matrix-valued]")
                    continue
                mark = "  [matrix-valued]" if param.is_matrix else ""
                shown.append(f"    {param.name} = {param.value[:60]}"
                             f"  in {param.source}{mark}")
            line += "\n  parameters it sets:\n" + "\n".join(shown)
        blocks.append(line)
    return "\n".join(blocks) or "(none)"


def build_prompt(project: Project, project_id: str, node_id: str, name: str,
                 sheet: dict, component_ids: list[str],
                 spans: list[dict] | None = None, links: list | None = None,
                 repo: Path | None = None, own_spans: list[dict] | None = None) -> str:
    symbols = "\n".join(
        f"  {s.get('symbol')} = {s.get('meaning', '')}"
        f"{'  [' + s.get('dimensions', '') + ']' if s.get('dimensions') else ''}"
        for s in sheet.get("symbols", [])
    ) or "(none listed)"
    spans = spans or []
    names = {c[len(project_id) + 2:] for c in component_ids}
    parameters = sorted({p.name for c in project.components if c.name in names
                         for p in c.parameters})
    return TEMPLATE.format(
        project=project.name, name=name,
        statement=sheet.get("statement", "")[:800],
        symbols=symbols,
        components=_render_components(project, component_ids, project_id,
                                      state_order(spans)),
        library=library_excerpts(spans, parameters, own=own_spans or [])
        or "(no library source shown)",
        upstream=upstream(links or [], repo, names),
    )


# A note claiming a mechanism the code does not have. The Taylor notes said
# the dropped remainder "is accounted for by adding delta_sec * Q": the line
# adds a fixed Q every step and computes no remainder. The `invented` check
# could not see it -- every name in the sentence was real.
UNSUPPORTED = re.compile(r"(?i)\b(account(s|ed)?\s+for|compensat\w*|absorb\w*|"
                         r"correct(s|ed)?\s+for|makes?\s+up\s+for)\b")


def unsupported(instantiation: Instantiation) -> list[str]:
    """Sentences claiming a parameter compensates for something.

    Flagged rather than trusted: the shown library lines say what a
    parameter does, never what it is for.
    """
    text = " ".join([instantiation.why_here, instantiation.notation,
                     *(str(c.get("in_your_project", "")) for c in instantiation.concrete),
                     *(str(k.get("means", "")) for k in instantiation.knobs)])
    return [s.strip() for s in re.split(r"(?<=[.;])\s+", text) if UNSUPPORTED.search(s)]


def spans_for(node_id: str, is_algorithm: bool, path: Path = LIBRARY_CODE) -> list[dict]:
    """Library source a note on this node may be shown.

    All of it for an algorithm the library implements; otherwise only the
    excerpts filed under this node. Shown `P += delta_sec * Q` while writing
    about Taylor's theorem, the note connected the two and invented the
    connection.
    """
    if not path.exists():
        return []
    groups = json.loads(path.read_text()).get("spans", {})
    if is_algorithm:
        return [s for group in groups.values() for s in group]
    return list(groups.get(node_id, []))


def instantiate(provider: Provider, project: Project, project_id: str,
                graph: Graph, node_id: str, sheet: dict,
                repo: Path | None = None, links: list | None = None) -> Instantiation:
    components = components_reaching(graph, project_id, node_id)
    if not components:
        return Instantiation(node_id=node_id, project=project.name)

    name = graph.nodes[node_id].name if node_id in graph.nodes else node_id
    from .schema import NodeType
    is_algorithm = node_id in graph.nodes and graph.nodes[node_id].type is NodeType.ALGORITHM
    payload = complete_json(
        provider,
        build_prompt(project, project_id, node_id, name, sheet, components,
                     spans_for(node_id, is_algorithm), links, repo,
                     own_spans=spans_for(node_id, False)),
        system=SYSTEM,
    )
    return Instantiation(
        node_id=node_id,
        project=project.name,
        components=[c[len(project_id) + 2:] for c in components],
        why_here=str(payload.get("why_here", "")),
        concrete=list(payload.get("concrete", [])),
        knobs=list(payload.get("knobs", [])),
        notation=str(payload.get("notation", "") or ""),
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


def thin(instantiation: Instantiation, sheet: dict) -> str:
    """Why this note says nothing, or "" if it says something.

    `invented` asks whether the claims are false. A note that makes no claims
    passes it trivially -- schur_complement came back with prose and nothing
    else, no symbol mapped and no parameter named, and was recorded clean.

    The two checks are different axes and both are needed: one catches a note
    that says too much, this one catches a note that says nothing.
    """
    if not instantiation.components:
        return ""  # nothing reaches it; an empty note is the honest answer
    if instantiation.concrete or instantiation.knobs:
        return ""
    if sheet.get("symbols"):
        return (f"names no parameter and maps none of the "
                f"{len(sheet['symbols'])} symbols the sheet defines")
    return "names no parameter and maps no symbol"
