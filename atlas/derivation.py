"""Stage 4b: restate a retrieved derivation, and check what comes back.

The model is never asked to recall a derivation. It is given source text and
asked to restate what is there, citing per step. Three things are then checked
mechanically, because the reader cannot check any of them:

  1. every citation names a source actually retrieved in this run
  2. every concept a step invokes resolves to a canon node
  3. the node's prerequisite closure covers everything the derivation used

Check 3 is the completeness criterion from design doc section 4. It is what
turns "what does this require?" from a judgment into a set difference.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from .graph import Graph
from .llm import LLMError, Provider, complete_json
from .retrieval import Source, cited_but_not_retrieved

MAX_SOURCE_CHARS = 24000

# Reference articles put derivations well past the introduction, so truncating
# from the front feeds the model the setup and cuts off the part it needs.
DERIVATION_MARKERS = (
    "deriv", "proof", "formulation", "solution", "estimation", "algorithm",
    "update", "predict", "optimal", "minimiz", "least squares",
)

SYSTEM = """You restate mathematics from supplied source text.

Some nodes are RESULTS with a derivation (the Kalman update, Gaussian
conditioning). Others are DEFINITIONS or objects (a covariance matrix, a
Jacobian) -- there is nothing to derive, and deriving some property of them
instead answers a question the reader did not ask. Decide which you have been
given and produce the matching shape.

Absolute rules:
- Use ONLY the supplied sources. If they do not contain a derivation, say so.
- Never introduce a step from memory. Never cite a source not supplied.
- Cite the source id for every step.

Scope: derive the object; do not prove theorems about it. Include what is
needed to reproduce the result on paper and to know when it fails. Exclude
existence and uniqueness proofs, convergence theorems, and measure-theoretic
or functional-analytic framing.

Reply with JSON only."""

TEMPLATE = """Concept: {name}
Canon id: {node_id}

Sources (cite by id):
{sources}

{expectation}

Then produce that shape from the sources.

Return JSON:
{{
  "kind": "derivation" | "definition",
  "grounded": true | false,
  "statement": "the result or the definition, stated precisely",
  "symbols": [{{"symbol": "P", "meaning": "state covariance", "dimensions": "n x n"}}],

  "steps": [
    {{"n": 1,
      "text": "the step, with its equation",
      "invokes": ["canon concept or technique this step uses"],
      "cites": ["source id"]}}
  ],

  "properties": [
    {{"property": "for a definition: a key fact that follows from it",
      "why": "one clause", "cites": ["source id"]}}
  ],

  "assumptions": [{{"assumption": "...", "breaks_when": "..."}}],
  "failure_modes": [{{"mode": "...", "signature": "...", "cause": "..."}}]
}}

For "derivation": fill steps, leave properties empty. Do NOT derive a property
of the object in place of the object itself.
For "definition": leave steps empty, give the defining equation in "statement",
and list the facts a reader needs in "properties". Every property cites a source.

Set "grounded" false only if the sources do not support the concept at all.
"""


@dataclass
class Step:
    n: int
    text: str
    invokes: list[str] = field(default_factory=list)
    cites: list[str] = field(default_factory=list)


@dataclass
class Derivation:
    node_id: str
    grounded: bool
    kind: str = "derivation"  # "derivation" | "definition"
    statement: str = ""
    symbols: list[dict] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    properties: list[dict] = field(default_factory=list)
    assumptions: list[dict] = field(default_factory=list)
    failure_modes: list[dict] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)

    @property
    def is_definition(self) -> bool:
        return self.kind == "definition"

    @property
    def invoked(self) -> set[str]:
        return {item for step in self.steps for item in step.invokes}

    @property
    def citations(self) -> list[str]:
        cited = [c for step in self.steps for c in step.cites]
        for prop in self.properties:
            cited += [str(c) for c in prop.get("cites", [])]
        return cited

    def to_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, indent=2) + "\n"


def select_text(text: str, budget: int) -> str:
    """Keep the contiguous region of a source most likely to carry a derivation.

    Two failure modes, both observed:
    - Front-truncation feeds the model the setup and cuts the derivation off. A
      329k-char Kalman filter article puts its derivations far past any budget.
    - Selecting high-scoring paragraphs independently is worse still. A
      derivation is a continuous argument; scattered fragments read as no
      derivation at all, and the model correctly refuses.

    So: score paragraphs, then keep the best *contiguous* window, plus a short
    head for context.
    """
    if len(text) <= budget:
        return text

    paragraphs = [p for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return text[:budget]

    scores = []
    for paragraph in paragraphs:
        lowered = paragraph.lower()
        score = sum(3 for marker in DERIVATION_MARKERS if marker in lowered)
        score += lowered.count("=") + lowered.count("\\")
        scores.append(score)

    head = paragraphs[:2]
    head_text = "\n".join(head)
    window_budget = max(budget - len(head_text), budget // 2)

    best_start, best_score, best_end = 0, -1, 1
    for start in range(len(paragraphs)):
        used, total, end = 0, 0, start
        while end < len(paragraphs) and used + len(paragraphs[end]) <= window_budget:
            used += len(paragraphs[end]) + 1
            total += scores[end]
            end += 1
        if end > start and total > best_score:
            best_start, best_score, best_end = start, total, end

    window = paragraphs[best_start:best_end]
    if best_start <= 2:
        return "\n".join(window)
    return head_text + "\n\n[...]\n\n" + "\n".join(window)


EXPECT_DERIVATION = (
    "This node is expected to be a RESULT with a derivation: the canon records "
    "that it is reached using {moves}, which are manipulation moves you perform "
    "in a derivation. Restate that derivation. Only call it a definition if the "
    "sources genuinely contain nothing to derive, and say so in the statement."
)
EXPECT_DEFINITION = (
    "This node is expected to be a DEFINITION or object -- the canon records no "
    "manipulation moves under it, so there is likely nothing to derive. State it "
    "precisely. Do NOT derive a property of the object in place of the object. "
    "Only produce a derivation if the sources clearly derive this node itself."
)


def expected_kind(graph: Graph | None, node_id: str) -> tuple[str, list[str]]:
    """Guess result-vs-definition from the canon's own structure.

    A node that requires techniques is derived: techniques are the moves you
    perform in a derivation, so needing one means there is something to perform.
    Without this hint the model takes the easier shape and calls everything a
    definition.
    """
    if graph is None or node_id not in graph.nodes:
        return "derivation", []
    from .schema import NodeType, Relation

    # A technique IS a move, so it is always performed rather than stated --
    # whatever its own prerequisites look like. Asking whether a node *requires*
    # techniques never asked whether it *is* one, which had 17 of 33 techniques,
    # integration_by_parts and linearization among them, stated as definitions.
    if graph.nodes[node_id].type is NodeType.TECHNIQUE:
        return "derivation", [node_id]

    moves = sorted(
        e.dst for e in graph.edges.values()
        if e.src == node_id and e.rel is Relation.REQUIRES
        and graph.nodes[e.dst].type is NodeType.TECHNIQUE
    )
    return ("derivation" if moves else "definition"), moves


def build_prompt(node_id: str, name: str, sources: list[Source],
                 vocabulary: list[str] | None = None,
                 graph: Graph | None = None) -> str:
    """Render sources into the prompt, keeping derivation-bearing sections."""
    budget = MAX_SOURCE_CHARS // max(len(sources), 1)
    blocks = [
        f"[{source.id}] {source.title}\n{select_text(source.text, budget)}"
        for source in sources
    ]
    vocab = ""
    if vocabulary:
        vocab = (
            "\n\nWhen naming what a step invokes, use ids from this list wherever "
            "one applies. Only invent a name if nothing here fits:\n"
            + ", ".join(sorted(vocabulary))
        )
    kind, moves = expected_kind(graph, node_id)
    expectation = (
        EXPECT_DERIVATION.format(moves=", ".join(moves))
        if kind == "derivation" else EXPECT_DEFINITION
    )
    return TEMPLATE.format(
        name=name, node_id=node_id, sources="\n\n".join(blocks) or "(none)",
        expectation=expectation,
    ) + vocab


def candidate_vocabulary(graph: Graph, node_id: str, limit: int = 60) -> list[str]:
    """Canon ids a derivation of this node plausibly invokes.

    Constraining invocations to real ids is what makes the closure check work:
    free-form prose like "sum of squared residuals definition" resolves to
    nothing, so every check downstream goes inert.
    """
    from .schema import NodeType

    nearby = set(graph.requires_closure(node_id)) if node_id in graph.nodes else set()
    if len(nearby) < limit:
        nearby |= {
            n.id for n in graph.nodes.values()
            if n.type in (NodeType.CONCEPT, NodeType.TECHNIQUE)
        }
    return sorted(nearby)[:limit] if len(nearby) > limit else sorted(nearby)


def generate(provider: Provider, node_id: str, name: str, sources: list[Source],
             vocabulary: list[str] | None = None,
             graph: Graph | None = None) -> Derivation:
    """Ask a provider to restate the derivation carried by `sources`."""
    if not sources:
        return Derivation(node_id=node_id, grounded=False)

    prompt = build_prompt(node_id, name, sources, vocabulary, graph)
    try:
        payload = complete_json(provider, prompt, system=SYSTEM)
    except LLMError:
        # Derivations are LaTeX-heavy and backslashes break JSON encoding often
        # enough that one retry is worth a call. A second failure is systematic.
        payload = complete_json(
            provider,
            prompt + "\n\nReturn STRICT JSON. Escape every backslash in LaTeX "
                     "as \\\\, and do not wrap the object in prose or fences.",
            system=SYSTEM,
        )
    steps = [
        Step(
            n=int(raw.get("n", index + 1)),
            text=str(raw.get("text", "")),
            invokes=[str(x) for x in raw.get("invokes", [])],
            cites=[str(x) for x in raw.get("cites", [])],
        )
        for index, raw in enumerate(payload.get("steps", []))
    ]
    kind = str(payload.get("kind", "derivation")).lower()
    if kind not in ("derivation", "definition"):
        kind = "definition" if not steps else "derivation"

    return Derivation(
        node_id=node_id,
        grounded=bool(payload.get("grounded", False)),
        kind=kind,
        statement=str(payload.get("statement", "")),
        properties=list(payload.get("properties", [])),
        symbols=list(payload.get("symbols", [])),
        steps=steps,
        assumptions=list(payload.get("assumptions", [])),
        failure_modes=list(payload.get("failure_modes", [])),
        source_ids=[s.id for s in sources],
    )


# ---------- checks the reader cannot perform ----------

def resolve_invoked(graph: Graph, invoked: set[str]) -> tuple[set[str], set[str]]:
    """Split invoked names into (resolved canon ids, unrecognised).

    Unrecognised names are not errors. They are candidate canon nodes -- the
    same provisional-node path the design doc uses when expansion meets a
    concept the canon lacks.
    """
    by_name = {n.name.lower(): n.id for n in graph.nodes.values()}
    resolved: set[str] = set()
    unknown: set[str] = set()

    for item in invoked:
        key = item.strip()
        slug = key.lower().replace(" ", "_").replace("-", "_")
        candidate = graph.resolve(slug)
        if candidate in graph.nodes:
            resolved.add(candidate)
        elif key.lower() in by_name:
            resolved.add(by_name[key.lower()])
        else:
            unknown.add(key)
    return resolved, unknown


def check(derivation: Derivation, sources: list[Source], graph: Graph,
          known: set[str] | None = None) -> dict:
    """Run every mechanical check on a generated derivation."""
    from .schema import Tier

    ungrounded = cited_but_not_retrieved(derivation.citations, sources)
    resolved, unknown = resolve_invoked(graph, derivation.invoked)

    # Assumed-tier nodes are the canon's floor. A derivation invoking
    # `derivative` has not found a hole; it has reached the bottom.
    floor = {n.id for n in graph.nodes.values() if n.tier is Tier.ASSUMED}
    holes: set[str] = set()
    if derivation.node_id in graph.nodes:
        holes = graph.closure_holes(
            derivation.node_id, resolved, (known or set()) | floor
        )

    uncited = [step.n for step in derivation.steps if not step.cites]

    # A definition has no steps to check; it must instead say what the thing is
    # and cite the facts a reader needs. Requiring steps of it is what made
    # covariance_matrix derive a property of itself to satisfy the shape.
    if derivation.is_definition:
        shape = []
        if not derivation.statement.strip():
            shape.append("definition has no statement")
        if not derivation.properties:
            shape.append("definition lists no properties")
        if derivation.steps:
            shape.append("definition carries derivation steps")
    else:
        shape = [] if derivation.steps else ["derivation carries no steps"]

    return {
        "node_id": derivation.node_id,
        "kind": derivation.kind,
        "grounded": derivation.grounded,
        "ungrounded_citations": sorted(ungrounded),
        "uncited_steps": uncited,
        "unknown_invocations": sorted(unknown),
        "closure_holes": sorted(holes),
        "shape": shape,
        "ok": (not ungrounded and not uncited and not shape
               and derivation.grounded),
    }
