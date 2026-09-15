"""The path as something you can work through on paper.

An ordered list of nodes is not a learning path, it is an inventory. Reading
one tells you marginalization comes before gaussian_conditioning; it never
tells you the extended Kalman filter needs both, which is the only reason
either is on your desk. Without that, a path reads as a pile of unrelated
mathematics -- which is exactly what the first worksheet was.

Every node here carries three things before its exercises:

  the chain   -- why it is on this path, as REQUIRES edges from the algorithm
                 your project actually runs down to this node
  in project  -- what it is doing in this specific codebase, from layer 2
  knobs       -- the config parameters it reaches, when it reaches any

All three already existed. The first worksheet simply threw them away.
"""

from __future__ import annotations

from .exercise import exercises_for, maskable_steps, step_exercise
from .graph import Graph


def _section(graph: Graph, node_id: str, sheet: dict, algorithm: str,
             include_symbols: bool, limit: int) -> tuple[list[str], list[str]]:
    """One node's questions and answers, as markdown fragments."""
    out: list[str] = [f"\n## {node_id}\n"]

    chain = graph.requires_chain(algorithm, node_id)
    if len(chain) > 1:
        out.append(f"\n**Why it is here:** `" + "` → `".join(chain) + "`\n")
    elif chain:
        out.append(f"\n**Why it is here:** `{algorithm}` requires it directly\n")

    note = sheet.get("instantiation") or {}
    if note.get("why_here"):
        out.append(f"\n**In your project:** {note['why_here']}\n")
    for knob in note.get("knobs", [])[:4]:
        out.append(f"\n- `{knob.get('parameter')}` "
                   f"in `{knob.get('file')}` — {knob.get('means')}\n")

    if include_symbols:
        found = exercises_for(sheet, limit=limit)
    else:
        found = [step_exercise(sheet, i) for i in maskable_steps(sheet)[:limit]]
    if not found:
        out.append("\n_No derivation steps on this sheet, so nothing to"
                   " reproduce. It is here as a prerequisite._\n")
    return out, found


def build(graph: Graph, path: list[str], sheets: dict[str, dict],
          algorithm: str, project_name: str, subsystem: str,
          limit: int = 10, include_symbols: bool = False) -> str:
    """Render the whole path: context, questions, then answers at the end."""
    questions: list[str] = []
    answers: list[str] = []
    count = 0

    for node_id in path:
        sheet = sheets.get(node_id)
        if sheet is None:
            continue
        section, found = _section(graph, node_id, sheet, algorithm,
                                  include_symbols, limit)
        questions += section
        for exercise in found:
            count += 1
            questions.append(f"\n**{count}. {exercise.question}**\n")
            if exercise.hint:
                questions.append(f"\n_{exercise.hint}_\n")
            questions.append("\n```\n")
            questions += [f"{line}\n" for line in exercise.context]
            questions.append("```\n")
            answers.append(f"\n**{count}. {node_id}**\n\n```\n"
                           f"{exercise.answer}\n```\n")

    header = (
        f"# {project_name} — {subsystem}\n\n"
        f"The mathematics behind `{algorithm}`, which is what "
        f"{subsystem.lower()} in this project actually runs.\n\n"
        f"Each node below says why it is on the path (the chain of "
        f"prerequisites from the algorithm down to it) and what it is doing "
        f"in this codebase, then asks you to reproduce the steps.\n\n"
        f"{count} exercises. Answers are in one block at the end.\n"
    )
    return header + "".join(questions) + "\n\n---\n\n# Answers\n" + "".join(answers)
