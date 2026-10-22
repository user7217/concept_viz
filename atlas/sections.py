"""Sections of a pinned document that another sheet already derives.

The EKF sheet restates section 4 of a paper whose section 4 hands the
closed-form Gaussian algebra to section 8: "see Section 8". The model,
seeing only section 4, wrote exactly that -- correct, and a dead end, because
section 8 is derived in full on the linear Kalman filter sheet one step down
the path. All four acceptance readers followed the pointer out of the atlas.

Which sheet derives which section is already recorded by hand, independently
of any sheet: source_aliases.json grounds linear_kalman_filter on
"arxiv:2406.06427#8", and source_documents.json lists that entry's sections.
Generation is told the owners; check() fails a sheet that points at an owned
section without naming the sheet that owns it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# "Section 8", "Sec. 8.1", "§8", "§ 8.2" -- the number is what matters.
REFERENCE = re.compile(r"(?:\bsections?|\bsec\.|§)\s*(\d+(?:\.\d+)*)", re.IGNORECASE)
NUMBER = re.compile(r"^\s*(\d+(?:\.\d+)*)")


@dataclass(frozen=True)
class Owner:
    document: str   # base id, e.g. "arxiv:2406.06427"
    section: str    # "8.1"
    node_id: str
    name: str


def _base(document_id: str) -> str:
    return document_id.split("#", 1)[0]


def _numbers(entry: dict) -> list[str]:
    titles = entry.get("sections") or [entry.get("section", "")]
    return [m.group(1) for m in map(NUMBER.match, titles) if m]


def owners(graph, node_id: str, aliases: dict | None = None,
           documents: dict | None = None) -> list[Owner]:
    """Pinned sections derived on this node's prerequisites.

    Only prerequisites count: a pointer to a sheet the reader has already
    passed on the way here is the one worth following.
    """
    from .retrieval import load_aliases, load_documents

    aliases = load_aliases() if aliases is None else aliases
    documents = load_documents() if documents is None else documents
    if graph is None or node_id not in graph.nodes:
        return []
    below = graph.requires_closure(node_id)
    found = []
    for other in sorted(below):
        for alias in aliases.get(other, []):
            if alias not in documents:
                continue
            name = graph.nodes[other].name if other in graph.nodes else other
            found.extend(Owner(_base(alias), number, other, name)
                         for number in _numbers(documents[alias]))
    return found


def _covers(reference: str, section: str) -> bool:
    """"8" covers 8.1, and 8.1 is inside 8; 8.3 is not 8.1."""
    return (section == reference or section.startswith(reference + ".")
            or reference.startswith(section + "."))


def prompt_block(found: list[Owner], source_ids: set[str]) -> str:
    """Tell generation which referenced sections it should hand to a sheet."""
    relevant = [o for o in found if o.document in {_base(s) for s in source_ids}]
    if not relevant:
        return ""
    rows = "\n".join(f"  section {o.section} of [{o.document}] -> {o.node_id} ({o.name})"
                     for o in relevant)
    return ("\n\nSections derived on other sheets. A source here may refer to a "
            "section outside its excerpt. These sections are derived in full on "
            f"prerequisite sheets:\n{rows}\n"
            "Where the source sends the reader to one of them, do not write "
            "\"see Section N\". Say the step is derived on that sheet, name it, "
            "and list its id in the step's invokes.")


def dangling(texts: list[str], cited: set[str], found: list[Owner]) -> list[str]:
    """Section references that name neither the owning sheet's id nor its name.

    A text that says "Section 8" and also names linear_kalman_filter has
    already made the hand-off; one that says only "Section 8" sends the
    reader out of the atlas.
    """
    bases = {_base(c) for c in cited}
    relevant = [o for o in found if o.document in bases]
    out = []
    for text in texts:
        lowered = text.lower()
        for match in REFERENCE.finditer(text):
            reference = match.group(1)
            for owner in relevant:
                if not _covers(reference, owner.section):
                    continue
                if owner.node_id in lowered or owner.name.lower() in lowered:
                    break
                problem = (f"refers to section {reference} of [{owner.document}], "
                           f"which {owner.node_id} derives; name that sheet instead")
                if problem not in out:
                    out.append(problem)
                break
    return out
