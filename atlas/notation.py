"""One notation per family of sheets, and a check that a sheet keeps it.

Each sheet restates one source, and sources disagree about letters. The
linear Kalman filter came from a derivation in the Probabilistic Robotics
convention -- R for process noise, Q for measurement noise -- while the EKF
sheet beside it, Wikipedia and robot_localization use the reverse. Every
symbol was right on its own sheet; a reader moving between them could not
tell which Q was which. All four acceptance readers named it.

The convention lives in data/notation.json. Generation is told to use it,
and the check compares a sheet's symbol table against it -- the file, not
the sheet, is the independent thing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

NOTATION = Path(__file__).resolve().parent.parent / "data" / "notation.json"


@dataclass(frozen=True)
class Convention:
    family: str
    roots: tuple[str, ...]
    symbols: tuple[dict, ...]


def load_conventions(path: Path = NOTATION) -> list[Convention]:
    if not path.exists():
        return []
    return [Convention(c["family"], tuple(c["roots"]), tuple(c["symbols"]))
            for c in json.loads(path.read_text())["conventions"]]


def convention_for(graph, node_id: str,
                   conventions: list[Convention] | None = None) -> Convention | None:
    """The convention whose roots are this node or require it."""
    for convention in load_conventions() if conventions is None else conventions:
        for root in convention.roots:
            if node_id == root or (graph is not None and root in graph.nodes
                                   and node_id in graph.requires_closure(root)):
                return convention
    return None


def prompt_block(convention: Convention | None) -> str:
    if convention is None:
        return ""
    rows = "\n".join(f"  {s['symbol']} = {s['means']}" for s in convention.symbols)
    return ("\n\nNotation. These sheets share one convention for "
            f"{convention.family}. Wherever these quantities appear, use these "
            f"symbols, even where the source uses others:\n{rows}\n"
            "If the source writes a quantity with a different letter, rename it "
            "consistently and say so once in the statement, e.g. \"(the source "
            "writes R for process noise)\".")


def _letter(symbol: str) -> str:
    """The bare letter of a symbol: \\mathbf{R}_t, R_k and R all give R."""
    plain = re.sub(r"\\(mathbf|boldsymbol|bm)\{(\w+)\}", r"\2", str(symbol)).strip()
    found = re.match(r"([A-Za-z])(?![A-Za-z])", plain)
    return found.group(1) if found else ""


def conflicts(symbols: list[dict], convention: Convention | None) -> list[str]:
    """Symbols that use a convention's letter for a different quantity.

    Only a meaning the convention recognises counts: R as "remainder of
    order k" is a different object, not a swapped covariance. Recognition
    asks for covariance, noise and its kind together -- F's meaning says
    "motion model", and keyword matching called it process noise.
    """
    if convention is None:
        return []
    owner = {s["symbol"]: s for s in convention.symbols}
    found = []
    for row in symbols:
        if not isinstance(row, dict):
            continue  # a bare symbol carries no meaning to contradict
        letter = _letter(row.get("symbol", ""))
        meaning = str(row.get("meaning", "")).lower()
        for other in convention.symbols:
            if other["symbol"] == letter or letter not in owner:
                continue
            pattern = other.get("recognise") or ""
            if pattern and re.search(pattern, meaning):
                found.append(f"{row.get('symbol')} is {other['means']} here; "
                             f"this family writes that {other['symbol']}")
                break
    return found
