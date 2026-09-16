"""Where a concept actually shows up in the project's code.

A sheet says a covariance matrix is symmetric positive semidefinite. The
project says:

    c = [0.0] * 36
    c[0]  = 0.05    # vx
    c[7]  = 0.05    # vy
    c[35] = 0.02    # vyaw

which is the same object with its indices exposed -- 0, 7 and 35 are the
diagonal of a 6x6 laid out row-major, and seeing that is worth more than
another sentence about positive semidefiniteness.

Matching is mechanical on purpose. Every span returned is a real file, real
line numbers, and a term that genuinely occurs there, so nothing here can
invent evidence the way a model asked to "find where this is used" would.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

CODE_SUFFIXES = {".py": "python", ".cpp": "cpp", ".hpp": "cpp", ".c": "cpp",
                 ".h": "cpp", ".yaml": "yaml", ".yml": "yaml", ".xml": "xml",
                 ".launch": "python", ".cmake": "cmake", ".sh": "shell"}
SKIP_DIRS = {"build", "install", "log", ".git", "__pycache__", "node_modules",
             ".venv", "venv"}

# Words that match everything and identify nothing.
GENERIC = frozenset({
    # too broad to identify anything in a robotics tree
    "matrix", "model", "function", "value", "state", "data", "type", "form",
    "rule", "space", "vector", "transform", "distribution", "variable",
    "identity", "theorem", "expansion", "linear", "general", "probability",
    # ordinary English that appears in configuration as something else:
    # "change" matched change_penalty, a Hybrid-A* cost for reversing
    # direction, and offered it as evidence of change of variables
    "change", "update", "total", "order", "point", "index", "limit", "scale",
    "offset", "delta", "angle", "penalty", "length", "count", "range",
})

CONTEXT = 5        # lines either side of a hit
MAX_SPANS = 3      # per node; more is a search result, not an explanation

# An import line names a module, never a use of the mathematics: "import
# random" is not evidence of a random variable. A licence header matched
# `basis` on "on an AS IS BASIS".
NEVER_EVIDENCE = re.compile(
    r"^\s*(?:#|//|\*|import\b|from\b\s+\S+\s+import\b|\s*<!--)")


def _is_wiring(path: str) -> bool:
    """Test scaffolding, package manifests and launch wiring carry no maths."""
    name = path.rsplit("/", 1)[-1]
    return (name.startswith("test_") or "/test/" in path
            or path.endswith(".xml") or name.endswith(".launch.py"))


@dataclass
class CodeFile:
    path: str
    language: str
    lines: list[str] = field(default_factory=list)


@dataclass
class Span:
    path: str
    language: str
    start: int             # 1-indexed line of lines[0]
    hit: int               # 1-indexed line the term occurs on
    term: str
    strength: str          # "parameter" | "name"
    lines: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"path": self.path, "language": self.language,
                "start": self.start, "hit": self.hit, "term": self.term,
                "strength": self.strength, "lines": self.lines}


def index_repo(root: Path, max_bytes: int = 200_000) -> list[CodeFile]:
    """Every source file in the project, generated build output excluded."""
    found: list[CodeFile] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        language = CODE_SUFFIXES.get(path.suffix)
        if language is None or path.stat().st_size > max_bytes:
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        found.append(CodeFile(path=str(path.relative_to(root)),
                              language=language,
                              lines=text.splitlines()))
    return found


def terms_for(name: str, instantiation: dict | None) -> list[tuple[str, str]]:
    """Search terms, strongest first.

    A parameter name from layer 2 is an exact identifier the project really
    sets, so a hit on it is evidence. A word from the node's name is weaker and
    generic words are dropped entirely -- searching for "matrix" in a robotics
    repo returns everything and means nothing.
    """
    out: list[tuple[str, str]] = []
    for knob in (instantiation or {}).get("knobs", []):
        parameter = str(knob.get("parameter", "")).strip()
        if parameter:
            out.append((parameter, "parameter"))
    for word in re.findall(r"[a-z]+", name.lower()):
        if len(word) > 4 and word not in GENERIC:
            out.append((word, "name"))
    seen: set[str] = set()
    unique = []
    for term, strength in out:
        if term.lower() in seen:
            continue
        seen.add(term.lower())
        unique.append((term, strength))
    return unique


def evidence_for(name: str, files: list[CodeFile],
                 instantiation: dict | None = None,
                 limit: int = MAX_SPANS) -> list[Span]:
    """Code spans where this concept is visibly at work."""
    wanted = terms_for(name, instantiation)
    if not wanted:
        return []
    # A bare name word is only worth searching for when the project actually
    # depends on this node. Without that gate, `joint_distribution` matched
    # joint_state_publisher and `random_variable` matched `import random` --
    # evidence that looks right and is about something else entirely.
    reaches = bool((instantiation or {}).get("components"))
    if not reaches:
        wanted = [(t, k) for t, k in wanted if k == "parameter"]
        if not wanted:
            return []

    # Executable code outranks configuration. A YAML block is already shown
    # as a knob in layer 2, so leading with it says nothing new -- while
    # odom_cov_relay.py setting c[0], c[7], c[35] shows a 6x6 covariance
    # flattened row-major, which is the indexing the sheet never spells out.
    rank = {"python": 0, "cpp": 0, "shell": 1, "cmake": 2, "yaml": 2, "xml": 3}
    strength_rank = {"parameter": 0, "name": 1}

    found: list[Span] = []
    claimed: set[tuple[str, int]] = set()
    for term, strength in wanted:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        for source in files:
            if strength == "name" and _is_wiring(source.path):
                continue
            for index, line in enumerate(source.lines):
                if not pattern.search(line) or NEVER_EVIDENCE.match(line):
                    continue
                hit = index + 1
                # One span per neighbourhood: a parameter set on three
                # consecutive lines is one piece of evidence, not three.
                if any(source.path == p and abs(hit - h) <= CONTEXT
                       for p, h in claimed):
                    continue
                claimed.add((source.path, hit))
                start = max(0, index - CONTEXT)
                end = min(len(source.lines), index + CONTEXT + 1)
                found.append(Span(
                    path=source.path, language=source.language,
                    start=start + 1, hit=hit, term=term, strength=strength,
                    lines=source.lines[start:end],
                ))
                break        # one span per file per term

    found.sort(key=lambda s: (rank.get(s.language, 4),
                              strength_rank.get(s.strength, 2), s.path))
    return found[:limit]
