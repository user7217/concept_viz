"""Discover what mathematics a codebase implements.

The library map only sees dependencies, which works for integration code -- a
ROS2 project *imports* robot_localization and the EKF arrives with it. Research
code is the opposite: the maths is written in the repo. V1 imports only numpy,
scipy and matplotlib while implementing Fisher information, natural gradient
and Tikhonov regularisation itself.

So concepts are read out of the source, not inferred from the manifest.

Three layers, keyed separately so caching works:
  1  concept store   global, grows monotonically, shared across projects
  2  project links   why a concept is here + which module evidences it
  3  project scope   discovered concepts plus their prerequisite closure
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .llm import Provider, complete_json

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache"}
MAX_UNITS = 120
MAX_DOC_CHARS = 300

SYSTEM = """You identify the mathematics a codebase implements.

Report only what the code gives evidence for. A module named fisher.py defining
a class that builds J^T W J is evidence for Fisher information. A docstring
mentioning a library that is never imported is not evidence of anything.

Name concepts as they are known in the literature ("Fisher information matrix",
"natural gradient"), not as the repo names them.

Distinguish:
  algorithm  a named procedure the code runs
  concept    a mathematical object or result it relies on
  technique  a manipulation move it performs

Reply with JSON only."""

TEMPLATE = """Repository: {name}
Imported third-party libraries: {libraries}

Code units (path :: kind :: name :: docstring):
{units}

Identify the mathematics this codebase implements.

Return JSON:
{{
  "concepts": [
    {{"name": "Fisher information matrix",
      "kind": "concept",
      "evidence": ["relative/path.py"],
      "note": "one clause on what it does here"}}
  ]
}}

Cite as evidence only paths from the list above. Omit anything the code does
not support, however likely it seems for a project of this kind."""


@dataclass
class CodeUnit:
    path: str
    kind: str  # module | class | function
    name: str
    doc: str = ""

    def render(self) -> str:
        doc = " ".join(self.doc.split())[:MAX_DOC_CHARS]
        return f"{self.path} :: {self.kind} :: {self.name} :: {doc}"


@dataclass
class ScanScope:
    """What was actually looked at.

    A discovery claim is only valid inside the scope that produced it. Scanning
    part of a repo and reporting "this project does not use CAMB" is the same
    error as treating a rate-limited request as a missing article -- absence of
    evidence recorded as evidence of absence.
    """

    root: str
    files_scanned: int
    files_skipped: int
    units_kept: int
    units_total: int
    skipped_dirs: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.files_skipped == 0 and self.units_kept == self.units_total

    def caveat(self) -> str:
        if self.complete:
            return ""
        return (
            f"partial scan: {self.units_kept}/{self.units_total} code units from "
            f"{self.files_scanned} files"
            + (f", skipped {', '.join(self.skipped_dirs)}" if self.skipped_dirs else "")
        )


@dataclass
class Discovery:
    name: str
    kind: str = "concept"
    evidence: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def slug(self) -> str:
        """A canon-compatible id.

        Accents must fold to ASCII: str.isalnum() accepts "é", but the node
        schema does not, so "Cramer-Rao" arriving as "Cramér-Rao" would produce
        an id that fails validation on insert.
        """
        import unicodedata

        folded = unicodedata.normalize("NFKD", self.name.lower())
        folded = folded.encode("ascii", "ignore").decode()
        keep = [c if c.isalnum() else "_" for c in folded]
        return "_".join(w for w in "".join(keep).split("_") if w)


def scan_repo(root: Path, limit: int = MAX_UNITS,
              exclude: set[str] | None = None) -> tuple[list[CodeUnit], set[str], ScanScope]:
    """Walk Python sources for module/class/function names and docstrings.

    Parsed, never imported: importing a research repo runs whatever is at module
    scope and needs its dependencies present.
    """
    root = Path(root)
    exclude = exclude or set()
    units: list[CodeUnit] = []
    libraries: set[str] = set()
    scanned = skipped = 0

    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if any(part in exclude for part in path.parts):
            skipped += 1
            continue
        scanned += 1
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except (SyntaxError, OSError):
            continue

        rel = str(path.relative_to(root))
        units.append(CodeUnit(rel, "module", path.stem, ast.get_docstring(tree) or ""))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                libraries.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                libraries.add(node.module.split(".")[0])
            elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                if not node.name.startswith("_"):
                    units.append(
                        CodeUnit(rel, kind, node.name, ast.get_docstring(node) or "")
                    )

    stdlib = {"os", "sys", "json", "math", "typing", "dataclasses", "pathlib",
              "logging", "itertools", "collections", "abc", "re", "time",
              "functools", "warnings", "urllib", "gzip", "unittest", "argparse",
              "csv", "importlib", "traceback", "random", "copy", "enum", "io"}
    # A repo's own packages import as bare names too. Reporting them as
    # third-party would tell the model this project depends on itself.
    local = {p.stem for p in root.iterdir()} | {
        Path(u.path).parts[0].removesuffix(".py") for u in units
    }
    scope = ScanScope(
        root=str(root), files_scanned=scanned, files_skipped=skipped,
        units_kept=min(len(units), limit), units_total=len(units),
        skipped_dirs=sorted(exclude),
    )
    return units[:limit], {
        lib for lib in libraries if lib not in stdlib and lib not in local
    }, scope


def build_prompt(name: str, units: list[CodeUnit], libraries: set[str]) -> str:
    return TEMPLATE.format(
        name=name,
        libraries=", ".join(sorted(libraries)) or "(none)",
        units="\n".join(u.render() for u in units),
    )


def discover(provider: Provider, root: Path, name: str | None = None) -> list[Discovery]:
    """Read a repository and report the mathematics it implements."""
    root = Path(root)
    units, libraries, _ = scan_repo(root)
    if not units:
        return []

    payload = complete_json(
        provider, build_prompt(name or root.name, units, libraries), system=SYSTEM
    )
    return [
        Discovery(
            name=str(item.get("name", "")).strip(),
            kind=str(item.get("kind", "concept")),
            evidence=[str(e) for e in item.get("evidence", [])],
            note=str(item.get("note", "")),
        )
        for item in payload.get("concepts", [])
        if str(item.get("name", "")).strip()
    ]


# ---------- the check: claimed evidence must exist ----------

def unsupported(discoveries: list[Discovery], units: list[CodeUnit]) -> dict[str, list[str]]:
    """Discoveries citing a path that is not in the repo.

    The same discipline as citations: a concept attributed to a file nobody
    scanned is the model reasoning from what a project like this usually
    contains, which is exactly what must not reach the map.
    """
    known = {u.path for u in units}
    problems: dict[str, list[str]] = {}
    for item in discoveries:
        missing = [e for e in item.evidence if e not in known]
        if missing or not item.evidence:
            problems[item.name] = missing or ["(no evidence cited)"]
    return problems


def file_hashes(root: Path, units: list[CodeUnit]) -> dict[str, str]:
    """Content hashes for layer-2 cache invalidation."""
    root = Path(root)
    hashes: dict[str, str] = {}
    for path in {u.path for u in units}:
        full = root / path
        if full.exists():
            hashes[path] = hashlib.sha256(full.read_bytes()).hexdigest()[:16]
    return hashes


@dataclass
class ProjectCache:
    """Layers 2 and 3 for one project."""

    project: str
    links: list[Discovery] = field(default_factory=list)   # layer 2
    scope: list[str] = field(default_factory=list)         # layer 3
    hashes: dict[str, str] = field(default_factory=dict)

    def stale(self, current: dict[str, str]) -> list[str]:
        """Files whose content moved since discovery ran."""
        return sorted(
            path for path, digest in current.items()
            if self.hashes.get(path) != digest
        ) + sorted(set(self.hashes) - set(current))

    def to_json(self) -> str:
        return json.dumps(
            {
                "project": self.project,
                "links": [asdict(link) for link in self.links],
                "scope": self.scope,
                "hashes": self.hashes,
            },
            indent=2,
        ) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "ProjectCache":
        payload = json.loads(text)
        return cls(
            project=payload["project"],
            links=[Discovery(**link) for link in payload.get("links", [])],
            scope=payload.get("scope", []),
            hashes=payload.get("hashes", {}),
        )


def build_scope(graph, discoveries: list[Discovery], resolve) -> list[str]:
    """Layer 3: discovered concepts plus everything they require.

    Strictly larger than layer 2. Fisher information has code evidence;
    eigendecomposition is on the path because Fisher needs it, and no line of
    code implements it.
    """
    scope: set[str] = set()
    for item in discoveries:
        node_id = resolve(item)
        if node_id is None:
            continue
        scope.add(node_id)
        scope |= graph.requires_closure(node_id)
    return sorted(scope)
