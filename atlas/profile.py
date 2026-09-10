"""What the reader already knows.

The floor that stops expansion is personal, not domain-specific: a path ends
where your knowledge starts. This is the one artifact in the system that cannot
be regenerated -- the canon can be rebuilt from seeds and derivations
re-retrieved, but what someone has actually worked through exists nowhere else.

Entries are per node, never per topic. "I know linear algebra" is unanswerable;
`eigendecomposition` is not, and the canon has already done the decomposition.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROFILE = ROOT / "profile.json"
EXAMPLE = ROOT / "profile.example.json"


class State(str, Enum):
    """How the claim is backed. These are worth very different amounts."""

    DECLARED = "declared"          # you said so at setup; people over-claim
    READ = "read"                  # you viewed the sheet; feels like knowing
    DEMONSTRATED = "demonstrated"  # you did the masked step


ORDER = {State.DECLARED: 0, State.READ: 1, State.DEMONSTRATED: 2}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Entry:
    node_id: str
    state: State
    at: str = field(default_factory=_now)
    evidence: str = ""
    suspect: bool = False  # a prerequisite was reopened beneath it

    @property
    def strength(self) -> int:
        return ORDER[State(self.state)]


@dataclass
class Profile:
    entries: dict[str, Entry] = field(default_factory=dict)
    denied: dict[str, str] = field(default_factory=dict)  # node -> when denied

    @property
    def not_known(self) -> set[str]:
        """Nodes explicitly denied, which override the canon's assumed tier.

        Without this the floor is a claim the system makes on the reader's
        behalf: assumed nodes vanish from every path whether or not the reader
        has ever seen them.
        """
        return set(self.denied)

    def mark_unknown(self, node_id: str) -> None:
        """Record not knowing something, even if the canon assumes it."""
        self.entries.pop(node_id, None)
        self.denied[node_id] = _now()

    # ---------- reading ----------

    def known(self, minimum: State = State.DECLARED,
              include_suspect: bool = False) -> set[str]:
        """The floor to cut a learning path at.

        Raising `minimum` is how you ask a harder question: with
        DEMONSTRATED, only nodes you have actually executed count as known.
        """
        floor = ORDER[minimum]
        return {
            node_id for node_id, entry in self.entries.items()
            if entry.strength >= floor and (include_suspect or not entry.suspect)
        }

    def weakest_first(self) -> list[Entry]:
        """Entries most likely to be wrong, first.

        When a derivation confuses you, the declared claims are where the
        profile is least trustworthy -- start debugging there.
        """
        return sorted(self.entries.values(), key=lambda e: (e.strength, e.at))

    def stale(self, days: int = 180) -> list[Entry]:
        """Entries untouched for a while. Recorded, not decayed automatically."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        out = []
        for entry in self.entries.values():
            try:
                when = datetime.fromisoformat(entry.at)
            except ValueError:
                continue
            if when < cutoff:
                out.append(entry)
        return sorted(out, key=lambda e: e.at)

    # ---------- writing ----------

    def mark(self, node_id: str, state: State = State.DEMONSTRATED,
             evidence: str = "") -> Entry:
        """Record knowing a node. Never downgrades an existing stronger claim."""
        self.denied.pop(node_id, None)
        existing = self.entries.get(node_id)
        if existing is not None and existing.strength > ORDER[state]:
            existing.suspect = False
            return existing
        entry = Entry(node_id=node_id, state=state, evidence=evidence)
        self.entries[node_id] = entry
        return entry

    def reopen(self, node_id: str, graph=None) -> list[str]:
        """Withdraw a claim, and flag everything that rested on it.

        Over-claiming once would otherwise silently remove things from every
        future path with no signal. Reopening a prerequisite makes its
        dependents suspect rather than deleting them: you may well still know
        them, but the profile can no longer vouch for the reason.
        """
        touched: list[str] = []
        if node_id in self.entries:
            del self.entries[node_id]
            touched.append(node_id)
        if graph is None:
            return touched

        for dependent in dependents_of(graph, node_id):
            entry = self.entries.get(dependent)
            if entry is not None and not entry.suspect:
                entry.suspect = True
                touched.append(dependent)
        return touched

    # ---------- storage ----------

    def to_json(self) -> str:
        return json.dumps(
            {
                "_note": "What you already know, per canon node. Not regenerable "
                         "-- keep a copy somewhere that survives this machine.",
                "entries": [asdict(e) for e in sorted(
                    self.entries.values(), key=lambda e: e.node_id)],
                "denied": self.denied,
            },
            indent=2,
        ) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "Profile":
        payload = json.loads(text)
        entries = {}
        for raw in payload.get("entries", []):
            entry = Entry(
                node_id=raw["node_id"],
                state=State(raw.get("state", "declared")),
                at=raw.get("at", _now()),
                evidence=raw.get("evidence", ""),
                suspect=bool(raw.get("suspect", False)),
            )
            entries[entry.node_id] = entry
        return cls(entries=entries, denied=payload.get("denied", {}))

    @classmethod
    def load(cls, path: Path = PROFILE) -> "Profile":
        """Missing profile is an empty one, not an error: a first run has none."""
        if not Path(path).exists():
            return cls()
        return cls.from_json(Path(path).read_text())

    def save(self, path: Path = PROFILE) -> None:
        Path(path).write_text(self.to_json())


def dependents_of(graph, node_id: str) -> set[str]:
    """Everything that transitively requires this node.

    The reverse of requires_closure: not what it needs, but what needed it.
    """
    from .schema import Relation

    reverse: dict[str, list[str]] = {}
    for edge in graph.edges.values():
        if edge.rel is Relation.REQUIRES:
            reverse.setdefault(edge.dst, []).append(edge.src)

    seen: set[str] = set()
    queue = list(reverse.get(node_id, []))
    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        queue.extend(reverse.get(current, []))
    return seen


def declare(profile: Profile, graph, names: list[str]) -> tuple[list[str], list[str]]:
    """Seed a profile from coarse claims, resolving names to canon ids."""
    by_name = {n.name.lower(): n.id for n in graph.nodes.values()}
    matched, unmatched = [], []
    for name in names:
        key = name.strip().lower()
        slug = graph.resolve(key.replace(" ", "_").replace("-", "_"))
        node_id = slug if slug in graph.nodes else by_name.get(key)
        if node_id:
            profile.mark(node_id, State.DECLARED, evidence=f"declared as {name!r}")
            matched.append(node_id)
        else:
            unmatched.append(name)
    return matched, unmatched
