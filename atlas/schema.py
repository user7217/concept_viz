"""Core types for the Derivation Atlas knowledge graph.

Two layers meet at algorithm nodes:
  - architecture (project-local): SYSTEM / SUBSYSTEM / COMPONENT
  - derivation   (global):        CONCEPT / TECHNIQUE

ALGORITHM nodes are boundary nodes: they have project-local parents via
CONTAINS/USES and global prerequisites via REQUIRES.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class NodeType(str, Enum):
    SYSTEM = "system"
    SUBSYSTEM = "subsystem"
    COMPONENT = "component"
    ALGORITHM = "algorithm"
    CONCEPT = "concept"
    TECHNIQUE = "technique"
    FAILURE_MODE = "failure_mode"


class Tier(str, Enum):
    """How deeply a node must be understood. See design doc section 2."""

    DERIVATION = "derivation"  # reproduce from blank paper
    STATEMENT = "statement"    # know the claim, conditions, failure modes
    CLAIM = "claim"            # know the claim and its implication only
    ASSUMED = "assumed"        # below the reader's floor; not rendered


class Relation(str, Enum):
    CONTAINS = "contains"        # architecture composition, project-local
    USES = "uses"                # architecture -> algorithm; the seam
    REQUIRES = "requires"        # prerequisite; strictly acyclic
    RELATED_TO = "related_to"    # weak; carries no learning semantics


PROJECT_TYPES = frozenset({NodeType.SYSTEM, NodeType.SUBSYSTEM, NodeType.COMPONENT})
GLOBAL_TYPES = frozenset(
    {NodeType.ALGORITHM, NodeType.CONCEPT, NodeType.TECHNIQUE, NodeType.FAILURE_MODE}
)


@dataclass(frozen=True)
class Node:
    id: str
    type: NodeType
    name: str
    tier: Tier = Tier.STATEMENT
    canonical: bool = False   # False => provisional, awaiting curation
    verified: bool = False    # derivation human-checked
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not ID_PATTERN.match(self.id):
            raise ValueError(f"invalid node id: {self.id!r} (want lower_snake_case)")
        if not self.name:
            raise ValueError(f"node {self.id} has no name")

    @property
    def is_global(self) -> bool:
        return self.type in GLOBAL_TYPES


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    rel: Relation
    invoked_at: tuple[str, ...] = ()  # derivation steps that use dst

    def key(self) -> tuple[str, str, Relation]:
        return (self.src, self.dst, self.rel)


@dataclass
class Seed:
    """One algorithm used to bootstrap the canon (design doc section 5)."""

    id: str
    name: str
    domain: str
    invokes: list[tuple[str, NodeType]] = field(default_factory=list)
