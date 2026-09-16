"""What each running component is for. Writes data/components.json.

    python3 scripts/describe_components.py <repo>

Grounded entirely in stage 1's extraction, and every identifier the model
names is checked back against it. A note that invents one is stored with the
problem attached rather than silently trusted.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.architecture import apply, propose
from atlas.component import describe, invented
from atlas.ingest import ingest_repo
from atlas.llm import AuthFailure, Dropped, LLMError, QuotaExhausted, get_provider
from atlas.schema import Relation

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "components.json"

project = ingest_repo(Path(sys.argv[1]))
proposal = propose(project)
proposal.unclassified, proposal.confirmed = [], True
graph = apply(proposal)
slug = "".join(c if c.isalnum() else "_" for c in project.name.lower()).strip("_")

uses = {}
for edge in graph.edges.values():
    if edge.rel is Relation.USES and edge.src.startswith(slug):
        uses.setdefault(edge.src[len(slug) + 2:], []).append(edge.dst)

stored = json.loads(OUT.read_text())["components"] if OUT.exists() else {}
wanted = [c for c in project.components if c.name.lower() in
          {k.lower() for k in uses} and c.name not in stored]
print(f"{len(wanted)} components to describe "
      f"({len(stored)} already done)\n", flush=True)

provider = get_provider()
for index, component in enumerate(wanted, 1):
    algorithms = next((v for k, v in uses.items()
                       if k.lower() == component.name.lower()), [])
    try:
        note = describe(provider, project, component, algorithms)
    except (AuthFailure, QuotaExhausted, Dropped) as exc:
        print(f"\n  stopped at {index}/{len(wanted)}: {exc}")
        break
    except LLMError as exc:
        print(f"  {index}/{len(wanted)} {component.name}: {str(exc)[:70]}", flush=True)
        continue

    problems = invented(note, project, component, algorithms)
    stored[component.name] = {**note.to_dict(), "algorithms": algorithms,
                              "invented": problems}
    flag = f"  INVENTED: {problems}" if problems else ""
    print(f"  {index}/{len(wanted)} {component.name}: "
          f"{len(note.evidence)} cited{flag}", flush=True)
    OUT.write_text(json.dumps({"components": stored}, indent=2) + "\n")

print(f"\nwrote {OUT.name}: {len(stored)} components")
