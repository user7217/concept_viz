"""Turn what the sheets reported into canon curation work.

A derivation that invokes something the canon cannot reach is evidence about
the CANON, not a defect in the sheet -- failing the sheet would force a
regeneration that fixes nothing, and would say nothing about whether the rest
of it is sound. So the signal is collected here instead.

Convergence is what makes a signal trustworthy: a name invoked by one sheet is
usually a model artefact, one invoked by several is a node the canon is
missing. That is the same test that made the round-2 canon flags worth acting
on -- rank and cross_product were each flagged independently by two clusters.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.bootstrap import build_graph, load

ROOT = Path(__file__).resolve().parent.parent
graph = build_graph(load())

unknown: dict[str, list[str]] = defaultdict(list)   # invented name -> sheets
holes: dict[str, list[str]] = defaultdict(list)     # unreachable node -> sheets
sheets = 0

for path in sorted((ROOT / "data" / "derivations").glob("*.json")):
    sheet = json.loads(path.read_text())
    check = sheet.get("check", {})
    if not check:
        continue
    sheets += 1
    node_id = sheet.get("node_id", path.stem)
    for name in check.get("unknown_invocations", []):
        unknown[name].append(node_id)
    for name in check.get("closure_holes", []):
        holes[name].append(node_id)

print(f"scanned {sheets} sheets\n")

print("MISSING EDGES -- a sheet reached a canon node the graph cannot get to")
if not holes:
    print("  none")
for name, users in sorted(holes.items(), key=lambda kv: (-len(kv[1]), kv[0])):
    mark = "  <- corroborated" if len(users) > 1 else ""
    print(f"  {name:<32} needed by {', '.join(users)}{mark}")

print("\nCANDIDATE NODES -- named by a derivation, absent from the canon")
if not unknown:
    print("  none")
for name, users in sorted(unknown.items(), key=lambda kv: (-len(kv[1]), kv[0])):
    mark = "  <- corroborated" if len(users) > 1 else "  (single sheet: likely an artefact)"
    print(f"  {name:<40} from {', '.join(users)}{mark}")

corroborated = sum(1 for u in unknown.values() if len(u) > 1)
print(f"\n{len(holes)} missing edges, {len(unknown)} candidate nodes "
      f"({corroborated} corroborated by more than one sheet)")
