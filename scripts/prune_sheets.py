"""Remove derivation sheets that a later fix invalidated.

Regenerating everything wastes calls on sheets that are still good, so this
removes only the ones a change actually broke:

  no "kind"                 predates definition/derivation branching, so it was
                            asked to derive whatever it was
  definition, no properties an empty definition sheet
  definition that is a      a technique is a move and must be performed; those
    technique node          were stated instead
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.bootstrap import build_graph, load
from atlas.schema import NodeType

ROOT = Path(__file__).resolve().parent.parent
SHEETS = ROOT / "data" / "derivations"
dry = "--dry-run" in sys.argv

graph = build_graph(load())
removed, kept = [], 0

for path in sorted(SHEETS.glob("*.json")):
    try:
        sheet = json.loads(path.read_text())
    except json.JSONDecodeError:
        removed.append((path, "unreadable"))
        continue

    kind = sheet.get("kind")
    node_id = sheet.get("node_id", path.stem)
    node = graph.nodes.get(node_id)

    if kind is None:
        reason = "predates kind branching"
    elif kind == "definition" and not sheet.get("properties"):
        reason = "definition with no properties"
    elif kind == "definition" and node is not None and node.type is NodeType.TECHNIQUE:
        reason = "technique stated instead of performed"
    elif sheet.get("check", {}).get("vacuous_invocations"):
        reason = "every step invokes only floor-level trivia"
    elif kind == "derivation" and sheet.get("steps") and all(
        not [i for i in st.get("invokes", []) if i not in ("algebraic_rearrangement",
                                                           "index_notation")]
        for st in sheet["steps"]):
        reason = "no substantive invocations, closure check was vacuous"
    else:
        kept += 1
        continue
    removed.append((path, reason))

for path, reason in removed:
    print(f"  remove {path.name:<40} {reason}")
    if not dry:
        path.unlink()

print(f"\n{'would remove' if dry else 'removed'} {len(removed)}, kept {kept}")
if dry:
    print("rerun without --dry-run to apply")
