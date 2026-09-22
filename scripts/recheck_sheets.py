"""Re-run the checks against sheets already on disk.

Guards change more often than sheets do, and checking costs nothing -- it is
local. Regenerating a sheet because a guard was tightened (or, as happened with
transpose_identities, because a guard was wrong) wastes a call on work that was
already correct.

Citation checking compares ids, so the stored ids would be enough on their own
-- but check() also asks source_derives() whether any retrieved article carries
a proof or derivation heading, and that reads the text. Standing in for it with
placeholder text answered "no source derives this" for every sheet on disk,
which is the branch that turns a kind mismatch into a canon note instead of a
failure. Every "expected a derivation, produced a definition" would have
flipped to ok against text that was literally "xxxx...".

So the sources are retrieved for real. It costs a fetch per sheet and no model
call. A sheet whose sources cannot be fetched is left alone rather than
rechecked against nothing -- a transport failure is not evidence about an
article.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.bootstrap import build_graph, load
from atlas.derivation import Derivation, Step, check
from atlas.retrieval import RetrievalError, retrieve_node

ROOT = Path(__file__).resolve().parent.parent
SHEETS = ROOT / "data" / "derivations"
graph = build_graph(load())
changed = still_failing = skipped = 0

for path in sorted(SHEETS.glob("*.json")):
    sheet = json.loads(path.read_text())
    if "kind" not in sheet:
        continue

    derivation = Derivation(
        node_id=sheet["node_id"],
        grounded=sheet.get("grounded", False),
        kind=sheet.get("kind", "derivation"),
        statement=sheet.get("statement", ""),
        symbols=sheet.get("symbols", []),
        steps=[Step(s.get("n", i + 1), s.get("text", ""), s.get("invokes", []),
                    s.get("cites", []))
               for i, s in enumerate(sheet.get("steps", []))],
        properties=sheet.get("properties", []),
        source_ids=sheet.get("source_ids", []),
    )
    node = graph.nodes.get(derivation.node_id)
    try:
        sources = retrieve_node(node.id, node.name) if node else []
    except RetrievalError:
        sources = []
    if not sources:
        skipped += 1
        print(f"  {derivation.node_id:<34} skipped: no source to check against")
        continue

    fresh = check(derivation, sources, graph)
    was = sheet.get("check", {}).get("ok")
    if fresh != sheet.get("check"):
        sheet["check"] = fresh
        path.write_text(json.dumps(sheet, indent=2) + "\n")
        changed += 1
        if was != fresh["ok"]:
            print(f"  {derivation.node_id:<34} ok {was} -> {fresh['ok']}")
    if not fresh["ok"]:
        still_failing += 1
        reasons = [k for k in ("shape", "ungrounded_citations", "uncited_steps",
                               "closure_holes") if fresh.get(k)]
        if fresh.get("vacuous_invocations"):
            reasons.append("vacuous_invocations")
        if not fresh["grounded"]:
            reasons.append("not grounded")
        print(f"  {derivation.node_id:<34} FAILS: {', '.join(reasons)}")

print(f"\nrewrote {changed}; {still_failing} still failing; {skipped} skipped")
