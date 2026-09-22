"""Generate sheets for named canon nodes, whatever their grounding state.

`generate_path.py` covers a project's path and `generate_all.py` covers
everything the grounding sweep called groundable. Neither can reach a node the
sweep never rated -- the assumed tier is excluded from it by construction --
and neither can be pointed at one sheet you want rebuilt.

    python3 scripts/generate_nodes.py matrix_inverse expectation
    python3 scripts/generate_nodes.py --tier assumed
    python3 scripts/generate_nodes.py --failing        # sheets whose check is not ok

Existing sheets are replaced, not skipped: naming a node is the request.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.bootstrap import build_graph, load
from atlas.derivation import candidate_vocabulary, check, generate
from atlas.llm import (AuthFailure, Dropped, LLMError, QuotaExhausted,
                       get_provider, get_rotating_provider)
from atlas.retrieval import RetrievalError, ground_from_consumers, retrieve_node

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "derivations"
OUT.mkdir(parents=True, exist_ok=True)

graph = build_graph(load())
args = sys.argv[1:]
delay = 2.0

if not args:
    print(__doc__)
    raise SystemExit(1)

if args[0] == "--tier":
    wanted = args[1]
    targets = sorted(n.id for n in graph.nodes.values()
                     if n.tier.value == wanted
                     and not (OUT / f"{n.id}.json").exists())
elif args[0] == "--failing":
    targets = []
    for path in sorted(OUT.glob("*.json")):
        payload = json.loads(path.read_text())
        if not payload.get("check", {}).get("ok", True):
            targets.append(path.stem)
else:
    targets = args
    unknown = [t for t in targets if t not in graph.nodes]
    if unknown:
        print(f"not canon nodes: {', '.join(unknown)}")
        raise SystemExit(1)

if not targets:
    print("nothing to do")
    raise SystemExit(0)

provider = get_provider()
if provider.name == "gemini":
    provider = get_rotating_provider()
print(f"{len(targets)} nodes | {provider.name}\n", flush=True)

wrote = refused = failed = 0
for index, node_id in enumerate(targets, 1):
    node = graph.nodes[node_id]
    try:
        sources = retrieve_node(node.id, node.name)
        if not sources:
            sources = ground_from_consumers(graph, node.id, node.name)
    except RetrievalError as exc:
        failed += 1
        print(f"  {index}/{len(targets)} {node_id}: retrieval failed", flush=True)
        continue
    if not sources:
        failed += 1
        print(f"  {index}/{len(targets)} {node_id}: no source", flush=True)
        continue

    try:
        derivation = generate(provider, node.id, node.name, sources,
                              vocabulary=candidate_vocabulary(graph, node.id),
                              graph=graph)
    except (AuthFailure, QuotaExhausted, Dropped) as exc:
        print(f"\n  stopped at {index}/{len(targets)}: {str(exc)[:90]}", flush=True)
        break
    except LLMError as exc:
        failed += 1
        print(f"  {index}/{len(targets)} {node_id}: LLM error {str(exc)[:60]}",
              flush=True)
        time.sleep(delay * 3)
        continue

    report = check(derivation, sources, graph)
    payload = json.loads(derivation.to_json())
    payload["check"] = report
    (OUT / f"{node_id}.json").write_text(json.dumps(payload, indent=2) + "\n")

    if derivation.grounded:
        wrote += 1
        body = (f"{len(derivation.steps)} steps" if not derivation.is_definition
                else f"definition, {len(derivation.properties)} properties")
        print(f"  {index}/{len(targets)} {node_id}: {body}"
              f"{' OK' if report['ok'] else ' CHECK FAILED'}", flush=True)
    else:
        refused += 1
        print(f"  {index}/{len(targets)} {node_id}: refused", flush=True)
    time.sleep(delay)

print(f"\nwrote {wrote}  refused {refused}  failed {failed}")
