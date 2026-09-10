"""Generate a derivation for every groundable canon node, and count what lands.

Coverage measured in grounding.json is "a source exists", which overstates
derivability: the Wikipedia EKF article is long, substantive, and contains no
derivation. This measures the number that matters -- how many nodes a grounded
derivation can actually be written for.

Resumable: results are written per node, and existing ones are skipped.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.bootstrap import build_graph, load
from atlas.derivation import candidate_vocabulary, check, generate
from atlas.llm import (AuthFailure, Dropped, LLMError, QuotaExhausted,
                       get_rotating_provider)
from atlas.retrieval import RetrievalError, ground_from_consumers, retrieve_node

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "derivations"
OUT.mkdir(parents=True, exist_ok=True)
DELAY = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0

graph = build_graph(load())
grounding = {r["id"]: r for r in json.loads((ROOT / "data" / "grounding.json").read_text())["rows"]}
targets = [nid for nid, row in grounding.items() if row["state"] == "grounded"]
targets.sort()

provider = get_rotating_provider()
print(f"{len(targets)} groundable nodes | rotation "
      f"{[p.model for p in provider.providers]} | {DELAY}s between calls\n", flush=True)

done = grounded = refused = failed = 0
for index, node_id in enumerate(targets, 1):
    path = OUT / f"{node_id}.json"
    if path.exists():
        done += 1
        grounded += json.loads(path.read_text()).get("grounded", False)
        continue

    node = graph.nodes[node_id]
    try:
        sources = retrieve_node(node.id, node.name)
        if not sources:
            sources = ground_from_consumers(graph, node.id, node.name)
    except RetrievalError as exc:
        failed += 1
        print(f"  {index}/{len(targets)} {node_id}: retrieval failed", flush=True)
        continue

    try:
        derivation = generate(provider, node.id, node.name, sources,
                              vocabulary=candidate_vocabulary(graph, node.id),
                              graph=graph)
    except AuthFailure as exc:
        print(f"\n  stopped at {index}: {exc}")
        break
    except (QuotaExhausted, Dropped) as exc:
        # Every model's daily bucket is spent, or the CLI is dropping
        # requests outright. Stop rather than burn the run producing
        # failures; the results written so far are resumable.
        print(f"\n  stopped at {index}/{len(targets)}: {str(exc)[:90]}", flush=True)
        break
    except LLMError as exc:
        failed += 1
        print(f"  {index}/{len(targets)} {node_id}: LLM error {str(exc)[:60]}", flush=True)
        time.sleep(DELAY * 3)
        continue

    report = check(derivation, sources, graph)
    payload = json.loads(derivation.to_json())
    payload["check"] = report
    path.write_text(json.dumps(payload, indent=2) + "\n")

    done += 1
    if derivation.grounded:
        grounded += 1
        body = (f"{len(derivation.steps)} steps" if not derivation.is_definition
                else f"definition, {len(derivation.properties)} properties")
        print(f"  {index}/{len(targets)} {node_id}: {body}"
              f"{' OK' if report['ok'] else ' CHECK FAILED'}", flush=True)
    else:
        refused += 1
        print(f"  {index}/{len(targets)} {node_id}: refused", flush=True)
    time.sleep(DELAY)

print(f"\nattempted {done}  grounded {grounded}  refused {refused}  failed {failed}")
if done:
    print(f"derivable: {100*grounded/done:.0f}% of nodes with a source")
