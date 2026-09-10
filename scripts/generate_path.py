"""Generate derivation sheets for one project path.

Scoped deliberately. Generating the whole canon was an artefact of building it
eagerly; what a reader needs is the nodes their own path touches, minus what
they already know.

    python3 scripts/generate_path.py <repo> [subsystem] [delay]

Resumable: one file per node, existing ones skipped.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.architecture import apply, propose
from atlas.derivation import candidate_vocabulary, check, generate
from atlas.ingest import ingest_repo
from atlas.llm import LLMError, QuotaExhausted, get_provider, get_rotating_provider
from atlas.profile import Profile
from atlas.retrieval import RetrievalError, ground_from_consumers, retrieve_node

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "derivations"
OUT.mkdir(parents=True, exist_ok=True)

repo = Path(sys.argv[1])
want = sys.argv[2] if len(sys.argv) > 2 else None
delay = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0

proposal = propose(ingest_repo(repo))
if proposal.unclassified:
    print(f"note: {len(proposal.unclassified)} components unclassified, dropped: "
          f"{', '.join(proposal.unclassified[:6])}...")
    proposal.unclassified = []
proposal.confirmed = True
graph = apply(proposal)
profile = Profile.load()

system = "".join(c if c.isalnum() else "_" for c in proposal.project.lower()).strip("_")
target = f"{system}__{want.lower()}" if want else system
if target not in graph.nodes:
    print(f"no such node: {target}")
    print("try: " + ", ".join(sorted(
        n.id for n in graph.nodes.values() if n.id.startswith(system))))
    raise SystemExit(1)

path = graph.learning_path(target, known=profile.known(),
                           not_known=profile.not_known)

try:
    provider = get_provider()
    if provider.name == "gemini":
        provider = get_rotating_provider()
except LLMError as exc:
    print(f"no provider: {exc}")
    raise SystemExit(1)

print(f"{target}: {len(path)} nodes to generate "
      f"({len(profile.known())} already known)\n", flush=True)

grounded = refused = skipped = 0
for index, node_id in enumerate(path, 1):
    out = OUT / f"{node_id}.json"
    if out.exists():
        skipped += 1
        continue

    node = graph.nodes[node_id]
    try:
        sources = retrieve_node(node.id, node.name)
        if not sources:
            sources = ground_from_consumers(graph, node.id, node.name)
    except RetrievalError:
        print(f"  {index}/{len(path)} {node_id}: retrieval failed", flush=True)
        continue

    try:
        derivation = generate(provider, node.id, node.name, sources,
                              vocabulary=candidate_vocabulary(graph, node.id),
                              graph=graph)
    except QuotaExhausted as exc:
        print(f"\n  stopped at {index}/{len(path)}: {str(exc)[:80]}")
        print("  rerun to resume -- finished nodes are skipped")
        break
    except LLMError as exc:
        print(f"  {index}/{len(path)} {node_id}: {str(exc)[:60]}", flush=True)
        continue

    report = check(derivation, sources, graph)
    payload = json.loads(derivation.to_json())
    payload["check"] = report
    out.write_text(json.dumps(payload, indent=2) + "\n")

    if derivation.grounded:
        grounded += 1
        print(f"  {index}/{len(path)} {node_id}: {len(derivation.steps)} steps"
              f"{'' if report['ok'] else '  CHECK FAILED'}", flush=True)
    else:
        refused += 1
        print(f"  {index}/{len(path)} {node_id}: refused (source has no derivation)",
              flush=True)
    time.sleep(delay)

done = grounded + refused
print(f"\ngenerated {grounded}  refused {refused}  already had {skipped}")
if done:
    print(f"derivable: {100*grounded/done:.0f}% of what was attempted")
print(f"sheets in {OUT.relative_to(ROOT)}/")
