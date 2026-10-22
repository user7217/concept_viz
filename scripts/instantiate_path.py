"""Attach project-specific notes to the sheets on one path.

    python3 scripts/instantiate_path.py <repo> [subsystem]

Needs no retrieval and sends a fraction of a derivation prompt, so it is cheap.
Resumable: a sheet that already carries an instantiation is skipped.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.architecture import apply, propose
from atlas.brief import resolve
from atlas.instantiate import instantiate, invented, thin, unsupported
from atlas.topics import endpoints, links
from atlas.llm import (AuthFailure, Dropped, LLMError, QuotaExhausted,
                       get_provider)
from atlas.profile import Profile

ROOT = Path(__file__).resolve().parent.parent
SHEETS = ROOT / "data" / "derivations"

args = [a for a in sys.argv[1:] if not a.startswith("--")]
repo = Path(args[0])
want = args[1] if len(args) > 1 else None
# --redo=a,b annotates these again even though they carry a note.
redo = {n for a in sys.argv[1:] if a.startswith("--redo=")
        for n in a.split("=", 1)[1].split(",") if n}

project, proposal, dropped = resolve(repo)
graph = apply(proposal)
profile = Profile.load()

project_id = "".join(c if c.isalnum() else "_" for c in project.name.lower()).strip("_")
target = f"{project_id}__{want.lower()}" if want else project_id
if target not in graph.nodes:
    print(f"no such node: {target}")
    raise SystemExit(1)

path = graph.learning_path(target, known=profile.known(),
                           not_known=profile.not_known)
provider = get_provider()
pending = [n for n in path if (SHEETS / f"{n}.json").exists()
           and (n in redo or
                "instantiation" not in json.loads((SHEETS / f"{n}.json").read_text()))]
topic_links = links(endpoints(project, repo))
print(f"{target}: {len(pending)} sheets to annotate\n", flush=True)

done = skipped = 0
for index, node_id in enumerate(pending, 1):
    path_json = SHEETS / f"{node_id}.json"
    sheet = json.loads(path_json.read_text())
    try:
        note = instantiate(provider, project, project_id, graph, node_id, sheet,
                           repo=repo, links=topic_links)
    except (AuthFailure, QuotaExhausted, Dropped) as exc:
        print(f"\n  stopped at {index}/{len(pending)}: {exc}")
        break
    except LLMError as exc:
        print(f"  {index}/{len(pending)} {node_id}: {str(exc)[:60]}", flush=True)
        continue

    if not note.components:
        skipped += 1
        continue

    problems = invented(note, project, sheet)
    weak = thin(note, sheet)
    claims = unsupported(note)
    sheet["instantiation"] = json.loads(note.to_json())
    sheet["instantiation"]["invented"] = problems
    sheet["instantiation"]["thin"] = weak
    sheet["instantiation"]["unsupported"] = claims
    path_json.write_text(json.dumps(sheet, indent=2) + "\n")
    done += 1
    flag = f"  INVENTED: {problems}" if problems else ""
    flag += f"  THIN: {weak}" if weak else ""
    flag += f"  UNSUPPORTED: {len(claims)} claim(s)" if claims else ""
    print(f"  {index}/{len(pending)} {node_id}: {len(note.knobs)} knobs, "
          f"{len(note.concrete)} symbols{flag}", flush=True)

print(f"\nannotated {done}, no components reach {skipped}")
