"""Drill the sheets you have read, and record what you could actually do.

    python3 scripts/drill.py                          # every sheet
    python3 scripts/drill.py --node marginalization   # one
    python3 scripts/drill.py <repo> <subsystem>       # in learning-path order

Reading a sheet marks it READ, which the profile already warns "feels like
knowing". This is the only thing that produces DEMONSTRATED, and it is
deliberately stingy: every exercise a sheet offers has to come back right,
because one recalled symbol is not a reproduced derivation.

No model, no network. Hiding a line and showing it again is the whole method.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.exercise import available_count, earned_state, exercises_for
from atlas.profile import Profile, State

ROOT = Path(__file__).resolve().parent.parent
SHEETS = ROOT / "data" / "derivations"

argv = [a for a in sys.argv[1:]]
only = None
if "--node" in argv:
    only = argv[argv.index("--node") + 1]
    argv = [a for i, a in enumerate(argv)
            if i not in (argv.index("--node"), argv.index("--node") + 1)]
dry_run = "--dry-run" in argv
argv = [a for a in argv if a != "--dry-run"]
worksheet = None
if "--worksheet" in argv:
    worksheet = Path(argv[argv.index("--worksheet") + 1])
    argv = (argv[:argv.index("--worksheet")]
            + argv[argv.index("--worksheet") + 2:])
limit = 3
if "--limit" in argv:
    limit = int(argv[argv.index("--limit") + 1])
    argv = argv[:argv.index("--limit")] + argv[argv.index("--limit") + 2:]

order: list[str]
if only:
    order = [only]
elif len(argv) >= 2:
    from atlas.architecture import apply, propose
    from atlas.ingest import ingest_repo

    project = ingest_repo(Path(argv[0]))
    proposal = propose(project)
    proposal.unclassified, proposal.confirmed = [], True
    graph = apply(proposal)
    pid = "".join(c if c.isalnum() else "_"
                  for c in project.name.lower()).strip("_")
    target = f"{pid}__{argv[1].lower()}"
    if target not in graph.nodes:
        print(f"no such node: {target}")
        raise SystemExit(1)
    saved = Profile.load()
    order = graph.learning_path(target, known=saved.known(),
                               not_known=saved.not_known)
else:
    order = [f.stem for f in sorted(SHEETS.glob("*.json"))]

profile = Profile.load()
asked = right = 0
demonstrated: list[str] = []

if worksheet is not None:
    # Paper form: every question first, every answer in a block at the end,
    # so working through it cannot accidentally reveal the next answer.
    questions: list[str] = []
    answers: list[str] = []
    count = 0
    for node_id in order:
        path = SHEETS / f"{node_id}.json"
        if not path.exists():
            continue
        sheet = json.loads(path.read_text())
        found = exercises_for(sheet, limit=limit)
        if not found:
            continue
        questions.append(f"\n## {node_id}\n")
        for exercise in found:
            count += 1
            questions.append(f"\n**{count}. {exercise.question}**\n")
            if exercise.hint:
                questions.append(f"\n_{exercise.hint}_\n")
            questions.append("\n```\n")
            questions += [f"{line}\n" for line in exercise.context]
            questions.append("```\n")
            answers.append(f"\n**{count}. {node_id}**\n\n```\n"
                           f"{exercise.answer}\n```\n")
    worksheet.write_text(
        "# Derivation Atlas drill\n\n"
        f"{count} exercises. Work them on paper; answers are at the end.\n"
        "Record what you actually got with:\n\n"
        "```\npython3 scripts/drill.py <repo> <subsystem> --limit "
        f"{limit}\n```\n"
        + "".join(questions)
        + "\n\n---\n\n# Answers\n" + "".join(answers))
    print(f"wrote {count} exercises to {worksheet}")
    raise SystemExit(0)

for node_id in order:
    path = SHEETS / f"{node_id}.json"
    if not path.exists():
        continue
    sheet = json.loads(path.read_text())
    exercises = exercises_for(sheet, limit=limit)
    if not exercises:
        continue

    print(f"\n{'=' * 72}\n{node_id}\n{'=' * 72}")
    passed = 0
    for exercise in exercises:
        print(f"\n{exercise.question}")
        if exercise.hint:
            print(f"  ({exercise.hint})")
        print()
        for line in exercise.context:
            print(f"  {line}")
        try:
            input("\n[enter to reveal] ")
        except (EOFError, KeyboardInterrupt):
            print("\nstopped.")
            order = []
            break
        print(f"\n  ANSWER: {exercise.answer}\n")
        try:
            verdict = input("did you get it? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nstopped.")
            order = []
            break
        asked += 1
        if verdict == "y":
            right += 1
            passed += 1
    else:
        # Stingy on purpose, and measured against what the sheet could ask
        # rather than what --limit chose to ask.
        available = available_count(sheet)
        earned = earned_state(passed, len(exercises), available)
        if earned == "demonstrated":
            profile.mark(node_id, State.DEMONSTRATED,
                         evidence=f"{passed}/{available} masked exercises")
            demonstrated.append(node_id)
        elif earned == "read":
            profile.mark(node_id, State.READ,
                         evidence=f"{passed}/{available} masked exercises")
        continue
    break

if not dry_run:
    profile.save()
else:
    print("\n(dry run: profile not written)")
print(f"\n{right}/{asked} correct; demonstrated: "
      f"{', '.join(demonstrated) if demonstrated else 'none'}")
