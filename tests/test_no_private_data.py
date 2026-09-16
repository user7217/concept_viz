"""Nothing about a private codebase may reach a tracked file.

Briefs are gitignored, but a project note is written *inside* a sheet, and
sheets are tracked. Instantiating against a private repo therefore writes its
component names and file paths into files that get committed, and no gitignore
can prevent that -- it is part of a file, not a file.

Tests gate every commit in this repo, so this is the check that can actually
stop it. PUBLIC lists the projects whose details are fine to publish; add to
it deliberately, never to make a failure go away.
"""

import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = {"agrios_ws"}          # github.com/user7217/agrios_ws


def _tracked(path: Path) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))],
        cwd=ROOT, capture_output=True, text=True, check=False)
    return result.returncode == 0


class TestNoPrivateProjectData(unittest.TestCase):
    def test_no_tracked_sheet_carries_a_private_project_note(self) -> None:
        offenders = []
        for sheet in sorted((ROOT / "data" / "derivations").glob("*.json")):
            note = json.loads(sheet.read_text()).get("instantiation") or {}
            project = note.get("project")
            if project and project not in PUBLIC and _tracked(sheet):
                offenders.append(f"{sheet.name} -> {project}")
        self.assertEqual(
            offenders, [],
            "tracked sheets carry notes about a project not in PUBLIC. "
            "Either add the project to PUBLIC deliberately, or strip the "
            "notes with scripts/strip_private_notes.py before committing:\n  "
            + "\n  ".join(offenders))

    def test_no_tracked_component_description_is_private(self) -> None:
        path = ROOT / "data" / "components.json"
        if not path.exists() or not _tracked(path):
            self.skipTest("no tracked components.json")
        bad = sorted({body.get("project") for body
                      in json.loads(path.read_text())["components"].values()
                      if body.get("project") and body["project"] not in PUBLIC})
        self.assertEqual(bad, [], f"components.json describes {bad}")

    def test_briefs_are_not_tracked(self) -> None:
        briefs = ROOT / "data" / "briefs"
        tracked = [p.name for p in briefs.glob("*.json") if _tracked(p)] \
            if briefs.exists() else []
        self.assertEqual(tracked, [],
                         "a brief describes a private repo's layout and must "
                         "stay untracked")


if __name__ == "__main__":
    unittest.main()
