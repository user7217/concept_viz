"""Remove project notes for anything but the public reference project.

    python3 scripts/strip_private_notes.py [--dry-run]

Instantiating against a private repo writes its component names and file
paths into tracked sheets. Run this before committing; the notes are cheap to
regenerate and the repo is public.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = {"agrios_ws"}
dry = "--dry-run" in sys.argv

stripped = []
for sheet in sorted((ROOT / "data" / "derivations").glob("*.json")):
    payload = json.loads(sheet.read_text())
    note = payload.get("instantiation") or {}
    if not note.get("project") or note["project"] in PUBLIC:
        continue
    stripped.append(f"{sheet.stem} ({note['project']})")
    if not dry:
        payload.pop("instantiation", None)
        sheet.write_text(json.dumps(payload, indent=2) + "\n")

print(f"{'would strip' if dry else 'stripped'} {len(stripped)} notes")
for name in stripped:
    print("  " + name)
