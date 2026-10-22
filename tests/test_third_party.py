"""Other people's code and text travel with what their licences require.

Library excerpts are redistributed source: BSD asks that they keep the
copyright notice, the conditions and the disclaimer of the file they came
from. A hand-written credit line had already dropped one of the two copyright
holders in src/ekf.cpp, so the notice is now read from each file and these
tests refuse an excerpt that arrives without it.
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LIBRARY = json.loads((ROOT / "data" / "library_code.json").read_text())
NOTICES_DOC = (ROOT / "THIRD_PARTY_NOTICES.md").read_text()
TEMPLATE = (ROOT / "viz" / "canon_map_template.html").read_text()

CONDITION = "Redistributions of source code must retain the above copyright"
DISCLAIMER = 'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS'


class LibraryExcerpts(unittest.TestCase):
    def paths(self):
        return {s["path"] for spans in LIBRARY["spans"].values() for s in spans}

    def test_every_excerpted_file_has_its_notice(self):
        missing = self.paths() - set(LIBRARY.get("notices", {}))
        self.assertEqual(missing, set())

    def test_each_notice_keeps_holder_conditions_and_disclaimer(self):
        for path, notice in LIBRARY["notices"].items():
            with self.subTest(path=path):
                self.assertIn("Copyright", notice)
                self.assertIn(CONDITION, " ".join(notice.split()))
                self.assertIn(DISCLAIMER, " ".join(notice.split()))

    def test_every_holder_named_in_a_notice_is_credited(self):
        """The credit line once named Charles River Analytics and left out
        Locus Robotics, whose copyright is on src/ekf.cpp."""
        for notice in LIBRARY["notices"].values():
            for line in notice.splitlines():
                if line.startswith("Copyright"):
                    holder = " ".join(line.split())
                    self.assertIn(holder, LIBRARY["source"]["copyright"])

    def test_the_notices_document_carries_every_holder(self):
        flat = " ".join(NOTICES_DOC.split())
        for notice in LIBRARY["notices"].values():
            for line in notice.splitlines():
                if line.startswith("Copyright"):
                    self.assertIn(" ".join(line.split()), flat)

    def test_the_excerpts_are_pinned_to_a_commit(self):
        self.assertRegex(LIBRARY["source"].get("commit", ""), r"^[0-9a-f]{40}$")


class PublishedMap(unittest.TestCase):
    def test_the_map_renders_each_files_licence(self):
        self.assertIn("DATA.library.notices", TEMPLATE)

    def test_the_map_attributes_restated_wikipedia_text(self):
        self.assertIn("d-attrib", TEMPLATE)
        self.assertIn("creativecommons.org/licenses/by-sa/4.0", TEMPLATE)


class ToolsAreCredited(unittest.TestCase):
    def test_graphify_is_linked_with_its_licence(self):
        self.assertIn("https://github.com/Graphify-Labs/graphify", NOTICES_DOC)
        self.assertIn("Apache-2.0", NOTICES_DOC)


if __name__ == "__main__":
    unittest.main()
