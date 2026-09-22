"""recheck_sheets.py has to check against the real articles.

It once stood the sources in as `Source(..., "x" * 2000)`, which was harmless
while check() only compared citation ids. Then check() started asking
source_derives() whether any article carries a proof heading, and placeholder
text answers "no" for every sheet at once -- the exact branch that downgrades a
kind mismatch from a failure to a canon note. Running it would have marked
every "expected a derivation, produced a definition" sheet ok against text that
was literally the letter x.

This is the codebase's recurring failure mode: a guard compared against
something fabricated instead of something independent. The test is static
because the script does its work at import time, and the invariant is about
what it compares against, not what it computes.
"""

import ast
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "recheck_sheets.py"


class RecheckUsesRealSources(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = ast.parse(SCRIPT.read_text())
        self.text = SCRIPT.read_text()

    def test_it_retrieves_rather_than_fabricating(self) -> None:
        called = {
            node.func.id
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("retrieve_node", called,
                      "recheck must fetch the article it judges a sheet against")

    def test_no_source_is_built_from_invented_text(self) -> None:
        for node in ast.walk(self.tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if node.func.id != "Source":
                continue
            self.fail("recheck constructs a Source itself; the text of a "
                      "fabricated source is what source_derives reads, and it "
                      "answers for every sheet at once")

    def test_a_sheet_it_cannot_fetch_is_left_alone(self) -> None:
        self.assertIn("skipped", self.text,
                      "a transport failure is not evidence about an article, "
                      "so an unfetchable sheet must be skipped, not rechecked "
                      "against nothing")


if __name__ == "__main__":
    unittest.main()
