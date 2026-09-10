"""Tests for concept discovery from source."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.discover import (
    Discovery,
    ProjectCache,
    build_prompt,
    discover,
    file_hashes,
    scan_repo,
    unsupported,
)

MODULE = '''"""Fisher information utilities."""
import numpy as np
from scipy import linalg
from mypkg.helpers import thing

class FisherMatrix:
    """Build F = J^T W J and forecast constraints."""
    def forecast(self):
        """One-sigma errors from inv(F)."""

def _private_helper():
    """Should not be reported."""
'''


class TestScan(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "mypkg").mkdir()
        (self.root / "mypkg" / "fisher.py").write_text(MODULE)
        (self.root / "__pycache__").mkdir()
        (self.root / "__pycache__" / "junk.py").write_text("x = 1")
        (self.root / "skipme").mkdir()
        (self.root / "skipme" / "other.py").write_text("import torch")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_units_are_found(self) -> None:
        units, _, _ = scan_repo(self.root)
        names = {(u.kind, u.name) for u in units}
        self.assertIn(("class", "FisherMatrix"), names)
        self.assertIn(("function", "forecast"), names)

    def test_private_names_are_skipped(self) -> None:
        units, _, _ = scan_repo(self.root)
        self.assertNotIn("_private_helper", {u.name for u in units})

    def test_cache_dirs_are_skipped(self) -> None:
        units, _, _ = scan_repo(self.root)
        self.assertFalse(any("__pycache__" in u.path for u in units))

    def test_third_party_libraries_only(self) -> None:
        """A repo's own packages must not be reported as dependencies."""
        _, libs, _ = scan_repo(self.root)
        self.assertIn("numpy", libs)
        self.assertIn("scipy", libs)
        self.assertNotIn("mypkg", libs)

    def test_excluded_directory_is_not_scanned(self) -> None:
        _, libs, scope = scan_repo(self.root, exclude={"skipme"})
        self.assertNotIn("torch", libs)
        self.assertEqual(scope.files_skipped, 1)

    def test_scope_records_an_incomplete_scan(self) -> None:
        """'Not found' is only meaningful inside the scope that looked."""
        _, _, scope = scan_repo(self.root, exclude={"skipme"})
        self.assertFalse(scope.complete)
        self.assertIn("skipme", scope.caveat())

    def test_scope_is_complete_when_nothing_is_excluded(self) -> None:
        _, _, scope = scan_repo(self.root)
        self.assertTrue(scope.complete)
        self.assertEqual(scope.caveat(), "")

    def test_unit_limit_marks_the_scan_partial(self) -> None:
        _, _, scope = scan_repo(self.root, limit=1)
        self.assertFalse(scope.complete)
        self.assertIn("1/", scope.caveat())

    def test_docstrings_reach_the_prompt(self) -> None:
        units, libs, _ = scan_repo(self.root)
        prompt = build_prompt("demo", units, libs)
        self.assertIn("Fisher information utilities", prompt)
        self.assertIn("numpy", prompt)


class TestUnsupported(unittest.TestCase):
    def setUp(self) -> None:
        self.units, _, _ = scan_repo(Path(__file__).resolve().parent.parent / "atlas")

    def test_evidence_outside_the_repo_is_caught(self) -> None:
        """A concept attributed to a file nobody scanned is inference, not evidence."""
        bad = Discovery("Fisher information", evidence=["nonexistent/fisher.py"])
        self.assertIn("Fisher information", unsupported([bad], self.units))

    def test_missing_evidence_is_caught(self) -> None:
        self.assertIn("Kalman", unsupported([Discovery("Kalman")], self.units))

    def test_real_evidence_passes(self) -> None:
        good = Discovery("Graph traversal", evidence=[self.units[0].path])
        self.assertEqual(unsupported([good], self.units), {})


class TestSlug(unittest.TestCase):
    def test_name_becomes_a_canon_style_id(self) -> None:
        self.assertEqual(Discovery("Fisher Information Matrix").slug,
                         "fisher_information_matrix")
        # accents must fold, or the id fails the node schema on insert
        self.assertEqual(Discovery("Cramér-Rao bound").slug, "cramer_rao_bound")

    def test_slug_is_accepted_by_the_node_schema(self) -> None:
        from atlas.schema import Node, NodeType

        for name in ("Cramér-Rao bound", "Fisher Information Matrix", "A* search"):
            node = Node(Discovery(name).slug, NodeType.CONCEPT, name)
            self.assertTrue(node.id)


class TestProjectCache(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "a.py").write_text("x = 1")
        (self.root / "b.py").write_text("y = 2")
        units, _, _ = scan_repo(self.root)
        self.hashes = file_hashes(self.root, units)
        self.cache = ProjectCache("demo", hashes=dict(self.hashes))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_unchanged_files_are_not_stale(self) -> None:
        self.assertEqual(self.cache.stale(self.hashes), [])

    def test_edited_file_is_stale(self) -> None:
        (self.root / "a.py").write_text("x = 99")
        units, _, _ = scan_repo(self.root)
        self.assertEqual(self.cache.stale(file_hashes(self.root, units)), ["a.py"])

    def test_deleted_file_is_stale(self) -> None:
        (self.root / "b.py").unlink()
        units, _, _ = scan_repo(self.root)
        self.assertIn("b.py", self.cache.stale(file_hashes(self.root, units)))

    def test_round_trips_through_json(self) -> None:
        self.cache.links = [Discovery("Fisher", evidence=["a.py"], note="n")]
        self.cache.scope = ["fisher", "eigenvalue"]
        restored = ProjectCache.from_json(self.cache.to_json())
        self.assertEqual(restored.scope, ["fisher", "eigenvalue"])
        self.assertEqual(restored.links[0].name, "Fisher")


class TestScope(unittest.TestCase):
    def test_scope_includes_prerequisites_without_code_evidence(self) -> None:
        """Layer 3 is strictly larger than layer 2."""
        from atlas.bootstrap import build_graph, load
        from atlas.discover import build_scope

        graph = build_graph(load())
        found = [Discovery("Principal Component Analysis", evidence=["pca.py"])]
        scope = build_scope(graph, found, lambda d: "pca")
        self.assertIn("pca", scope)
        self.assertIn("eigendecomposition", scope)
        self.assertGreater(len(scope), len(found))


if __name__ == "__main__":
    unittest.main(verbosity=2)
