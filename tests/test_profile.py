"""Tests for the profile: the personal floor a learning path is cut at."""

import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.bootstrap import build_graph, load
from atlas.profile import Entry, Profile, State, declare, dependents_of


class TestStates(unittest.TestCase):
    def setUp(self) -> None:
        self.p = Profile()

    def test_states_are_ordered_by_strength(self) -> None:
        self.p.mark("a", State.DECLARED)
        self.p.mark("b", State.READ)
        self.p.mark("c", State.DEMONSTRATED)
        self.assertEqual(self.p.known(State.DECLARED), {"a", "b", "c"})
        self.assertEqual(self.p.known(State.READ), {"b", "c"})
        self.assertEqual(self.p.known(State.DEMONSTRATED), {"c"})

    def test_marking_never_downgrades_a_stronger_claim(self) -> None:
        """Viewing a sheet you already demonstrated must not weaken the record."""
        self.p.mark("a", State.DEMONSTRATED)
        self.p.mark("a", State.READ)
        self.assertEqual(self.p.entries["a"].state, State.DEMONSTRATED)

    def test_marking_upgrades_a_weaker_claim(self) -> None:
        self.p.mark("a", State.DECLARED)
        self.p.mark("a", State.DEMONSTRATED)
        self.assertEqual(self.p.entries["a"].state, State.DEMONSTRATED)

    def test_weakest_first_puts_declared_before_demonstrated(self) -> None:
        self.p.mark("strong", State.DEMONSTRATED)
        self.p.mark("weak", State.DECLARED)
        self.assertEqual(self.p.weakest_first()[0].node_id, "weak")


class TestReopen(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = build_graph(load())
        self.p = Profile()

    def test_reopening_removes_the_claim(self) -> None:
        self.p.mark("covariance_matrix")
        self.p.reopen("covariance_matrix", self.graph)
        self.assertNotIn("covariance_matrix", self.p.entries)

    def test_reopening_makes_dependents_suspect(self) -> None:
        """Over-claiming once must not silently thin every future path."""
        self.p.mark("covariance_matrix", State.DECLARED)
        self.p.mark("gaussian_conditioning", State.DEMONSTRATED)
        touched = self.p.reopen("covariance_matrix", self.graph)
        self.assertIn("gaussian_conditioning", touched)
        self.assertTrue(self.p.entries["gaussian_conditioning"].suspect)

    def test_suspect_entries_drop_out_of_the_floor(self) -> None:
        self.p.mark("covariance_matrix")
        self.p.mark("gaussian_conditioning")
        self.p.reopen("covariance_matrix", self.graph)
        self.assertNotIn("gaussian_conditioning", self.p.known())
        self.assertIn("gaussian_conditioning", self.p.known(include_suspect=True))

    def test_unrelated_entries_are_untouched(self) -> None:
        self.p.mark("covariance_matrix")
        self.p.mark("priority_queue")
        self.p.reopen("covariance_matrix", self.graph)
        self.assertFalse(self.p.entries["priority_queue"].suspect)

    def test_remarking_clears_suspicion(self) -> None:
        self.p.mark("covariance_matrix")
        self.p.mark("gaussian_conditioning", State.DEMONSTRATED)
        self.p.reopen("covariance_matrix", self.graph)
        self.p.mark("gaussian_conditioning", State.DEMONSTRATED)
        self.assertFalse(self.p.entries["gaussian_conditioning"].suspect)

    def test_reopen_without_a_graph_does_not_propagate(self) -> None:
        self.p.mark("covariance_matrix")
        self.assertEqual(self.p.reopen("covariance_matrix"), ["covariance_matrix"])


class TestDependents(unittest.TestCase):
    def test_dependents_are_the_reverse_of_prerequisites(self) -> None:
        graph = build_graph(load())
        self.assertIn("gaussian_conditioning", dependents_of(graph, "covariance_matrix"))
        self.assertNotIn("covariance_matrix",
                         dependents_of(graph, "gaussian_conditioning"))


class TestPathIntegration(unittest.TestCase):
    def test_profile_shortens_a_learning_path(self) -> None:
        graph = build_graph(load())
        full = graph.learning_path("extended_kalman_filter")
        p = Profile()
        for node_id in full[:5]:
            p.mark(node_id)
        cut = graph.learning_path("extended_kalman_filter", known=p.known())
        self.assertEqual(len(cut), len(full) - 5)


class TestDeclare(unittest.TestCase):
    def test_names_resolve_to_canon_ids(self) -> None:
        graph = build_graph(load())
        p = Profile()
        matched, unmatched = declare(
            p, graph, ["Eigendecomposition", "covariance matrix", "quantum chromodynamics"]
        )
        self.assertIn("eigendecomposition", matched)
        self.assertIn("covariance_matrix", matched)
        self.assertEqual(unmatched, ["quantum chromodynamics"])

    def test_declared_entries_are_the_weakest_state(self) -> None:
        graph = build_graph(load())
        p = Profile()
        declare(p, graph, ["Eigendecomposition"])
        self.assertEqual(p.entries["eigendecomposition"].state, State.DECLARED)


class TestStorage(unittest.TestCase):
    def test_round_trips(self) -> None:
        p = Profile()
        p.mark("a", State.DEMONSTRATED, evidence="masked step")
        p.entries["a"].suspect = True
        restored = Profile.from_json(p.to_json())
        self.assertEqual(restored.entries["a"].state, State.DEMONSTRATED)
        self.assertTrue(restored.entries["a"].suspect)
        self.assertEqual(restored.entries["a"].evidence, "masked step")

    def test_missing_file_loads_empty_rather_than_raising(self) -> None:
        """A first run has no profile; that is not an error."""
        self.assertEqual(Profile.load(Path("/nonexistent/profile.json")).entries, {})

    def test_saves_and_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            p = Profile()
            p.mark("a")
            p.save(path)
            self.assertEqual(Profile.load(path).known(), {"a"})

    def test_example_template_parses(self) -> None:
        example = Path(__file__).resolve().parent.parent / "profile.example.json"
        self.assertTrue(Profile.from_json(example.read_text()).entries)


class TestStale(unittest.TestCase):
    def test_old_entries_are_reported(self) -> None:
        p = Profile()
        p.mark("old")
        p.entries["old"].at = (
            datetime.now(timezone.utc) - timedelta(days=400)
        ).isoformat(timespec="seconds")
        p.mark("fresh")
        self.assertEqual([e.node_id for e in p.stale(180)], ["old"])


class TestFloorOverride(unittest.TestCase):
    """The assumed tier is a default for silence, not an override of a claim."""

    def setUp(self) -> None:
        self.graph = build_graph(load())
        self.p = Profile()

    def test_assumed_nodes_are_hidden_by_default(self) -> None:
        path = self.graph.learning_path("extended_kalman_filter")
        self.assertNotIn("partial_derivative", path)

    def test_denying_an_assumed_node_surfaces_it(self) -> None:
        self.p.mark_unknown("partial_derivative")
        path = self.graph.learning_path(
            "extended_kalman_filter", known=self.p.known(),
            not_known=self.p.not_known)
        self.assertIn("partial_derivative", path)

    def test_denial_removes_a_known_claim(self) -> None:
        self.p.mark("expectation")
        self.p.mark_unknown("expectation")
        self.assertNotIn("expectation", self.p.known())
        self.assertIn("expectation", self.p.not_known)

    def test_marking_known_clears_a_denial(self) -> None:
        self.p.mark_unknown("expectation")
        self.p.mark("expectation")
        self.assertIn("expectation", self.p.known())
        self.assertEqual(self.p.not_known, set())

    def test_denials_survive_a_round_trip(self) -> None:
        self.p.mark_unknown("matrix_inverse")
        self.assertEqual(Profile.from_json(self.p.to_json()).not_known,
                         {"matrix_inverse"})
