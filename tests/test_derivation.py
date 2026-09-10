"""Tests for derivation generation and the checks the reader cannot perform."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.bootstrap import build_graph, load
from atlas.derivation import (
    Derivation,
    Step,
    build_prompt,
    candidate_vocabulary,
    check,
    generate,
    resolve_invoked,
    select_text,
)
from atlas.retrieval import Source


class FakeProvider:
    name = "fake"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.prompts.append(prompt)
        return self.reply


GOOD = """{"grounded": true, "statement": "b = (X'X)^-1 X'y",
  "symbols": [{"symbol": "X", "meaning": "design matrix", "dimensions": "n x p"}],
  "steps": [{"n": 1, "text": "minimise the sum of squares",
             "invokes": ["quadratic_form"], "cites": ["wikipedia:OLS"]}],
  "assumptions": [], "failure_modes": []}"""


def source(text: str = "x" * 2000) -> Source:
    return Source("wikipedia:OLS", "OLS", "u", "encyclopedia", text)


class TestSelectText(unittest.TestCase):
    LONG = (
        "Intro paragraph about the topic.\nSecond intro paragraph.\n"
        + "filler with nothing useful\n" * 150
        + "Derivation. We minimise S the sum of squares.\n"
        "Setting the gradient to zero gives X^T X b = X^T y.\n"
        "Hence b = (X^T X)^-1 X^T y as required.\n"
    )

    def test_short_text_is_untouched(self) -> None:
        self.assertEqual(select_text("short", 500), "short")

    def test_derivation_survives_truncation(self) -> None:
        """Front-truncation would cut exactly this."""
        kept = select_text(self.LONG, 700)
        self.assertIn("Derivation. We minimise", kept)

    def test_window_is_contiguous(self) -> None:
        """Scattered paragraphs read as no derivation; the argument must hold."""
        kept = select_text(self.LONG, 700)
        self.assertIn("gradient to zero", kept)
        self.assertIn("as required", kept)

    def test_respects_the_budget(self) -> None:
        self.assertLessEqual(len(select_text(self.LONG, 700)), 900)


class TestPrompt(unittest.TestCase):
    def test_vocabulary_is_offered_when_supplied(self) -> None:
        prompt = build_prompt("ols", "OLS", [source()], vocabulary=["quadratic_form"])
        self.assertIn("quadratic_form", prompt)

    def test_source_ids_are_shown_for_citation(self) -> None:
        self.assertIn("[wikipedia:OLS]", build_prompt("ols", "OLS", [source()]))

    def test_no_sources_skips_the_call_entirely(self) -> None:
        provider = FakeProvider(GOOD)
        result = generate(provider, "ols", "OLS", [])
        self.assertFalse(result.grounded)
        self.assertEqual(provider.prompts, [])


class TestChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = build_graph(load())
        self.sources = [source()]

    def test_a_clean_derivation_passes(self) -> None:
        result = generate(FakeProvider(GOOD), "ordinary_least_squares", "OLS", self.sources)
        report = check(result, self.sources, self.graph)
        self.assertTrue(report["ok"])
        self.assertEqual(report["ungrounded_citations"], [])

    def test_citation_from_memory_is_caught(self) -> None:
        reply = GOOD.replace('"wikipedia:OLS"', '"Rao, Linear Models, 1973"')
        result = generate(FakeProvider(reply), "ordinary_least_squares", "OLS", self.sources)
        report = check(result, self.sources, self.graph)
        self.assertEqual(report["ungrounded_citations"], ["Rao, Linear Models, 1973"])
        self.assertFalse(report["ok"])

    def test_uncited_step_is_caught(self) -> None:
        reply = GOOD.replace('"cites": ["wikipedia:OLS"]', '"cites": []')
        result = generate(FakeProvider(reply), "ordinary_least_squares", "OLS", self.sources)
        self.assertEqual(check(result, self.sources, self.graph)["uncited_steps"], [1])

    def test_refusal_is_reported_not_treated_as_success(self) -> None:
        reply = '{"grounded": false, "statement": "no derivation here", "steps": []}'
        result = generate(FakeProvider(reply), "ordinary_least_squares", "OLS", self.sources)
        report = check(result, self.sources, self.graph)
        self.assertFalse(report["ok"])
        self.assertFalse(report["grounded"])

    def test_floor_nodes_are_not_closure_holes(self) -> None:
        """Reaching `derivative` is hitting the bottom, not finding a gap."""
        result = Derivation(
            node_id="ordinary_least_squares", grounded=True,
            steps=[Step(1, "t", ["derivative"], ["wikipedia:OLS"])],
        )
        self.assertEqual(check(result, self.sources, self.graph)["closure_holes"], [])


class TestResolveInvoked(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = build_graph(load())

    def test_canon_id_resolves(self) -> None:
        resolved, unknown = resolve_invoked(self.graph, {"quadratic_form"})
        self.assertEqual(resolved, {"quadratic_form"})
        self.assertEqual(unknown, set())

    def test_display_name_resolves(self) -> None:
        resolved, _ = resolve_invoked(self.graph, {"Jacobian Matrix"})
        self.assertEqual(resolved, {"jacobian_matrix"})

    def test_merged_id_resolves_through_its_alias(self) -> None:
        resolved, _ = resolve_invoked(self.graph, {"logit"})
        self.assertEqual(resolved, {"log_odds"})

    def test_prose_becomes_a_candidate_not_an_error(self) -> None:
        _, unknown = resolve_invoked(self.graph, {"Sum of squared residuals definition"})
        self.assertEqual(unknown, {"Sum of squared residuals definition"})


class TestVocabulary(unittest.TestCase):
    def test_vocabulary_is_drawn_from_the_canon(self) -> None:
        graph = build_graph(load())
        vocab = candidate_vocabulary(graph, "extended_kalman_filter")
        self.assertTrue(vocab)
        self.assertTrue(all(v in graph.nodes for v in vocab))


if __name__ == "__main__":
    unittest.main(verbosity=2)
