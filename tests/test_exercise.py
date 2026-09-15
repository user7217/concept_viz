import unittest

from atlas.exercise import (Exercise, exercises_for, maskable_steps,
                            maskable_symbols, step_exercise, symbol_exercise,
                            unique_content, unique_symbol_content)

SHEET = {
    "node_id": "marginalization",
    "statement": "The marginal recovers one variable from a joint distribution.",
    "steps": [
        {"n": 1, "text": "Start from the joint distribution of X and Y.",
         "invokes": []},
        {"n": 2, "text": "Integrate the joint density over every value Y takes,"
                         " discarding Y's identity.",
         "invokes": ["algebraic_rearrangement"]},
        {"n": 3, "text": "Substitute the conditional factorisation p(x,y) ="
                         " p(x|y)p(y) to reach a conditional expectation.",
         "invokes": ["conditional_probability", "expectation"]},
    ],
    "symbols": [
        {"symbol": "p", "meaning": "a probability density", "dimensions": ""},
        {"symbol": "Y", "meaning": "the nuisance variable eliminated",
         "dimensions": "scalar"},
    ],
    "assumptions": [],
    "failure_modes": [],
}


class TestVacuity(unittest.TestCase):
    def test_a_step_restated_elsewhere_contributes_nothing(self) -> None:
        # The guard's independent comparison: not "is this step well formed"
        # but "can the reader read it off the page around it".
        sheet = {
            "node_id": "x",
            "statement": "Integrate the joint density over every value Y takes.",
            "steps": [
                {"n": 1, "text": "Integrate the joint density over every value"
                                 " Y takes.", "invokes": []},
            ],
            "symbols": [], "assumptions": [], "failure_modes": [],
        }
        self.assertEqual(unique_content(sheet, 0), set())
        self.assertEqual(maskable_steps(sheet), [])

    def test_a_step_with_its_own_content_is_maskable(self) -> None:
        self.assertIn(2, maskable_steps(SHEET))
        self.assertTrue(unique_content(SHEET, 2))

    def test_hardest_step_comes_first(self) -> None:
        order = maskable_steps(SHEET)
        scores = [len(unique_content(SHEET, i)) for i in order]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_out_of_range_is_empty_not_an_error(self) -> None:
        self.assertEqual(unique_content(SHEET, 99), set())
        self.assertEqual(unique_symbol_content(SHEET, 99), set())


class TestStepExercise(unittest.TestCase):
    def test_the_masked_step_is_hidden_and_the_rest_shown(self) -> None:
        ex = step_exercise(SHEET, 1)
        self.assertEqual(ex.kind, "step")
        self.assertIn("[2] ???", ex.context)
        self.assertIn("Start from the joint", ex.context[0])
        # the answer must not leak into the visible context
        self.assertNotIn(ex.answer, ex.context)

    def test_the_hint_names_the_move_without_performing_it(self) -> None:
        ex = step_exercise(SHEET, 2)
        self.assertIn("conditional_probability", ex.hint)
        self.assertNotIn("Substitute", ex.hint)

    def test_a_step_invoking_nothing_gets_no_hint(self) -> None:
        self.assertEqual(step_exercise(SHEET, 0).hint, "")


class TestSymbolExercise(unittest.TestCase):
    def test_asks_only_what_the_sheet_can_answer(self) -> None:
        ex = symbol_exercise(SHEET, 1)
        self.assertEqual(ex.question, "What is Y?")
        # not "what changes if you change it" -- the sheet has no such field
        self.assertNotIn("changes", ex.question)
        self.assertEqual(ex.answer, "the nuisance variable eliminated")

    def test_dimensions_become_the_hint(self) -> None:
        self.assertIn("scalar", symbol_exercise(SHEET, 1).hint)


class TestSelection(unittest.TestCase):
    def test_steps_are_preferred_over_symbols(self) -> None:
        kinds = [e.kind for e in exercises_for(SHEET, limit=2)]
        self.assertEqual(kinds, ["step", "step"])

    def test_a_stepless_sheet_still_yields_symbol_exercises(self) -> None:
        stepless = dict(SHEET, steps=[])
        out = exercises_for(stepless, limit=3)
        self.assertTrue(out)
        self.assertTrue(all(e.kind == "symbol" for e in out))

    def test_a_sheet_with_nothing_to_recall_yields_nothing(self) -> None:
        empty = {"node_id": "x", "statement": "", "steps": [], "symbols": [],
                 "assumptions": [], "failure_modes": []}
        self.assertEqual(exercises_for(empty), [])


if __name__ == "__main__":
    unittest.main()


class TestEarnedState(unittest.TestCase):
    def test_a_display_limit_cannot_buy_demonstrated(self) -> None:
        from atlas.exercise import earned_state
        # 1 of 1 shown, but the sheet could have asked 5
        self.assertEqual(earned_state(passed=1, presented=1, available=5), "read")

    def test_answering_everything_the_sheet_can_ask_is_demonstrated(self) -> None:
        from atlas.exercise import earned_state
        self.assertEqual(earned_state(passed=5, presented=5, available=5),
                         "demonstrated")

    def test_a_partial_pass_is_reading(self) -> None:
        from atlas.exercise import earned_state
        self.assertEqual(earned_state(passed=3, presented=5, available=5), "read")

    def test_getting_nothing_earns_nothing(self) -> None:
        from atlas.exercise import earned_state
        self.assertIsNone(earned_state(passed=0, presented=5, available=5))

    def test_available_count_ignores_the_limit(self) -> None:
        from atlas.exercise import available_count
        self.assertEqual(available_count(SHEET),
                         len(maskable_steps(SHEET)) + len(maskable_symbols(SHEET)))


class TestSymbolRanking(unittest.TestCase):
    def test_a_term_in_the_equation_outranks_a_rarer_aside(self) -> None:
        # F_s is a filtration: it appears nowhere else, so rarity alone ranked
        # it first, surfacing measure theory the project excludes on purpose.
        sheet = {
            "node_id": "markov_assumption",
            "statement": "P(X_t | X_s) does not depend on history before s.",
            "steps": [],
            "symbols": [
                {"symbol": "F_s", "meaning": "filtration sigma-algebra"
                                             " encoding information up to s"},
                {"symbol": "X_t", "meaning": "the process state random"
                                             " variable at index t"},
            ],
            "assumptions": [], "failure_modes": [],
        }
        order = maskable_symbols(sheet)
        self.assertEqual(order[0], 1, "X_t appears in the statement, F_s does not")


class TestForwardLeaks(unittest.TestCase):
    LEAKY = {
        "node_id": "marginalization",
        "statement": "The marginal recovers one variable.",
        "steps": [
            {"n": 1, "text": "Start from the joint distribution.", "invokes": []},
            {"n": 2, "text": "Substitute p(x,y) = p(x|y)p(y) to reach a"
                             " conditional expectation form.",
             "invokes": ["expectation"]},
            {"n": 3, "text": "As a special case, Step 2 becomes"
                             " p_X(x) = int delta(x-g(y)) p_Y(y) dy.",
             "invokes": ["dirac_delta"]},
        ],
        "symbols": [], "assumptions": [], "failure_modes": [],
    }

    def test_a_step_naming_the_masked_one_is_found(self) -> None:
        from atlas.exercise import forward_leaks
        self.assertEqual(forward_leaks(self.LEAKY, 1), [2])

    def test_only_later_steps_count(self) -> None:
        from atlas.exercise import forward_leaks
        # step 3 is last: nothing can leak it
        self.assertEqual(forward_leaks(self.LEAKY, 2), [])

    def test_the_leaking_step_is_hidden_too(self) -> None:
        ex = step_exercise(self.LEAKY, 1)
        self.assertEqual(ex.masked, [1, 2])
        self.assertIn("[2] ???", ex.context)
        self.assertIn("[3] ???", ex.context)
        # and the answer covers both, so the reader is asked for what is hidden
        self.assertIn("conditional expectation", ex.answer)
        self.assertIn("special case", ex.answer)

    def test_the_question_says_why_two_are_hidden(self) -> None:
        ex = step_exercise(self.LEAKY, 1)
        self.assertIn("2", ex.question)
        self.assertIn("3", ex.question)
        self.assertIn("restates", ex.question)

    def test_hints_merge_without_duplicates(self) -> None:
        ex = step_exercise(self.LEAKY, 1)
        self.assertIn("expectation", ex.hint)
        self.assertIn("dirac_delta", ex.hint)
        self.assertEqual(ex.hint.count("expectation"), 1)

    def test_a_clean_step_still_masks_only_itself(self) -> None:
        ex = step_exercise(SHEET, 1)
        self.assertEqual(ex.masked, [1])
        self.assertEqual(ex.question, "Reproduce step 2.")
