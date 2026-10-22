"""Choosing a model for Graphify, and the commands a build runs.

Nothing here runs Graphify or a model: `which`, the environment and the
subprocess runner are all passed in.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas import graphify_build
from atlas.graphify_build import Choice, GraphifyUnavailable, choose_backend, commands


def which_of(*present):
    return lambda name: f"/usr/bin/{name}" if name in present else None


class ChoosingABackend(unittest.TestCase):
    def test_an_explicit_backend_wins_over_everything(self):
        choice = choose_backend({"ATLAS_GRAPHIFY_BACKEND": "ollama",
                                 "ATLAS_PROVIDER": "claude_code"},
                                which=which_of("claude"))
        self.assertEqual(choice.backend, "ollama")

    def test_claude_code_maps_to_graphifys_claude_cli(self):
        choice = choose_backend({"ATLAS_PROVIDER": "claude_code"},
                                which=which_of("claude"))
        self.assertEqual(choice, Choice("claude-cli", "ATLAS_PROVIDER=claude_code"))

    def test_gemini_maps_to_gemini_when_the_key_is_set(self):
        choice = choose_backend({"ATLAS_PROVIDER": "gemini", "GEMINI_API_KEY": "k"},
                                which=which_of())
        self.assertEqual(choice.backend, "gemini")

    def test_present_is_not_working_so_every_model_is_a_candidate(self):
        """The claude CLI was on PATH with an expired session. A Gemini key
        in .env must still get its turn."""
        from atlas.graphify_build import candidates
        found = candidates({"ATLAS_PROVIDER": "antigravity", "GEMINI_API_KEY": "k"},
                           which=which_of("claude"))
        self.assertEqual([c.backend for c in found], ["claude-cli", "gemini"])

    def test_the_configured_provider_is_tried_first(self):
        from atlas.graphify_build import candidates
        found = candidates({"ATLAS_PROVIDER": "gemini", "GEMINI_API_KEY": "k"},
                           which=which_of("claude"))
        self.assertEqual([c.backend for c in found], ["gemini", "claude-cli"])

    def test_antigravity_falls_back_and_says_why(self):
        """Graphify cannot talk to agy. Falling back is fine; doing it
        without saying so is not."""
        choice = choose_backend({"ATLAS_PROVIDER": "antigravity"},
                                which=which_of("claude", "agy"))
        self.assertEqual(choice.backend, "claude-cli")
        self.assertIn("no backend for antigravity", choice.reason)

    def test_gemini_without_a_key_does_not_pretend(self):
        choice = choose_backend({"ATLAS_PROVIDER": "gemini"}, which=which_of())
        self.assertIsNone(choice.backend)
        self.assertIn("code only", choice.reason)

    def test_with_no_model_at_all_it_builds_from_code(self):
        choice = choose_backend({}, which=which_of())
        self.assertIsNone(choice.backend)


class Commands(unittest.TestCase):
    def test_a_model_build_extracts_labels_and_draws(self):
        steps = commands("graphify", Path("/r"), Choice("claude-cli", ""))
        self.assertEqual([s[1] for s in steps], ["extract", "label", "export"])
        self.assertIn("--backend", steps[0])
        self.assertIn("--missing-only", steps[1])

    def test_code_only_skips_the_model_entirely(self):
        steps = commands("graphify", Path("/r"), Choice(None, ""))
        self.assertIn("--code-only", steps[0])
        self.assertNotIn("label", [s[1] for s in steps])

    def test_force_reaches_extract(self):
        steps = commands("graphify", Path("/r"), Choice(None, ""), force=True)
        self.assertIn("--force", steps[0])


class Building(unittest.TestCase):
    def test_when_every_model_fails_the_error_says_so(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(graphify_build, "find_graphify", return_value="graphify"), \
                mock.patch.dict("os.environ", {"ATLAS_GRAPHIFY_BACKEND": "gemini"}):
            failing = lambda step, **kw: subprocess.CompletedProcess(step, 2)
            with self.assertRaises(GraphifyUnavailable) as caught:
                graphify_build.build(Path(tmp), env_file=None, runner=failing)
        self.assertIn("every available model failed", str(caught.exception))

    def test_a_failing_model_hands_over_to_the_next(self):
        tried = []
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(graphify_build, "find_graphify", return_value="graphify"), \
                mock.patch.object(graphify_build, "candidates", return_value=[
                    Choice("claude-cli", "a"), Choice("gemini", "b")]):
            out = Path(tmp) / "graphify-out"
            out.mkdir()
            (out / "graph.json").write_text("{}")

            def run(step, **kw):
                if step[1] == "extract":
                    tried.append(step[step.index("--backend") + 1])
                    return subprocess.CompletedProcess(step, 1 if tried[-1] == "claude-cli" else 0)
                return subprocess.CompletedProcess(step, 0)

            choice, _ = graphify_build.build(Path(tmp), env_file=None, runner=run)
        self.assertEqual(tried, ["claude-cli", "gemini"])
        self.assertEqual(choice.backend, "gemini")

    def test_success_without_a_graph_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(graphify_build, "find_graphify", return_value="graphify"), \
                mock.patch.dict("os.environ", {"ATLAS_GRAPHIFY_BACKEND": "gemini"}):
            ok = lambda step, **kw: subprocess.CompletedProcess(step, 0)
            with self.assertRaises(GraphifyUnavailable):
                graphify_build.build(Path(tmp), env_file=None, runner=ok)

    def test_every_step_runs_from_inside_the_repo(self):
        seen = []
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(graphify_build, "find_graphify", return_value="graphify"), \
                mock.patch.dict("os.environ", {"ATLAS_GRAPHIFY_BACKEND": "gemini"}):
            out = Path(tmp) / "graphify-out"
            out.mkdir()
            (out / "graph.json").write_text("{}")

            def record(step, **kw):
                seen.append(kw["cwd"])
                return subprocess.CompletedProcess(step, 0)

            graphify_build.build(Path(tmp), env_file=None, runner=record)
        self.assertEqual(set(seen), {Path(tmp).resolve()})


class MissingGraphify(unittest.TestCase):
    def test_the_error_says_how_to_install_it(self):
        with mock.patch("shutil.which", return_value=None):
            with self.assertRaises(GraphifyUnavailable) as caught:
                graphify_build.find_graphify()
        self.assertIn("graphifyy", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
