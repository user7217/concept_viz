"""Tests for the provider seam. No network: transports are injected."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.llm import (
    ClaudeCodeProvider,
    GeminiProvider,
    LLMError,
    complete_json,
    extract_json,
    get_provider,
    load_env,
)


def gemini_reply(text: str) -> bytes:
    return json.dumps(
        {"candidates": [{"content": {"parts": [{"text": text}]}}]}
    ).encode()


class TestExtractJson(unittest.TestCase):
    def test_plain_object(self) -> None:
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_code_fenced(self) -> None:
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_object_surrounded_by_prose(self) -> None:
        self.assertEqual(
            extract_json('Sure, here it is:\n{"a": 1}\nHope that helps.'), {"a": 1}
        )

    def test_no_object_raises(self) -> None:
        with self.assertRaises(LLMError):
            extract_json("I cannot answer that.")

    def test_malformed_object_raises(self) -> None:
        with self.assertRaises(LLMError):
            extract_json('{"a": }')


class TestGemini(unittest.TestCase):
    def test_sends_prompt_and_parses_reply(self) -> None:
        seen = {}

        def transport(url: str, data: bytes) -> bytes:
            seen["url"] = url
            seen["body"] = json.loads(data)
            return gemini_reply('{"ok": true}')

        provider = GeminiProvider(api_key="secret123", transport=transport)
        self.assertEqual(complete_json(provider, "derive it"), {"ok": True})
        self.assertEqual(
            seen["body"]["contents"][0]["parts"][0]["text"], "derive it"
        )
        self.assertEqual(
            seen["body"]["generationConfig"]["responseMimeType"], "application/json"
        )

    def test_system_instruction_is_sent_separately(self) -> None:
        seen = {}

        def transport(url: str, data: bytes) -> bytes:
            seen.update(json.loads(data))
            return gemini_reply("{}")

        GeminiProvider(api_key="k", transport=transport).complete("p", system="s")
        self.assertEqual(seen["systemInstruction"]["parts"][0]["text"], "s")

    def test_blocked_prompt_reports_the_reason(self) -> None:
        provider = GeminiProvider(
            api_key="k",
            transport=lambda u, d: json.dumps(
                {"promptFeedback": {"blockReason": "SAFETY"}}
            ).encode(),
        )
        with self.assertRaises(LLMError) as ctx:
            provider.complete("x")
        self.assertIn("SAFETY", str(ctx.exception))

    def test_unexpected_shape_raises_rather_than_returning_none(self) -> None:
        provider = GeminiProvider(
            api_key="k", transport=lambda u, d: json.dumps({"candidates": []}).encode()
        )
        with self.assertRaises(LLMError):
            provider.complete("x")

    def test_api_key_never_appears_in_an_error(self) -> None:
        import urllib.error

        def transport(url: str, data: bytes) -> bytes:
            raise urllib.error.HTTPError(url, 400, "Bad", {}, None)

        provider = GeminiProvider(api_key="SUPERSECRET", transport=transport)
        with self.assertRaises(Exception) as ctx:
            provider.complete("x")
        self.assertNotIn("SUPERSECRET", str(ctx.exception))


class TestClaudeCode(unittest.TestCase):
    def test_builds_a_print_mode_command(self) -> None:
        seen = {}

        def runner(command: list[str], text: str) -> str:
            seen["command"], seen["text"] = command, text
            return '{"ok": true}'

        provider = ClaudeCodeProvider(model="claude-opus-5", runner=runner)
        self.assertEqual(complete_json(provider, "derive", system="be terse"),
                         {"ok": True})
        # The prompt belongs in argv. `-p "..."` is the user message; stdin is
        # supplementary data, so piping it sent attached data and no question.
        self.assertEqual(seen["command"][:2], ["claude", "-p"])
        self.assertIn("derive", seen["command"][2])
        self.assertTrue(seen["command"][2].startswith("be terse"))
        self.assertEqual(seen["command"][3:], ["--model", "claude-opus-5"])

    def test_auth_failure_is_its_own_error(self) -> None:
        """A run that cannot authenticate must stop, not repeat the failure."""
        import subprocess
        from atlas.llm import AuthFailure

        class Result:
            returncode, stdout, stderr = 1, "Failed to authenticate: OAuth session expired", ""

        real = subprocess.run
        subprocess.run = lambda *a, **k: Result()
        try:
            with self.assertRaises(AuthFailure) as ctx:
                ClaudeCodeProvider().complete("x")
            self.assertIn("setup-token", str(ctx.exception))
        finally:
            subprocess.run = real

    def test_missing_binary_explains_the_fix(self) -> None:
        provider = ClaudeCodeProvider(binary="claude_not_installed_xyz")
        with self.assertRaises(LLMError) as ctx:
            provider.complete("x")
        self.assertIn("ATLAS_PROVIDER=gemini", str(ctx.exception))


class TestEnv(unittest.TestCase):
    def test_parses_comments_and_quotes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text('# note\nGEMINI_API_KEY="abc"\nEMPTY=\nBAD LINE\n')
            values = load_env(path)
        self.assertEqual(values["GEMINI_API_KEY"], "abc")
        self.assertEqual(values["EMPTY"], "")
        self.assertNotIn("BAD LINE", values)

    def test_does_not_override_a_real_environment_variable(self) -> None:
        os.environ["ATLAS_TEST_KEY"] = "from_environment"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / ".env"
                path.write_text("ATLAS_TEST_KEY=from_file\n")
                load_env(path)
            self.assertEqual(os.environ["ATLAS_TEST_KEY"], "from_environment")
        finally:
            del os.environ["ATLAS_TEST_KEY"]

    def test_missing_file_is_not_an_error(self) -> None:
        self.assertEqual(load_env(Path("/nonexistent/.env")), {})


class TestGetProvider(unittest.TestCase):
    def test_missing_key_explains_the_fix(self) -> None:
        """Must not read the developer's own .env to decide this."""
        saved = os.environ.pop("GEMINI_API_KEY", None)
        try:
            with self.assertRaises(LLMError) as ctx:
                get_provider("gemini", env_file=None)
            self.assertIn(".env.example", str(ctx.exception))
        finally:
            if saved is not None:
                os.environ["GEMINI_API_KEY"] = saved

    def test_default_model_is_configurable(self) -> None:
        from atlas.llm import DEFAULT_GEMINI_MODEL, GeminiProvider

        self.assertEqual(GeminiProvider(api_key="k").model, DEFAULT_GEMINI_MODEL)
        self.assertEqual(GeminiProvider(api_key="k", model="other").model, "other")

    def test_unknown_provider_raises(self) -> None:
        with self.assertRaises(LLMError):
            get_provider("gpt5", env_file=None)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestRotation(unittest.TestCase):
    """Free-tier quota is per model, so several small allowances combine."""

    def make(self, model: str, replies):
        import itertools
        from atlas.llm import GeminiProvider

        stream = iter(replies)

        def transport(url: str, data: bytes) -> bytes:
            item = next(stream)
            if isinstance(item, Exception):
                raise item
            return item

        return GeminiProvider(api_key="k", model=model, transport=transport)

    def test_falls_through_to_the_next_model_when_quota_is_spent(self) -> None:
        from atlas.llm import QuotaExhausted, RotatingProvider

        first = self.make("a", [QuotaExhausted("a: spent")])
        second = self.make("b", [gemini_reply('{"ok": true}')])
        rotation = RotatingProvider([first, second])
        self.assertEqual(rotation.complete("p"), '{"ok": true}')

    def test_an_exhausted_model_is_not_retried(self) -> None:
        from atlas.llm import QuotaExhausted, RotatingProvider

        calls = []

        def counting(url: str, data: bytes) -> bytes:
            calls.append(url)
            raise QuotaExhausted("spent")

        from atlas.llm import GeminiProvider

        first = GeminiProvider(api_key="k", model="a", transport=counting)
        second = self.make("b", [gemini_reply("{}"), gemini_reply("{}")])
        rotation = RotatingProvider([first, second])
        rotation.complete("one")
        rotation.complete("two")
        self.assertEqual(len(calls), 1, "exhausted model was called again")

    def test_all_exhausted_raises(self) -> None:
        from atlas.llm import QuotaExhausted, RotatingProvider

        rotation = RotatingProvider([
            self.make("a", [QuotaExhausted("spent")]),
            self.make("b", [QuotaExhausted("spent")]),
        ])
        with self.assertRaises(QuotaExhausted):
            rotation.complete("p")

    def test_a_non_quota_error_is_not_treated_as_exhaustion(self) -> None:
        """A malformed request must not silently burn the whole rotation."""
        import urllib.error

        from atlas.llm import GeminiProvider, LLMError, RotatingProvider

        def bad_request(url: str, data: bytes) -> bytes:
            raise urllib.error.HTTPError(url, 400, "Bad", {}, None)

        rotation = RotatingProvider([GeminiProvider(api_key="k", model="a",
                                                    transport=bad_request)])
        with self.assertRaises(LLMError):
            rotation.complete("p")
        self.assertEqual(rotation.exhausted, set())

    def test_empty_rotation_is_rejected(self) -> None:
        from atlas.llm import LLMError, RotatingProvider

        with self.assertRaises(LLMError):
            RotatingProvider([])


class TestGeminiCli(unittest.TestCase):
    def test_prompt_and_model_reach_argv(self) -> None:
        from atlas.llm import GeminiCliProvider

        seen = {}

        def runner(command, text):
            seen["command"] = command
            return '{"ok": true}'

        p = GeminiCliProvider(model="gemini-3-pro", runner=runner)
        self.assertEqual(complete_json(p, "derive", system="be terse"), {"ok": True})
        self.assertEqual(seen["command"][:2], ["gemini", "-p"])
        self.assertIn("derive", seen["command"][2])
        self.assertIn("--skip-trust", seen["command"])
        # read-only: this needs text back, never an edit or a command
        self.assertEqual(seen["command"][seen["command"].index("--approval-mode") + 1],
                         "plan")
        self.assertEqual(seen["command"][-2:], ["-m", "gemini-3-pro"])

    def test_missing_binary_explains_the_install(self) -> None:
        from atlas.llm import GeminiCliProvider

        with self.assertRaises(LLMError) as ctx:
            GeminiCliProvider(binary="gemini_not_installed_xyz").complete("x")
        self.assertIn("npm install", str(ctx.exception))

    def test_quota_message_raises_quota_exhausted(self) -> None:
        import subprocess

        from atlas.llm import GeminiCliProvider, QuotaExhausted

        class Result:
            returncode, stdout, stderr = 1, "", "Quota exceeded for this model"

        real = subprocess.run
        subprocess.run = lambda *a, **k: Result()
        try:
            with self.assertRaises(QuotaExhausted):
                GeminiCliProvider().complete("x")
        finally:
            subprocess.run = real

    def test_selectable_by_env(self) -> None:
        import os

        from atlas.llm import GeminiCliProvider, get_provider

        os.environ["GEMINI_CLI_MODEL"] = "gemini-3-pro"
        try:
            p = get_provider("gemini_cli", env_file=None)
            self.assertIsInstance(p, GeminiCliProvider)
            self.assertEqual(p.model, "gemini-3-pro")
        finally:
            del os.environ["GEMINI_CLI_MODEL"]


class TestAntigravity(unittest.TestCase):
    def test_command_shape(self) -> None:
        from atlas.llm import AntigravityProvider

        seen = {}

        def runner(command, text):
            seen["command"] = command
            return '{"ok": true}'

        p = AntigravityProvider(model="gemini-3-pro", effort="high", runner=runner)
        self.assertEqual(complete_json(p, "derive", system="be terse"), {"ok": True})
        self.assertEqual(seen["command"][:2], ["agy", "-p"])
        self.assertIn("derive", seen["command"][2])
        # read-only: one prompt, JSON back, never an edit or a command
        self.assertEqual(seen["command"][seen["command"].index("--mode") + 1], "plan")
        self.assertIn("--model", seen["command"])
        self.assertIn("high", seen["command"])
        # The CLI voids --mode plan when slash expansion is off, and says so
        # on stderr. Passing both silently gave up the read-only guarantee.
        self.assertNotIn("--disable-slash-commands", seen["command"])

    def test_empty_output_is_retried_not_failed(self) -> None:
        import subprocess

        from atlas.llm import AntigravityProvider

        class Result:
            def __init__(self, stdout):
                self.returncode, self.stdout, self.stderr = 0, stdout, ""

        replies = ["", "", '{"ok": true}']
        slept: list[float] = []
        real = subprocess.run
        subprocess.run = lambda *a, **k: Result(replies.pop(0))
        try:
            p = AntigravityProvider(sleeper=slept.append)
            self.assertEqual(p.complete("x"), '{"ok": true}')
        finally:
            subprocess.run = real
        # backed off between attempts rather than hammering the throttle
        self.assertEqual(slept, [20.0, 40.0])

    def test_persistent_drop_is_not_a_content_failure(self) -> None:
        import subprocess

        from atlas.llm import AntigravityProvider, Dropped

        class Result:
            returncode, stdout, stderr = 0, "", ""

        real = subprocess.run
        subprocess.run = lambda *a, **k: Result()
        try:
            with self.assertRaises(Dropped) as caught:
                AntigravityProvider(sleeper=lambda _: None).complete("x" * 4000)
        finally:
            subprocess.run = real
        # the message has to name the size, since that is what predicts it
        self.assertIn("4000", str(caught.exception))

    def test_quota_is_its_own_error(self) -> None:
        import subprocess

        from atlas.llm import AntigravityProvider, QuotaExhausted

        class Result:
            returncode, stdout, stderr = 1, "", "Rate limit exceeded"

        real = subprocess.run
        subprocess.run = lambda *a, **k: Result()
        try:
            with self.assertRaises(QuotaExhausted):
                AntigravityProvider().complete("x")
        finally:
            subprocess.run = real

    def test_selectable_by_env(self) -> None:
        import os

        from atlas.llm import AntigravityProvider, get_provider

        os.environ["AGY_EFFORT"] = "high"
        try:
            p = get_provider("antigravity", env_file=None)
            self.assertIsInstance(p, AntigravityProvider)
            self.assertEqual(p.effort, "high")
        finally:
            del os.environ["AGY_EFFORT"]
