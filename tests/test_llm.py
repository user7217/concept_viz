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
        self.assertEqual(seen["command"], ["claude", "-p", "--model", "claude-opus-5"])
        self.assertTrue(seen["text"].startswith("be terse"))

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
