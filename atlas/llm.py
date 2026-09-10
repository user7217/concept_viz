"""Provider seam for the LLM-backed pipeline stages.

Two backends, chosen by task rather than preference:

  claude_code  -- headless `claude -p`, running on a Claude subscription.
                  Used for derivation extraction: low volume, cached forever,
                  and the one place where a dropped transpose is unrecoverable
                  because the reader cannot check it (design doc section 9).

  gemini       -- free tier via Google AI Studio. Used for everything else:
                  identity resolution, extracting which concepts a given
                  derivation invokes, tagging. High volume, low stakes,
                  cheaply verified.

No SDK dependencies: urllib and subprocess only.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


class LLMError(RuntimeError):
    """A provider call failed in a way retrying will not fix."""


def load_env(path: Path = ENV_FILE) -> dict[str, str]:
    """Read a .env file into os.environ without overwriting real env vars.

    Deliberately tiny: KEY=value, # comments, optional quotes. The file is
    gitignored; .env.example is the committed template.
    """
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        values[key] = value
        os.environ.setdefault(key, value)
    return values


def _redact(text: str) -> str:
    """Never let a key reach a log line or an exception message."""
    return re.sub(r"(key=)[A-Za-z0-9_\-]+", r"\1***", text)


def extract_json(text: str) -> dict:
    """Parse a JSON object out of a model response.

    Models wrap JSON in code fences often enough that stripping them is not
    defensive programming, it is the normal path.
    """
    cleaned = _FENCE.sub("", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise LLMError(f"no JSON object in response: {cleaned[:200]!r}")
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"malformed JSON in response: {exc}") from exc


class Provider(Protocol):
    name: str

    def complete(self, prompt: str, *, system: str | None = None) -> str: ...


@dataclass
class GeminiProvider:
    """Google AI Studio free tier.

    Free-tier inputs may be used to improve Google's products, which is fine
    for this workload -- it sends textbook mathematics, never project secrets.
    """

    api_key: str
    model: str = DEFAULT_GEMINI_MODEL
    json_mode: bool = True
    max_retries: int = 4
    transport: Callable[[str, bytes], bytes] | None = None
    name: str = field(default="gemini", init=False)

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        body: dict = {"contents": [{"parts": [{"text": prompt}]}]}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if self.json_mode:
            body["generationConfig"] = {"responseMimeType": "application/json"}

        url = GEMINI_ENDPOINT.format(model=self.model) + f"?key={self.api_key}"
        raw = self._post(url, json.dumps(body).encode())
        payload = json.loads(raw)

        try:
            return payload["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            blocked = payload.get("promptFeedback", {}).get("blockReason")
            if blocked:
                raise LLMError(f"gemini blocked the prompt: {blocked}") from exc
            raise LLMError(f"unexpected gemini response shape: {payload}") from exc

    def _post(self, url: str, data: bytes) -> bytes:
        if self.transport is not None:
            return self.transport(url, data)

        delay = 2.0
        for attempt in range(self.max_retries):
            request = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                # 429 is the expected free-tier signal, not an error condition.
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise LLMError(
                    f"gemini HTTP {exc.code}: {_redact(exc.read().decode()[:300])}"
                ) from exc
            except urllib.error.URLError as exc:
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise LLMError(f"gemini unreachable: {exc.reason}") from exc
        raise LLMError("gemini: retries exhausted")


@dataclass
class ClaudeCodeProvider:
    """Headless `claude -p`, running on a Claude subscription rather than API credits.

    Reserved for derivation work: subscription rate limits are built for
    interactive use, so this is not a bulk backend.
    """

    binary: str = "claude"
    model: str | None = None
    timeout: int = 300
    runner: Callable[[list[str], str], str] | None = None
    name: str = field(default="claude_code", init=False)

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        command = [self.binary, "-p"]
        if self.model:
            command += ["--model", self.model]
        text = f"{system}\n\n{prompt}" if system else prompt

        if self.runner is not None:
            return self.runner(command, text)

        try:
            result = subprocess.run(
                command, input=text, capture_output=True, text=True,
                timeout=self.timeout, check=False,
            )
        except FileNotFoundError as exc:
            raise LLMError(
                f"{self.binary!r} not found on PATH; install Claude Code or set "
                "ATLAS_PROVIDER=gemini"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise LLMError(f"claude -p timed out after {self.timeout}s") from exc

        if result.returncode != 0:
            raise LLMError(f"claude -p failed: {result.stderr.strip()[:300]}")
        return result.stdout


def get_provider(name: str | None = None) -> Provider:
    """Build a provider from the environment.

    ATLAS_PROVIDER selects; GEMINI_API_KEY and GEMINI_MODEL configure Gemini;
    CLAUDE_MODEL configures the headless backend.
    """
    load_env()
    name = name or os.environ.get("ATLAS_PROVIDER", "gemini")

    if name == "gemini":
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            raise LLMError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add a "
                "key from Google AI Studio."
            )
        return GeminiProvider(
            api_key=key,
            model=os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        )

    if name == "claude_code":
        return ClaudeCodeProvider(model=os.environ.get("CLAUDE_MODEL") or None)

    raise LLMError(f"unknown provider {name!r}; expected 'gemini' or 'claude_code'")


def complete_json(provider: Provider, prompt: str, *, system: str | None = None) -> dict:
    """Call a provider and parse a JSON object out of the reply."""
    return extract_json(provider.complete(prompt, system=system))
