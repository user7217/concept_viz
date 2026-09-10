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


class AuthFailure(LLMError):
    """Credentials are expired or absent. Every later call fails the same way.

    Worth its own type: a run that cannot authenticate should stop at the first
    node, not grind through the whole path producing identical failures.
    """


class QuotaExhausted(LLMError):
    """This model's quota is spent. A different model may still have some.

    Free-tier quota is per project *per model*, so exhausting one model says
    nothing about the others -- which is what makes rotation worth doing.
    """


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
    backoff_on_quota: bool = True
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
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                # An injected transport goes through the same error handling as
                # a real request, so tests exercise the paths that actually run.
                if self.transport is not None:
                    return self.transport(url, data)
                request = urllib.request.Request(
                    url, data=data, headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                # An HTTPError can carry no readable body; failing to read one
                # must not mask the status code that actually matters.
                try:
                    body = _redact(exc.read().decode()[:300])
                except Exception:
                    body = f"(no response body) {exc.reason}"
                if exc.code == 429:
                    # Backing off is pointless when the bucket is daily; let the
                    # caller try another model instead.
                    if "PerDay" in body or not self.backoff_on_quota:
                        raise QuotaExhausted(f"{self.model}: {body}") from exc
                    if attempt < self.max_retries - 1:
                        time.sleep(delay)
                        delay *= 2
                        continue
                    raise QuotaExhausted(f"{self.model}: {body}") from exc
                if exc.code in (500, 502, 503, 504) and attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise LLMError(f"gemini HTTP {exc.code}: {body}") from exc
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
        # The prompt goes in argv, not stdin. `-p "..."` is the user message;
        # stdin is supplementary data, so piping the instruction sent a request
        # with attached data and no actual question.
        text = f"{system}\n\n{prompt}" if system else prompt
        command = [self.binary, "-p", text]
        if self.model:
            command += ["--model", self.model]

        if self.runner is not None:
            return self.runner(command, text)

        try:
            result = subprocess.run(
                command, capture_output=True, text=True,
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
            # Report everything: an empty stderr with a non-zero exit says
            # nothing, and the useful message is often on stdout instead.
            detail = (result.stderr.strip() or result.stdout.strip()
                      or "(no output on either stream)")
            lowered = detail.lower()
            if "authenticate" in lowered or "oauth" in lowered or "login" in lowered:
                raise AuthFailure(
                    f"{detail[:200]}\n"
                    "Print mode cannot run /login. Issue a long-lived token:\n"
                    "  claude setup-token\n"
                    "  export CLAUDE_CODE_OAUTH_TOKEN=<token>"
                )
            raise LLMError(
                f"claude -p exited {result.returncode}: {detail[:400]}"
            )
        if not result.stdout.strip():
            raise LLMError("claude -p exited 0 but produced no output")
        return result.stdout


@dataclass
class RotatingProvider:
    """Try each provider in turn, moving on when one's quota is spent.

    Free-tier quota is granted per model, so several small daily allowances
    combine into a usable one. Order matters: put the model you most want
    answering first, since later ones are only reached once earlier buckets
    are empty.
    """

    providers: list[Provider]
    name: str = field(default="rotating", init=False)

    def __post_init__(self) -> None:
        if not self.providers:
            raise LLMError("RotatingProvider needs at least one provider")
        self.exhausted: set[str] = set()

    @property
    def model(self) -> str:
        live = [p for p in self.providers if getattr(p, "model", "") not in self.exhausted]
        return getattr(live[0], "model", "-") if live else "(all exhausted)"

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        errors = []
        for provider in self.providers:
            model = getattr(provider, "model", provider.name)
            if model in self.exhausted:
                continue
            try:
                return provider.complete(prompt, system=system)
            except QuotaExhausted as exc:
                self.exhausted.add(model)
                errors.append(f"{model}: quota spent")
                continue
        raise QuotaExhausted("every model's quota is spent: " + "; ".join(errors))


ROTATION = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
]


def get_rotating_provider(models: list[str] | None = None,
                          env_file: Path | None = ENV_FILE) -> RotatingProvider:
    """Build a rotation over several Gemini models sharing one API key."""
    if env_file is not None:
        load_env(env_file)
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise LLMError("GEMINI_API_KEY is not set.")
    chosen = models or [
        m.strip() for m in os.environ.get("GEMINI_ROTATION", ",".join(ROTATION)).split(",")
        if m.strip()
    ]
    return RotatingProvider([GeminiProvider(api_key=key, model=m) for m in chosen])


@dataclass
class GeminiCliProvider:
    """Google's `gemini -p`, running on a Google account rather than an API key.

    Same shape as ClaudeCodeProvider and far better quota than the API free
    tier: a personal account gets 1000 requests/day against 20/day per model
    on the key-based free tier, and an AI Pro subscription reaches Pro models.

    Reported limit trouble mostly comes from agentic sessions where one prompt
    fans out into dozens of model requests. These calls are single-turn with no
    tool use, so one prompt is one request.

    BLOCKED as of gemini-cli 0.46.0 (Sept 2026): every auth path, personal
    account and API key alike, fails setup with

        IneligibleTierError: This client is no longer supported for Gemini Code
        Assist for individuals. Migrate to the Antigravity suite.

    Kept because the code is correct and the blocker is a account/version gate
    that may lift. Do not spend time rediscovering it -- test the CLI by hand
    before wiring a run to this provider.
    """

    binary: str = "gemini"
    model: str | None = None
    timeout: int = 300
    runner: Callable[[list[str], str], str] | None = None
    name: str = field(default="gemini_cli", init=False)

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        text = f"{system}\n\n{prompt}" if system else prompt
        # --approval-mode plan is read-only: the model cannot edit or execute
        # anything, which is all this needs -- one prompt, text back. That makes
        # --skip-trust (required to run headless at all) a much smaller thing to
        # grant than it would be with tools enabled.
        command = [self.binary, "-p", text, "--skip-trust", "--approval-mode", "plan"]
        if self.model:
            command += ["-m", self.model]

        if self.runner is not None:
            return self.runner(command, text)

        try:
            result = subprocess.run(
                command, capture_output=True, text=True,
                timeout=self.timeout, check=False,
            )
        except FileNotFoundError as exc:
            raise LLMError(
                f"{self.binary!r} not found on PATH. Install with "
                "`npm install -g @google/gemini-cli`, then run `gemini` once to "
                "sign in."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise LLMError(f"gemini -p timed out after {self.timeout}s") from exc

        if result.returncode != 0:
            detail = (result.stderr.strip() or result.stdout.strip()
                      or "(no output on either stream)")
            lowered = detail.lower()
            if "quota" in lowered or "rate limit" in lowered or "429" in lowered:
                raise QuotaExhausted(f"{self.model or 'gemini-cli'}: {detail[:200]}")
            if "auth" in lowered or "sign in" in lowered or "login" in lowered:
                raise AuthFailure(
                    f"{detail[:200]}\nRun `gemini` once interactively to sign in."
                )
            raise LLMError(f"gemini -p exited {result.returncode}: {detail[:400]}")
        if not result.stdout.strip():
            raise LLMError("gemini -p exited 0 but produced no output")
        return result.stdout


@dataclass
class AntigravityProvider:
    """Google's `agy -p`, the client that replaced gemini-cli for individuals.

    Runs on a Google AI subscription rather than an API key, so it is not
    bound by the API free tier's 20 requests/day/model.

    `--mode plan` is read-only: the model cannot edit or execute anything,
    which is all this needs -- one prompt, JSON back.
    """

    binary: str = "agy"
    model: str | None = None
    effort: str | None = None  # low | medium | high
    timeout: int = 600
    runner: Callable[[list[str], str], str] | None = None
    name: str = field(default="antigravity", init=False)

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        text = f"{system}\n\n{prompt}" if system else prompt
        command = [self.binary, "-p", text, "--mode", "plan",
                   "--disable-slash-commands"]
        if self.model:
            command += ["--model", self.model]
        if self.effort:
            command += ["--effort", self.effort]

        if self.runner is not None:
            return self.runner(command, text)

        try:
            result = subprocess.run(
                command, capture_output=True, text=True,
                timeout=self.timeout, check=False,
            )
        except FileNotFoundError as exc:
            raise LLMError(
                f"{self.binary!r} not found on PATH. Install the Antigravity "
                "CLI and run `agy` once to sign in."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise LLMError(f"agy -p timed out after {self.timeout}s") from exc

        if result.returncode != 0:
            detail = (result.stderr.strip() or result.stdout.strip()
                      or "(no output on either stream)")
            lowered = detail.lower()
            if "quota" in lowered or "rate limit" in lowered or "429" in lowered:
                raise QuotaExhausted(f"{self.model or 'antigravity'}: {detail[:200]}")
            if "auth" in lowered or "sign in" in lowered or "login" in lowered:
                raise AuthFailure(
                    f"{detail[:200]}\nRun `agy` once interactively to sign in."
                )
            raise LLMError(f"agy -p exited {result.returncode}: {detail[:400]}")
        if not result.stdout.strip():
            raise LLMError("agy -p exited 0 but produced no output")
        return result.stdout


def get_provider(
    name: str | None = None, env_file: Path | None = ENV_FILE
) -> Provider:
    """Build a provider from the environment.

    ATLAS_PROVIDER selects; GEMINI_API_KEY and GEMINI_MODEL configure Gemini;
    CLAUDE_MODEL configures the headless backend.

    `env_file` is explicit so callers -- tests especially -- are not silently
    reading whatever .env happens to sit in the working tree.
    """
    if env_file is not None:
        load_env(env_file)
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

    if name == "gemini_cli":
        return GeminiCliProvider(model=os.environ.get("GEMINI_CLI_MODEL") or None)

    if name == "antigravity":
        return AntigravityProvider(
            model=os.environ.get("AGY_MODEL") or None,
            effort=os.environ.get("AGY_EFFORT") or None,
        )

    raise LLMError(
        f"unknown provider {name!r}; expected 'gemini', 'antigravity', "
        f"'claude_code' or 'gemini_cli'"
    )


def complete_json(provider: Provider, prompt: str, *, system: str | None = None) -> dict:
    """Call a provider and parse a JSON object out of the reply."""
    return extract_json(provider.complete(prompt, system=system))
