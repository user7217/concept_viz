"""Build a repo's Graphify graph from inside Atlas, with whatever model is here.

Graphify is a separate tool with its own licence, and it stays one: this
module runs its command-line interface as a subprocess and never imports it,
so Atlas keeps to the standard library and redistributes none of Graphify's
code. All Atlas needs afterwards is the graph.json it leaves in the repo.

Graphify reads code with a parser and no model at all. Only its second pass --
docs, READMEs and config files -- needs one, and it can talk to Claude Code,
Gemini, Ollama or an OpenAI-compatible server, but not to Antigravity. So the
model is chosen like this, and the choice is always printed with its reason:

    ATLAS_GRAPHIFY_BACKEND   any Graphify backend name, used as given
    ATLAS_PROVIDER           claude_code -> claude-cli, gemini -> gemini
    otherwise                every model that is present, in order: the
                             claude CLI, then a Gemini key

Present is not the same as working. The first version picked the claude CLI
because it was on PATH, and its headless session had expired, so a forced
rebuild failed outright with a Gemini key sitting unused in .env. Each
candidate is now tried in turn, every failure is printed, and the build moves
on. Only with no model present at all does it build from code.

Code only is a real fallback, not a quiet one: the graph keeps every symbol and
call, but loses the descriptions that let Atlas recognise library components
(ekf_local, controller_server) that have no source in the repo.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .llm import ENV_FILE, load_env

INSTALL_HINT = "uv tool install graphifyy   (or: pip install graphifyy)"

# Atlas provider -> the Graphify backend that reaches the same model.
NATIVE = {"claude_code": "claude-cli", "gemini": "gemini"}


class GraphifyUnavailable(RuntimeError):
    """Graphify is not installed, or a step of it failed."""


@dataclass(frozen=True)
class Choice:
    backend: str | None      # None means code only
    reason: str


def find_graphify() -> str:
    path = shutil.which("graphify")
    if path is None:
        raise GraphifyUnavailable(
            f"graphify is not on PATH. Install it separately: {INSTALL_HINT}")
    return path


def candidates(env: dict[str, str] | None = None,
               which=shutil.which) -> list[Choice]:
    """Every backend worth trying, best first. Empty means code only."""
    env = dict(os.environ if env is None else env)
    explicit = env.get("ATLAS_GRAPHIFY_BACKEND", "").strip()
    if explicit:
        return [Choice(explicit, "ATLAS_GRAPHIFY_BACKEND")]

    provider = env.get("ATLAS_PROVIDER", "").strip()
    native = NATIVE.get(provider)
    present = []
    if which("claude"):
        present.append("claude-cli")
    if env.get("GEMINI_API_KEY", "").strip():
        present.append("gemini")

    found: list[Choice] = []
    if native in present:
        found.append(Choice(native, f"ATLAS_PROVIDER={provider}"))
    why = (f"Graphify has no backend for {provider}" if provider and not native
           else f"ATLAS_PROVIDER={provider} is unavailable" if native not in present
           and provider else "fallback")
    for backend in present:
        if all(c.backend != backend for c in found):
            source = "the claude CLI" if backend == "claude-cli" else "GEMINI_API_KEY"
            found.append(Choice(backend, f"{why}; using {source}"))
    return found


def choose_backend(env: dict[str, str] | None = None, which=shutil.which) -> Choice:
    """The first candidate, or code only when there is none."""
    found = candidates(env, which)
    if found:
        return found[0]
    return Choice(None, "no model available (no claude CLI, no Gemini key); "
                        "building from code only")


def commands(graphify: str, repo: Path, choice: Choice,
             force: bool = False) -> list[list[str]]:
    """The Graphify invocations a build runs, in order."""
    extract = [graphify, "extract", str(repo)]
    extract += ["--backend", choice.backend] if choice.backend else ["--code-only"]
    if force:
        extract.append("--force")
    steps = [extract]
    if choice.backend:
        steps.append([graphify, "label", str(repo), "--backend", choice.backend,
                      "--missing-only"])
    steps.append([graphify, "export", "html"])
    return steps


def build(repo: Path, force: bool = False, env_file: Path | None = ENV_FILE,
          runner=subprocess.run) -> tuple[Choice, Path]:
    """Build or incrementally update <repo>/graphify-out/graph.json."""
    repo = Path(repo).resolve()
    if not repo.is_dir():
        raise GraphifyUnavailable(f"{repo} is not a directory")
    if env_file is not None:
        load_env(env_file)          # a Gemini key in Atlas's .env reaches Graphify
    graphify = find_graphify()

    tries = candidates() or [choose_backend()]
    failures: list[str] = []
    for choice in tries:
        print(f"graphify: {'model ' + choice.backend if choice.backend else 'code only'}"
              f" ({choice.reason})", flush=True)
        steps = commands(graphify, repo, choice, force)
        # Graphify's export writes into graphify-out/ under the working
        # directory, so every step runs from inside the repo.
        if runner(steps[0], cwd=repo, check=False).returncode != 0:
            failures.append(choice.backend or "code only")
            print(f"graphify: {choice.backend} failed; its output above says why",
                  flush=True)
            continue
        for step in steps[1:]:
            result = runner(step, cwd=repo, check=False)
            if result.returncode != 0:
                raise GraphifyUnavailable(
                    f"`{' '.join(step[1:3])}` exited {result.returncode}; "
                    "its own output above says why")
        graph = repo / "graphify-out" / "graph.json"
        if not graph.exists():
            raise GraphifyUnavailable(f"graphify finished but wrote no {graph}")
        return choice, graph

    raise GraphifyUnavailable(
        f"every available model failed ({', '.join(failures)}). The existing "
        "graph, if any, is untouched. Usual fixes: for claude-cli, run "
        "`claude setup-token` once and set CLAUDE_CODE_OAUTH_TOKEN; for gemini, "
        "Graphify needs its extra: uv tool install \"graphifyy[gemini]\" --force")
