# Derivation Atlas

Maps the mathematics you must be able to derive on paper to understand — not
merely operate — the libraries a specific project depends on, and shows it as
a 3D map joined to the project's own code.

The governing constraint is **derive the object, don't prove theorems about
it**: deep enough to reproduce the mechanics on paper and know when they fail,
not so deep it becomes theory. Existence proofs, convergence theorems and
measure-theoretic framing are out of scope by construction.

Nothing is generated from memory. Every sheet is restated from a retrieved
source, and every citation is checked against what was actually fetched.

## How it works

```
your repo ──► Graphify (separate tool)     code structure, rationale, communities
          ──► Atlas ingest                 launched nodes, package.xml, parameter values
          ──► Atlas topics                 who publishes and reads which topic
                     │
                     ▼  joined per component
          architecture   components → subsystems → algorithms   (human gate)
          canon          246-node prerequisite DAG, shared by every project
          path           the nodes one algorithm needs, cut at what you know
          retrieval      a source for each node (Wikipedia, checked)
          generation     the sheet: statement, steps, symbols   (the only LLM step)
          project notes  what each node means in *your* config
          map            3D, drill-down, with code and licences attached
```

The **canon** is the product: mathematics and its prerequisites, shared by
every project. The **project layer** is cheap and disposable, rebuilt whenever
the code changes. The two meet at algorithm nodes.

[Graphify](https://github.com/Graphify-Labs/graphify) reads the code. Atlas
keeps its own deterministic extractor for what the maths hangs on — which nodes
a launch file starts and the exact values of the parameters they set — and
states the links between components from code, parameters and launch files
rather than inferring them. See `ARCHITECTURE.md` for every module.

## Setup

Python 3.11+. No packages to install; Atlas uses the standard library only.

```bash
git clone <this repo> && cd concept_viz
python3 -m unittest discover -s tests      # should end in OK
cp .env.example .env                       # only if you generate sheets
cp profile.example.json profile.json       # what you already know
```

**A model, only for generating sheets.** Set `ATLAS_PROVIDER` in `.env`:

| Provider | Needs |
|---|---|
| `antigravity` (recommended) | the `agy` CLI, signed in once interactively |
| `claude_code` | `claude setup-token`, then `CLAUDE_CODE_OAUTH_TOKEN` |
| `gemini` | `GEMINI_API_KEY` (free tier, 20 requests/day/model) |

**Graphify, for the code layer.** A separate project with its own licence,
installed on its own. Atlas runs it for you; it never imports it.

```bash
uv tool install graphifyy        # or: pip install graphifyy
```

Graphify reads code with a parser and needs no model. Its pass over docs and
configs does, and `build_codegraph.py` picks one for it:

| Set | Graphify uses |
|---|---|
| `ATLAS_GRAPHIFY_BACKEND=<name>` | that backend (`claude-cli`, `gemini`, `ollama`, `openai`, ...) |
| `ATLAS_PROVIDER=claude_code` | `claude-cli`, your Claude subscription |
| `ATLAS_PROVIDER=gemini` | `gemini`, with `GEMINI_API_KEY` from `.env` |
| `ATLAS_PROVIDER=antigravity` | Graphify has no Antigravity backend, so the claude CLI if installed, else a Gemini key |
| none of the above | code only: every symbol and call, but no descriptions |

The choice and its reason are printed on every run.

## Run

**Study a repo**

```bash
REPO=~/path/to/your_ws

python3 scripts/build_codegraph.py $REPO         # Graphify graph, with whatever model is here
python3 scripts/graphify_join.py $REPO           # what the graph adds, both sides' misses
python3 scripts/topics.py $REPO                  # every topic link, with file:line evidence

python3 scripts/generate_path.py $REPO localization      # sheets for one subsystem (LLM)
python3 scripts/instantiate_path.py $REPO localization   # what each means in your config
python3 scripts/drill.py $REPO localization              # masked-step exercises
```

**Build the map**

```bash
python3 scripts/export_graph.py $REPO localization > /tmp/graph.json
python3 scripts/build_map.py /tmp/graph.json viz/canon_map.html
open viz/canon_map.html
```

Pass the subsystem. Without it no learning path is computed and the map's
path rings and filter are empty.

**Keep it current**

```bash
python3 scripts/build_codegraph.py $REPO   # incremental: only changed files reach a model
python3 scripts/recheck_sheets.py  # re-run every guard on stored sheets; no LLM
```

Use `ATLAS_MIN_INTERVAL=3` for anything that fetches across the whole canon:
Wikipedia rate-limits long runs, and a throttled request must never read as a
missing article. Long generation runs need the machine plugged in with the lid
open; a sleeping Mac stalls them silently.

## State

| | |
|---|---|
| Canon | **246** nodes (186 concepts, 37 techniques, 23 algorithms), **926** edges, 23 seeds |
| Sheets | **241** of 246 nodes; 221 pass every check; 47 carry worked derivations |
| Reference project | `agrios_ws`: 26 components, 196 parameters, 15 topic links |
| Tests | 395, all passing |

The five nodes without a sheet have nothing to restate: four are vision
conventions with no article, and `greedy_choice` appears only in passing
inside Dijkstra's. The failing sheets are listed by
`recheck_sheets.py`; most fail because the retrieved article states a result
without deriving it, which the guards report rather than paper over.

## What is enforced in code, not trusted to prompts

- `REQUIRES` is strictly acyclic; an edge closing a cycle is demoted to
  `related_to`, so the information survives.
- Citations are checked against sources retrieved in that run. A plausible
  reference nobody fetched is worse than none.
- A sheet's shape is checked against what the article actually performs, read
  from the body of its sections, not from their headings.
- A disambiguation page is not a source. `matrix`, `span` and `norm` all
  resolved to one before this was checked.
- A transport failure is never recorded as a missing article.
- Model-inferred links are kept apart from stated ones and never feed a guard.
- Third-party code carries its licence notice, or the tests fail.

## Third-party work and licences

Atlas uses [Graphify](https://github.com/Graphify-Labs/graphify) (Apache-2.0)
as a separate tool, shows excerpts of
[robot_localization](https://github.com/cra-ros-pkg/robot_localization)
(BSD-3-Clause) with each file's full notice, and restates mathematics from
Wikipedia (CC BY-SA 4.0) with every source named and linked. What each
licence asks, and how it is met, is in `THIRD_PARTY_NOTICES.md`.

