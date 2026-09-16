# Architecture

Maps the mathematics you must be able to derive **on paper** to understand — not
merely operate — the libraries a project depends on.

Point it at a repository. It reads what the code imports and configures, works
out which algorithms are running, walks a prerequisite graph down from those
algorithms to the mathematics beneath them, retrieves a source for each node,
and produces a sheet you can work through.

## The pipeline

```
1  ingest         repo ──▶ components, libraries, parameters       no LLM
2  architecture   components ──▶ subsystems + algorithms           no LLM, human gate
3  canon          226 nodes, prerequisite DAG                      prebuilt
4  path           project ──▶ ordered node list, cut at profile    no LLM
5  retrieval      node ──▶ grounded source text                    no LLM
6  generation     source ──▶ restated sheet + mechanical checks     LLM
7  instantiation  node ──▶ what it means in THIS repo               LLM
8  exercise       sheet ──▶ masked drills                          no LLM
```

Only steps 6 and 7 call a model. Everything else is plain code running offline
in milliseconds, which is why the expensive steps can be cached, resumed and
re-checked for free.

Two layers meet at **algorithm** nodes. Above them is one project's structure
(system → subsystem → component); below them is the shared canon of
mathematics, which no project owns. That boundary is what lets the canon be
reused across repositories while the layer above is thrown away per project.

## `atlas/` — the library

Zero dependencies. stdlib only: `urllib`, `subprocess`, `ast`, `json`.

| module | what it owns |
|---|---|
| `schema.py` | Node, Edge, Seed; the NodeType / Tier / Relation enums. Rejects ids that are not `lower_snake_case`. |
| `graph.py` | The DAG. `requires_closure`, `learning_path`, `requires_chain`, `syllabus`, `merge_node`, `validate`. REQUIRES is kept acyclic by demoting any cycle-closing edge to RELATED_TO. |
| `bootstrap.py` | Builds the canon from `seeds.json` + `canon_edges.json`, then applies amendments in a fixed order: add nodes → load edges → **drops → retargets** → add_edges → retier → merges. An amendment that cannot apply raises. |
| `merge.py` | Folds independently authored edge proposals into one canon and surfaces the cross-cluster problems: unknown ids, branching-cap violations, cycles that only exist once two proposals meet. |
| `ingest.py` | Stage 1. Parses launch files with `ast` (never executes them), finds config at any depth, extracts parameters and marks the matrix-valued ones. |
| `discover.py` | The other half of stage 1. Dependency scanning only finds mathematics a project *imports*; research code writes it inline, so this reads the source itself. |
| `architecture.py` | Stage 2 and the pipeline's only human gate. `propose` → confirm → `apply`. Resolves a component by plugin class, then `package/executable`, then package. |
| `retrieval.py` | Stage 5. Throttled fetching with backoff. A transport error raises rather than looking like a missing article. |
| `derivation.py` | Stage 6. `expected_kind` decides derivation-vs-definition from canon shape *and* the retrieved source; `select_text` keeps the window most likely to carry a derivation; `check` runs the mechanical guards. |
| `instantiate.py` | Stage 7. What a canon node means in one repo: which components reach it, which parameters it governs. `invented()` and `thin()` guard it from opposite sides. |
| `code.py` | Where a concept is visibly at work in the source. Mechanical matching only, so it cannot fabricate evidence. |
| `exercise.py` | Stage 8. Masks a step or a symbol and checks the answer is not readable off the surrounding page. |
| `profile.py` | What the reader knows: declared / read / demonstrated. **The one artifact that cannot be regenerated** — the canon rebuilds from seeds, this does not. |
| `worksheet.py` | Renders a path as paper: why each node is on it, what it does in the project, then the drills. |
| `llm.py` | The provider seam. Gemini API, headless Claude, Antigravity, and a rotating provider. Distinguishes `AuthFailure`, `QuotaExhausted` and `Dropped` because each deserves a different response. |

## `scripts/` — the entry points

**Running the pipeline**

| script | what it does |
|---|---|
| `generate_path.py <repo> <subsystem>` | Generate sheets for one subsystem. Resumable. |
| `generate_all.py` | The whole canon. Prefer `generate_path.py`. |
| `instantiate_path.py <repo> <subsystem>` | Attach project meaning to each sheet on a path. |
| `drill.py [<repo> <subsystem>]` | The masked-exercise drill. `--worksheet` writes paper; `--dry-run` leaves the profile alone. |

**Inspecting what exists**

| script | what it does |
|---|---|
| `report.py` | Canon bootstrap and the saturation curve. |
| `grounding.py` | Which nodes have a retrievable source. |
| `recheck_sheets.py` | Re-runs every guard over stored sheets. Free — no model. |
| `prune_sheets.py --dry-run` | What a fix just invalidated. |
| `canon_gaps.py` | Holes and candidate nodes across all sheets. |

**Building the canon** (rarely needed once it is stable)

`write_canon.py` is the **sole writer** to `data/`. `merge_report.py`,
`retry_grounding.py`, `technique_grounding.py` support it.

**The map**

| script | what it does |
|---|---|
| `export_graph.py <repo> <subsystem>` | The whole graph as one JSON: nodes, edges with their rationales, walkthrough levels, per-node measures, code spans. |
| `build_map.py <graph.json> <out.html>` | Injects that payload into the page template. |
| `extract_library.py <clone>` | Lifts verified snippets out of a third-party library so the clone can be deleted. |

## `data/` — the durable state

| file | contents |
|---|---|
| `seeds.json` | 20 seed algorithms and the assumed floor. |
| `canon_edges.json` | 550 curated prerequisite edges, **each with the reason it exists**. |
| `canon_amendments.json` | Every correction applied on top of the raw canon, kept separate so the bootstrap stays reproducible. |
| `canon_flags.json` | Adjudicated and pending canon questions. |
| `source_aliases.json` | What reference sources actually call things. An entry of `[]` means deliberately unretrievable. |
| `grounding.json`, `technique_grounding.json` | Which nodes resolved to a source. |
| `library_map.json` | Package/plugin → algorithm. |
| `library_code.json` | Verified excerpts from third-party source, with commit and licence. |
| `derivations/*.json` | The sheets: statement, symbols, steps, assumptions, failure modes, project instantiation. |
| `../profile.json` | Yours. Not regenerable. |

## `viz/` and `tests/`

`viz/canon_map_template.html` is the 3D map — one file, no framework, hand-rolled
projection on a 2D canvas. `build_map.py` injects the data.

`tests/` holds 310 tests and gates every commit:

```bash
python3 -m unittest discover -s tests || exit 1
```

## The invariants

- **Zero dependencies.** stdlib only.
- **Guards are code, not prompts.** Acyclicity, branching caps, citation
  checking, shape validation are enforced at write time.
- **Never fail silently.** An amendment that cannot apply raises. A transport
  error is not a missing article. A skipped directory is not an absent import.
- **A guard must compare against something independent.** Checking a sheet
  against itself is the default, and it is always wrong. It has been the cause
  of five separate defects here.
