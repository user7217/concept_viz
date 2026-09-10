# Derivation Atlas

Maps the mathematics you must be able to derive on paper to understand — not
merely operate — the libraries a specific project depends on.

Design doc: https://claude.ai/code/artifact/87787bf4-2df0-4916-8b2c-48cce3d703dd

The governing constraint is **derive the object, don't prove theorems about it**:
deep enough to reproduce the mechanics on paper and know when they fail, not so
deep it becomes theory. Existence proofs, convergence theorems and
measure-theoretic framing are out of scope by construction.

## Status

Stages 1–4 build and run end to end. Nothing is generated from memory: a
derivation must be located in a retrieved source and restated, and every
citation is checked against what was actually fetched.

```
atlas/schema.py       node/edge types, four tiers, layer membership
atlas/graph.py        DAG enforcement, cycle demotion, closure, syllabus,
                      learning_path, alias resolution, transitive reduction
atlas/bootstrap.py    canon assembly from seeds + edges + amendments
atlas/merge.py        merges independently-authored edge proposals
atlas/ingest.py       stage 1 — repo and brief ingest, leak detection
atlas/architecture.py stage 2 — architecture layer behind a confirmation gate
atlas/retrieval.py    stage 4a — grounded sources + the citation guard
atlas/derivation.py   stage 4b — restate a derivation, then check it
atlas/llm.py          provider seam: Gemini, model rotation, headless Claude
```

Run:

```
python3 scripts/report.py            # canon bootstrap + saturation
python3 scripts/merge_report.py DIR  # merge edge proposals
python3 scripts/grounding.py         # which nodes have a source
python3 scripts/generate_all.py      # derivations across the canon (resumable)
python3 -m unittest discover -s tests
```

No dependencies. Python 3.11+. Copy `.env.example` to `.env` for an API key.

## The canon

Grown from derivations rather than authored top-down: 20 seed algorithms were
expanded with no floor, and the union of what their derivations invoke became
the canon.

| | |
|---|---|
| Nodes | **226** — 173 concepts, 33 techniques, 20 algorithms |
| Edges | **851**, depth 6–11 below each seed algorithm |
| Saturation | growth crossed 5% at k=15 seeds |
| Source coverage | **179/191** nodes have a retrievable source |

Internal prerequisite edges were authored in eight independent cluster passes
and merged under a single writer, then a second round filled the gaps the first
found. 473 + 92 proposed, 2 demoted to `related_to` where two clusters picked
opposite directions on a genuine co-definition.

All 64 round-1 flags are adjudicated in `data/canon_amendments.json`, with a
reason recorded for every decision including the eight rejected.

## What is enforced in code, not trusted to prompts

- `REQUIRES` is strictly acyclic. An edge closing a cycle is **demoted** to
  `related_to`, so the information survives and topological sort stays valid.
- Unknown node ids are rejected, never auto-created.
- Branching is capped at 5; drops apply at merge time so an edge removed to free
  a slot actually frees it.
- Amendments **fail loudly**. One that cannot be applied raises rather than
  no-opping — this caught a merge that silently never happened because a
  retarget had deleted its survivor node.
- Node ids are never reassigned. A merged id stays resolvable through an alias.
- Citations are checked against sources retrieved in that run
  (`cited_but_not_retrieved`). A plausible reference nobody fetched is worse
  than none, because the reader cannot tell.
- `closure_holes` is the completeness criterion: what a derivation invoked, minus
  what the node's closure reaches, minus the floor. It must be empty.

`Graph.transitive_reduction` exists but is deliberately **not applied**: it would
remove 145 individually-justified edges, and edges out of ALGORITHM nodes must
never be reduced because they record what a derivation invokes.

## Grounding

Coverage went 7% → 47% → 70% → 94% across four measurements. Every jump was a
measurement defect, not a coverage gain:

- HTTP errors were swallowed as missing articles, so rate limiting read as
  "cannot be grounded" — the state that would licence a fallback to recall.
- Wikipedia titles are sentence case; generated title-case names missed.
- Node ids are not what sources call things (100 alias entries now).

**Techniques ground at 100%** (32/32) — 27 by alias, 5 by extracting the passages
where their consumers perform the move, with provenance preserved so the citation
guard still holds. `transpose_identities` draws 2546 chars from three articles
that each perform the identity.

Four nodes are correctly marked `known_absent`: `anchor_box`, `objectness`,
`bounding_box_parameterization`, `non_max_suppression` are engineering
conventions with no derivation to ground, so under the grounding rule they get
no sheet.

**Coverage is not derivability.** 94% means a source exists; it does not mean the
source contains a derivation. Asked to derive the EKF from its Wikipedia article,
the model correctly returned `grounded: false` — the article states the equations
and Jacobians but never derives them. Of the first 9 nodes generated, 7 grounded
and 2 refused, both graph-search nodes that get described rather than derived.

## Generation cost

Each call sends ~6,500 input tokens (a windowed extract of a reference article).
Two independent bottlenecks:

- **Hosted**: quota, not latency. Gemini free tier allows 20 requests per *day*
  per model. `RotatingProvider` combines several models' separate daily buckets;
  billing removes the wall entirely for a few dollars over the whole canon.
- **Local**: memory bandwidth. Generation requires reading every weight per
  token, so a 7B-Q4 ceiling is ~27 tok/s on an M4 and ~23 tok/s on a Jetson Orin
  NX 16GB (102 GB/s). Prefill of a 6.5k-token prompt dominates per-call latency.

The lever is prompt size, which is a choice rather than a requirement: most of a
reference article is irrelevant to the derivation it contains.

Derivations are global and cached, so this is a one-time cost paid once for all
users and projects — which is also what makes human verification of a core set
affordable.

## Next

- Trim prompt size and re-measure whether the grounding rate holds.
- Run generation across the canon and count how many come back grounded. That
  number, not source coverage, is the real ceiling on how much of the atlas can
  exist.
- 32 round-2 flags remain unadjudicated in `data/canon_flags.json`, mostly
  combinatorics and sampling primitives.
