# Derivation Atlas

Maps the mathematics you must be able to derive on paper to understand — not
merely operate — the libraries a specific project depends on.

Design doc: https://claude.ai/code/artifact/87787bf4-2df0-4916-8b2c-48cce3d703dd

## Status

Stage 0 of the build: the canonical foundations ontology. Nothing else can be
built until expansion has a floor to terminate against.

```
atlas/schema.py      node/edge types, tiers, layer membership
atlas/graph.py       DAG enforcement, cycle demotion, closure, syllabus
atlas/bootstrap.py   canon union + saturation analysis
data/seeds.json      20 seed algorithms and what their derivations invoke
scripts/report.py    bootstrap report
tests/               15 tests over the graph invariants
```

Run:

```
python3 scripts/report.py
python3 -m unittest discover -s tests
```

No dependencies. Python 3.11+.

## Bootstrap result (measured, 20 seeds)

| | |
|---|---|
| Canon nodes | **179** (148 concepts, 31 techniques) |
| Total invocations | 301 |
| Mean reuse | 1.68 seeds per node (techniques 1.97, concepts 1.62) |
| Saturation | growth crosses 5% at **k=15**, still 3.3%/seed at k=20 |
| Single-seed nodes | 111 / 179 (62%) |

Estimate in the design doc was 120–200 nodes with 40–60 techniques. Node count
held; technique count came in low at 31.

### Caveats on that number

1. **Convergence is weaker than the design doc claims.** 62% of canon nodes are
   invoked by exactly one seed. "A small set repeated endlessly" is only true of
   the techniques and the linear-algebra core; the long tail is real.
2. **Saturation is domain-conditional.** The seeds span estimation, planning,
   geometry, optimization, control and ML. A genuinely new domain (signal
   processing, RF, graphics) will spike the curve again. k=15 is saturation
   *within these domains*, not universally.
3. **Naming consistency is artificially high.** Seeds and extraction were both
   produced in one pass, so ids are consistent by construction. Independent LLM
   extraction will produce far more alias variance and identity resolution will
   do much more work. Treat 179 as an optimistic lower bound.

## Canon internal edges (bootstrap step 3)

Authored in eight independent cluster passes — probability/estimation, linear
algebra, calculus/optimization, geometry/robotics, graphs/planning,
control/signals, ML concepts, techniques — then merged under a single writer.

| | |
|---|---|
| Edges proposed | 473 |
| Accepted as `REQUIRES` | 471 |
| Demoted to `related_to` | 2 (would have closed a cycle) |
| Rejected | 0 |
| Depth below seed algorithms | **6–11** (was 1 for all) |
| Total graph edges | 774 |
| Open flags | 64 |

The two demotions are `backward_recursion -> dynamic_programming` and
`marginalization -> total_probability`: cases where two clusters independently
picked opposite directions on a genuine co-definition. Both survive as
`related_to` rather than being dropped.

`syllabus()` is no longer vacuous. The EKF path is 35 nodes floored at a
linear-algebra background, ordered so nothing appears before its prerequisites.

Merge invariants are enforced in code, not trusted to the authoring pass:
unknown ids are rejected rather than auto-created, edges out of floor nodes are
rejected, branching is capped at 5, and any edge closing a cycle is demoted.
`merge()` starts from the seed graph so re-running is idempotent.

## Flag adjudication (round 2)

All 64 round-1 flags were adjudicated in `data/canon_amendments.json`, which
records a reason for every decision including the rejections.

| | |
|---|---|
| Nodes added | 28 (7 floor, 17 concepts, 4 techniques) |
| Merges | 2, losing ids kept resolvable |
| Tier corrections | 12 |
| Edges dropped | 9 |
| Splits | 2, with 7 consumer retargets |
| Flags rejected with reasons | 8 |
| Canon after round 2 | **226 nodes, 851 edges**, depth 6-11 |

Rejected rather than fixed: three flags wanted concepts for content that
belongs on an existing ALGORITHM node (round 1's brief omitted the algorithm
layer, so the clusters could not see it); one wanted Chebyshev's inequality for
`law_of_large_numbers`, which is proof machinery the ceiling excludes - resolved
by retiering LLN to `claim` instead; four wanted concepts no seeded algorithm
needs.

Two splits carried real risk, since retargeting the wrong consumer weakens a
derivation without breaking anything:

- `taylor_expansion` -> univariate + `multivariate_taylor_expansion`. Five
  consumers retargeted; `step_size` and `exponential_map` deliberately left on
  the univariate form.
- `normalization` -> probability + `unit_norm_normalization`.

### Invariants added

- **Amendments fail loudly.** An amendment that cannot be applied raises rather
  than no-oping. This caught a merge that silently did not happen because a
  retarget had deleted its survivor node.
- **Drops apply at merge time**, so an edge removed to free branching-cap space
  actually frees it.
- **Transitive reduction is reporting-only.** It exists (`Graph.transitive_reduction`)
  but is not applied: it would remove 145 true, individually-justified edges,
  and edges out of ALGORITHM nodes must never be reduced because they record
  what a derivation invokes, not a minimal cover.
- **Orphan checking is optional**, since it is a whole-graph property and
  meaningless on a partially merged graph.

## Next

32 round-2 flags remain unadjudicated in `data/canon_flags.json`. The largest
group is combinatorics and sampling primitives (`factorial`, `permutation`,
`uniform_distribution`, `cumulative_distribution_function`) that
`binomial_coefficient` and `sampling` currently bottom out without.

The canon is now usable as a termination floor, so the next build stage is
project ingest and the architecture layer.
