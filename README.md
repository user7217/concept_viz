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

## Next: resolve the 64 flags

`data/canon_flags.json` records what the authoring passes could not settle.
Three groups:

1. **Vocabulary gaps.** Concepts the derivations genuinely need that the hand
   extraction missed. `rank` was flagged independently by two clusters;
   `skew_symmetric_matrix` is what so(3) literally *is*; `cross_product` was
   flagged by two. Also `characteristic_polynomial`, `null_space`,
   `measurement_model`, `cross_entropy`, `sigmoid`, `euler_angles`,
   `homogeneous_coordinates`, `path`, `inverse_laplace_transform`, complex
   numbers, and a plain `function` node.
2. **Merge candidates.** `log_odds_transformation`/`logit`/`log_odds` are three
   ids for one object; `svd_factorization`/`singular_value_decomposition` are
   procedure vs object; `weight_normalization` is a subset of `normalization`;
   `quadratic_form_differentiation` a subset of `matrix_calculus_gradient`.
3. **Mis-specified nodes.** `information_matrix` conflates the precision matrix
   with Fisher information. `taylor_expansion` conflates univariate and
   multivariate. `inequality_bounding` is too coarse. `sparse_matrix`,
   `algebraic_rearrangement`, `priority_queue` and `error_signal` may belong in
   FLOOR or not in a mechanics canon at all.

That the gap list exists is itself evidence for the caveat above: 179 was an
optimistic lower bound, and independent authoring found the holes that one
extraction pass did not.
