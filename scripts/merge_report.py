"""Report on merging cluster proposals into the canon."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.merge import depth_profile, merge, read_proposals
from atlas.schema import NodeType, Relation

PROPOSALS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("proposals")

proposals = read_proposals(PROPOSALS)
print(f"proposals found: {len(proposals)}")
for p in proposals:
    print(f"  {p.cluster:<22} {len(p.edges):>4} edges  {len(p.flags):>2} flags")

graph, report = merge(proposals)

print()
print("=" * 60)
print(f"accepted  {len(report.accepted)}")
print(f"demoted   {len(report.demoted)}   (would have closed a cycle)")
print(f"rejected  {report.rejected_count}")
for reason, items in sorted(report.rejected.items()):
    print(f"  {reason}: {len(items)}")
    for item in items[:6]:
        print(f"      {item}")
    if len(items) > 6:
        print(f"      ... {len(items) - 6} more")

if report.demoted:
    print()
    print("DEMOTED TO related_to (cycle would have formed):")
    for src, dst in report.demoted:
        print(f"  {src} -> {dst}")

depths = depth_profile(graph)
print()
print("DEPTH BELOW EACH SEED ALGORITHM  (was 1 for all before internal edges)")
for node_id, d in sorted(depths.items(), key=lambda kv: -kv[1]):
    print(f"  {d:>2}  {node_id}")

problems = graph.validate()
print()
print(f"VALIDATION  {'clean' if not problems else str(len(problems)) + ' problems'}")
for problem in problems[:10]:
    print(f"  - {problem}")

known = {"matrix", "vector", "matrix_transpose", "matrix_inverse", "partial_derivative",
         "derivative", "function_of_several_variables", "dot_product", "norm",
         "summation", "determinant", "basis", "linear_independence", "integral",
         "logarithm", "exponential_function", "vector_space", "matrix_multiplication"}
for target in ("extended_kalman_filter", "gauss_newton", "pca"):
    path = graph.syllabus(target, known=known)
    print()
    print(f"SYLLABUS  {target}  ({len(path)} nodes, dependency-ordered)")
    for i in range(0, len(path), 6):
        print("  " + " -> ".join(path[i:i + 6]))

if report.flags:
    print()
    print(f"FLAGS RAISED BY CLUSTERS ({len(report.flags)})")
    for cluster, flag in report.flags:
        node = flag.get("node", "?")
        issue = flag.get("issue", "")
        print(f"  [{cluster}] {node}: {issue}")
