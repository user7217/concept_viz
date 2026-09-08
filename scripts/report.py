"""Report the canon bootstrap result: does the saturation hypothesis hold?"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.bootstrap import build_graph, load, mean_curve, reuse, saturation
from atlas.schema import NodeType

payload = load()
graph = build_graph(payload)
curve = saturation(payload)
mean = mean_curve(payload)
counts = reuse(payload)

print("=" * 66)
print("CANON BOOTSTRAP")
print("=" * 66)
print(f"seed algorithms      {len(payload['seeds'])}")
print(f"total invocations    {sum(len(s['concepts']) + len(s['techniques']) for s in payload['seeds'])}")
print(f"distinct canon nodes {len(counts)}")
for node_type, label in [(NodeType.CONCEPT, "concepts"), (NodeType.TECHNIQUE, "techniques")]:
    print(f"  {label:<18} {len(graph.by_type(node_type))}")
print(f"reuse factor         {sum(counts.values()) / len(counts):.2f} invocations per node")

print()
print("SATURATION (listed order)")
print(f"{'k':>3}  {'seed':<34} {'canon':>6} {'new':>5} {'growth':>8}")
for point, seed in zip(curve, payload["seeds"]):
    print(f"{point.n_seeds:>3}  {seed['name'][:34]:<34} {point.canon_size:>6} {point.added:>5} {point.added_pct:>7.1f}%")

print()
print("SATURATION (mean over 500 random orderings)")
print(f"{'k':>3}  {'canon':>7}  {'growth':>8}")
for k, size in enumerate(mean, start=1):
    growth = 100.0 * (size - mean[k - 2]) / mean[k - 2] if k > 1 else 100.0
    flag = "  <- under 5%" if k > 1 and growth < 5 else ""
    print(f"{k:>3}  {size:>7.1f}  {growth:>7.1f}%{flag}")

print()
print("HIGHEST-REUSE NODES")
for node_id, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:14]:
    kind = graph.nodes[node_id].type.value
    print(f"  {count:>2} seeds  {node_id:<34} {kind}")

print()
problems = graph.validate()
print(f"VALIDATION  {'clean' if not problems else str(len(problems)) + ' problems'}")
for problem in problems[:8]:
    print(f"  - {problem}")

print()
known = {"matrix", "vector", "matrix_transpose", "partial_derivative", "derivative",
         "function_of_several_variables", "dot_product", "norm", "summation"}
path = graph.syllabus("extended_kalman_filter", known=known)
print(f"SYLLABUS for EKF, floored at a linear-algebra background  ({len(path)} nodes)")
print("  " + " -> ".join(path[:9]))
print("  " + " -> ".join(path[9:]))
