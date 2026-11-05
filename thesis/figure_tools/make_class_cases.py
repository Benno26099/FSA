# AI assistance: written with Claude (Anthropic; Claude Code, Claude Opus 5/5.5),
# then reviewed, run and validated by Ben Colbourne. See the thesis's
# Statement on the Use of Generative AI.
"""Write the 137-equation class suite as run_cases.py JSONL, one file per SA size.

Same shapes as python/test_einsum_class.py (every letter at n, plus one seeded
random size per equation), with each case labelled by the planner rule and the
number of pre-reduced operands, so precision and counters can be grouped.

    python/.venv/bin/python3 make_class_cases.py <python_dir> <out_dir>
"""
import json
import os
import sys

import numpy as np

# The planner lives in python/; importers (make_scaling_cases.py) put it on
# sys.path first.
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(sys.argv[1]))
from einsum import roles_helper          # noqa: E402
from test_einsum_class import load_expected, operand_shapes   # noqa: E402


def rule_of(eq):
    """The planner's core rule and how many operands it pre-reduces (mirrors mx_einsum)."""
    lhs, out = eq.split("->")
    ins = lhs.split(",")
    roles = roles_helper(ins, out)
    reduced = ["".join(c for c in op if not roles[c].startswith("R")) for op in ins]
    n_reduced = sum(r != op for r, op in zip(reduced, ins))
    if len(ins) == 1:
        return "unary", n_reduced
    v = list(roles_helper(reduced, out).values())
    c, e, fa, fb = v.count("C"), v.count("E"), v.count("Fa"), v.count("Fb")
    if e == 2:
        rule = "hadamard"
    elif e == 1 and c == 1:
        rule = "diag"
    elif e == 1 and fa + fb == 1:
        rule = "broadcast"
    elif e == 1:
        rule = "vec_hadamard"
    elif c == 2:
        rule = "trace"
    elif fa <= 1 and fb <= 1:
        rule = "matmul"
    else:
        rule = "sI"
    return rule, n_reduced


if __name__ == "__main__":
    py = os.path.abspath(sys.argv[1])
    out_dir = os.path.abspath(sys.argv[2])
    expected = load_expected(os.path.join(py, "einsum_class_expected_passes.tsv"))
    for n in (4, 8, 16, 32):
        path = os.path.join(out_dir, f"class_cases_sa{n}.jsonl")
        seen = set()
        with open(path, "w") as f:
            for seed, (eq, passes) in enumerate(expected.items()):
                rule, n_reduced = rule_of(eq)
                rng = np.random.default_rng(seed)
                rand = {letter: int(rng.integers(1, n + 1)) for letter in "ijkl"}
                for kind, sizes in (("full", {letter: n for letter in "ijkl"}), ("random", rand)):
                    shapes = operand_shapes(eq, sizes)
                    if (eq, tuple(shapes)) in seen:
                        continue
                    seen.add((eq, tuple(shapes)))
                    f.write(json.dumps({
                        "instance": f"class_{kind}", "family": rule, "step": seed,
                        "group": f"{n_reduced} reduced", "equation": eq, "raw_equation": eq,
                        "shapes": [list(s) for s in shapes], "count": passes,
                    }) + "\n")
        print(path, len(seen))
