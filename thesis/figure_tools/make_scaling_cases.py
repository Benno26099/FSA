# AI assistance: written with Claude (Anthropic; Claude Code, Claude Opus 5/5.5),
# then reviewed, run and validated by Ben Colbourne. See the thesis's
# Statement on the Use of Generative AI.
"""Write the Layer 3 scaling sweep as run_cases.py JSONL, one file per SA size.

One representative equation per core form (TESTING_PLAN.md, Layer 3), every
letter at size k for k = 1..N, at N = 4, 8, 16, 32. Each case is labelled with
the planner rule (as make_class_cases.rule_of computes it) and the form name.
Forms that use TRANSPOSE_SRAM are in group "transpose" so they can be kept out
of the scaling curves (B3 is future work: port-0 routing, one AXI read per
element) and reported separately.

    python/.venv/bin/python3 make_scaling_cases.py <python_dir> <out_dir>
"""
import json
import os
import sys

py = os.path.abspath(sys.argv[1])
out_dir = os.path.abspath(sys.argv[2])
sys.path.insert(0, py)
from make_class_cases import rule_of       # noqa: E402
from test_einsum_class import load_expected   # noqa: E402

# (form, canonical equation, group)
FORMS = [
    ("matmul",        "ij,jk->ik", "core"),
    ("mat-vec",       "ij,j->i",   "core"),
    ("outer",         "i,j->ij",   "core"),
    ("dot",           "i,i->",     "core"),
    ("hadamard",      "ij,ij->ij", "core"),
    ("frobenius",     "ij,ij->",   "core"),
    ("broadcast",     "i,ij->ij",  "core"),
    ("vec-hadamard",  "i,i->i",    "core"),
    ("scaling",       ",ij->ij",   "core"),
    ("diag",          "ij,ji->i",  "core"),
    ("trace",         "ij,ji->",   "core"),
    ("row-sum",       "ij->i",     "core"),
    ("total-sum",     "ij->",      "core"),
    ("pre-reduce",    "ij,kl->ik", "core"),
    ("transpose",     "ij->ji",    "transpose"),
    ("matmul-TN",     "ij,ik->jk", "transpose"),
]

expected = load_expected(os.path.join(py, "einsum_class_expected_passes.tsv"))
for _, eq, _ in FORMS:
    assert eq in expected, eq

for n in (4, 8, 16, 32):
    path = os.path.join(out_dir, f"scaling_cases_sa{n}.jsonl")
    count = 0
    with open(path, "w") as f:
        for form, eq, group in FORMS:
            rule, n_reduced = rule_of(eq)
            lhs = eq.split("->")[0]
            for k in range(1, n + 1):
                shapes = [[k] * len(op) for op in lhs.split(",")]
                f.write(json.dumps({
                    "suite": "scaling", "op": form, "k": k, "family": rule,
                    "group": group, "equation": eq, "raw_equation": eq,
                    "shapes": shapes, "count": expected[eq],
                }) + "\n")
                count += 1
    print(path, count)
