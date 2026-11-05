# AI assistance: written and run by Claude (Anthropic; Claude Code, Claude Opus 5.5), 2026-11-05,
# at Ben Colbourne's request. Reads thesis/results/final_new; no simulation.
"""Maximum normwise relative error by error level, over every executed case of every suite and size.

    python per_rule_errors.py [results_dir]     (default ../final_new)

Each case's rule and pre-reduction count come from the class suite, keyed by canonical equation;
the BLAS3 TN/TT cases (written non-canonically) are one-pass matrix products.
"""
import csv, glob, os, sys, collections

d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "final_new")
rule = {}
for f in glob.glob(os.path.join(d, "class_cases_*.csv")):
    for r in csv.DictReader(open(f)):
        rule[r["equation"]] = (r["family"], int(r["group"][0]))
rule["ji,jk->ik"] = rule["ji,kj->ik"] = ("matmul", 0)

def level(rl, nred, eq):
    if rl == "unary" or (rl in ("broadcast", "sI", "vec_hadamard") and nred == 0):
        return "exact"
    if rl in ("matmul", "diag", "trace") and nred == 0:
        return "fp32"
    return "fp16, scalar output" if eq.endswith("->") else "fp16, matrix/vector output"

errs = collections.defaultdict(list)
for f in glob.glob(os.path.join(d, "*.csv")):
    for r in csv.DictReader(open(f)):
        if r.get("rel_err") in (None, ""):
            continue
        rl, nred = rule[r["equation"]]
        errs[level(rl, nred, r["equation"])].append(float(r["rel_err"]))

print("cases:", sum(len(v) for v in errs.values()))
for k in ("exact", "fp32", "fp16, matrix/vector output", "fp16, scalar output"):
    v = errs[k]
    extra = f", {sum(e <= 1e-6 for e in v)} <= 1e-6" if k == "fp32" else ""
    print(f"{k:28} {len(v):5} cases, max {max(v):.2e}{extra}")
