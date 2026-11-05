# AI assistance: written with Claude (Anthropic; Claude Code, Claude Opus 5/5.5),
# then reviewed, run and validated by Ben Colbourne. See the thesis's
# Statement on the Use of Generative AI.
"""Write the FuseMax 2D subset as run_cases.py JSONL, one file per SA size, plus the
full cascade with each Einsum's status (fusemax_cascade.tsv).

Source: Nayak et al., "FuseMax: Leveraging Extended Einsums to Optimize
Attention Accelerator Design", MICRO 2024, arXiv:2406.10491, Einsums 31-55
(cascade 4, 3-pass attention; cascade 5, 1-pass attention; the division-reduced
form 31-32). The paper already drops batch and head. Indices: p = query
position, m = key/value position (softmax reduces over m), e = Q/K embedding,
f = V embedding; m0 = position within a key block of the 1-pass cascade.
Per-iteration tensors of cascade 5 are taken at a fixed m1, which makes them
2D. Running tensors (RM, RD, RNV) are read at m1 or m1+1 accordingly.

In class = a plain sum-of-products Einsum with rank <= 2 operands. Max
reductions, exp, division and tensor addition are extended-Einsum operators
outside the class; they are listed, not run.

    python/.venv/bin/python3 make_fusemax_cases.py <python_dir> <out_dir>
"""
import json
import os
import sys

py = os.path.abspath(sys.argv[1])
out_dir = os.path.abspath(sys.argv[2])
sys.path.insert(0, py)
from extract_einsum_cases import parse_eq, planner_canonical   # noqa: E402

# (Einsum number, tensor, cascade, expression as in the paper, einsum or None, why not)
CASCADE = [
    ("33", "QK",   "3-pass", "QK[m,p] = Q[e,p] x K[e,m]",            "ep,em->mp", None),
    ("34", "GM",   "3-pass", "GM[p] = max over m of QK[m,p]",        None, "max reduction"),
    ("35", "SN",   "3-pass", "SN[m,p] = exp(QK[m,p] - GM[p])",       None, "exp / subtraction"),
    ("36", "SD",   "3-pass", "SD[p] = sum over m of SN[m,p]",        "mp->p", None),
    ("37", "A",    "3-pass", "A[m,p] = SN[m,p] / SD[p]",             None, "division"),
    ("38", "AV",   "3-pass", "AV[f,p] = A[m,p] x V[f,m]",            "mp,fm->fp", None),
    ("31", "SNV",  "division-reduced", "SNV[f,p] = SN[m,p] x V[f,m]", "mp,fm->fp", None),
    ("32", "AV",   "division-reduced", "AV[f,p] = SNV[f,p] / SD[p]", None, "division"),
    ("44", "BQK",  "1-pass", "BQK[m0,p] = Q[e,p] x BK[e,m0]",        "ep,eM->Mp", None),
    ("45", "LM",   "1-pass", "LM[p] = max over m0 of BQK[m0,p]",     None, "max reduction"),
    ("46", "RM",   "1-pass", "RM'[p] = max(RM[p], LM[p])",           None, "elementwise max"),
    ("47", "SLN",  "1-pass", "SLN[m0,p] = exp(BQK[m0,p] - RM'[p])",  None, "exp / subtraction"),
    ("48", "SLD",  "1-pass", "SLD[p] = sum over m0 of SLN[m0,p]",    "Mp->p", None),
    ("49", "SLNV", "1-pass", "SLNV[f,p] = SLN[m0,p] x BV[f,m0]",     "Mp,fM->fp", None),
    ("50", "PRM",  "1-pass", "PRM[p] = exp(RM[p] - RM'[p])",         None, "exp / subtraction"),
    ("51", "SPD",  "1-pass", "SPD[p] = RD[p] x PRM[p]",              "p,p->p", None),
    ("52", "RD",   "1-pass", "RD'[p] = SLD[p] + SPD[p]",             None, "tensor addition"),
    ("53", "SPNV", "1-pass", "SPNV[f,p] = RNV[f,p] x PRM[p]",        "fp,p->fp", None),
    ("54", "RNV",  "1-pass", "RNV'[f,p] = SLNV[f,p] + SPNV[f,p]",    None, "tensor addition"),
    ("55", "AV",   "1-pass", "AV[f,p] = RNV[f,p] / RD[p]",           None, "division"),
]

with open(os.path.join(out_dir, "fusemax_cascade.tsv"), "w") as f:
    f.write("einsum\ttensor\tcascade\texpression\tequation\tcanonical\tstatus\n")
    for num, name, cascade, expr, eq, why in CASCADE:
        canon = ""
        if eq:
            terms, out = parse_eq(eq)
            canon = planner_canonical(terms, out)
            assert canon is not None, eq
        f.write("\t".join([num, name, cascade, expr, eq or "", canon or "",
                           "in class" if eq else f"outside: {why}"]) + "\n")

for n in (4, 8, 16, 32):
    path = os.path.join(out_dir, f"fusemax_cases_sa{n}.jsonl")
    count = 0
    with open(path, "w") as f:
        # Every index at n; the 1-pass block index M (m0) at n/2, the block
        # size the cascade would use for a 2-block key sequence.
        sizes = {"m": n, "p": n, "e": n, "f": n, "M": max(n // 2, 1)}
        for num, name, cascade, expr, eq, _ in CASCADE:
            if not eq:
                continue
            terms, out = parse_eq(eq)
            f.write(json.dumps({
                "suite": "fusemax", "op": f"({num}) {name}", "family": cascade,
                "group": "in class", "equation": planner_canonical(terms, out),
                "raw_equation": eq, "shapes": [[sizes[c] for c in t] for t in terms],
            }) + "\n")
            count += 1
    print(path, count)
