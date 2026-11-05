# AI assistance: written with Claude (Anthropic; Claude Code, Claude Opus 5/5.5),
# then reviewed, run and validated by Ben Colbourne. See the thesis's
# Statement on the Use of Generative AI.
"""What each addition buys: equations of the 2D class and einsum-benchmark steps that depend on it.

    python capability_ablation.py <results_dir> [N ...]

Reads the planner's per-equation pass counts (the 137) and the distinct in-class benchmark cases
(benchmark_distinct_sa{N}_{N}x{N}.csv, one row per distinct case with the number of steps it stands
for). An equation depends on a primitive when its plan cannot run without it:
  elementwise  -- the Hadamard rule (both indices elementwise); every other rule is a matrix product
  transpose    -- the plan uses the transposed load (list from planner_transpose_audit.py)
  scalar       -- an operand of rank 0
Without the primitive those equations need the host (or are not run at all); nothing else changes.
"""
import csv
import os
import sys
from collections import Counter

HADAMARD = {"ij,ij->ij", "ij,ij->ji", "ij,ji->ij", "ij,ji->ji"}
# planner_transpose_audit.py, 21 of 137
TRANSPOSED_LOAD = {
    ",ij->ji", "i,ij->ji", "i,jk->kj", "ij,->ji", "ij,i->ji", "ij,ij->j", "ij,ij->ji", "ij,ik->ji",
    "ij,ik->jk", "ij,ik->ki", "ij,ik->kj", "ij,ji->ij", "ij,ji->ji", "ij,jk->ki", "ij,jk->kj",
    "ij,k->ji", "ij,ki->ji", "ij,ki->jk", "ij,kl->ji", "ij,kl->lk", "ij->ji",
}


def has_scalar(eq):
    return "" in eq.split("->")[0].split(",")


def real_2d(eq):
    ins = eq.split("->")[0].split(",")
    return len(ins) == 2 and all(len(op) == 2 for op in ins)


def main():
    res = sys.argv[1]
    sizes = [int(x) for x in sys.argv[2:]] or [16]
    here = os.path.dirname(os.path.abspath(__file__))
    tsv = os.path.join(here, "einsum_class_expected_passes.tsv")
    equations = [l.split("\t")[0] for l in open(tsv) if l.strip() and not l.startswith("#")]
    assert len(equations) == 137 and TRANSPOSED_LOAD <= set(equations) and HADAMARD <= set(equations)
    groups = {"elementwise multiply": HADAMARD, "transposed load": TRANSPOSED_LOAD,
              "scalar operands": {e for e in equations if has_scalar(e)}}

    for n in sizes:
        steps = Counter()
        with open(os.path.join(res, f"benchmark_distinct_sa{n}_{n}x{n}.csv")) as f:
            for row in csv.DictReader(f):
                steps[row["equation"]] += int(row["count"])
        total = sum(steps.values())
        total_2d = sum(c for e, c in steps.items() if real_2d(e))
        print(f"{n}x{n}: {total:,} in-class benchmark steps, {total_2d:,} of them real 2D "
              f"(both operands rank 2); {len(steps)} distinct equations occur")
        print(f"  {'without':<22}{'equations':>12}{'steps lost':>14}{'% in-class':>12}{'real-2D lost':>14}{'% real 2D':>11}")
        union = set()
        for name, eqs in groups.items():
            union |= eqs
            lost = sum(c for e, c in steps.items() if e in eqs)
            lost_2d = sum(c for e, c in steps.items() if e in eqs and real_2d(e))
            print(f"  {name:<22}{len(eqs):>8} /137{lost:>14,}{100 * lost / total:>11.1f}%"
                  f"{lost_2d:>14,}{100 * lost_2d / total_2d:>10.1f}%")
        lost = sum(c for e, c in steps.items() if e in union)
        lost_2d = sum(c for e, c in steps.items() if e in union and real_2d(e))
        print(f"  {'all three':<22}{len(union):>8} /137{lost:>14,}{100 * lost / total:>11.1f}%"
              f"{lost_2d:>14,}{100 * lost_2d / total_2d:>10.1f}%")
        top = sorted(((c, e) for e, c in steps.items() if e in union), reverse=True)[:6]
        print("  largest dependent equations: " + ", ".join(f"{e} {c:,}" for c, e in top))


if __name__ == "__main__":
    main()
