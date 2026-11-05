# AI assistance: written and run by Claude (Anthropic; Claude Code, Claude Opus 5.5), 2026-11-05.
# Run with the miniforge3 Python (has einsum_benchmark) from the python/ folder.
import os, sys, collections
sys.path.insert(0, "/home/ben/chipyard/chipyard-fsa/generators/fsa/python")
import extract_einsum_cases as X
d = X.DEFAULT_INSTANCES
files = sorted(f for f in os.listdir(d) if f.startswith("lm_") and f.endswith(".pkl"))
print("lm instances:", files)
for sa in (16, 32):
    buckets = collections.Counter(); groups = collections.Counter(); ranks = collections.Counter(); maxdim = collections.Counter()
    tot = 0
    for f in files:
        for k, terms, out, sizes in X.iter_steps(os.path.join(d, f)):
            tot += 1
            b, _ = X.classify(terms, out, sizes, sa); buckets[b] += 1
            ranks[(tuple(len(t) for t in terms), len(out))] += 1
            maxdim[max(sizes[c] for t in terms for c in t) if any(terms) else 0] += 1
            ceq = X.planner_canonical(terms, out)
            if ceq is not None and not any(sizes[c] > sa for t in terms for c in t):
                groups[X.planner_group(terms)] += 1
    print(f"\nSA {sa}: steps {tot}")
    print(" classify buckets:", buckets.most_common())
    print(" in-class planner groups:", groups.most_common(), "total in class:", sum(groups.values()))
    if sa == 16:
        print(" operand ranks / output rank:", ranks.most_common(12))
        print(" largest dimension per step:", sorted(maxdim.items())[:5], "...", sorted(maxdim.items())[-5:])
