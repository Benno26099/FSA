import argparse
import json
import os
import pickle
import random
import sys
from collections import Counter, defaultdict

DEFAULT_INSTANCES = "/mnt/c/Users/benac/Downloads/instances/instances"

# The old pattern table mx_einsum dispatched on
SUPPORTED = {
    # two rank-2 operands, one contracted index
    "ij,jk->ik", "ji,jk->ik", "ij,kj->ik", "ji,kj->ik",
    "ij,jk->ki", "ji,jk->ki", "ij,kj->ki", "ji,kj->ki",
    # two rank-2 operands, elementwise
    "ij,ij->ij", "ij,ij->",
    # single rank-2 operand
    "ij->ji", "ij->ij", "ij->i", "ij->j", "ij->",
}


def parse_eq(eq):
    """'ab,bc->ac' -> (['ab','bc'], 'ac'). Handles the implicit-output form."""
    if "->" in eq:
        lhs, rhs = eq.split("->")
    else:
        lhs, rhs = eq, None
    return lhs.split(","), rhs


def canonical(terms, out):
    """Relabel a step to one of SUPPORTED, or return None.
    """
    if out is None:
        return None

    for t in list(terms) + [out]:
        if len(set(t)) != len(t):
            return None

    if len(terms) == 1:
        (a,) = terms
        if len(a) != 2:
            return None
        i, j = a
        if out == a:
            return "ij->ij"
        if out == a[::-1]:
            return "ij->ji"
        if out == i:
            return "ij->i"
        if out == j:
            return "ij->j"
        if out == "":
            return "ij->"
        return None

    if len(terms) != 2:
        return None
    a, b = terms
    if len(a) != 2 or len(b) != 2:
        return None

    sa, sb, so = set(a), set(b), set(out)

    # Elementwise: identical index sets on both operands. 
    if sa == sb:
        if b != a:
            return None
        if so == sa and out == a:
            return "ij,ij->ij"
        if out == "":
            return "ij,ij->"
        return None

    # Matmul
    shared = sa & sb
    if len(shared) != 1:
        return None
    (k,) = shared
    if k in so:
        return None
    free_a = (sa - shared).pop()
    free_b = (sb - shared).pop()
    if so != {free_a, free_b} or len(out) != 2:
        return None

   
    lhs_a = "ij" if a[1] == k else "ji"     # contracted index trailing -> 'ij'
    lhs_b = "jk" if b[0] == k else "kj"     # contracted index leading  -> 'jk'
    rhs = "ik" if out[0] == free_a else "ki"
    return f"{lhs_a},{lhs_b}->{rhs}"


def classify(terms, out, dims, sa_size):
    """Return (bucket, canonical_eq_or_None) for the coverage breakdown."""
    ranks = [len(t) for t in terms]
    if any(r > 2 for r in ranks):
        return "rank>2", None
    if any(dims[c] > sa_size for t in terms for c in t):
        return "dim>sa", None
    if any(len(set(t)) != len(t) for t in terms):
        return "repeated-index", None
    if all(r <= 1 for r in ranks):
        return "scalar/vector", None
    if len(ranks) == 2 and sorted(ranks) == [1, 2]:
        return "mat-vec", None

    eq = canonical(terms, out)
    if eq is None:
        if len(terms) == 2 and out is not None and len(out) > 2:
            return "rank2-out-rank3+", None
        return "rank2-unsupported", None
    if eq in ("ij,ij->ij", "ij,ij->"):
        return "elementwise", eq
    if len(terms) == 1:
        return "unary", eq
    return "matmul", eq


def planner_canonical(terms, out):
    """Canonical equation of a step in the planner's class, or None if outside it.
    """
    if out is None or len(terms) > 2 or len(out) > 2:
        return None
    for t in list(terms) + [out]:
        if len(t) > 2 or len(set(t)) != len(t):
            return None
    letters = "".join(terms)
    if not set(out) <= set(letters):
        return None
    rename = {}
    for c in letters:
        rename.setdefault(c, "ijkl"[len(rename)])
    return ",".join("".join(rename[c] for c in t) for t in terms) + "->" + \
        "".join(rename[c] for c in out)


def planner_group(terms):
    """Reporting group for an in-class step, by operand ranks."""
    ranks = sorted(len(t) for t in terms)
    if ranks[-1] <= 1:
        return "scalar/vector"
    if ranks == [1, 2] or ranks == [0, 2]:
        return "mat-vec/scaling"
    if ranks == [2, 2]:
        return "real 2D"
    return "unary"


def index_sizes(format_string, tensors):
    """Map every index label to its dimension, from the input tensor shapes."""
    lhs, _ = parse_eq(format_string)
    sizes = {}
    for term, t in zip(lhs, tensors):
        for c, d in zip(term, t.shape):
            sizes[c] = d
    return sizes


def iter_steps(path):
    """Yield (step_index, terms, out) for every pairwise step of one instance."""
    from einsum_benchmark.meta import to_ssa_path, to_annotated_ssa_path

    fs, tensors, path_metas, _ = pickle.load(open(path, "rb"))
    sizes = index_sizes(fs, tensors)

    linear = path_metas[0][0]
    ssa = to_ssa_path(linear)
    annotated = to_annotated_ssa_path(fs, ssa, is_ascii=False)
    for k, (_id_a, _id_b, eq) in enumerate(annotated):
        terms, out = parse_eq(eq)
        yield k, terms, out, sizes


def family_of(name):
    return name.split("_", 1)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", default=DEFAULT_INSTANCES)
    ap.add_argument("--sa-size", type=int, default=16,
                    help="max operand dimension that fits the SA (default 16)")
    ap.add_argument("--mode", choices=("summary", "extract"), default="summary")
    ap.add_argument("--out", default="einsum_cases.jsonl")
    ap.add_argument("--sample", type=int, default=0,
                    help="extract mode: cap per (family, pattern) stratum. "
                         "0 = no cap, which is ~1M records.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--distinct", action="store_true",
                    help="extract mode: one record per distinct (equation, shapes), "
                         "with the number of steps it stands for. Ignores --sample.")
    ap.add_argument("--limit-instances", type=int, default=0,
                    help="debug: only read the first N instance files")
    args = ap.parse_args()

    if not os.path.isdir(args.instances):
        sys.exit(f"instances directory not found: {args.instances}")

    files = sorted(f for f in os.listdir(args.instances) if f.endswith(".pkl"))
    if args.limit_instances:
        files = files[:args.limit_instances]
    if not files:
        sys.exit(f"no .pkl files in {args.instances}")

    rng = random.Random(args.seed)
    buckets = Counter()
    patterns = Counter()
    per_family = defaultdict(Counter)
    strata = defaultdict(list)
    distinct = {}
    total = 0

    planner = Counter()
    planner_eqs = Counter()
    planner_family = defaultdict(Counter)

    for n, fname in enumerate(files, 1):
        name = fname[:-4]
        fam = family_of(name)
        print(f"[{n}/{len(files)}] {name}", file=sys.stderr, flush=True)
        try:
            steps = list(iter_steps(os.path.join(args.instances, fname)))
        except Exception as e:                      # keep going; report at the end
            print(f"    SKIPPED ({type(e).__name__}: {e})", file=sys.stderr)
            buckets["read-error"] += 1
            continue

        for k, terms, out, sizes in steps:
            total += 1
            bucket, eq = classify(terms, out, sizes, args.sa_size)
            buckets[bucket] += 1
            per_family[fam][bucket] += 1
            if eq is not None:
                patterns[eq] += 1

            ceq = planner_canonical(terms, out)
            if ceq is None or any(sizes[c] > args.sa_size for t in terms for c in t):
                continue
            group = planner_group(terms)
            planner[group] += 1
            planner_eqs[ceq] += 1
            planner_family[fam][group] += 1
            if args.mode == "extract" and args.distinct:
                shapes = [[sizes[c] for c in t] for t in terms]
                key = (ceq, str(shapes))
                if key not in distinct:
                    distinct[key] = {
                        "instance": name, "family": fam, "step": k, "group": group,
                        "equation": ceq, "raw_equation": ",".join(terms) + "->" + out,
                        "shapes": shapes, "count": 0, "families": [],
                    }
                distinct[key]["count"] += 1
                if fam not in distinct[key]["families"]:
                    distinct[key]["families"].append(fam)
            elif args.mode == "extract":
                shapes = [[sizes[c] for c in t] for t in terms]
                strata[(fam, ceq)].append({
                    "instance": name, "family": fam, "step": k, "group": group,
                    "equation": ceq, "raw_equation": ",".join(terms) + "->" + out,
                    "shapes": shapes,
                })

    print(f"\nTotal pairwise contraction steps: {total:,}")
    print(f"SA size filter: dims <= {args.sa_size}\n")
    executable = sum(patterns.values())
    for b, c in buckets.most_common():
        print(f"  {b:<20} {c:>10,}  {100*c/max(total,1):5.1f}%")
    print(f"\n  FSA-EXECUTABLE       {executable:>10,}  {100*executable/max(total,1):5.1f}%\n")
    for eq, c in patterns.most_common():
        print(f"    {eq:<14} {c:>10,}")

    rank2 = (buckets["elementwise"] + buckets["matmul"]
             + buckets["rank2-unsupported"])
    if rank2 > 0:
        print(f"\n  Both operands rank 2, output rank <= 2 ('real 2D work'): {rank2:,}")
        covered = buckets["elementwise"] + buckets["matmul"]
        print(f"  Of which FSA executes: {covered:,} ({100*covered/rank2:.1f}%)")

    in_class = sum(planner.values())
    nontrivial = in_class - planner["scalar/vector"]
    print(f"\nRole planner (Task 3b): steps in the 2D class with dims <= {args.sa_size}")
    for g in ("real 2D", "mat-vec/scaling", "unary", "scalar/vector"):
        print(f"  {g:<20} {planner[g]:>10,}  {100*planner[g]/max(total,1):5.1f}%")
    print(f"\n  IN CLASS             {in_class:>10,}  {100*in_class/max(total,1):5.1f}%")
    print(f"  excl. scalar/vector  {nontrivial:>10,}  {100*nontrivial/max(total,1):5.1f}%")
    if rank2 > 0:

        print(f"  real 2D covered: {planner['real 2D']:,} / {planner['real 2D']:,} (100.0%)"
              f"  [old table: {covered:,} / {rank2:,}; {rank2 - planner['real 2D']:,}"
              f" of those are scalar x matrix]")
    print(f"  distinct canonical equations seen: {len(planner_eqs)} of 137")
    for eq, c in planner_eqs.most_common(20):
        print(f"    {eq:<14} {c:>10,}")
    print("\n  per family (in class excl. scalar/vector / all steps):")
    for fam in sorted(per_family):
        fam_total = sum(per_family[fam].values())
        fam_nt = sum(planner_family[fam].values()) - planner_family[fam]["scalar/vector"]
        print(f"    {fam:<6} {fam_nt:>9,} / {fam_total:>9,}  {100*fam_nt/max(fam_total,1):5.1f}%")

    if args.mode == "extract" and args.distinct:
        recs = sorted(distinct.values(), key=lambda r: -r["count"])
        with open(args.out, "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        print(f"\nWrote {len(recs):,} distinct (equation, shapes) records to {args.out}, "
              f"standing for {sum(r['count'] for r in recs):,} steps.")
    elif args.mode == "extract":
        written = 0
        with open(args.out, "w") as fh:
            for key, recs in sorted(strata.items(), key=lambda kv: str(kv[0])):
                chosen = recs
                if args.sample and len(recs) > args.sample:
                    chosen = rng.sample(recs, args.sample)
                for r in chosen:
                    fh.write(json.dumps(r) + "\n")
                    written += 1
        print(f"\nWrote {written:,} records to {args.out} "
              f"across {len(strata)} (family, pattern) strata.")


if __name__ == "__main__":
    main()
