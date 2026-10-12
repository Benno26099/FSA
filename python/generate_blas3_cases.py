import argparse
import json
import sys

BLAS3_OPS = [
    ("GEMM_NN", "ij,jk->ik", ["random", "random"],      "C = A B"),
    ("GEMM_TN", "ji,jk->ik", ["random", "random"],      "C = A^T B"),
    ("GEMM_NT", "ij,kj->ik", ["random", "random"],      "C = A B^T"),
    ("GEMM_TT", "ji,kj->ik", ["random", "random"],      "C = A^T B^T"),
    ("SYRK",    "ij,kj->ik", ["random", "alias:0"],     "C = A A^T (result symmetric)"),
    ("SYMM",    "ij,jk->ik", ["symmetric", "random"],   "C = A B, A symmetric"),
    ("TRMM_U",  "ij,jk->ik", ["triu", "random"],        "B = A B, A upper triangular"),
    ("TRMM_L",  "ij,jk->ik", ["tril", "random"],        "B = A B, A lower triangular"),
]

SYMMETRIC_RESULT = {"SYRK"}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sa-size", type=int, default=16,
                    help="SA dimension N; operands are generated at k <= N")
    ap.add_argument("--sizes", type=str, default=None,
                    help="comma-separated k values (default: full sweep 1..N)")
    ap.add_argument("--full-only", action="store_true",
                    help="only k == N (full occupancy), skip the occupancy sweep")
    ap.add_argument("--seed", type=int, default=42,
                    help="base seed; each operand gets seed + a stable offset")
    ap.add_argument("--out", default="blas3_cases.jsonl")
    args = ap.parse_args()

    if args.sizes:
        ks = [int(x) for x in args.sizes.split(",") if x.strip()]
    elif args.full_only:
        ks = [args.sa_size]
    else:
        ks = list(range(1, args.sa_size + 1))

    bad = [k for k in ks if k < 1 or k > args.sa_size]
    if bad:
        sys.exit(f"sizes must satisfy 1 <= k <= sa_size ({args.sa_size}); got {bad}")

    n = 0
    with open(args.out, "w") as fh:
        for k in ks:
            for op, eq, ctors, note in BLAS3_OPS:
                shapes = [[k, k] for _ in ctors]
                operands = []
                for i, c in enumerate(ctors):
                    if c.startswith("alias:"):
                        operands.append({"gen": "alias", "index": int(c.split(":")[1])})
                    else:
                        operands.append({"gen": c, "seed": args.seed + 1000 * i + k})
                fh.write(json.dumps({
                    "suite": "blas3",
                    "op": op,
                    "k": k,
                    "sa_size": args.sa_size,
                    "occupancy": round((k * k) / (args.sa_size ** 2), 4),
                    "equation": eq,
                    "shapes": shapes,
                    "operands": operands,
                    "symmetric_result": op in SYMMETRIC_RESULT,
                    "note": note,
                }) + "\n")
                n += 1

    print(f"Wrote {n} cases to {args.out}")
    print(f"  SA size {args.sa_size}, k in {ks[0]}..{ks[-1]} ({len(ks)} sizes)"
          f" x {len(BLAS3_OPS)} ops")
    print(f"  equations exercised: {sorted({e for _, e, _, _ in BLAS3_OPS})}")


if __name__ == "__main__":
    main()
