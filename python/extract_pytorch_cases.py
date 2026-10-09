#!/usr/bin/env python
"""Extract the 2D contractive subset of PyTorch's einsum test cases.

Runs under the project venv (`uv run python`) -- numpy only.

Source: pytorch/pytorch, test/test_linalg.py, `test_einsum`, tag v2.6.0
(matching the torch 2.6.0 in this project's venv). PyTorch ships no tests in
its pip wheel, so the case list below is transcribed from that file rather than
imported. The upstream comment credits the cases to
https://gist.github.com/rockt/15ee013889d65342088e9260a377dc8f -- the widely
cited einsum tutorial gist, which is what makes these *recognisable* cases
rather than ones we invented.

Deliberately excluded:
  test_einsum_random        -- generates random equations, not a fixed set
  test_einsum_sublist_format -- the same cases through a different API
  test_einsum_corner_cases / _error_cases -- exercise error handling and
                               degenerate shapes, not contraction patterns

Classification reuses canonical() from extract_einsum_cases.py, so a case is
only emitted if it maps onto a pattern mx_einsum actually dispatches.
"""

import argparse
import json
import sys

from extract_einsum_cases import parse_eq, canonical

# (equation, [operand shapes]) exactly as test_einsum instantiates them.
#   x (5,)   y (7,)    A (3,5)   B (2,5)   C (2,3,5)  D (2,5,7)
#   E (7,9)  F (2,3,3,5)  G (5,4,6)  H (4,4)  I (2,3,2)
PYTORCH_CASES = [
    # Vector operations
    ("i->",            [(5,)],              "sum"),
    ("i,i->",          [(5,), (5,)],        "dot"),
    ("i,i->i",         [(5,), (5,)],        "vector elementwise mul"),
    ("i,j->ij",        [(5,), (7,)],        "outer"),
    # Matrix operations
    ("ij->ji",         [(3, 5)],            "transpose"),
    ("ij->j",          [(3, 5)],            "column sum"),
    ("ij->i",          [(3, 5)],            "row sum"),
    ("ij,ij->ij",      [(3, 5), (3, 5)],    "matrix elementwise mul"),
    ("ij,j->i",        [(3, 5), (5,)],      "matrix-vector multiply"),
    ("ij,kj->ik",      [(3, 5), (2, 5)],    "matmul"),
    ("ij,ab->ijab",    [(3, 5), (7, 9)],    "matrix outer product"),
    # Tensor operations
    ("Aij,Ajk->Aik",   [(2, 3, 5), (2, 5, 7)],  "batch matmul"),
    ("ijk,jk->i",      [(2, 3, 5), (3, 5)],     "tensor-matrix contraction"),
    ("aij,jk->aik",    [(2, 5, 7), (7, 9)],     "tensor-matrix contraction"),
    ("abCd,dFg->abCFg", [(2, 3, 3, 5), (5, 4, 6)], "tensor-tensor contraction"),
    ("ijk,jk->ik",     [(2, 3, 5), (3, 5)],     "double indices"),
    ("ijk,jk->ij",     [(2, 3, 5), (3, 5)],     "double indices"),
    ("ijk,ik->j",      [(2, 3, 5), (2, 5)],     "non contiguous"),
    ("ijk,ik->jk",     [(2, 3, 5), (2, 5)],     "non contiguous, double indices"),
    # Diagonals -- all rejected (FSA has no diagonal path; A2 dropped on the data)
    ("ii->",           [(4, 4)],            "trace"),
    ("ii->i",          [(4, 4)],            "diagonal"),
    ("iji->j",         [(2, 3, 2)],         "non-contiguous trace"),
    # Ellipsis forms -- resolved against the concrete shapes below
    ("i...->...",      [(4, 4)],            "ellipsis sum"),
    ("ki,...k->i...",  [(5, 3), (2, 5)],    "ellipsis, A.t() and B"),
    ("k...,jk->...",   [(5, 3), (2, 5)],    "ellipsis, A.t() and B"),
]

_LABELS = "ZYXWVUTSRQPON"  # fresh labels for ellipsis dims, unlikely to collide


def resolve_ellipsis(eq, shapes):
    """Replace '...' with concrete fresh labels, using the operand shapes.

    Returns (terms, out) with no ellipsis, or None if the equation cannot be
    resolved (ragged broadcast etc. -- not worth handling for this subset).
    """
    terms, out = parse_eq(eq)
    if out is None:
        return None
    if "..." not in eq:
        return terms, out

    # Width of the ellipsis for each term = rank - number of named indices.
    widths = []
    for t, sh in zip(terms, shapes):
        if "..." not in t:
            widths.append(0)
            continue
        named = len(t.replace("...", ""))
        w = len(sh) - named
        if w < 0:
            return None
        widths.append(w)
    nmax = max(widths) if widths else 0
    if nmax > len(_LABELS):
        return None
    # Broadcasting aligns ellipsis dims to the right.
    full = _LABELS[:nmax]

    def sub(term, w):
        return term.replace("...", full[nmax - w:] if w else "")

    new_terms = [sub(t, w) for t, w in zip(terms, widths)]
    new_out = sub(out, nmax)
    return new_terms, new_out


def index_sizes(terms, shapes):
    sizes = {}
    for t, sh in zip(terms, shapes):
        if len(t) != len(sh):
            return None
        for c, d in zip(t, sh):
            sizes[c] = d
    return sizes


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sa-size", type=int, default=16,
                    help="reject any case with an operand dimension > N")
    ap.add_argument("--out", default="pytorch_cases.jsonl")
    ap.add_argument("--show-rejected", action="store_true",
                    help="print why each excluded case was excluded")
    args = ap.parse_args()

    kept, rejected = [], []
    for eq, shapes, note in PYTORCH_CASES:
        resolved = resolve_ellipsis(eq, shapes)
        if resolved is None:
            rejected.append((eq, note, "unresolvable ellipsis"))
            continue
        terms, out = resolved
        sizes = index_sizes(terms, shapes)
        if sizes is None:
            rejected.append((eq, note, "rank/shape mismatch after resolution"))
            continue
        if any(d > args.sa_size for sh in shapes for d in sh):
            rejected.append((eq, note, f"dimension > {args.sa_size}"))
            continue
        if len(terms) > 2:
            rejected.append((eq, note, "more than 2 operands"))
            continue
        if any(len(t) > 2 for t in terms):
            rejected.append((eq, note, "operand rank > 2"))
            continue
        canon = canonical(terms, out)
        if canon is None:
            rejected.append((eq, note, "not a pattern mx_einsum dispatches"))
            continue
        kept.append({
            "suite": "pytorch",
            "source": "test/test_linalg.py::test_einsum @ v2.6.0",
            "raw_equation": eq,
            "resolved_equation": ",".join(terms) + "->" + out,
            "equation": canon,
            "shapes": [list(s) for s in shapes],
            "sa_size": args.sa_size,
            "note": note,
        })

    with open(args.out, "w") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")

    print(f"PyTorch einsum cases: {len(PYTORCH_CASES)} total, "
          f"{len(kept)} executable at SA size {args.sa_size}\n")
    for r in kept:
        print(f"  {r['raw_equation']:<16} -> {r['equation']:<12} "
              f"{r['shapes']}   # {r['note']}")
    if args.show_rejected:
        print(f"\n  rejected ({len(rejected)}):")
        for eq, note, why in rejected:
            print(f"    {eq:<18} {why:<36} # {note}")
    print(f"\nWrote {len(kept)} records to {args.out}")


if __name__ == "__main__":
    main()
