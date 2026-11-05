# AI assistance: written with Claude (Anthropic; Claude Code, Claude Opus 5/5.5),
# then reviewed, run and validated by Ben Colbourne. See the thesis's
# Statement on the Use of Generative AI.
"""Numpy model of the proposed role planner. Verifies the rule set covers all 137.

Primitives modelled (each = 1 chip pass):
  MM(X, W)  : matmul of two tiles (operand transposes free; output transpose by swap)
  HAD(X, Y) : Hadamard of two equally-shaped tiles
  RED       : ones-reduction of one operand along its R axes
Host work allowed: placing data (pad, row/col/1x1, diag, s*I), reading row/col/[0,0]/diagonal, summing <= n values.
"""
import numpy as np, sys
sys.path.insert(0, ".")
from einsum_class_enumerate import eqs   # the 137 canonical equations

def roles(ins, out):
    a = ins[0]; b = ins[1] if len(ins) == 2 else None
    R = {}
    for c in set(a) | set(b or ""):
        ina, inb, ino = c in a, b is not None and c in b, c in out
        R[c] = ("E" if ino else "C") if (ina and inb) else (("F" if ino else "R") + ("a" if ina else "b"))
    return R

def mat(X, s, row, col):
    """Arrange operand X (letters s) as a 2-D matrix with axes (row, col); None = size-1 axis."""
    if len(s) == 0:                       # scalar -> 1x1
        return X.reshape(1, 1)
    if len(s) == 1:
        v = X.reshape(-1)
        return v.reshape(-1, 1) if row == s else v.reshape(1, -1)
    return X if s == row + col else X.T   # transposes are free

def plan(eq, ops):
    lhs, out = eq.split("->"); ins = lhs.split(",") if lhs else [""]
    R = roles(ins, out)
    passes = 0
    # 1. pre-reduce R letters, operand by operand (one chip pass per operand that has any)
    rem = []
    for s, X in zip(ins, ops):
        red = [k for k, c in enumerate(s) if R[c].startswith("R")]
        if red:
            X = X.sum(axis=tuple(red)); passes += 1
            s = "".join(c for c in s if not R[c].startswith("R"))
        rem.append((s, X))
    if len(ins) == 1:                     # unary: only layout left
        s, X = rem[0]
        if len(s) == 2 and s != out:
            X = X.T; passes += 1
        return X, passes
    (sa, A), (sb, B) = rem
    C = [c for c in sa if R[c] == "C"]; E = [c for c in sa if R[c] == "E"]
    Fa = [c for c in sa if R[c] == "Fa"]; Fb = [c for c in sb if R[c] == "Fb"]
    if len(E) == 2:                                    # Hadamard
        Bt = B if sb == sa else B.T
        passes += 1 + (sb != sa)                       # transposed B costs a transpose pass
        Y = A * Bt
        if out != sa: Y = Y.T; passes += 1
        return Y, passes
    if len(E) == 1:
        e = E[0]
        if C:                                          # out[e] = sum_c A[e,c]B[e,c] = diag(A' B'^T)
            c = C[0]
            M = mat(A, sa, e, c) @ mat(B, sb, c, e); passes += 1
            return np.diag(M).copy(), passes
        # broadcast: e plus at most one free letter on one side
        if Fa:   # A has (e, f); out[...] = A[e,f] * v[e]
            f = Fa[0]; M = np.diag(B.reshape(-1)) @ mat(A, sa, e, f)   # rows e, cols f
        elif Fb:
            f = Fb[0]; M = np.diag(A.reshape(-1)) @ mat(B, sb, e, f)
        else:    # vector Hadamard: diag(u) @ v-as-column
            M = np.diag(A.reshape(-1)) @ B.reshape(-1, 1); passes += 1
            return M[:, 0], passes
        passes += 1
        return (M if out == e + f else M.T), passes    # output order by swap (free)
    if len(C) == 2:                                    # trace(A B^T) aligned
        M = mat(A, sa, C[0], C[1]) @ mat(B, sb, C[1], C[0]); passes += 1
        return np.trace(M), passes
    # general matmul: |C|<=1, |Fa|<=2, |Fb|<=2 ; scalars as 1x1, two free letters vs scalar via s*I
    if len(Fa) == 2:                                   # B is a scalar (or C-free)
        s_eye = np.eye(A.shape[sa.index(out[1])]) * B.reshape(())    # s*I, placed on the host
        M = mat(A, sa, out[0], out[1]) @ s_eye
        passes += 1; return M, passes
    if len(Fb) == 2:
        M = (np.eye(B.shape[sb.index(out[0])]) * A.reshape(())) @ mat(B, sb, out[0], out[1])
        passes += 1; return M, passes
    c = C[0] if C else None; fa = Fa[0] if Fa else None; fb = Fb[0] if Fb else None
    M = mat(A, sa, fa, c) @ mat(B, sb, c, fb); passes += 1
    if out == "":                return M[0, 0], passes
    if len(out) == 1:            return (M[:, 0] if out == fa else M[0, :]), passes
    return (M if out == fa + fb else M.T), passes      # swap for output order

rng = np.random.default_rng(0)
bad, hist = [], {}
for eq in sorted(eqs):
    for trial in range(20):
        lhs, out = eq.split("->"); ins = lhs.split(",")
        size = {c: int(rng.integers(1, 5)) for c in "ijkl"}
        ops = [rng.uniform(-1, 1, [size[c] for c in s]) for s in ins]
        got, p = plan(eq, ops)
        ref = np.einsum(eq, *ops)
        if np.shape(got) != np.shape(ref) or not np.allclose(got, ref):
            bad.append((eq, np.shape(got), np.shape(ref))); break
    hist[p] = hist.get(p, 0) + 1
print("equations:", len(eqs), " failures:", len(bad))
for b in bad[:20]: print("  FAIL", b)
print("passes histogram:", dict(sorted(hist.items())))
