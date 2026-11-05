# AI assistance: written with Claude (Anthropic; Claude Code, Claude Opus 5/5.5),
# then reviewed, run and validated by Ben Colbourne. See the thesis's
# Statement on the Use of Generative AI.
"""List class equations whose plan uses TRANSPOSE_SRAM (transpose_a or mx_transpose). Host only."""
import sys, os
import numpy as np
py = sys.argv[1]
sys.path.insert(0, py)
import einsum
from test_einsum_class import load_expected, operand_shapes

log = []
real_matmul = einsum.mx_matmul
def fake_matmul(A, B, engine, transpose_a=False, transpose_b=False, transpose_output=False):
    log.append(("matmul_ta" if transpose_a else "matmul") + ("+tout" if transpose_output else ""))
    a = A.T if transpose_a else A
    b = B.T if transpose_b else B
    r = (a.astype(np.float32) @ b.astype(np.float32))
    return r.T if transpose_output else r
def fake_transpose(A, engine):
    log.append("mx_transpose"); return A.T.copy()
def fake_hadamard(A, B, engine):
    log.append("hadamard"); return (A.astype(np.float32) * B.astype(np.float32))
einsum.mx_matmul = fake_matmul
einsum.mx_transpose = fake_transpose
if hasattr(einsum, "mx_hadamard"): einsum.mx_hadamard = fake_hadamard
einsum.set_config_file = lambda *_: None

class Cfg: sa_rows = sa_cols = 8
import fsa.config
fsa.config.get_config = lambda: Cfg
n = 8
exp = load_expected(os.path.join(py, "einsum_class_expected_passes.tsv"))
hits = []
for eq in exp:
    log.clear()
    shapes = operand_shapes(eq, {c: n for c in "ijkl"})
    ops = [np.random.default_rng(0).random(s).astype(np.float16) for s in shapes]
    try:
        got = einsum.mx_einsum(eq, *ops, engine=None)
        ok = np.allclose(got, np.einsum(eq, *[o.astype(np.float64) for o in ops]), rtol=1e-2, atol=1e-2)
    except Exception as e:
        print("ERR", eq, type(e).__name__, e); continue
    if any(x.startswith("matmul_ta") or x == "mx_transpose" or "tout" in x for x in log):
        hits.append((eq, log.copy(), ok))
for h in hits: print(h)
print(len(hits), "of", len(exp))
