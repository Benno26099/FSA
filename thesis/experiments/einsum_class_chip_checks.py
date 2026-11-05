# AI assistance: written with Claude (Anthropic; Claude Code, Claude Opus 5/5.5),
# then reviewed, run and validated by Ben Colbourne. See the thesis's
# Statement on the Use of Generative AI.
"""Chip checks for EINSUM_CLASSIFICATION.md: compositions using EXISTING kernels only.

Run from generators/fsa/python with the project venv:
    .venv/bin/python3 ../thesis/experiments/einsum_class_chip_checks.py
Calls the mx_* functions directly (no mx_einsum changes needed). 4x4 config.
"""
import numpy as np, einsum as e, fsa as F
from fsa.config import reset

CFG = '../../../sims/verilator/generated-src/chipyard.harness.TestHarness.FSA4X4Fp16Config/chipyard.harness.TestHarness.FSA4X4Fp16Config.FSAConfig.json'
SIM = '../../../sims/verilator/simulator-chipyard.harness-FSA4X4Fp16Config-debug'
F.init(CFG); e.set_config_file(CFG)

class Counting:
    def __init__(s, i): s.i, s.n = i, 0
    def execute(s, k): s.n += 1; return s.i.execute(k)

eng = Counting(F.VerilatorSimulator(SIM, output_dir='/tmp', max_cycles=10000000))
n = 4; rng = np.random.default_rng(1)
g = lambda *s: rng.uniform(-1, 1, s).astype(np.float16)
A, B = g(n, n), g(n, n); u, v = g(n), g(n)
f = lambda x: x.astype(np.float32)
def col(x): M = np.zeros((n, n), np.float16); M[:, 0] = x; return M
def row(x): M = np.zeros((n, n), np.float16); M[0, :] = x; return M

CHECKS = [
    ('ij,jk->ki in ONE pass: matmul(B,A,tA,tB)', lambda: e.mx_matmul(B, A, eng, transpose_a=True, transpose_b=True), np.einsum('ij,jk->ki', f(A), f(B))),
    ('ij,ij->j from existing Frobenius kernel row 0', lambda: e.mx_hadamard(A, B, eng, stream_ones=True)[0, :], np.einsum('ij,ij->j', f(A), f(B))),
    ('ij,j->i  mat-vec = matmul(A, v as column 0)', lambda: e.mx_matmul(A, col(v), eng)[:, 0], np.einsum('ij,j->i', f(A), f(v))),
    ('i,j->ij  outer = matmul(u col 0, v row 0)', lambda: e.mx_matmul(col(u), row(v), eng), np.einsum('i,j->ij', f(u), f(v))),
    ('i,i->    dot = matmul(u row 0, v col 0)[0,0]', lambda: e.mx_matmul(row(u), col(v), eng)[0, 0], np.einsum('i,i->', f(u), f(v))),
    ('i,i->i   vector Hadamard = hadamard rows 0', lambda: e.mx_hadamard(row(u), row(v), eng)[0, :], np.einsum('i,i->i', f(u), f(v))),
    ('ij,ji->ij via existing kernels: transpose(B) then hadamard', lambda: e.mx_hadamard(A, e.mx_transpose(B, eng), eng), np.einsum('ij,ji->ij', f(A), f(B))),
]
for name, fn, ref in CHECKS:
    reset(); F.init(CFG); eng.n = 0
    got = fn()
    ok = np.shape(got) == np.shape(ref) and np.allclose(got, ref, rtol=1e-2, atol=1e-2)
    print(f'{"PASS" if ok else "FAIL"}  passes={eng.n}  maxerr={np.max(np.abs(np.asarray(got, np.float64) - ref)):.2e}  {name}')
