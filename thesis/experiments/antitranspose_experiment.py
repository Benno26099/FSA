# AI assistance: written with Claude (Anthropic; Claude Code, Claude Opus 5/5.5),
# then reviewed, run and validated by Ben Colbourne. See the thesis's
# Statement on the Use of Generative AI.
"""
Experiment: where does the SA's anti_transpose come from?

Hypothesis (derived from RTL reading, METHOD_WRITING_HANDOFF.md section 3.2):
  1. LoadStationary's shift chain puts spad row k into PE column n-1-k
     -> reg[i][c] = S[n-1-c][i]
  2. revInput=True on the stream reverses each stream vector
  3. OutputDelayer only de-skews (its two reversals cancel)

Predictions for out = f(X, S), stream tile X, stationary spad tile S:
  A  LS normal,   stream revInput=True   -> X @ anti_transpose(S)     (known)
  B  LS normal,   stream revInput=False  -> (X @ S.T)[:, ::-1]        (known earlier finding)
  C  LS reversed, stream revInput=True   -> X @ S[:, ::-1].T
  D  LS reversed, stream revInput=False  -> X @ S.T

If OutputDelayer were the source of a reversal that "cannot be disabled",
D would NOT come out as a clean X @ S.T.

Runs one kernel per variant; nothing in the repo is modified.
"""
import os
import sys
import argparse
import numpy as np

REPO_PY = "/home/ben/chipyard/chipyard-fsa/generators/fsa/python"
sys.path.insert(0, REPO_PY)
os.chdir(REPO_PY)

import fsa as F
import einsum
from einsum import make_dma_load, make_dma_store, anti_transpose
from fsa.instructions import (
    FenceInstruction, MatrixInstructionHeader, MatrixInstructionSpad,
    MatrixInstrucionAcc, MatrixInstruction, MxFunc,
)
from fsa.kernel import Kernel
from fsa.config import get_config


def load_stationary(spad_tile, n, e_itemsize, reverse):
    row_bytes = spad_tile.shape[-1] * e_itemsize
    row0 = spad_tile.data_ptr // row_bytes
    return MatrixInstruction(
        MatrixInstructionHeader(
            semId=2, acquireValid=True, acquireSemValue=1,
            releaseValid=True, releaseSemValue=2,
            func=MxFunc.LOAD_STATIONARY.value, waitPrevAcc=False,
        ),
        MatrixInstructionSpad(
            addr=row0 + (n - 1 if reverse else 0),
            stride=-1 if reverse else 1,
            revInput=False, revOutput=False, delayOutput=False,
        ),
        MatrixInstrucionAcc(addr=0, stride=0, zero=False),
    )


def tensor_mul(spad_tile, acc_tile, e_itemsize, a_itemsize, rev_input):
    spad_row0 = spad_tile.data_ptr // (spad_tile.shape[-1] * e_itemsize)
    acc_row0 = acc_tile.data_ptr // (acc_tile.shape[-1] * a_itemsize)
    return MatrixInstruction(
        MatrixInstructionHeader(
            semId=1, acquireValid=False, acquireSemValue=0,
            releaseValid=True, releaseSemValue=1,
            func=MxFunc.TENSOR_MULTIPLICATION.value, waitPrevAcc=False,
        ),
        MatrixInstructionSpad(
            addr=spad_row0, stride=1,
            revInput=rev_input, revOutput=False, delayOutput=True,
        ),
        MatrixInstrucionAcc(addr=acc_row0, stride=1, zero=True),
    )


def run(engine, S, X, ls_reverse, rev_input):
    einsum._reset_allocator_if_configured()
    cfg = get_config()
    n = cfg.sa_rows
    e, a = cfg.e_type.itemsize, cfg.a_type.itemsize

    spad_x = F.alloc_spad(X.shape)
    spad_s = F.alloc_spad(S.shape)
    acc = F.alloc_accumulator((n, n))
    x_mem, s_mem = F.from_numpy(X), F.from_numpy(S)
    out_mem = F.alloc_mem((n, n), F.fp32)

    insts = [
        make_dma_load(x_mem, spad_x, sem_id=0, rel_val=1, rows=n, cols=n, e_itemsize=e),
        make_dma_load(s_mem, spad_s, sem_id=2, rel_val=1, rows=n, cols=n, e_itemsize=e),
        load_stationary(spad_s, n, e, ls_reverse),
        tensor_mul(spad_x, acc, e, a, rev_input),
        make_dma_store(acc, out_mem, sem_id=1, acq_val=1, rel_val=2,
                       rows=n, cols=n, a_itemsize=a),
        FenceInstruction(mx=True, dma=True, stop=True),
    ]
    k = Kernel(instructions=insts, input=[x_mem, s_mem, out_mem], output=out_mem)
    return F.to_numpy(engine.execute(k)).astype(np.float64)


def candidates(X, S):
    X, S = X.astype(np.float64), S.astype(np.float64)
    return {
        "X @ S.T":                 X @ S.T,
        "X @ anti_transpose(S)":   X @ anti_transpose(S),
        "(X @ S.T)[:, ::-1]":      (X @ S.T)[:, ::-1],
        "X @ S[:, ::-1].T":        X @ S[:, ::-1].T,
        "(X @ S.T)[::-1, :]":      (X @ S.T)[::-1, :],
        "X @ S":                   X @ S,
        "rot180(X @ S.T)":         (X @ S.T)[::-1, ::-1],
    }


VARIANTS = [
    ("A", False, True,  "X @ anti_transpose(S)"),
    ("B", False, False, "(X @ S.T)[:, ::-1]"),
    ("C", True,  True,  "X @ S[:, ::-1].T"),
    ("D", True,  False, "X @ S.T"),
]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="FSA4X4Fp16Config")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sims = "/home/ben/chipyard/chipyard-fsa/sims/verilator"
    long = "chipyard.harness.TestHarness." + args.config
    cfg_file = os.path.join(sims, "generated-src", long, long + ".FSAConfig.json")
    binary = os.path.join(sims, f"simulator-chipyard.harness-{args.config}-debug")
    os.makedirs(args.out, exist_ok=True)

    F.init(cfg_file)
    einsum.set_config_file(cfg_file)
    n = get_config().sa_rows
    engine = F.VerilatorSimulator(binary, output_dir=args.out, verbose=False)

    rng_x, rng_s = np.random.default_rng(42), np.random.default_rng(100)
    X = rng_x.uniform(-1, 1, (n, n)).astype(np.float16)
    S = rng_s.uniform(-1, 1, (n, n)).astype(np.float16)
    cands = candidates(X, S)

    all_ok = True
    print(f"\n=== {args.config} (n={n}) ===")
    for name, ls_rev, rev_in, predicted in VARIANTS:
        got = run(engine, S, X, ls_rev, rev_in)
        matches = [k for k, v in cands.items() if np.allclose(got, v, atol=1e-2, rtol=1e-2)]
        ok = predicted in matches
        all_ok &= ok
        print(f"{name}  LS {'reversed' if ls_rev else 'normal  '}  revInput={str(rev_in):5}  "
              f"predicted {predicted:24}  -> {'PASS' if ok else 'FAIL'}  matches: {matches}")
    print("ALL PREDICTIONS HOLD" if all_ok else "SOME PREDICTIONS FAILED")
