
import os
import sys
import argparse
import numpy as np
import fsa as F
import string

import einsum
from einsum import TestResult, print_results, mx_einsum

SEEDS = (42, 100, 69) 

def shape_generator(shape, seed):
    rng = np.random.default_rng(seed)
    operand_shape = rng.uniform(low=-1, high=1, size=(shape))
    operand_shape_f16 = operand_shape.astype(np.float16)
    return operand_shape_f16

def random_letter_generator(equation, seed):
    rng = np.random.default_rng(seed)
    old = sorted({c for c in equation if c.isalpha()})
    new = rng.choice(list(string.ascii_letters), size=len(old), replace=False)
    mapping = dict(zip(old, new))
    return "".join(mapping.get(c, c) for c in equation)


class CountingEngine:
    def __init__(self, inner):
        self.inner = inner
        self.count = 0

    def execute(self, kernel):
        self.count += 1
        return self.inner.execute(kernel)

    def reset(self):
        self.count = 0


def run_case(engine, eq, shapes, expected_calls, sa_rows, name=None):
    """Run one einsum pattern against numpy and check its invocation count.

    `engine` must be the CountingEngine wrapper, not the raw simulator.
    """
    # Distinct seeds per operand: with A == B an operand-order bug would pass
    # unnoticed (mx_col_sum deliberately calls mx_matmul with them swapped).
    
    operands = tuple(shape_generator(shape, seed) for shape, seed in zip(shapes, SEEDS))

    engine.reset()
    got = mx_einsum(eq, *operands, engine=engine)   # engine is KEYWORD-ONLY
    calls = engine.count

    # fp64 reference: np.einsum on fp16 operands accumulates in fp16, which on
    # long sums (16x16 and up) is less accurate than the chip's fp32 accumulate.
    expected = np.einsum(eq, *(o.astype(np.float64) for o in operands))

    problems = []
    # Shape first: numpy will happily broadcast a wrong-shaped result and give a
    # misleading comparison rather than an error 
    if np.shape(got) != np.shape(expected):
        problems.append(
            f"shape mismatch: got {np.shape(got)}, expected {np.shape(expected)}"
        )
    elif not np.allclose(got, expected, rtol=1e-2, atol=1e-2):
        max_err = np.max(np.abs(np.asarray(got, dtype=np.float64)
                                - np.asarray(expected, dtype=np.float64)))
        problems.append(
            f"value mismatch, max error {max_err}\n"
            f"Got:\n{got}\nExpected:\n{expected}"
        )
    # Reported separately from the value check: "right answer, wrong count" is a
    # fusion regression, not a dispatch or algebra bug.
    if calls != expected_calls:
        problems.append(
            f"invocation count: expected {expected_calls}, got {calls}"
        )

    if name is None:
        name = eq
    if expected_calls == 0:
        name += "  (host passthrough, no chip pass)"
    return TestResult(
        name=name,
        passed=not problems,
        expected=expected,
        actual=got,
        error_msg="\n".join(problems),
    )

def run_reject_case(engine, eq, shapes, expected_msg, name):
    operands = tuple(shape_generator(shape, seed) for shape, seed in zip(shapes, SEEDS))
    
    engine.reset()

    problems = []

    try:
        mx_einsum(eq, *operands, engine=engine)
        problems.append("accepted an invalid equation")
    except ValueError as e:
        if expected_msg not in str(e):
            problems.append(f"wrong rejection: expected {expected_msg!r} in {str(e)!r}")

    if engine.count != 0:
        problems.append(f"chip invoked {engine.count} times before rejecting")

    return TestResult(name=name, passed=not problems, error_msg="\n".join(problems))
    

def get_tests(engine, sa_rows, sa_cols):
    """Return {test_name: callable() -> TestResult}, as test_matmul.get_tests does."""
    assert sa_rows == sa_cols, \
        f"Square SA required, got {sa_rows}x{sa_cols}"

    CASES = [ # (equation, n_operands, expected_invocations)
         ("ij,jk->ik", 2, 1),
         ("ji,jk->ik", 2, 1),
         ("ij,kj->ik", 2, 1),
         ("ji,kj->ik", 2, 1),
         ("ij,jk->ki", 2, 1),
         ("ji,jk->ki", 2, 1),
         ("ij,kj->ki", 2, 1),
         ("ji,kj->ki", 2, 1),
         ("ij->ji", 1, 1),
         ("ij->ij", 1, 0),
         ("ij->i", 1, 1),
         ("ij->j", 1, 1),
         ("ij->", 1, 1),
         ("ij,ij->ij", 2, 1),
         ("ij,ij->", 2, 1),
         ("ij,jk", 2, 1),
         ("ji", 1, 1),
         ("ij", 1, 0),
         ("ij,ij", 2, 1)
        ]


    RECT_CASES = [ # (equation, operand shapes, expected_invocations)
         ("ij,jk->ik", [(3, 4), (4, 2)], 1),
         ("ji,kj->ki", [(4, 3), (2, 4)], 1),
         ("ij,jk->ki", [(3, 4), (4, 2)], 1),
         ("ij,ij->ij", [(2, 3), (2, 3)], 1),
         ("ij->ji",    [(3, 2)],         1),
         ("ij->i",     [(3, 2)],         1),
         ("ij->j",     [(3, 2)],         1),
         ("ij->",      [(3, 2)],         1),
         ("ij,ij->",   [(2, 3), (2, 3)], 1),
         ("ij,jk->ik", [(1, 4), (4, 1)], 1),   
         ("ab,bc->ac", [(2, 3), (3, 4)], 1),
         ("ij,j->i",   [(3, 4), (4,)],   1),
         ("ij,i->j",   [(3, 2), (3,)],   1),
         ("i,ij->j",   [(3,), (3, 2)],   1),
         ("j,ij->i",   [(4,), (3, 4)],   1),
         ("i,i->",     [(3,), (3,)],     1),
         ("i,j->ij",   [(3,), (2,)],     1),
         ("i,j->ji",   [(3,), (2,)],     1),
         # diagonal read 
         ("ij,ij->i",  [(3, 4), (3, 4)], 1),
         ("ij,ij->j",  [(3, 4), (3, 4)], 1),
         ("ij,ji->i",  [(3, 4), (4, 3)], 1),
         ("ij,ji->j",  [(3, 4), (4, 3)], 1),
         # trace 
         ("ij,ji->",   [(3, 4), (4, 3)], 1),
         # broadcast diag(v) 
         ("ij,j->ij",  [(3, 4), (4,)],   1),
         ("ij,j->ji",  [(3, 4), (4,)],   1),
         ("i,ij->ji",  [(3,), (3, 4)],   1),
         # vector Hadamard (E=1)
         ("i,i->i",    [(3,), (3,)],     1),
         # Hadamard with transposes 
         ("ij,ij->ji", [(2, 3), (2, 3)], 2),
         ("ij,ji->ij", [(2, 3), (3, 2)], 2),
         ("ij,ji->ji", [(2, 3), (3, 2)], 2),
         # pre-reduction of R letters (+1 pass per operand that has any)
         ("ij,jk->i",  [(3, 4), (4, 2)], 2),   # row-sum B, then mat-vec
         ("ij,jk->k",  [(3, 4), (4, 2)], 2),   # col-sum A, then vec-mat
         ("ij,jk->",   [(3, 4), (4, 2)], 3),   # reduce both, then dot
         ("ij,kl->ik", [(3, 4), (2, 3)], 3),   # reduce both, then outer
         ("ij,ik->ij", [(3, 4), (3, 2)], 2),   # row-sum B, then broadcast
         ("ij,i->i",   [(3, 4), (3,)],   2),   # row-sum A, then vector Hadamard
         ("i,jk->j",   [(3,), (2, 4)],   3),   # sum u, row-sum B, then scale
         # s*I (two free letters vs a scalar from a reduction) -- needs 3d
         ("ij,k->ij",  [(3, 4), (2,)],   2),
         ("ij,kl->ji", [(3, 4), (2, 3)], 2),
         # unary through the pipeline
         ("i->i",      [(3,)],           0),
         ("i->",       [(3,)],           1),
         ("->",        [()],             0),
         # scalar (rank-0) operands
         (",ij->ij",   [(), (3, 4)],     1),   # s*I
         ("ij,->ji",   [(3, 4), ()],     1),   # s*I, transposed output
         (",i->i",     [(), (3,)],       1),   # scalar x vector (matmul)
         (",->",       [(), ()],         1),   # scalar x scalar (1x1 matmul)
         (",ij->i",    [(), (3, 4)],     2),   # row-sum, then scale
        ]

    N = sa_rows
    REJECT_CASES = [
        ("ijk,kl->il",   [(N, N), (N, N)],          "operand rank > 2"),
        ("ii->i",        [(N, N)],                  "repeated within one operand"),
        ("ij,jk,kl->il", [(N, N)] * 3,              "at most two operands"),
        ("ij,jk->ikj",   [(N, N), (N, N)],          "output rank > 2"),
        ("ij->k",        [(N, N)],                  "not in any input"),
        ("...ij->ij",    [(N, N)],                  "ellipsis"),
        ("ij,jk->ik",    [(N, N)],                  "expects 2 operands"),
        ("ij->ij",       [(N,)],                    "expects ndim"),
        ("i1->i",        [(N, N)],                  "non-letter index"),
        (",ij->ij",      [(N, N), (N, N)],          "expects ndim"),   # scalar given a matrix
        ("ij,jk->ik",    [(3, 4), (3, 2)],          "has size"),
        ("ij->ij",       [(N + 1, N + 1)],          "exceeds SA size")
    ]

    tests = {}
    for eq, n, expected_calls in CASES:
        tests[eq] = (
            lambda eq=eq, shapes=[(sa_rows, sa_rows)] * n, c=expected_calls:
                run_case(engine, eq, shapes, c, sa_rows)
        )

    for eq, shapes, expected_calls in RECT_CASES:
        name = f"{eq}  [{', '.join('x'.join(map(str, s)) for s in shapes)}]"
        tests[name] = (
            lambda eq=eq, shapes=shapes, c=expected_calls, name=name:
                run_case(engine, eq, shapes, c, sa_rows, name=name)
        )

    for eq, shapes, msg in REJECT_CASES:
        reject_name = f"reject: {eq}  [{', '.join('x'.join(map(str, s)) for s in shapes)}]"
        tests[reject_name] = (
            lambda eq=eq, shapes=shapes, msg=msg, name=reject_name:
                run_reject_case(engine, eq, shapes, msg, name)
        )

    for seed, (eq, n, expected_calls) in enumerate(CASES):
        if "->" not in eq:
            continue
        relabelled = random_letter_generator(eq, seed)
        name = f"{relabelled}  (relabel of {eq})"
        tests[name] = (
            lambda eq=relabelled, shapes=[(sa_rows, sa_rows)] * n,
                   c=expected_calls, name=name:
                run_case(engine, eq, shapes, c, sa_rows, name=name)
        )
    return tests


# Main

def reinit_fsa(config_file: str):
    """Reset and re-initialize FSA state (memory allocators, etc.)."""
    from fsa.config import reset
    reset()
    import fsa as F
    F.init(config_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FSA einsum dispatcher test suite")
    parser.add_argument("--config", type=str, default="FSA4X4Fp16Config")
    parser.add_argument("--build_dir", type=str, default=None)
    parser.add_argument("--simulator_bin", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="/tmp")
    parser.add_argument("--test", type=str, default=None, help="Run a specific test by name")
    parser.add_argument("--list", action="store_true", help="List available tests")
    parser.add_argument("--max_cycles", type=int, default=10000000)
    parser.add_argument("--vcdfile", type=str, default=None,
                        help="Path to write VCD waveform (requires -debug simulator)")
    args = parser.parse_args()

    if args.list:
        print("Available tests:")
        for name in get_tests(None, 4, 4).keys():
            print(f"  {name}")
        sys.exit(0)

    if args.build_dir is None:
        build_dir = os.path.join("..", "..", "..", "sims", "verilator")
    else:
        build_dir = args.build_dir

    long_name = "chipyard.harness.TestHarness." + args.config
    config_file = os.path.join(
        build_dir, "generated-src", long_name,
        long_name + ".FSAConfig.json"
    )

    if not os.path.isfile(config_file):
        print(f"Config file not found: {config_file}")
        sys.exit(1)

    simulator_bin = args.simulator_bin
    if not simulator_bin or not os.path.isfile(simulator_bin):
        print(f"Simulator binary not found: {simulator_bin}")
        sys.exit(1)

    from fsa.config import get_config
    F.init(config_file)
    
    einsum.set_config_file(config_file)
    cfg = get_config()
    sa_rows, sa_cols = cfg.sa_rows, cfg.sa_cols

    engine = F.VerilatorSimulator(
        simulator_bin, output_dir=args.output_dir,
        max_cycles=args.max_cycles,
        vcdfile=args.vcdfile,
    )
    
    counting = CountingEngine(engine)

    tests = get_tests(counting, sa_rows, sa_cols)

    if args.test:
        if args.test not in tests:
            print(f"Unknown test: {args.test}")
            print(f"Available: {', '.join(tests.keys())}")
            sys.exit(1)
        reinit_fsa(config_file)
        results = [tests[args.test]()]
    else:
        results = []
        for name, test_fn in tests.items():
            print(f"\n{'='*60}")
            print(f"Running: {name}")
            print(f"{'='*60}")
            reinit_fsa(config_file)
            try:
                results.append(test_fn())
            except Exception as e:
                results.append(TestResult(name=name, passed=False, error_msg=str(e)))

    all_passed = print_results(results)
    sys.exit(0 if all_passed else 1)
