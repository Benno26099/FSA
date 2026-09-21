
import os
import sys
import argparse
import numpy as np
import fsa as F

import einsum
from einsum import TestResult, print_results, mx_einsum

# tensor_generator already exists in test_matmul.py (and a second copy in
# test_transpose_dma.py). Importing rather than adding a third copy.
from test_matmul import tensor_generator


SEEDS = (42, 100) 

class CountingEngine:
    def __init__(self, inner):
        self.inner = inner
        self.count = 0

    def execute(self, kernel):
        self.count += 1
        return self.inner.execute(kernel)

    def reset(self):
        self.count = 0


def run_case(engine, eq, n, expected_calls, sa_rows):
    """Run one einsum pattern against numpy and check its invocation count.

    `engine` must be the CountingEngine wrapper, not the raw simulator.
    """
    # Distinct seeds per operand: with A == B an operand-order bug would pass
    # unnoticed (mx_col_sum deliberately calls mx_matmul with them swapped).
    operands = tuple(tensor_generator(sa_rows, s) for s in SEEDS[:n])

    engine.reset()
    got = mx_einsum(eq, *operands, engine=engine)   # engine is KEYWORD-ONLY
    calls = engine.count

    expected = np.einsum(eq, *operands)

    problems = []
    # Shape first: numpy will happily broadcast a wrong-shaped result and give a
    # misleading comparison rather than an error (cf. Bug 3 in the thesis log).
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

    name = eq if expected_calls > 0 else f"{eq}  (host passthrough, no chip pass)"
    return TestResult(
        name=name,
        passed=not problems,
        expected=expected,
        actual=got,
        error_msg="\n".join(problems),
    )


def get_tests(engine, sa_rows, sa_cols):
    """Return {test_name: callable() -> TestResult}, as test_matmul.get_tests does."""
    assert sa_rows == sa_cols, \
        f"Square SA required, got {sa_rows}x{sa_cols}"

    CASES = [ # (equation, n_operands, expected_invocations)
         ("ij,jk->ik", 2, 1),
         ("ji,jk->ik", 2, 1),
         ("ij,kj->ik", 2, 1),
         ("ji,kj->ik", 2, 1),
         ("ij,jk->ki", 2, 2),
         ("ji,jk->ki", 2, 2),
         ("ij,kj->ki", 2, 2),
         ("ji,kj->ki", 2, 2),
         ("ij->ji", 1, 1),
         ("ij->ij", 1, 0),
         ("ij->i", 1, 1),
         ("ij->j", 1, 1),
         ("ij->", 1, 1),
         ("ij,ij->ij", 2, 1),
         ("ij,ij->", 2, 1)
        ]

    tests = {}
    for eq, n, expected_calls in CASES:
        # Default arguments bind the CURRENT values. A bare `lambda:` would
        # capture the loop variables themselves, so all 13 entries would run
        # the last case -- silently, and all green.
        tests[eq] = (
            lambda eq=eq, n=n, c=expected_calls:
                run_case(engine, eq, n, c, sa_rows)
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
    # REQUIRED: without this einsum._reset_allocator_if_configured is a no-op,
    # the allocator never resets between kernels, and the run dies partway
    # through with "allocation failed: not enough memory".
    einsum.set_config_file(config_file)
    cfg = get_config()
    sa_rows, sa_cols = cfg.sa_rows, cfg.sa_cols

    engine = F.VerilatorSimulator(
        simulator_bin, output_dir=args.output_dir,
        max_cycles=args.max_cycles,
        vcdfile=args.vcdfile,
    )
    
    # Pass the WRAPPER, not the raw simulator -- otherwise the counter never
    # sees anything and every case reports 0 invocations.
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
