
import os
import sys
import argparse
import numpy as np
import fsa as F

import einsum
from einsum import print_results
from dispatcher import CountingEngine, run_case, reinit_fsa

EXPECTED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "einsum_class_expected_passes.tsv")


def load_expected(path):
    expected = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            eq, passes = line.rstrip("\n").split("\t")
            expected[eq] = int(passes)
    return expected


def operand_shapes(eq, sizes):
    inputs = eq.split("->")[0].split(",")
    return [tuple(sizes[letter] for letter in op) for op in inputs]


def get_tests(engine, sa_rows, expected):
    """Every canonical equation at full size and at one random size, seeded by row."""
    tests = {}
    for seed, (eq, passes) in enumerate(expected.items()):
        full = {letter: sa_rows for letter in "ijkl"}
        rng = np.random.default_rng(seed)
        rand = {letter: int(rng.integers(1, sa_rows + 1)) for letter in "ijkl"}

        for sizes in (full, rand):
            shapes = operand_shapes(eq, sizes)
            name = f"{eq}  [{', '.join('x'.join(map(str, s)) for s in shapes)}]"
            tests[name] = (
                lambda eq=eq, shapes=shapes, c=passes, name=name:
                    run_case(engine, eq, shapes, c, sa_rows, name=name)
            )
    return tests


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FSA einsum class test: all 137 canonical equations")
    parser.add_argument("--config", type=str, default="FSA4X4Fp16Config")
    parser.add_argument("--build_dir", type=str, default=None)
    parser.add_argument("--simulator_bin", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="/tmp")
    parser.add_argument("--test", type=str, default=None, help="Run a specific test by name")
    parser.add_argument("--list", action="store_true", help="List available tests")
    parser.add_argument("--max_cycles", type=int, default=10000000)
    args = parser.parse_args()

    expected = load_expected(EXPECTED_FILE)

    if args.list:
        print("Available tests:")
        for name in get_tests(None, 4, expected).keys():
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
    sa_rows = get_config().sa_rows

    engine = F.VerilatorSimulator(
        simulator_bin, output_dir=args.output_dir,
        max_cycles=args.max_cycles,
    )

    counting = CountingEngine(engine)

    tests = get_tests(counting, sa_rows, expected)

    if args.test:
        if args.test not in tests:
            print(f"Unknown test: {args.test}")
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
                results.append(einsum.TestResult(name=name, passed=False, error_msg=str(e)))

    all_passed = print_results(results)
    sys.exit(0 if all_passed else 1)
