
import os
import sys
import csv
import json
import argparse
import numpy as np
import fsa as F

import einsum
from einsum import mx_einsum
from dispatcher import CountingEngine, shape_generator, reinit_fsa, SEEDS

COUNTERS = ("execTime", "mxActive", "dmaActive", "mxInst", "dmaInst")


class PerfEngine(CountingEngine):
    """CountingEngine that also sums the simulator's perf counters over a case."""
    def reset(self):
        super().reset()
        self.totals = dict.fromkeys(COUNTERS, 0)

    def execute(self, kernel):
        result = super().execute(kernel)
        for name in COUNTERS:
            self.totals[name] += self.inner.last_counters.get(name, 0)
        return result


def make_operands(record, shapes):
    """Operands for one case: random per SEEDS, or as the record's 'operands' list asks.

    'operands' (BLAS3) gives each operand a generator: random, symmetric
    ((X + X^T) / 2), triu / tril, or alias (the same tensor as an earlier operand).
    """
    if "operands" not in record:
        return tuple(shape_generator(shape, seed) for shape, seed in zip(shapes, SEEDS))
    operands = []
    for spec, shape in zip(record["operands"], shapes):
        gen = spec["gen"]
        if gen == "alias":
            operands.append(operands[spec["index"]])
            continue
        x = shape_generator(shape, spec["seed"])
        if gen == "symmetric":
            x = ((x.astype(np.float32) + x.T.astype(np.float32)) / 2).astype(x.dtype)
        elif gen == "triu":
            x = np.triu(x)
        elif gen == "tril":
            x = np.tril(x)
        elif gen != "random":
            raise ValueError(f"unknown operand generator {gen!r}")
        operands.append(x)
    return tuple(operands)


def run_record(engine, record):
    """Run one JSONL case; returns its CSV row."""
    eq = record["equation"]
    shapes = [tuple(s) for s in record["shapes"]]
    operands = make_operands(record, shapes)

    row = {k: record.get(k, "") for k in ("suite", "op", "k", "instance", "family", "step", "group",
                                          "equation", "raw_equation", "count")}
    row["shapes"] = " ".join("x".join(map(str, s)) or "scalar" for s in shapes)

    engine.reset()
    try:
        got = mx_einsum(eq, *operands, engine=engine)
        expected = np.einsum(eq, *(o.astype(np.float64) for o in operands))
        err = np.abs(np.asarray(got, dtype=np.float64) - expected)
        row["passed"] = np.shape(got) == np.shape(expected) and \
            bool(np.allclose(got, expected, rtol=1e-2, atol=1e-2))
        row["max_abs_err"] = float(np.max(err))
        row["rel_err"] = float(np.max(err) / max(np.max(np.abs(expected)), 1e-12))
        row["error"] = ""
        row["symmetric_ok"] = bool(np.allclose(got, np.asarray(got).T, rtol=1e-2, atol=1e-2)) \
            if record.get("symmetric_result") else ""
    except Exception as e:
        row["passed"] = False
        row["max_abs_err"] = row["rel_err"] = row["symmetric_ok"] = ""
        row["error"] = f"{type(e).__name__}: {e}"
    row["passes"] = engine.count
    row.update(engine.totals)
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run JSONL einsum cases on the FSA simulator, write CSV")
    parser.add_argument("--cases", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--config", type=str, default="FSA4X4Fp16Config")
    parser.add_argument("--build_dir", type=str, default=None)
    parser.add_argument("--simulator_bin", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="/tmp")
    parser.add_argument("--max_cycles", type=int, default=10000000)
    parser.add_argument("--limit", type=int, default=0, help="only run the first N cases")
    args = parser.parse_args()

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

    with open(args.cases) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        records = records[:args.limit]

    F.init(config_file)
    einsum.set_config_file(config_file)

    engine = PerfEngine(F.VerilatorSimulator(
        simulator_bin, output_dir=args.output_dir,
        max_cycles=args.max_cycles,
    ))

    fields = ["suite", "op", "k", "instance", "family", "step", "group", "equation", "raw_equation",
              "count", "shapes", "passed", "symmetric_ok", "passes", "max_abs_err", "rel_err",
              *COUNTERS, "error"]
    passed = 0

    with open(args.out, "w", newline="", encoding="utf-8", errors="backslashreplace") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for n, record in enumerate(records, 1):
            print(f"\n[{n}/{len(records)}] {record['equation']}  {record.get('instance', '')}")
            reinit_fsa(config_file)
            row = run_record(engine, record)
            writer.writerow(row)
            f.flush()
            passed += row["passed"]

    print(f"\n{passed}/{len(records)} cases passed -> {args.out}")
    sys.exit(0 if passed == len(records) else 1)
