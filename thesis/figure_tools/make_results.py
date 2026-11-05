# AI assistance: written with Claude (Anthropic; Claude Code, Claude Opus 5/5.5),
# then reviewed, run and validated by Ben Colbourne. See the thesis's
# Statement on the Use of Generative AI.
"""Build every results table (CSV + Markdown) and results plot from thesis/results.

    ~/.local/share/thesis-plots/bin/python make_results.py <results_dir> <figures_dir>

Inputs it reads (missing ones are skipped and reported):
  logs/*.log                                   suite pass counts, attention MAE + counters
  coverage_planner_sa{N}.txt                   extract_einsum_cases.py --mode summary
  benchmark_distinct_sa{N}_{N}x{N}.csv         run_cases.py over every distinct benchmark case
  class_cases_sa{N}_{N}x{N}.csv                run_cases.py over the 137-equation class
  scaling_cases_sa{N}_{N}x{N}.csv              run_cases.py over the Layer 3 sweep (make_scaling_cases.py)
  fusemax_cascade.tsv, fusemax_cases_{N}x{N}.csv  FuseMax cascade (make_fusemax_cases.py) and its run_cases.py runs

Palette: reference dataviz palette, light mode (print). Single-series charts use
slot 1; series = SA size or reduction count use slots 1-3 in fixed order.
"""
import csv
import os
import re
import statistics as st
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import matplotlib.ticker   # noqa: E402

SIZES = (4, 8, 16, 32)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a")
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.labelcolor": INK2})
missing = []
THOUSANDS = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")

def arr(n):
    """Array size as printed in figures."""
    return f"{n}\u00d7{n}"


RULE_NAMES = {"matmul": "Matrix product", "sI": "Scalar \u00d7 identity", "broadcast": "Broadcast",
              "vec_hadamard": "Vector Hadamard", "diag": "Diagonal", "trace": "Trace",
              "hadamard": "Hadamard", "unary": "Unary"}
FORM_NAMES = {"matmul": "Matrix product", "mat-vec": "Matrix\u2013vector", "outer": "Outer product",
              "dot": "Dot product", "hadamard": "Hadamard", "frobenius": "Frobenius inner product",
              "broadcast": "Broadcast", "vec-hadamard": "Vector Hadamard", "scaling": "Scalar scaling",
              "diag": "Diagonal", "trace": "Trace", "row-sum": "Row sum", "total-sum": "Total sum",
              "pre-reduce": "Pre-reduction", "transpose": "Transpose (transposed load)",
              "matmul-TN": "A\u1d40B (transposed load)"}


def read_csv(path):
    if not os.path.isfile(path):
        missing.append(os.path.basename(path))
        return None
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_table(tdir, name, header, rows, md):
    with open(os.path.join(tdir, name + ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    md.append(f"\n### {name}\n\n| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
    for r in rows:
        md.append("| " + " | ".join(str(x) for x in r) + " |\n")


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, fdir, name):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(os.path.join(fdir, f"{name}.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def suites(res, tdir, md):
    logs = os.path.join(res, "logs")
    pat = re.compile(r"(\d+)/(\d+) tests passed")
    rows = []
    for suite in ("dispatcher", "class_suite", "test_matmul", "test_transpose_dma"):
        row = [suite]
        for n in SIZES:
            p = os.path.join(logs, f"{suite}_{n}x{n}.log")
            m = pat.findall(open(p).read()) if os.path.isfile(p) else []
            row.append(f"{m[-1][0]}/{m[-1][1]}" if m else "")
            if not m:
                missing.append(os.path.basename(p))
        rows.append(row)
    write_table(tdir, "T1_suite_pass_counts", ["suite"] + [f"{n}x{n}" for n in SIZES], rows, md)

    rows = []
    for n in SIZES:
        p = os.path.join(logs, f"attention_{n}x{n}.log")
        if not os.path.isfile(p):
            missing.append(os.path.basename(p))
            continue
        txt = open(p).read()
        mae = re.search(r"'MAE': np.float32\(([0-9.e-]+)\)", txt)
        c = {k: int(v) for k, v in re.findall(r"FSA: (\w+) =\s+(\d+)", txt)}
        if mae and c:
            rows.append([f"{n}x{n}", mae.group(1), c["execTime"], c["mxActive"], c["dmaActive"],
                         f"{100 * c['mxActive'] / c['execTime']:.1f}%"])
    write_table(tdir, "T2_attention_regression",
                ["SA", "MAE vs torch", "execTime", "mxActive", "dmaActive", "mxActive/execTime"], rows, md)

    # T2b: the same attention on the upstream FSA (commit f4051d9) under the same simulation driver.
    up_dir, rows = os.path.join(res, "upstream"), []
    def attn(path):
        if not os.path.isfile(path):
            missing.append(os.path.relpath(path, res))
            return None, {}
        txt = open(path).read()
        mae = re.search(r"'MAE': np.float32\(([0-9.e-]+)\)", txt)
        return (mae.group(1) if mae else None), {k: int(v) for k, v in re.findall(r"FSA: (\w+) =\s+(\d+)", txt)}
    for n in SIZES:
        mu, cu = attn(os.path.join(up_dir, f"attention_upstream_{n}x{n}.log"))
        mf, cf = attn(os.path.join(up_dir, f"attention_fork_{n}x{n}.log"))
        if not (cu and cf):
            continue
        d = cf["execTime"] - cu["execTime"]
        rows.append([f"{n}x{n}", mu, mf, "yes" if mu == mf else "NO", cu["execTime"], cf["execTime"],
                     f"+{d} ({100 * d / cu['execTime']:.1f}%)", f"{cf['mxActive'] - cu['mxActive']:+d}",
                     f"{cf['dmaActive'] - cu['dmaActive']:+d}", f"{cu['rawInst']} / {cf['rawInst']}"])
    if rows:
        write_table(tdir, "T2b_attention_vs_upstream",
                    ["SA", "MAE upstream", "MAE extended", "MAE identical", "execTime upstream",
                     "execTime extended", "difference", "mxActive difference", "dmaActive difference",
                     "instruction words (upstream / extended)"], rows, md)
        md.append("\nFlashAttention (main.py, seq = N, seed 0) on the upstream FSA (commit f4051d9, its own "
                  "Python) and on the extended FSA, both built with the same simulation driver. Capability: "
                  "upstream runs 0 of the 137 equations of the 2D class; the extended FSA runs all 137.\n")


def coverage(res, tdir, fdir, md):
    rows, cats16 = [], None
    for n in SIZES:
        p = os.path.join(res, f"coverage_planner_sa{n}.txt")
        if not os.path.isfile(p):
            missing.append(os.path.basename(p))
            continue
        txt = open(p).read()
        num = lambda label: int(re.search(rf"^\s*{re.escape(label)}\s+([\d,]+)", txt, re.M).group(1).replace(",", ""))
        total = int(re.search(r"Total pairwise contraction steps: ([\d,]+)", txt).group(1).replace(",", ""))
        old = re.search(r"Of which FSA executes: ([\d,]+) \(([\d.]+)%\)", txt)
        real2d = num("real 2D")
        rows.append([f"{n}x{n}", f"{total:,}", f"{num('IN CLASS'):,}", f"{num('excl. scalar/vector'):,}",
                     f"{real2d:,}", f"{old.group(1)} ({old.group(2)}%)", f"{real2d:,} (100.0%)"])
        if n == 16:
            cats16 = [("Both operands rank 2", real2d),
                      ("Matrix\u2013vector or scalar\u2013matrix", num("mat-vec/scaling")),
                      ("Scalars and vectors only", num("scalar/vector")),
                      ("Outside: an operand of rank > 2", num("rank>2")),
                      # Every output of rank >= 3, not only the rank-2-operand subset that
                      # "rank2-out-rank3+" counts: the residual after the other categories.
                      ("Outside: output of rank \u2265 3", total - num("IN CLASS") - num("rank>2")
                       - num("dim>sa") - num("repeated-index")),
                      ("Outside: a dimension > 16", num("dim>sa")),
                      ("Outside: index repeated in an operand", num("repeated-index"))]
    write_table(tdir, "T3_benchmark_coverage",
                ["SA", "steps", "in 2D class", "in class excl. scalar/vector", "real 2D",
                 "old table executes (real 2D)", "planner executes (real 2D)"], rows, md)
    if cats16:
        fig, ax = plt.subplots(figsize=(6.4, 2.6))
        labels = [c for c, _ in cats16][::-1]
        vals = [v for _, v in cats16][::-1]
        colors = ["#a3a29b" if l.startswith("Outside") else SLOTS[0] for l in labels]
        ax.barh(labels, vals, color=colors, height=0.62)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        for y, v in enumerate(vals):
            ax.text(v + 8000, y, f"{v:,}", va="center", fontsize=8, color=INK2)
        ax.set_xlabel("Pairwise contraction steps in the einsum benchmark (1,615,342 in total)")
        ax.set_xlim(0, max(vals) * 1.18)
        style(ax)
        save(fig, fdir, "F1_benchmark_coverage_sa16")


def benchmark_exec(res, tdir, fdir, md):
    rows, dist = [], {}
    for n in SIZES:
        data = read_csv(os.path.join(res, f"benchmark_distinct_sa{n}_{n}x{n}.csv"))
        if not data:
            continue
        steps = sum(int(r["count"]) for r in data)
        ok = [r for r in data if r["passed"] == "True"]
        abs_e = [float(r["max_abs_err"]) for r in ok]
        rel_e = [float(r["rel_err"]) for r in ok]
        rows.append([f"{n}x{n}", len(data), f"{steps:,}", f"{len(ok)}/{len(data)}",
                     f"{sum(int(r['count']) for r in ok):,}", f"{max(abs_e):.1e}", f"{max(rel_e):.1e}"])
        d = Counter()
        for r in data:
            d[int(r["passes"])] += int(r["count"])
        dist[n] = d
    write_table(tdir, "T4_benchmark_execution",
                ["SA", "distinct cases", "steps they stand for", "cases passed", "steps passed",
                 "max abs err", "max rel err"], rows, md)
    if dist:
        n = max(dist)
        d, total = dist[n], sum(dist[n].values())
        fig, ax = plt.subplots(figsize=(5.2, 2.2))
        ks = sorted(d)
        ax.barh([f"{k} pass{'es' if k != 1 else ''}" for k in ks][::-1],
                [100 * d[k] / total for k in ks][::-1], color=SLOTS[0], height=0.6)
        for y, k in enumerate(ks[::-1]):
            ax.text(100 * d[k] / total + 1, y, f"{100 * d[k] / total:.1f}%  ({d[k]:,} steps)",
                    va="center", fontsize=8, color=INK2)
        ax.set_xlim(0, 115)
        ax.set_xlabel("Share of benchmark steps in the 2D class (%)")
        style(ax)
        save(fig, fdir, "F2_benchmark_passes")
        write_table(tdir, "T4b_benchmark_passes",
                    ["passes"] + [f"{m}x{m} steps" for m in sorted(dist)],
                    [[k] + [dist[m].get(k, 0) for m in sorted(dist)] for k in range(4)], md)


RULE_ORDER = ["matmul", "sI", "broadcast", "vec_hadamard", "diag", "trace", "hadamard", "unary"]


def class_runs(res, tdir, fdir, md):
    per_size = {n: read_csv(os.path.join(res, f"class_cases_sa{n}_{n}x{n}.csv")) for n in SIZES}
    rows = []
    for n, data in per_size.items():
        if not data:
            continue
        by = defaultdict(list)
        for r in data:
            if r["passed"] == "True":
                by[r["family"]].append(float(r["rel_err"]))
        for rule in RULE_ORDER:
            e = by.get(rule, [])
            if e:
                rows.append([f"{n}x{n}", rule, len(e), f"{st.median(e):.1e}", f"{max(e):.1e}"])
    write_table(tdir, "T5_precision_by_rule", ["SA", "rule", "cases", "median rel err", "max rel err"], rows, md)

    big = max((n for n, d in per_size.items() if d), default=None)
    if big:
        data = [r for r in per_size[big] if r["passed"] == "True" and float(r["rel_err"]) > 0]
        fig, ax = plt.subplots(figsize=(6.4, 3.0))
        groups = ["0 reduced", "1 reduced", "2 reduced"]
        for gi, g in enumerate(groups):
            pts = [(RULE_ORDER.index(r["family"]), float(r["rel_err"])) for r in data if r["group"] == g]
            if pts:
                ax.scatter([p[1] for p in pts], [p[0] + (gi - 1) * 0.2 for p in pts], s=16,
                           color=SLOTS[gi], edgecolors="white", linewidths=0.6, label={"0 reduced": "No pre-reduction", "1 reduced": "One operand pre-reduced", "2 reduced": "Both operands pre-reduced"}[g], zorder=3)
        ax.set_xscale("log")
        ax.set_yticks(range(len(RULE_ORDER)))
        ax.set_yticklabels([RULE_NAMES[x] for x in RULE_ORDER])
        ax.set_xlabel("Normwise relative error against an fp64 reference (log scale)")
        style(ax)
        exact = [r for r in per_size[big] if r["passed"] == "True" and float(r["rel_err"]) == 0]
        for rule in RULE_ORDER:
            n_exact = sum(r["family"] == rule for r in exact)
            if n_exact and not any(r["family"] == rule for r in data):
                ax.text(1.2e-8, RULE_ORDER.index(rule), f"All {n_exact} cases exact",
                        va="center", fontsize=8, color=INK2)
        ax.set_ylim(len(RULE_ORDER) - 0.4, -0.6)
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
        save(fig, fdir, "F3_precision_by_rule")

    reps = [("ij,jk->ik", "Matrix product"), ("ij,ij->ij", "Hadamard"), ("ij->ji", "Transpose")]
    rows, series = [], defaultdict(list)
    for n, data in per_size.items():
        if not data:
            continue
        for eq, label in reps + [("ij,kl->ik", "reduce both + outer, 3 passes")]:
            r = next((r for r in data if r["equation"] == eq and r["instance"] == "class_full"), None)
            if r:
                et, ma = int(r["execTime"]), int(r["mxActive"])
                rows.append([f"{n}x{n}", eq, r["passes"], et, ma, r["dmaActive"], f"{100 * ma / et:.1f}%"])
                series[eq].append((n, et / int(r["passes"])))
    write_table(tdir, "T6_counters_core_forms_full_size",
                ["SA", "equation", "passes", "execTime", "mxActive", "dmaActive", "mxActive/execTime"], rows, md)
    if series:
        fig, ax = plt.subplots(figsize=(5.6, 3.0))
        for i, (eq, label) in enumerate(reps):
            pts = series.get(eq, [])
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=SLOTS[i], linewidth=2,
                        marker="o", markersize=5, markeredgecolor="white", label=label)
        # End labels of series that finish close together are pushed apart (above / below).
        ends = sorted(((series[eq][-1][1], label) for eq, label in reps if series.get(eq)), reverse=True)
        for j, (y, label) in enumerate(ends):
            close_above = j > 0 and ends[j - 1][0] - y < 0.08 * ends[0][0]
            close_below = j + 1 < len(ends) and y - ends[j + 1][0] < 0.08 * ends[0][0]
            va = "top" if close_above else "bottom" if close_below else "center"
            ax.text(SIZES[-1] * 1.08, y, label, va=va, fontsize=8, color=INK2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(SIZES)
        ax.set_xticklabels([arr(n) for n in SIZES])
        ax.set_xlim(3.4, 80)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Execution time per pass (cycles)")
        ax.yaxis.set_major_formatter(THOUSANDS)
        ax.set_xlabel("Array size (operands at full size)")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        save(fig, fdir, "F4_exectime_vs_sa_size")


def pytorch(res, tdir, md):
    rows = []
    for n in (8, 16, 32):
        data = read_csv(os.path.join(res, f"pytorch_cases_{n}x{n}.csv"))
        if not data:
            continue
        for r in data:
            rows.append([f"{n}x{n}", r["raw_equation"], r["equation"], r["shapes"], r["passed"],
                         r["passes"], f"{float(r['rel_err']):.1e}", r["execTime"]])
    write_table(tdir, "T7_pytorch_test_einsum",
                ["SA", "PyTorch equation", "canonical", "shapes", "passed", "passes", "rel err", "execTime"], rows, md)


BLAS3_OPS = ["GEMM_NN", "GEMM_NT", "GEMM_TN", "GEMM_TT", "SYRK", "SYMM", "TRMM_U", "TRMM_L"]


def blas3(res, tdir, fdir, md):
    rows, series = [], defaultdict(list)
    summary = []
    for n in SIZES:
        data = read_csv(os.path.join(res, f"blas3_cases_{n}x{n}.csv"))
        if not data:
            continue
        ok = sum(r["passed"] == "True" and r["symmetric_ok"] != "False" for r in data)
        ks = {r["execTime"] for r in data if r["op"] == "GEMM_NN"}
        summary.append([f"{n}x{n}", len(data), f"{ok}/{len(data)}",
                        f"{max(float(r['rel_err']) for r in data):.1e}",
                        "yes" if len(ks) == 1 else "no"])
        for op in BLAS3_OPS:
            r = next(r for r in data if r["op"] == op and r["k"] == str(n))
            et, ma = int(r["execTime"]), int(r["mxActive"])
            rows.append([f"{n}x{n}", op, et, ma, r["dmaActive"], f"{100 * ma / et:.1f}%"])
            series["A transposed (GEMM TN, TT)" if op in ("GEMM_TN", "GEMM_TT") else "A as stored (other six routines)"].append((n, et))
    write_table(tdir, "T8_blas3_summary",
                ["SA", "cases (8 ops x k=1..N)", "passed (incl. SYRK symmetry)", "max rel err",
                 "execTime independent of k"], summary, md)
    write_table(tdir, "T8b_blas3_counters_full_size",
                ["SA", "op", "execTime", "mxActive", "dmaActive", "mxActive/execTime"], rows, md)
    if series:
        fig, ax = plt.subplots(figsize=(5.6, 3.0))
        for i, label in enumerate(["A as stored (other six routines)", "A transposed (GEMM TN, TT)"]):
            pts = sorted(set(series[label]))
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=SLOTS[i], linewidth=2, marker="o",
                    markersize=5, markeredgecolor="white", label=label)
            ax.text(pts[-1][0] * 1.08, pts[-1][1], f"{pts[-1][1]:,}", va="center", fontsize=8, color=INK2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(SIZES)
        ax.set_xticklabels([arr(n) for n in SIZES])
        ax.set_xlim(3.4, 48)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Execution time (cycles)")
        ax.yaxis.set_major_formatter(THOUSANDS)
        ax.set_xlabel("Array size")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        save(fig, fdir, "F5_blas3_transpose_cost")


# Ordinal SA-size ramp: reference palette blue, steps 250/400/550/700 (lightest
# step clears 2:1 on white). Every line is direct-labelled, so identity is
# never colour alone.
SA_RAMP = dict(zip(SIZES, ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")))
SA_MARK = dict(zip(SIZES, ("o", "s", "^", "D")))
CYCLES = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
SCALING_FORMS = ["matmul", "mat-vec", "outer", "dot", "hadamard", "frobenius", "broadcast",
                 "vec-hadamard", "scaling", "diag", "trace", "row-sum", "total-sum", "pre-reduce",
                 "transpose", "matmul-TN"]


# DMA ports (Configs.scala: fsa4x4/8x8 = 4, fsa16x16/32x32 = 8) and AXI beat bytes
# (FSAConfig.json mem_align: 8 / 8 / 8 / 16) -> theoretical DMA bytes per cycle.
DMA_PORTS = {4: 4, 8: 4, 16: 8, 32: 8}
BEAT_BYTES = {4: 8, 8: 8, 16: 8, 32: 16}


def useful_macs(eq, k):
    """Multiply-accumulates the einsum itself needs, all extents k. An index that appears in only
    one of two operands and not in the output is summed out of that operand first (k per letter of
    that operand); the remaining indices then cost k per distinct letter, multiplied (k^3 for
    matmul, k^2 for Hadamard/mat-vec/outer/reductions, 3k^2 for ij,kl->ik)."""
    lhs, out = eq.split("->")
    ops = lhs.split(",")
    if len(ops) < 2:
        return k ** len(set(lhs))
    macs, letters = 0, set(lhs.replace(",", ""))
    for i, op in enumerate(ops):
        others = set("".join(ops[:i] + ops[i + 1:]))
        private = {c for c in op if c not in others and c not in out}
        if private:
            macs += k ** len(op)
            letters -= private
    return macs + k ** len(letters)


def dma_bytes(r, n):
    """Bytes moved for one case: every DMA load is an n x n fp16 tile, every store (one
    per pass) an n x n fp32 tile."""
    passes, dma = int(r["passes"]), int(r["dmaInst"])
    return (dma - passes) * n * n * 2 + passes * n * n * 4


def scaling(res, tdir, fdir, md):
    per_size = {n: read_csv(os.path.join(res, f"scaling_cases_sa{n}_{n}x{n}.csv")) for n in SIZES}
    per_size = {n: d for n, d in per_size.items() if d}
    if not per_size:
        return
    rows, summary = [], []
    for n, data in per_size.items():
        ok = sum(r["passed"] == "True" for r in data)
        core = [r for r in data if r["group"] == "core"]
        summary.append([f"{n}x{n}", len(data), f"{ok}/{len(data)}",
                        f"{max(float(r['rel_err']) for r in data if r['rel_err']):.1e}",
                        sum(len({r["execTime"] for r in core if r["op"] == f}) == 1
                            for f in {r["op"] for r in core}),
                        len({r["op"] for r in core})])
        for form in SCALING_FORMS:
            fr = sorted((r for r in data if r["op"] == form), key=lambda r: int(r["k"]))
            if not fr:
                continue
            ets = [int(r["execTime"]) for r in fr]
            full = fr[-1]
            et, ma, da = int(full["execTime"]), int(full["mxActive"]), int(full["dmaActive"])
            macs = useful_macs(full["equation"], n)
            rows.append([f"{n}x{n}", form, full["equation"], full["family"], full["passes"],
                         f"{sum(r['passed'] == 'True' for r in fr)}/{len(fr)}",
                         f"{min(ets)}" if min(ets) == max(ets) else f"{min(ets)}-{max(ets)}",
                         et, da, f"{100 * ma / et:.1f}%",
                         f"{100 * macs / (n * n * et):.2f}%",
                         f"{100 * macs / (n * n * ma):.1f}%" if ma else "-",
                         f"{dma_bytes(full, n) / da:.1f} / {DMA_PORTS[n] * BEAT_BYTES[n]}" if da else "-",
                         "transpose (port 0)" if full["group"] == "transpose" else ""])
    write_table(tdir, "T9_scaling_summary",
                ["SA", "cases (16 forms x k=1..N)", "passed", "max rel err",
                 "core forms with execTime independent of k", "core forms"], summary, md)
    write_table(tdir, "T9b_scaling_by_form",
                ["SA", "form", "equation", "rule", "passes", "passed over k", "execTime over k",
                 "execTime at k=N", "dmaActive at k=N", "mxActive/execTime at k=N",
                 "useful MAC util at k=N (of N^2 x execTime)", "useful MAC util while MX busy",
                 "DMA bytes/cycle while active / peak", "note"], rows, md)
    md.append("\nUseful MACs: k per distinct index letter, multiplied (k^3 for matmul, k^2 for "
              "Hadamard/mat-vec/outer/reductions); an index summed out of one operand alone costs that "
              "operand's size first (3k^2 for ij,kl->ik). execTime is the accelerator's own time: the "
              "simulation driver preloads the instructions before activating it.\n")

    # F6: useful matmul throughput (k^3 MACs / execTime) against k, one line per SA size.
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for n, data in per_size.items():
        pts = sorted((int(r["k"]), int(r["k"]) ** 3 / int(r["execTime"]))
                     for r in data if r["op"] == "matmul" and r["passed"] == "True")
        if not pts:
            continue
        # Lines coincide where the SA sizes cost the same, so identity is carried by
        # marker shape + legend rather than direct labels (which would collide).
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=SA_RAMP[n], linewidth=2,
                marker=SA_MARK[n], markersize=5, markeredgecolor="white", markeredgewidth=0.8,
                label=f"{arr(n)} array")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(0.8, 48)
    ax.set_xlabel("Problem size k (k\u00d7k by k\u00d7k product, zero-padded to the array)")
    ax.set_ylabel("Useful MACs per cycle (k³ / execution time)")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, fdir, "F6_scaling_matmul_throughput")

    # F7: execTime per pass at full size, by form, one dot per SA size.
    forms = [f for f in SCALING_FORMS if any(r["op"] == f for d in per_size.values() for r in d)]
    fig, ax = plt.subplots(figsize=(6.4, 0.28 * len(forms) + 1.0))
    for n, data in per_size.items():
        xs, ys = [], []
        for i, form in enumerate(forms):
            fr = [r for r in data if r["op"] == form]
            if fr:
                full = max(fr, key=lambda r: int(r["k"]))
                xs.append(int(full["execTime"]) / int(full["passes"]))
                ys.append(i)
        ax.scatter(xs, ys, s=40, color=SA_RAMP[n], marker=SA_MARK[n], edgecolors="white",
                   linewidths=1.2, label=arr(n), zorder=3)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([50, 100, 200, 500, 1000, 2000]))
    ax.xaxis.set_major_formatter(CYCLES)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(range(len(forms)))
    ax.set_yticklabels([FORM_NAMES[f] for f in forms])
    ax.set_ylim(len(forms) - 0.4, -0.6)
    ax.set_xlabel("Execution time per pass (cycles, log scale), operands at full size")
    style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0),
              handletextpad=0.3, columnspacing=1.2)
    save(fig, fdir, "F7_scaling_exectime_by_form")


def fusemax(res, tdir, md):
    path = os.path.join(res, "fusemax_cascade.tsv")
    if not os.path.isfile(path):
        missing.append("fusemax_cascade.tsv")
        return
    with open(path) as f:
        cascade = list(csv.DictReader(f, delimiter="\t"))
    runs = {n: read_csv(os.path.join(res, f"fusemax_cases_{n}x{n}.csv")) for n in SIZES}
    runs = {n: {r["op"]: r for r in d} for n, d in runs.items() if d}
    rows = []
    for c in cascade:
        row = [c["einsum"], c["tensor"], c["cascade"], c["expression"], c["canonical"] or "-", c["status"]]
        for n in runs:
            r = runs[n].get(f"({c['einsum']}) {c['tensor']}")
            row.append("-" if r is None else
                       f"{'pass' if r['passed'] == 'True' else 'FAIL'}, {r['passes']}p, {r['execTime']}")
        rows.append(row)
    n_in = sum(c["status"] == "in class" for c in cascade)
    md.append(f"\nFuseMax cascades: {n_in} of {len(cascade)} Einsums are in the 2D class "
              f"({len({c['canonical'] for c in cascade if c['canonical']})} distinct canonical forms).\n")
    write_table(tdir, "T10_fusemax_cascade",
                ["Einsum", "tensor", "cascade", "expression (FuseMax, MICRO'24)", "canonical", "status",
                 *[f"{n}x{n}: result, passes, execTime" for n in runs]], rows, md)


def main(res, fdir):
    tdir = os.path.join(fdir, "tables")
    os.makedirs(tdir, exist_ok=True)
    md = ["# Results tables (generated by thesis/figure_tools/make_results.py — do not edit by hand)\n"]
    suites(res, tdir, md)
    coverage(res, tdir, fdir, md)
    benchmark_exec(res, tdir, fdir, md)
    class_runs(res, tdir, fdir, md)
    pytorch(res, tdir, md)
    blas3(res, tdir, fdir, md)
    scaling(res, tdir, fdir, md)
    fusemax(res, tdir, md)
    if missing:
        md.append("\n**Inputs missing at generation time:** " + ", ".join(sorted(set(missing))) + "\n")
    open(os.path.join(tdir, "TABLES.md"), "w").write("".join(md))
    print("missing:", sorted(set(missing)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
