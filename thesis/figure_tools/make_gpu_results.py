# AI assistance: written with Claude (Anthropic; Claude Code, Claude Opus 5/5.5),
# then reviewed, run and validated by Ben Colbourne. See the thesis's
# Statement on the Use of Generative AI.
"""Build the GPU-comparison tables (CSV + Markdown) and plots.

    ~/.local/share/thesis-plots/bin/python make_gpu_results.py <fsa_results_dir> <gpu_results_dir> <figures_dir>

<fsa_results_dir> holds the on-chip run CSVs under the new simulation driver (results/final_new);
<gpu_results_dir> holds the run_cases_gpu.py outputs (results/gpu). FSA time is execTime at an
assumed 1.5 GHz (the synthesis frequency reported for FSA). Tables go to <figures_dir>/tables
(T11-T16, appended to GPU_TABLES.md), plots to <figures_dir> (F8-F12).

Inputs (missing ones are skipped and reported):
  gpu/{suite}_sa{N}_gpu.csv                --measure accuracy (fp16, fp16_fullacc, fp32)
  gpu/scaling_cases_sa{N}_timing.csv       --measure timing
  gpu/scaling_cases_sa{N}_overhead.csv     --measure overhead (CUDA graph replay, torch.matmul)
  gpu/cpu_v2/scaling_cases_sa{N}_cpu_v2.csv --measure cpu, single BLAS thread (one row per variant:
                                           np.einsum, np.einsum with a precomputed path, direct numpy;
                                           the fastest is used, and 'floor' rows give the call cost)
  gpu/scaling_sa{N}_batchsweep.csv         --measure batchsweep (B = 1 ... 1024)
  gpu/scaling_sa{N}_power.csv              --measure power (bracketed idle baseline)
  gpu/large_sweep_gpu_timing.csv           --measure timing over BLAS3 at n = 32 ... 4096
"""
import csv
import os
import statistics as st
import sys
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker

import make_results as M   # shared style, palette, save(), write_table(), arr()

SIZES = M.SIZES
CLOCK_MHZ = 1500                     # assumed FSA clock (cycles per µs)
FPGA_MHZ = 70                        # clock of the FSA prototype on the U55C FPGA (sensitivity line)
GPU_PEAK_FP32_GFLOPS = 3584 * 2 * 1.582   # GTX 1080 Ti: 3,584 FP32 lanes at the 1,582 MHz boost clock
SUITES = [("class_cases", "class_cases_sa{n}_{n}x{n}.csv", "The 137-equation class"),
          ("benchmark_distinct", "benchmark_distinct_sa{n}_{n}x{n}.csv", "Einsum benchmark (distinct cases)"),
          ("blas3_cases", "blas3_cases_{n}x{n}.csv", "Level-3 BLAS"),
          ("fusemax_cases", "fusemax_cases_{n}x{n}.csv", "FuseMax 2D subset"),
          ("scaling_cases", "scaling_cases_sa{n}_{n}x{n}.csv", "Synthetic sweep"),
          ("pytorch_cases", "pytorch_cases_{n}x{n}.csv", "PyTorch test_einsum")]
missing = []
C_FSA, C_GPU, C_THIRD, C_FOURTH = M.SLOTS[0], M.SLOTS[1], M.SLOTS[2], "#eda100"


def read(path):
    if not os.path.isfile(path):
        missing.append(os.path.relpath(path))
        return None
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fsa_us(r):
    return int(r["execTime"]) / CLOCK_MHZ


CPU_ROUTINES = {"einsum": "np.einsum", "einsum_opt": "np.einsum, precomputed path"}
DIRECT_NAMES = {"ij->ji": "a.T.copy()", "ij,kl->ik": "np.outer(a.sum(1), b.sum(1))"}


def cpu_rows(gpu_dir, n):
    """The single-threaded CPU v2 rows for one size (None if missing)."""
    return read(os.path.join(gpu_dir, "cpu_v2", f"scaling_cases_sa{n}_cpu_v2.csv"))


def cpu_best(rows):
    """One row per case (in JSONL order): the fastest variant's median, and which routine it was."""
    best = []
    for r in rows:
        if r["suite"] == "floor":
            continue
        same = best and all(best[-1][k] == r[k] for k in ("op", "k", "instance", "equation", "shapes"))
        if same and float(r["cpu_us_med"]) >= float(best[-1]["cpu_us_med"]):
            continue
        r = dict(r, cpu_routine=CPU_ROUTINES.get(r["variant"]) or DIRECT_NAMES.get(r["equation"], "@ (BLAS)"))
        if same:
            best[-1] = r
        else:
            best.append(r)
    return best


def by_mode(rows, mode):
    return [r for r in rows if r["mode"] == mode]


def aligned(fsa, gpu, what):
    """FSA and GPU rows come from the same JSONL in the same order; check that, then pair them."""
    if len(fsa) != len(gpu):
        raise SystemExit(f"{what}: {len(fsa)} FSA rows vs {len(gpu)} GPU rows")
    for a, b in zip(fsa, gpu):
        if a["equation"] != b["equation"] or a["shapes"] != b["shapes"]:
            raise SystemExit(f"{what}: row mismatch {a['equation']} {a['shapes']} vs {b['equation']} {b['shapes']}")
    return list(zip(fsa, gpu))


def full_tile(rows, n, op=None):
    """Scaling-sweep rows at k = N (operands at full size), optionally for one form."""
    return [r for r in rows if int(r["k"]) == n and (op is None or r["op"] == op)]


def ratio(x):
    """A speed-up: one decimal below 10 (so a CPU win such as 0.6 does not round to 1)."""
    return f"{x:.1f}" if x < 10 else f"{x:,.0f}"


def fmt(x, nd=3):
    return f"{x:.{nd}f}" if x < 100 else f"{x:,.0f}"


def style_y(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=M.GRID, linewidth=0.8, which="major")
    ax.set_axisbelow(True)


def sa_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES)
    ax.set_xticklabels([M.arr(n) for n in SIZES])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(3.2, 52)
    ax.set_xlabel("Array size")


def line(ax, xs, ys, color, marker, label, dashed=False, end_label=True, dy=1.0):
    ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=6, markeredgecolor="white",
            markeredgewidth=0.8, linestyle="--" if dashed else "-", label=label, zorder=3)
    if end_label:
        ax.text(xs[-1] * 1.09, ys[-1] * dy, label, va="center", fontsize=9, color=M.INK2)


# ---------------------------------------------------------------- accuracy (T11, F8)
def accuracy(fsa_dir, gpu_dir, tdir, fdir, md):
    rows, f8 = [], None
    identical = differ = 0
    for key, fsa_pat, name in SUITES:
        for n in SIZES:
            if key == "pytorch_cases" and not os.path.isfile(os.path.join(gpu_dir, f"{key}_sa{n}_gpu.csv")):
                continue   # PyTorch cases exist only for the 16x16 case file
            fsa = read(os.path.join(fsa_dir, fsa_pat.format(n=n)))
            gpu = read(os.path.join(gpu_dir, f"{key}_sa{n}_gpu.csv"))
            if not fsa or not gpu:
                continue
            g16, gfa, g32 = by_mode(gpu, "fp16"), by_mode(gpu, "fp16_fullacc"), by_mode(gpu, "fp32")
            pairs16, pairs32 = aligned(fsa, g16, f"{key} {n}"), aligned(fsa, g32, f"{key} {n}")
            for a, b in zip(g16, gfa):
                identical += a["rel_err"] == b["rel_err"]
                differ += a["rel_err"] != b["rel_err"]
            e = lambda rs: [float(r["rel_err"]) for r in rs]
            ef, e16, e32 = e(fsa), e(g16), e(g32)
            rows.append([name, M.arr(n), len(fsa),
                         f"{sum(r['passed'] == 'True' for r in fsa)}/{len(fsa)}",
                         f"{sum(r['passed'] == 'True' for r in g16)}/{len(g16)}",
                         f"{sum(r['passed'] == 'True' for r in g32)}/{len(g32)}",
                         f"{st.median(ef):.1e}", f"{st.median(e16):.1e}", f"{st.median(e32):.1e}",
                         f"{max(ef):.1e}", f"{max(e16):.1e}", f"{max(e32):.1e}",
                         sum(a < b for a, b in zip(ef, e16)), sum(a > b for a, b in zip(ef, e16)),
                         sum(a > b for a, b in zip(ef, e32))])
            if key == "class_cases" and n == 32:
                f8 = (fsa, g16, g32)
    write_table(tdir, "T11_gpu_accuracy",
                ["suite", "array", "cases", "FSA passed", "GPU fp16 passed", "GPU fp32 passed",
                 "median rel err FSA", "median GPU fp16", "median GPU fp32",
                 "max rel err FSA", "max GPU fp16", "max GPU fp32",
                 "FSA lower than GPU fp16", "FSA higher than GPU fp16", "FSA higher than GPU fp32"], rows, md)
    md.append(f"\nRelative error = max|e| / max|y| against an fp64 reference on the same fp16 operands. "
              f"fp16 with fp32 reduction (`allow_fp16_reduced_precision_reduction` off) gave errors identical "
              f"to plain fp16 in {identical:,} of {identical + differ:,} cases, so it is not shown.\n")
    if not f8:
        return
    # F8: median relative error by planner rule, FSA vs GPU fp16 vs GPU fp32, 32x32 class cases.
    fsa, g16, g32 = f8
    rules = [r for r in M.RULE_NAMES if any(x["family"] == r for x in fsa)]
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(rules) + 1.0))
    off = {0: -0.22, 1: 0.0, 2: 0.22}
    for si, (rows_, label, color, marker) in enumerate([(fsa, "FSA", C_FSA, "o"), (g16, "GPU, fp16", C_GPU, "s"),
                                                         (g32, "GPU, fp32", C_THIRD, "^")]):
        for i, rule in enumerate(rules):
            errs = [float(x["rel_err"]) for x, f in zip(rows_, fsa) if f["family"] == rule and float(x["rel_err"]) > 0]
            if not errs:
                continue
            q = sorted(errs)
            lo, hi = q[int(0.1 * (len(q) - 1))], q[int(0.9 * (len(q) - 1))]
            ax.plot([lo, hi], [i + off[si]] * 2, color=color, linewidth=1.5, alpha=0.6, zorder=2)
            ax.scatter([st.median(errs)], [i + off[si]], s=42, color=color, marker=marker, edgecolors="white",
                       linewidths=1.0, zorder=3, label=label if i == 0 else None)
    ax.set_xscale("log")
    ax.set_yticks(range(len(rules)))
    ax.set_yticklabels([M.RULE_NAMES[r] for r in rules])
    ax.set_ylim(len(rules) - 0.5, -0.6)
    ax.set_xlabel("Normwise relative error against fp64 (log scale; dot = median, bar = 10th–90th percentile)")
    M.style(ax)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    M.save(fig, fdir, "F8_gpu_accuracy_by_rule")


# ---------------------------------------------------------------- latency (T12, F9)
def latency(fsa_dir, gpu_dir, tdir, fdir, md):
    rows, series = [], defaultdict(dict)
    for n in SIZES:
        fsa = read(os.path.join(fsa_dir, f"scaling_cases_sa{n}_{n}x{n}.csv"))
        tim = read(os.path.join(gpu_dir, f"scaling_cases_sa{n}_timing.csv"))
        ovh = read(os.path.join(gpu_dir, f"scaling_cases_sa{n}_overhead.csv"))
        cpu = cpu_rows(gpu_dir, n)
        if not (fsa and tim and ovh and cpu):
            continue
        cpu = cpu_best(cpu)
        t32, o32 = by_mode(tim, "fp32"), by_mode(ovh, "fp32")
        aligned(fsa, t32, f"timing {n}"), aligned(fsa, o32, f"overhead {n}"), aligned(fsa, cpu, f"cpu {n}")
        for f, t, o, c in zip(fsa, t32, o32, cpu):
            if int(f["k"]) != n:
                continue
            us = fsa_us(f)
            g, e, cp = float(o["graph_wall_us_med"]), float(t["lat_wall_us_med"]), float(c["cpu_us_med"])
            rows.append([M.arr(n), M.FORM_NAMES.get(f["op"], f["op"]), f["equation"], f["passes"], f["execTime"],
                         fmt(us), fmt(int(f["execTime"]) / FPGA_MHZ), fmt(cp), c["cpu_routine"], fmt(g), fmt(float(o["graph_gpu_us_med"])),
                         fmt(e), fmt(float(t["xfer_wall_us_med"])), fmt(float(t["batched_gpu_us_per_einsum"])),
                         ratio(cp / us), ratio(g / us), ratio(e / us)])
            if f["op"] == "matmul":
                series["FSA at 1.5 GHz"][n] = us
                series["FSA at 70 MHz"][n] = int(f["execTime"]) / FPGA_MHZ
                series["CPU, NumPy"][n] = cp
                series["GPU, CUDA graph replay"][n] = g
                series["GPU, PyTorch einsum"][n] = e
                series["_matmul"][n] = (float(o["matmul_wall_us_med"]) if o.get("matmul_wall_us_med") else None)
    write_table(tdir, "T12_latency_single_einsum",
                ["array", "form", "equation", "passes", "FSA cycles", "FSA us (1.5 GHz)", "FSA us (70 MHz)", "CPU us (NumPy, best)",
                 "CPU routine",
                 "GPU us, CUDA graph (wall)", "GPU us, CUDA graph (device)", "GPU us, PyTorch einsum",
                 "GPU us, einsum incl. transfers", "GPU us per einsum, batched (B<=1024)",
                 "FSA speed-up vs CPU", "vs GPU graph", "vs GPU einsum"], rows, md)
    md.append("\nOne einsum, operands at full size, fp32 on the GPU and CPU (GTX 1080 Ti, i7-8700K), medians of "
              "up to 500 timed calls after 50 warm-up calls. GPU wall times include the host's launch and "
              "synchronisation; FSA time is the accelerator's execTime at an assumed 1.5 GHz (and, as a "
              "sensitivity check, at the 70 MHz of the FSA prototype on the U55C FPGA) and excludes the host side. "
              "The CPU time is the fastest of np.einsum, np.einsum with a precomputed contraction path, and a "
              "direct NumPy routine where one exists (@ for matrix-product forms, a.T.copy() for the transpose, "
              "an outer product of row sums for ij,kl->ik), on one BLAS thread, every result materialised "
              "as a contiguous array.\n")
    if not series:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for (label, color, marker, dashed), dy in zip([("GPU, PyTorch einsum", C_GPU, "s", False),
                                                   ("GPU, CUDA graph replay", C_GPU, "s", True),
                                                   ("CPU, NumPy", C_FOURTH, "D", False),
                                                   ("FSA at 70 MHz", C_FSA, "o", True),
                                                   ("FSA at 1.5 GHz", C_FSA, "o", False)],
                                                  (1.12, 0.9, 1.0, 1.0, 1.0)):
        pts = sorted(series[label].items())
        line(ax, [p[0] for p in pts], [p[1] for p in pts], color, marker, label, dashed=dashed, dy=dy)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylabel("Time per einsum (µs, log scale)")
    sa_axis(ax)
    ax.set_xlim(3.2, 120)
    style_y(ax)
    M.save(fig, fdir, "F9_latency_single_matmul")


# ---------------------------------------------------------------- batch sweep (T13, F10)
def batch_sweep(fsa_dir, gpu_dir, tdir, fdir, md):
    rows, curves, fsa_line = [], {}, {}
    for n in SIZES:
        fsa = read(os.path.join(fsa_dir, f"scaling_cases_sa{n}_{n}x{n}.csv"))
        sw = read(os.path.join(gpu_dir, f"scaling_sa{n}_batchsweep.csv"))
        if not (fsa and sw):
            continue
        fsa_full = {r["op"]: r for r in full_tile(fsa, n)}
        for mode in ("fp32", "fp16"):
            by_op = defaultdict(list)
            for r in by_mode(sw, mode):
                if not r["error"]:
                    by_op[r["op"]].append((int(r["batch_B"]), float(r["gpu_us_per_einsum"])))
            for op in M.SCALING_FORMS:
                if op not in by_op or op not in fsa_full:
                    continue
                us = fsa_us(fsa_full[op])
                pts = sorted(by_op[op])
                hit = [b for b, t in pts if t <= us]
                rows.append([M.arr(n), mode, M.FORM_NAMES.get(op, op), fsa_full[op]["equation"], fmt(us),
                             fmt(pts[0][1]), fmt(pts[-1][1]), pts[-1][0],
                             f"{min(hit):,}" if hit else f"> {pts[-1][0]:,}"])
                if op == "matmul" and mode == "fp32":
                    curves[n], fsa_line[n] = pts, us
    write_table(tdir, "T13_gpu_batch_to_match_fsa",
                ["array", "GPU precision", "form", "equation", "FSA us (1.5 GHz)", "GPU us per einsum at B=1",
                 "GPU us per einsum at largest B", "largest B", "smallest B with GPU <= FSA"], rows, md)
    md.append("\nB einsums are stacked on a new leading index and run as one torch.einsum call (B real copies of "
              "the operands, at most 1 GiB). Time per einsum = device time of the call / B.\n")
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    for n, pts in curves.items():
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=M.SA_RAMP[n], linewidth=2, marker=M.SA_MARK[n],
                markersize=5, markeredgecolor="white", markeredgewidth=0.8, label=f"GPU, {M.arr(n)} operands", zorder=3)
        ax.hlines(fsa_line[n], 0.8, 1300, color=M.SA_RAMP[n], linewidth=1.2, linestyle="--", zorder=2)
        ax.text(1450, fsa_line[n], f"FSA, {M.arr(n)}", fontsize=9, color=M.INK2,
                va={4: "top", 8: "bottom"}.get(n, "center"))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 4, 16, 64, 256, 1024])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(0.8, 5000)
    ax.set_xlabel("Matrix products per GPU call (B)")
    ax.set_ylabel("Time per matrix product (µs, log scale)")
    style_y(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(0.72, 1.0))
    M.save(fig, fdir, "F10_gpu_batch_sweep_matmul")


# ---------------------------------------------------------------- large sizes (T14, F11)
def large(fsa_dir, gpu_dir, tdir, fdir, md):
    big = read(os.path.join(gpu_dir, "large_sweep_gpu_timing.csv"))
    rows, fsa_pts, fsa_slow, single, batched = [], [], [], {}, {}
    for n in SIZES:
        fsa = read(os.path.join(fsa_dir, f"scaling_cases_sa{n}_{n}x{n}.csv"))
        tim = read(os.path.join(gpu_dir, f"scaling_cases_sa{n}_timing.csv"))
        if not (fsa and tim):
            continue
        f = full_tile(fsa, n, "matmul")[0]
        fsa_pts.append((n, 2 * n ** 3 / (fsa_us(f) * 1e3)))
        fsa_slow.append((n, 2 * n ** 3 / (int(f["execTime"]) / FPGA_MHZ * 1e3)))
        if n < 32:   # n = 32 comes from the large sweep below
            t = [r for r in by_mode(tim, "fp32") if r["op"] == "matmul" and int(r["k"]) == n][0]
            single[n] = 2 * n ** 3 / (float(t["lat_wall_us_med"]) * 1e3)
            batched[n] = 2 * n ** 3 / (float(t["batched_gpu_us_per_einsum"]) * 1e3)
    if big:
        for r in big:
            if r["op"] != "GEMM_NN":
                continue
            n = int(r["shapes"].split("x")[0])
            fl = 2 * n ** 3
            rows.append([f"{n}×{n}", r["mode"], fmt(float(r["lat_wall_us_med"])), r["n_repeats"],
                         fmt(float(r["batched_gpu_us_per_einsum"])), r["batch_B"],
                         f"{fl / (float(r['lat_wall_us_med']) * 1e3):,.1f}",
                         f"{fl / (float(r['batched_gpu_us_per_einsum']) * 1e3):,.1f}"])
            if r["mode"] == "fp32":
                single[n] = fl / (float(r["lat_wall_us_med"]) * 1e3)
                batched[n] = fl / (float(r["batched_gpu_us_per_einsum"]) * 1e3)
        write_table(tdir, "T14_gpu_large_matmul",
                    ["matrix size", "precision", "GPU us per call (wall)", "timed calls", "GPU us per product, batched",
                     "B", "GFLOP/s, one call", "GFLOP/s, batched"], rows, md)
        md.append(f"\nGEMM (C = AB) on the GPU only, square operands. GFLOP/s = 2n^3 / time. GTX 1080 Ti fp32 peak "
                  f"at its boost clock: {GPU_PEAK_FP32_GFLOPS:,.0f} GFLOP/s. FSA peak at 1.5 GHz: 2 N^2 x 1.5 GHz "
                  f"(48 / 192 / 768 / 3,072 GFLOP/s at 4x4 ... 32x32).\n")
    if not fsa_pts or not single:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    s = sorted(single.items()); b = sorted(batched.items())
    line(ax, [p[0] for p in s], [p[1] for p in s], C_GPU, "s", "GPU, one call")
    line(ax, [p[0] for p in b], [p[1] for p in b], C_THIRD, "^", "GPU, batched")
    line(ax, [p[0] for p in fsa_pts], [p[1] for p in fsa_pts], C_FSA, "o", "FSA at 1.5 GHz, array = matrix size")
    line(ax, [p[0] for p in fsa_slow], [p[1] for p in fsa_slow], C_FSA, "o", "FSA at 70 MHz", dashed=True)
    ax.axhline(GPU_PEAK_FP32_GFLOPS, color=M.INK2, linewidth=1, linestyle=":")
    ax.text(4.2, GPU_PEAK_FP32_GFLOPS * 1.25, "GPU fp32 peak", fontsize=9, color=M.INK2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ticks = [4, 16, 64, 256, 1024, 4096]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(3.2, 4096 * 1.5)
    ax.set_xlabel("Matrix size n (n×n by n×n product, fp32 on the GPU)")
    ax.set_ylabel("GFLOP/s (2n³ / time, log scale)")
    style_y(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    for t in ax.texts[:4]:
        t.set_visible(False)   # legend carries identity: the end labels would run off the right edge
    M.save(fig, fdir, "F11_gpu_matmul_throughput_by_size")


# ---------------------------------------------------------------- energy (T15, F12)
def energy(gpu_dir, tdir, fdir, md):
    rows, pts = [], defaultdict(dict)
    for n in SIZES:
        p = read(os.path.join(gpu_dir, f"scaling_sa{n}_power.csv"))
        if not p:
            continue
        for mode in ("fp32", "fp16"):
            rs = [r for r in by_mode(p, mode) if not r["error"]]
            med = lambda k: st.median(float(r[k]) for r in rs)
            rows.append([M.arr(n), mode, len(rs), f"{med('idle_power_w'):.1f}", f"{med('avg_power_w'):.1f}",
                         f"{med('energy_uj_total') / 1e3:.2f}", f"{med('batched_avg_power_w'):.1f}",
                         f"{med('batched_energy_uj_total'):.2f}", f"{med('batched_energy_uj_dynamic'):.2f}",
                         sum(float(r["idle_spread_w"]) > 5 for r in rs)])
            if mode == "fp32":
                pts["GPU, one call: total"][n] = med("energy_uj_total")
                pts["GPU, batched: total"][n] = med("batched_energy_uj_total")
                pts["GPU, batched: above idle"][n] = med("batched_energy_uj_dynamic")
    write_table(tdir, "T15_gpu_energy",
                ["array", "precision", "forms", "idle W", "one call: avg W", "one call: mJ per einsum (total)",
                 "batched: avg W", "batched: uJ per einsum (total)", "batched: uJ per einsum (above idle)",
                 "forms with idle drift > 5 W"], rows, md)
    md.append("\nGPU board power from NVML every 10 ms while one einsum (or one batched call of up to 1,024) runs "
              "back to back for 4 s; idle measured for 2 s before, between and after the two loops, each loop "
              "using the mean of the idles around it. Medians over the 16 forms at full size. For one call the "
              "average power equals the idle power, so its energy above idle is below the measurement's "
              "resolution and only the total is reported. No FSA energy is measured.\n")
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for label, color, marker, dashed in [("GPU, one call: total", C_GPU, "s", False),
                                         ("GPU, batched: total", C_THIRD, "^", False),
                                         ("GPU, batched: above idle", C_THIRD, "^", True)]:
        s = sorted(pts[label].items())
        line(ax, [q[0] for q in s], [q[1] for q in s], color, marker, label, dashed=dashed)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylabel("Energy per einsum (µJ, log scale)")
    sa_axis(ax)
    ax.set_xlim(3.2, 140)
    style_y(ax)
    M.save(fig, fdir, "F12_gpu_energy_per_einsum")


# ---------------------------------------------------------------- environment (T16)
def environment(gpu_dir, tdir, md):
    clocks, temps = [], []
    for n in SIZES:
        for name in (f"scaling_cases_sa{n}_timing.csv", f"scaling_cases_sa{n}_overhead.csv"):
            for r in read(os.path.join(gpu_dir, name)) or []:
                if r.get("sm_clock_mhz"):
                    clocks.append(int(float(r["sm_clock_mhz"]))); temps.append(int(float(r["temp_c"])))
    cpu_all = [r for n in SIZES for r in (cpu_rows(gpu_dir, n) or [])]
    cpu = cpu_all[0] if cpu_all else {}
    floors = defaultdict(list)
    for r in cpu_all:
        if r["suite"] == "floor":
            floors[r["variant"]].append(float(r["cpu_us_med"]))
    floor = ", ".join(f"{name} {min(floors[v]):.2f}-{max(floors[v]):.2f} us"
                      for v, name in (("direct", "@"), ("einsum", "np.einsum"), ("timer", "timer alone"))
                      if floors[v])
    rows = [["GPU", "NVIDIA GeForce GTX 1080 Ti (Pascal, 16 nm, 11 GB)"],
            ["GPU driver / CUDA", "470.256.02 / 11.4 (driver); PyTorch 2.6.0+cu118 (CUDA 11.8 runtime)"],
            ["CPU", f"{cpu.get('device', '?')}; NumPy {cpu.get('numpy_version', '?')}, {cpu.get('blas_lib', '?')}, "
                    f"{cpu.get('blas_threads', '?').replace(' (env)', '')} BLAS thread (fp32)"],
            ["CPU fixed call cost (1-element call)", floor or "-"],
            ["GPU SM clock during timing", f"{min(clocks):,}-{max(clocks):,} MHz (median {st.median(clocks):,.0f})" if clocks else "-"],
            ["GPU temperature during timing", f"{min(temps)}-{max(temps)} C" if temps else "-"],
            ["Timing protocol", "50 warm-up calls; up to 500 timed calls, 2 s cap; median, 10th/90th percentile"],
            ["FSA clock (assumed)", "1.5 GHz (synthesis frequency reported for FSA, 16 nm)"]]
    write_table(tdir, "T16_gpu_environment", ["item", "value"], rows, md)


def write_table(tdir, name, header, rows, md):
    M.write_table(tdir, name, header, rows, md)


def main(fsa_dir, gpu_dir, fdir):
    tdir = os.path.join(fdir, "tables")
    os.makedirs(tdir, exist_ok=True)
    md = ["# GPU comparison tables (generated by thesis/figure_tools/make_gpu_results.py — do not edit by hand)\n"]
    accuracy(fsa_dir, gpu_dir, tdir, fdir, md)
    latency(fsa_dir, gpu_dir, tdir, fdir, md)
    batch_sweep(fsa_dir, gpu_dir, tdir, fdir, md)
    large(fsa_dir, gpu_dir, tdir, fdir, md)
    energy(gpu_dir, tdir, fdir, md)
    environment(gpu_dir, tdir, md)
    if missing:
        md.append("\n**Inputs missing at generation time:** " + ", ".join(sorted(set(missing))) + "\n")
    open(os.path.join(tdir, "GPU_TABLES.md"), "w").write("".join(md))
    print("missing:", sorted(set(missing)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
