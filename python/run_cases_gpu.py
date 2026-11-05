# AI assistance: written with Claude (Anthropic; Claude Code, Claude Opus 5/5.5),
# then reviewed, run and validated by Ben Colbourne. See the thesis's
# Statement on the Use of Generative AI.

import os
import sys
import csv
import json
import time
import string
import argparse
import threading
import numpy as np
import torch

from cases import make_operands

# fp16: PyTorch defaults; fp16_fullacc: fp16 matmuls reduce in fp32 (closest to the FSA's
# fp16 multiply / fp32 accumulate); fp32: operands upcast, the GPU's fast path on Pascal
MODES = ("fp16", "fp16_fullacc", "fp32")
MEASURES = ("accuracy", "timing", "power", "overhead", "cpu", "cpupower", "batchsweep")

ID_FIELDS = ["suite", "op", "k", "instance", "family", "step", "group", "equation", "raw_equation",
             "count", "shapes", "device", "mode"]
ACCURACY_FIELDS = ["passed", "symmetric_ok", "max_abs_err", "rel_err"]
TIMING_FIELDS = ["n_repeats", "lat_gpu_us_med", "lat_gpu_us_p10", "lat_gpu_us_p90",
                 "lat_wall_us_med", "lat_wall_us_p10", "lat_wall_us_p90",
                 "xfer_wall_us_med", "xfer_wall_us_p10", "xfer_wall_us_p90",
                 "batch_B", "batched_n_repeats", "batched_gpu_us_per_einsum", "batched_wall_us_per_einsum",
                 "sm_clock_mhz", "temp_c"]
POWER_FIELDS = ["idle_before_w", "idle_mid_w", "idle_after_w", "idle_spread_w",
                "idle_power_w", "avg_power_w", "iters", "loop_s", "energy_uj_total", "energy_uj_dynamic",
                "batch_B", "batched_idle_power_w", "batched_avg_power_w", "batched_iters", "batched_loop_s",
                "batched_energy_uj_total", "batched_energy_uj_dynamic",
                "sm_clock_mhz", "temp_c"]
# overhead: einsum as in timing, the same work replayed from a CUDA graph (one launch, no Python
# dispatch), and torch.matmul for plain matmuls -- separates PyTorch overhead from the GPU's own cost
OVERHEAD_FIELDS = ["n_repeats", "einsum_gpu_us_med", "einsum_wall_us_med",
                   "graph_gpu_us_med", "graph_wall_us_med", "graph_wall_us_p10", "graph_wall_us_p90",
                   "matmul_gpu_us_med", "matmul_wall_us_med", "sm_clock_mhz", "temp_c"]
# cpu: one row per variant (CPU_VARIANTS); every result is materialised contiguous, so views (ij->ji)
# do real work. Rows with suite "floor" are the timer and per-variant dispatch floors (1-element call).
CPU_FIELDS = ["variant", "n_repeats", "cpu_us_med", "cpu_us_p10", "cpu_us_p90", "matches_einsum",
              "numpy_version", "blas_lib", "blas_threads", "threads_env", "os_threads"]
# cpupower: one row per variant (CPU_VARIANTS), package energy from RAPL with the idle baseline measured
# on both sides of each loop, as in power; fp32 is the meaningful mode, as for cpu
CPU_POWER_FIELDS = ["variant", "idle_before_w", "idle_after_w", "idle_spread_w", "idle_power_w",
                    "avg_power_w", "iters", "loop_s", "energy_uj_total", "energy_uj_dynamic", "rapl_domains",
                    "numpy_version", "blas_lib", "blas_threads", "threads_env", "os_threads"]
SWEEP_FIELDS = ["batch_B", "n_repeats", "batched_gpu_us", "batched_wall_us",
                "gpu_us_per_einsum", "wall_us_per_einsum", "sm_clock_mhz", "temp_c"]

BATCH_BYTES = 1 << 30   # cap on the batched operands' memory (11 GB card; leaves room for the result)


def base_row(record, mode, device):
    """The identifying CSV columns shared by every measurement, so GPU and FSA rows join."""
    shapes = [tuple(s) for s in record["shapes"]]
    row = {k: record.get(k, "") for k in ("suite", "op", "k", "instance", "family", "step", "group",
                                          "equation", "raw_equation", "count")}
    row["shapes"] = " ".join("x".join(map(str, s)) or "scalar" for s in shapes)
    row["device"] = cpu_name() if device.type == "cpu" else torch.cuda.get_device_name(device)
    row["mode"] = mode
    return row


def cpu_name():
    """CPU model from /proc/cpuinfo (Linux), for the device column of CPU rows."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "cpu"


def prepare(record, mode, device):
    """Equation, numpy operands (for the fp64 reference) and the same operands on the GPU in mode's dtype.

    Sets the reduced-precision flag every call: it is global, so it must not leak between modes.
    """
    eq = record["equation"]
    shapes = [tuple(s) for s in record["shapes"]]
    operands = make_operands(record, shapes)

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (mode == "fp16")
    dtype = torch.float32 if mode == "fp32" else torch.float16
    tensors = [torch.as_tensor(np.asarray(o), device=device, dtype=dtype) for o in operands]
    return eq, operands, tensors


def run_record(record, mode, device):
    """Run one JSONL case through torch.einsum on the GPU; returns its CSV row."""
    row = base_row(record, mode, device)
    try:
        eq, operands, tensors = prepare(record, mode, device)
        got = torch.einsum(eq, *tensors).double().cpu().numpy()
        expected = np.einsum(eq, *(o.astype(np.float64) for o in operands))
        err = np.abs(got - expected)
        row["passed"] = np.shape(got) == np.shape(expected) and \
            bool(np.allclose(got, expected, rtol=1e-2, atol=1e-2))
        row["max_abs_err"] = float(np.max(err))
        row["rel_err"] = float(np.max(err) / max(np.max(np.abs(expected)), 1e-12))
        row["error"] = ""
        row["symmetric_ok"] = bool(np.allclose(got, got.T, rtol=1e-2, atol=1e-2)) \
            if record.get("symmetric_result") else ""
    except Exception as e:
        row["passed"] = False
        row["max_abs_err"] = row["rel_err"] = row["symmetric_ok"] = ""
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def batch_equation(equation):
    """Equation with a fresh leading batch letter on every operand and on the output."""
    no_whitespace = "".join(equation.split())
    inputs, outputs = no_whitespace.split("->")
    inputs = inputs.split(",")

    batch = next(letter for letter in string.ascii_lowercase if letter not in no_whitespace)
    batched_inputs = [batch + op for op in inputs]
    return ",".join(batched_inputs) + "->" + batch + outputs


def batch_tensors(tensors, B):
    """B real copies of each tensor stacked on a new leading dimension (not a view: the GPU must read B tiles)."""
    results = []
    for t in tensors:
        stacked = t.unsqueeze(0).repeat(B, *[1] * t.dim())
        results.append(stacked)

    return results


def batch_size(tensors, batch):
    """`batch`, reduced so the B copies of the operands fit in BATCH_BYTES (at least 1)."""
    one_copy = sum(t.numel() * t.element_size() for t in tensors)
    return max(1, min(batch, BATCH_BYTES // max(one_copy, 1)))


def batched(eq, tensors, B):
    """Batched equation and operands, checked once: slice 0 must equal the unbatched result."""
    beq, bt = batch_equation(eq), batch_tensors(tensors, B)
    single, many = torch.einsum(eq, *tensors), torch.einsum(beq, *bt)
    if not torch.allclose(many[0].float(), single.float(), rtol=1e-2, atol=1e-2):
        raise RuntimeError(f"batched einsum {beq!r} does not reproduce {eq!r}")
    return beq, bt


def time_calls(fn, warmup, repeats, max_s):
    """Time fn() on the GPU: median/p10/p90 in µs of CUDA-event (GPU) and wall-clock time.

    Runs up to `warmup` untimed calls, then up to `repeats` timed ones; each phase also stops after
    `max_s` seconds (at least 3 timed calls), so large slow cases finish. Synchronising before and
    after each call keeps the repeats from overlapping, so each is one einsum's latency.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    t0 = time.perf_counter()
    for _ in range(warmup):
        fn()
        if time.perf_counter() - t0 > max_s:
            break
    torch.cuda.synchronize()

    gpu_us = []
    wall_us = []
    t0 = time.perf_counter()
    for i in range(repeats):
        torch.cuda.synchronize()
        first = time.perf_counter()
        start.record()
        fn()
        end.record()
        end.synchronize()
        second = time.perf_counter()
        gpu_us.append(start.elapsed_time(end) * 1000)   # elapsed_time is in ms
        wall_us.append((second - first) * 1e6)          # perf_counter is in s
        if i >= 2 and second - t0 > max_s:
            break

    return {
        "n": len(gpu_us),
        "gpu_med": float(np.median(gpu_us)),
        "gpu_p10": float(np.percentile(gpu_us, 10)),
        "gpu_p90": float(np.percentile(gpu_us, 90)),
        "wall_med": float(np.median(wall_us)),
        "wall_p10": float(np.percentile(wall_us, 10)),
        "wall_p90": float(np.percentile(wall_us, 90)),
    }


def gpu_conditions(nvml, handle):
    """SM clock (MHz) and temperature (°C) now."""
    return {"sm_clock_mhz": nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM),
            "temp_c": nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)}


def time_record(record, mode, device, nvml, handle, args):
    """Latency (operands resident), latency with host transfers, and batched time per einsum."""
    row = base_row(record, mode, device)
    try:
        eq, operands, tensors = prepare(record, mode, device)
        dtype = tensors[0].dtype if tensors else torch.float32

        lat = time_calls(lambda: torch.einsum(eq, *tensors), args.warmup, args.repeats, args.max_s)

        def round_trip():
            on_gpu = [torch.as_tensor(np.asarray(o), device=device, dtype=dtype) for o in operands]
            return torch.einsum(eq, *on_gpu).cpu()
        xfer = time_calls(round_trip, args.warmup, args.repeats, args.max_s)

        B = batch_size(tensors, args.batch)
        beq, bt = batched(eq, tensors, B)
        bat = time_calls(lambda: torch.einsum(beq, *bt), args.warmup, args.repeats, args.max_s)
        del bt

        row.update({
            "n_repeats": lat["n"],
            "lat_gpu_us_med": lat["gpu_med"], "lat_gpu_us_p10": lat["gpu_p10"], "lat_gpu_us_p90": lat["gpu_p90"],
            "lat_wall_us_med": lat["wall_med"], "lat_wall_us_p10": lat["wall_p10"], "lat_wall_us_p90": lat["wall_p90"],
            "xfer_wall_us_med": xfer["wall_med"], "xfer_wall_us_p10": xfer["wall_p10"], "xfer_wall_us_p90": xfer["wall_p90"],
            "batch_B": B, "batched_n_repeats": bat["n"],
            "batched_gpu_us_per_einsum": bat["gpu_med"] / B, "batched_wall_us_per_einsum": bat["wall_med"] / B,
        })
        row.update(gpu_conditions(nvml, handle))
        row["error"] = ""
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    return row


class PowerSampler:
    """Background thread reading board power (W) from NVML every `period` seconds."""
    def __init__(self, nvml, handle, period=0.01):
        self.nvml, self.handle, self.period = nvml, handle, period
        self.samples = []   # (perf_counter time, watts)

    def __enter__(self):
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def run(self):
        while not self.stop.is_set():
            self.samples.append((time.perf_counter(), self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000))
            time.sleep(self.period)

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()

    def mean(self, t_from, t_to):
        watts = [w for t, w in self.samples if t_from <= t <= t_to]
        if not watts:
            raise RuntimeError("no power samples in the window; increase --power_s")
        return float(np.mean(watts))


def power_loop(fn, nvml, handle, seconds, ramp_s=0.1):
    """Call fn() back to back for `seconds` while sampling power; returns (mean W, iterations, loop s).

    The first `ramp_s` of samples are dropped while power rises. A synchronise every 64 calls stops
    the launch queue running far ahead of the clock check.
    """
    fn()
    torch.cuda.synchronize()
    with PowerSampler(nvml, handle) as sampler:
        t0 = time.perf_counter()
        iters = 0
        while time.perf_counter() - t0 < seconds:
            fn()
            iters += 1
            if iters % 64 == 0:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
    return sampler.mean(t0 + ramp_s, t1), iters, t1 - t0


def idle_power(nvml, handle, seconds):
    """Mean power with the GPU doing nothing, measured straight after a workload (active P-state)."""
    with PowerSampler(nvml, handle) as sampler:
        t0 = time.perf_counter()
        time.sleep(seconds)
        t1 = time.perf_counter()
    return sampler.mean(t0, t1)


def power_record(record, mode, device, nvml, handle, args):
    """Energy per einsum, unbatched and batched, total and above the active-idle baseline.

    The idle baseline drifts with the card's power state, so it is measured on both sides of each
    loop (idle, unbatched loop, idle, batched loop, idle) and each loop uses the mean of the idles
    around it. idle_spread_w (max - min of the three) flags cases where the baseline moved.
    """
    row = base_row(record, mode, device)
    try:
        eq, _, tensors = prepare(record, mode, device)
        single = lambda: torch.einsum(eq, *tensors)

        power_loop(single, nvml, handle, args.warm_s)     # bring the card to its active state
        idle_before = idle_power(nvml, handle, args.idle_s)
        watts, iters, loop_s = power_loop(single, nvml, handle, args.power_s)
        idle_mid = idle_power(nvml, handle, args.idle_s)

        B = batch_size(tensors, args.batch)
        beq, bt = batched(eq, tensors, B)
        b_watts, b_iters, b_loop_s = power_loop(lambda: torch.einsum(beq, *bt), nvml, handle, args.power_s)
        del bt
        cond = gpu_conditions(nvml, handle)
        idle_after = idle_power(nvml, handle, args.idle_s)

        idle = (idle_before + idle_mid) / 2
        b_idle = (idle_mid + idle_after) / 2
        per = loop_s / iters                # seconds per einsum
        b_per = b_loop_s / (b_iters * B)
        row.update({
            "idle_before_w": idle_before, "idle_mid_w": idle_mid, "idle_after_w": idle_after,
            "idle_spread_w": max(idle_before, idle_mid, idle_after) - min(idle_before, idle_mid, idle_after),
            "idle_power_w": idle,
            "avg_power_w": watts, "iters": iters, "loop_s": loop_s,
            "energy_uj_total": watts * per * 1e6, "energy_uj_dynamic": (watts - idle) * per * 1e6,
            "batch_B": B, "batched_idle_power_w": b_idle,
            "batched_avg_power_w": b_watts, "batched_iters": b_iters, "batched_loop_s": b_loop_s,
            "batched_energy_uj_total": b_watts * b_per * 1e6,
            "batched_energy_uj_dynamic": (b_watts - b_idle) * b_per * 1e6,
        })
        row.update(cond)
        row["error"] = ""
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    time.sleep(args.settle_s)
    return row


def graph_of(fn):
    """fn captured as a CUDA graph (warmed up on a side stream first, as capture requires)."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def overhead_record(record, mode, device, nvml, handle, args):
    """einsum latency vs the same work replayed from a CUDA graph, and vs torch.matmul for matmuls."""
    row = base_row(record, mode, device)
    try:
        eq, _, tensors = prepare(record, mode, device)
        einsum = time_calls(lambda: torch.einsum(eq, *tensors), args.warmup, args.repeats, args.max_s)
        graph = graph_of(lambda: torch.einsum(eq, *tensors))
        replay = time_calls(graph.replay, args.warmup, args.repeats, args.max_s)
        row.update({
            "n_repeats": einsum["n"],
            "einsum_gpu_us_med": einsum["gpu_med"], "einsum_wall_us_med": einsum["wall_med"],
            "graph_gpu_us_med": replay["gpu_med"], "graph_wall_us_med": replay["wall_med"],
            "graph_wall_us_p10": replay["wall_p10"], "graph_wall_us_p90": replay["wall_p90"],
        })
        if eq == "ij,jk->ik":
            a, b = tensors
            mm = time_calls(lambda: torch.matmul(a, b), args.warmup, args.repeats, args.max_s)
            row.update({"matmul_gpu_us_med": mm["gpu_med"], "matmul_wall_us_med": mm["wall_med"]})
        row.update(gpu_conditions(nvml, handle))
        row["error"] = ""
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def wall_calls(fn, warmup, repeats, max_s):
    """time_calls for the CPU: wall-clock µs only (median/p10/p90), same warm-up and time cap."""
    t0 = time.perf_counter()
    for _ in range(warmup):
        fn()
        if time.perf_counter() - t0 > max_s:
            break
    wall_us = []
    t0 = time.perf_counter()
    for i in range(repeats):
        first = time.perf_counter()
        fn()
        second = time.perf_counter()
        wall_us.append((second - first) * 1e6)
        if i >= 2 and second - t0 > max_s:
            break
    return {"n": len(wall_us), "med": float(np.median(wall_us)),
            "p10": float(np.percentile(wall_us, 10)), "p90": float(np.percentile(wall_us, 90))}


# The best plain-numpy routine for the matmul-shaped forms (BLAS through @), for the three-pass
# ij,kl->ik, which numpy's einsum path never pre-reduces (it stays a k^4 loop even when optimised),
# and for the transpose (a real copy, as the FSA moves the data)
DIRECT = {
    "ij->ji": lambda a: a.T.copy(),
    "ij,jk->ik": lambda a, b: a @ b,
    "ji,jk->ik": lambda a, b: a.T @ b,
    "ij,kj->ik": lambda a, b: a @ b.T,
    "ji,kj->ik": lambda a, b: a.T @ b.T,
    "ij,ik->jk": lambda a, b: a.T @ b,
    "ij,j->i": lambda a, b: a @ b,
    "i,i->": lambda a, b: a @ b,
    "ij,kl->ik": lambda a, b: np.outer(a.sum(axis=1), b.sum(axis=1)),
}
CPU_VARIANTS = ("einsum", "einsum_opt", "direct")


def cpu_variants(eq, arrays):
    """{variant: fn} for one case, each fn returning a materialised (contiguous) result.

    einsum_opt passes a contraction path found once with np.einsum_path, so the path search is not
    timed on every call (numpy still parses the equation each call; that cost is part of its overhead).
    """
    path = np.einsum_path(eq, *arrays, optimize="optimal")[0]
    fns = {"einsum": lambda: np.ascontiguousarray(np.einsum(eq, *arrays)),
           "einsum_opt": lambda: np.ascontiguousarray(np.einsum(eq, *arrays, optimize=path))}
    if eq in DIRECT:
        direct = DIRECT[eq]
        fns["direct"] = lambda: np.ascontiguousarray(direct(*arrays))
    return fns


def blas_info():
    """(library, threads) of the BLAS numpy calls: threadpoolctl if installed, else numpy's build config.

    Without threadpoolctl the thread count is read from the environment ("default" if unset).
    """
    try:
        from threadpoolctl import threadpool_info
        pools = [p for p in threadpool_info() if p["user_api"] == "blas"]
        ours = [p for p in pools if "numpy" in p["filepath"]] or pools
        if ours:
            return f"{ours[0]['internal_api']} {ours[0]['version']}", str(ours[0]["num_threads"])
    except ImportError:
        pass
    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        lib = f"{blas['name']} {blas.get('version', '')}".strip()
    except Exception:
        lib = "unknown"
    for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        if os.environ.get(var):
            return lib, os.environ[var] + " (env)"
    return lib, "default"


def threads_env():
    """The thread-count environment variables as set for this run."""
    return " ".join(f"{v}={os.environ.get(v, '-')}"
                    for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"))


def cpu_common():
    lib, threads = blas_info()
    # os_threads: threads in this process, an independent check on the BLAS pool (OpenBLAS starts
    # its workers at import, so default threads shows ~cpu_count more than a single-threaded run)
    return {"numpy_version": np.__version__, "blas_lib": lib, "blas_threads": threads,
            "threads_env": threads_env(), "os_threads": len(os.listdir("/proc/self/task"))}


def cpu_record(record, mode, device, args):
    """Each CPU_VARIANTS on the host CPU, one row each (fp16 is emulated by numpy, so fp32 is the meaningful mode).

    matches_einsum checks the variant's result against plain np.einsum on the same operands.
    """
    rows = []
    try:
        eq = record["equation"]
        shapes = [tuple(s) for s in record["shapes"]]
        dtype = np.float32 if mode == "fp32" else np.float16
        arrays = [np.asarray(o, dtype=dtype) for o in make_operands(record, shapes)]
        fns = cpu_variants(eq, arrays)
    except Exception as e:
        row = base_row(record, mode, device)
        row["error"] = f"{type(e).__name__}: {e}"
        return [row]
    expected = np.einsum(eq, *arrays)
    common = cpu_common()
    for variant, fn in fns.items():
        row = base_row(record, mode, device)
        row.update(common)
        row["variant"] = variant
        try:
            got = fn()
            row["matches_einsum"] = np.size(got) == np.size(expected) and \
                bool(np.allclose(np.reshape(got, np.shape(expected)), expected, rtol=1e-4, atol=1e-4))
            cpu = wall_calls(fn, args.warmup, args.repeats, args.max_s)
            row.update({"n_repeats": cpu["n"],
                        "cpu_us_med": cpu["med"], "cpu_us_p10": cpu["p10"], "cpu_us_p90": cpu["p90"]})
            row["error"] = "" if row["matches_einsum"] else "result differs from np.einsum"
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
    return rows


def cpu_floor_rows(mode, device, args, step):
    """Fixed-cost rows: the timer alone (empty lambda) and each variant on 1-element operands (i,i->)."""
    dtype = np.float32 if mode == "fp32" else np.float16
    x = np.ones(1, dtype=dtype)
    record = {"suite": "floor", "step": step, "equation": "i,i->", "raw_equation": "i,i->",
              "count": 1, "shapes": [[1], [1]]}
    fns = {"timer": lambda: None}
    fns.update(cpu_variants("i,i->", [x, x]))
    rows = []
    common = cpu_common()
    for variant, fn in fns.items():
        row = base_row(dict(record, op="timer" if variant == "timer" else "dispatch"), mode, device)
        row.update(common)
        row["variant"] = variant
        cpu = wall_calls(fn, args.warmup, args.repeats, args.max_s)
        row.update({"n_repeats": cpu["n"],
                    "cpu_us_med": cpu["med"], "cpu_us_p10": cpu["p10"], "cpu_us_p90": cpu["p90"],
                    "error": ""})
        rows.append(row)
    return rows


class Rapl:
    """CPU package energy from the Linux powercap interface (RAPL), summed over the packages.

    Only the top-level package domains are read (intel-rapl:0, not its intel-rapl:0:0 sub-domains),
    so the energy covers the cores and the uncore. Each counter wraps at max_energy_range_uj.
    Reading energy_uj usually needs root; the first read happens here, so a permission error stops
    the run at start-up rather than part-way through.
    """
    ROOT = "/sys/class/powercap"

    def __init__(self):
        self.domains = []   # (name, energy_uj path, wrap in µJ)
        entries = sorted(os.listdir(self.ROOT)) if os.path.isdir(self.ROOT) else []
        for entry in entries:
            path = os.path.join(self.ROOT, entry)
            if entry.count(":") != 1:
                continue
            try:
                with open(os.path.join(path, "name")) as f:
                    name = f.read().strip()
                with open(os.path.join(path, "max_energy_range_uj")) as f:
                    wrap = int(f.read())
            except OSError:
                continue
            if name.startswith("package"):
                self.domains.append((name, os.path.join(path, "energy_uj"), wrap))
        if not self.domains:
            raise RuntimeError(f"no RAPL package domains under {self.ROOT}")
        self.read()

    def read(self):
        """One counter value (µJ) per package."""
        values = []
        for _, path, _ in self.domains:
            with open(path) as f:
                values.append(int(f.read()))
        return values

    def joules(self, before, after):
        """Energy (J) between two read()s, allowing each counter to wrap once."""
        return sum((a - b) % wrap for (_, _, wrap), b, a in zip(self.domains, before, after)) / 1e6

    def names(self):
        return " ".join(name for name, _, _ in self.domains)


def rapl_loop(fn, rapl, seconds):
    """Call fn() back to back for `seconds`; returns (mean package W, iterations, loop s)."""
    fn()
    before = rapl.read()
    t0 = time.perf_counter()
    iters = 0
    while time.perf_counter() - t0 < seconds:
        fn()
        iters += 1
    t1 = time.perf_counter()
    after = rapl.read()
    return rapl.joules(before, after) / (t1 - t0), iters, t1 - t0


def rapl_idle(rapl, seconds):
    """Mean package power with this process asleep, measured straight after a workload."""
    before = rapl.read()
    t0 = time.perf_counter()
    time.sleep(seconds)
    t1 = time.perf_counter()
    after = rapl.read()
    return rapl.joules(before, after) / (t1 - t0)


def cpu_power_record(record, mode, device, rapl, args):
    """Energy per einsum on the host CPU for each CPU_VARIANTS, total and above the idle baseline.

    Mirrors power_record: an unmeasured warm-up, then idle, loop, idle for each variant, with the
    loop's baseline the mean of the idles around it. The package counter includes everything else
    running on the machine, so idle_spread_w flags cases where the baseline moved.
    """
    rows = []
    try:
        eq = record["equation"]
        shapes = [tuple(s) for s in record["shapes"]]
        dtype = np.float32 if mode == "fp32" else np.float16
        arrays = [np.asarray(o, dtype=dtype) for o in make_operands(record, shapes)]
        fns = cpu_variants(eq, arrays)
    except Exception as e:
        row = base_row(record, mode, device)
        row["error"] = f"{type(e).__name__}: {e}"
        return [row]
    common = cpu_common()
    for variant, fn in fns.items():
        row = base_row(record, mode, device)
        row.update(common)
        row["variant"] = variant
        row["rapl_domains"] = rapl.names()
        try:
            rapl_loop(fn, rapl, args.warm_s)
            idle_before = rapl_idle(rapl, args.idle_s)
            watts, iters, loop_s = rapl_loop(fn, rapl, args.power_s)
            idle_after = rapl_idle(rapl, args.idle_s)
            idle = (idle_before + idle_after) / 2
            per = loop_s / iters
            row.update({
                "idle_before_w": idle_before, "idle_after_w": idle_after,
                "idle_spread_w": abs(idle_after - idle_before), "idle_power_w": idle,
                "avg_power_w": watts, "iters": iters, "loop_s": loop_s,
                "energy_uj_total": watts * per * 1e6, "energy_uj_dynamic": (watts - idle) * per * 1e6,
            })
            row["error"] = ""
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        time.sleep(args.settle_s)
    return rows


def sweep_records(record, mode, device, nvml, handle, args):
    """Batched time per einsum for B = 1, 2, 4 ... --batch; one row per B (B = 1 is one call)."""
    rows = []
    try:
        eq, _, tensors = prepare(record, mode, device)
    except Exception as e:
        row = base_row(record, mode, device)
        row["error"] = f"{type(e).__name__}: {e}"
        return [row]
    B = 1
    while B <= batch_size(tensors, args.batch):
        row = base_row(record, mode, device)
        try:
            beq, bt = batched(eq, tensors, B)
            r = time_calls(lambda: torch.einsum(beq, *bt), args.warmup, args.repeats, args.max_s)
            del bt
            row.update({"batch_B": B, "n_repeats": r["n"],
                        "batched_gpu_us": r["gpu_med"], "batched_wall_us": r["wall_med"],
                        "gpu_us_per_einsum": r["gpu_med"] / B, "wall_us_per_einsum": r["wall_med"] / B})
            row.update(gpu_conditions(nvml, handle))
            row["error"] = ""
        except Exception as e:
            row.update({"batch_B": B, "error": f"{type(e).__name__}: {e}"})
        rows.append(row)
        B *= 2
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run JSONL einsum cases through torch.einsum on the GPU, write CSV")
    parser.add_argument("--cases", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--modes", type=str, nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit", type=int, default=0, help="only run the first N cases")
    parser.add_argument("--k", type=int, default=0, help="only cases whose 'k' equals this (e.g. full tiles)")
    parser.add_argument("--family", type=str, nargs="+", default=None, help="only cases of these families")
    parser.add_argument("--measure", type=str, choices=MEASURES, default="accuracy",
                        help="what to measure for each case (cpu: np.einsum variants and direct numpy on the host, no GPU needed; "
                             "cpupower: their energy from the CPU's RAPL counters)")
    parser.add_argument("--warmup", type=int, default=50, help="timing: untimed calls before measuring")
    parser.add_argument("--repeats", type=int, default=500, help="timing: timed calls per measurement")
    parser.add_argument("--max_s", type=float, default=2.0, help="timing: time cap per measurement phase (s)")
    parser.add_argument("--batch", type=int, default=1024, help="timing/power/batchsweep: einsums per batched call (capped by memory)")
    parser.add_argument("--power_s", type=float, default=4.0, help="power/cpupower: seconds per measurement loop")
    parser.add_argument("--idle_s", type=float, default=2.0, help="power/cpupower: seconds per idle-baseline measurement (3 per case for power, 2 per variant for cpupower)")
    parser.add_argument("--warm_s", type=float, default=0.5, help="power/cpupower: unmeasured run before the first idle baseline (s)")
    parser.add_argument("--settle_s", type=float, default=1.0, help="power/cpupower: pause between cases (s)")
    args = parser.parse_args()

    if args.measure in ("cpu", "cpupower"):
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            print("CUDA is not available")
            sys.exit(1)
        device = torch.device(args.device)

    with open(args.cases) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if args.k:
        records = [r for r in records if r.get("k") == args.k]
    if args.family:
        records = [r for r in records if r.get("family") in args.family]
    if args.limit:
        records = records[:args.limit]

    nvml = handle = None
    rapl = Rapl() if args.measure == "cpupower" else None
    if args.measure not in ("accuracy", "cpu", "cpupower"):
        import pynvml as nvml
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(device.index or 0)

    tail = {"accuracy": ACCURACY_FIELDS, "timing": TIMING_FIELDS, "power": POWER_FIELDS,
            "overhead": OVERHEAD_FIELDS, "cpu": CPU_FIELDS, "cpupower": CPU_POWER_FIELDS,
            "batchsweep": SWEEP_FIELDS}[args.measure]
    fields = ID_FIELDS + tail + ["error"]
    ok = total = 0

    if device.type == "cpu":
        lib, threads = blas_info()
        print(f"numpy {np.__version__} on {cpu_name()} ({os.cpu_count()} logical CPUs); "
              f"BLAS {lib}, threads {threads}; {threads_env()}")
    else:
        print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) on {torch.cuda.get_device_name(device)}")
    with open(args.out, "w", newline="", encoding="utf-8", errors="backslashreplace") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        if args.measure == "cpu":
            for mode in args.modes:
                writer.writerows(cpu_floor_rows(mode, device, args, "start"))
        for n, record in enumerate(records, 1):
            for mode in args.modes:
                if args.measure == "accuracy":
                    rows = [run_record(record, mode, device)]
                elif args.measure == "timing":
                    rows = [time_record(record, mode, device, nvml, handle, args)]
                elif args.measure == "power":
                    rows = [power_record(record, mode, device, nvml, handle, args)]
                elif args.measure == "overhead":
                    rows = [overhead_record(record, mode, device, nvml, handle, args)]
                elif args.measure == "cpu":
                    rows = cpu_record(record, mode, device, args)
                elif args.measure == "cpupower":
                    rows = cpu_power_record(record, mode, device, rapl, args)
                else:
                    rows = sweep_records(record, mode, device, nvml, handle, args)
                for row in rows:
                    good = row["passed"] if args.measure == "accuracy" else not row["error"]
                    writer.writerow(row)
                    ok += good
                    total += 1
                    if not good:
                        print(f"[{n}/{len(records)}] {mode} FAIL {record['equation']} {row['shapes']} {row['error']}")
                if args.measure != "accuracy" and all(not r["error"] for r in rows):
                    print(f"[{n}/{len(records)}] {mode} {record['equation']} {rows[0]['shapes']} done")
            f.flush()

    if args.measure == "cpu":
        with open(args.out, "a", newline="", encoding="utf-8", errors="backslashreplace") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            for mode in args.modes:
                writer.writerows(cpu_floor_rows(mode, device, args, "end"))

    print(f"\n{ok}/{total} rows {'passed' if args.measure == 'accuracy' else 'measured'} -> {args.out}")
    sys.exit(0 if ok == total else 1)
