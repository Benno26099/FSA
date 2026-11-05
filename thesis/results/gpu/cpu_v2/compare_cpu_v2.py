"""Form-matched FSA vs best-CPU at k = N from the *_cpu_v2.csv files.

usage: compare_cpu_v2.py <dir with *_cpu_v2*.csv> <results dir (has final_new/)> [suffix] [out.csv]
FSA time = execTime / 1.5 GHz. Ratio = best CPU median / FSA time (> 1: FSA faster).
"""
import csv
import sys
import statistics as st

cpu_dir, results = sys.argv[1], sys.argv[2]
suffix = sys.argv[3] if len(sys.argv) > 3 else ""
out = sys.argv[4] if len(sys.argv) > 4 else None
SPECIAL = ("pre-reduce", "transpose", "matmul-TN")
VARIANTS = ("einsum", "einsum_opt", "direct")


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


table = []
for N in (4, 8, 16, 32):
    for suite, fsa_file in (("scaling", f"scaling_cases_sa{N}_{N}x{N}.csv"), ("blas3", f"blas3_cases_{N}x{N}.csv")):
        fsa = {(r["op"], r["shapes"]): float(r["execTime"]) / 1.5e3
               for r in load(f"{results}/final_new/{fsa_file}") if r["k"] == str(N)}
        rows = load(f"{cpu_dir}/{suite}_cases_sa{N}_cpu_v2{suffix}.csv")
        bad = [r for r in rows if r["error"]]
        assert not bad, bad[:3]
        if suite == "scaling":
            floors = {(r["step"], r["variant"]): float(r["cpu_us_med"]) for r in rows if r["suite"] == "floor"}
            print(f"\n{N}x{N} floors (start/end, us): " + "  ".join(
                f"{v} {floors[('start', v)]:.2f}/{floors[('end', v)]:.2f}" for v in ("timer",) + VARIANTS))
            print(f"  threads: {sorted({(r['blas_lib'], r['blas_threads'], r['threads_env']) for r in rows})}")
        by_case = {}
        for r in rows:
            if r["suite"] != "floor" and r["k"] == str(N):
                by_case.setdefault((r["op"], r["shapes"], r["equation"]), {})[r["variant"]] = float(r["cpu_us_med"])
        for (op, shapes, eq), v in by_case.items():
            best = min(v, key=v.get)
            f_us = fsa[(op, shapes)]
            table.append({"N": N, "suite": suite, "op": op, "equation": eq, "shapes": shapes, "fsa_us": f_us,
                          **{f"{k}_us": v.get(k, "") for k in VARIANTS},
                          "best_variant": best, "best_cpu_us": v[best], "cpu_over_fsa": v[best] / f_us})

for N in (4, 8, 16, 32):
    sc = {t["op"]: t for t in table if t["N"] == N and t["suite"] == "scaling"}
    one = [t["cpu_over_fsa"] for op, t in sc.items() if op not in SPECIAL]
    print(f"\n{N}x{N}: one-pass core ({len(one)} forms) median {st.median(one):.1f}x, "
          f"range {min(one):.1f}-{max(one):.1f}x")
    for op in ("matmul",) + SPECIAL:
        t = sc[op]
        print(f"  {op:11s} {t['equation']:10s} FSA {t['fsa_us']:.3f} us  CPU {t['best_cpu_us']:.2f} us "
              f"({t['best_variant']})  -> {t['cpu_over_fsa']:.1f}x")
    b3 = [t for t in table if t["N"] == N and t["suite"] == "blas3"]
    r = [t["cpu_over_fsa"] for t in b3]
    print(f"  blas3 ({len(b3)}): median {st.median(r):.1f}x range {min(r):.1f}-{max(r):.1f}x; "
          f"best variants {sorted({t['best_variant'] for t in b3})}")
    print("  best variant per scaling form:", {op: t["best_variant"] for op, t in sc.items()})

if out:
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print(f"\n-> {out}")
