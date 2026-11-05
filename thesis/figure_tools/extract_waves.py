# AI assistance: written with Claude (Anthropic; Claude Code, Claude Opus 5/5.5),
# then reviewed, run and validated by Ben Colbourne. See the thesis's
# Statement on the Use of Generative AI.
"""Turn the captured VCDs into per-cycle CSVs of the signals each figure needs.

    ~/.local/share/thesis-plots/bin/python extract_waves.py <vcd_dir> <out_dir>

Cycle 0 is the first matrix-engine instruction issue (mxControl in.fire).
"""
import csv
import os
import sys

import vcdtools as v

P = "TOP.TestDriver.testHarness.chiptop0.system.domain.fsa.fsa."
MX = P + "mxControl."
FUNC = {0: "LOAD_STATIONARY", 6: "TENSOR_MULTIPLICATION", 7: "LOAD_STATIONARY_2", 8: "ELEMENTWISE_MUL"}


def signals(n_rows):
    s = {
        "clock": P + "clock",
        "issue_valid": MX + "io_in_valid",
        "issue_ready": MX + "io_in_ready",
        "func": MX + "io_in_bits_header_func",
        "sp_read": MX + "io_sp_read_valid",
        "sem_release": MX + "io_sem_release_valid",
        "sem_id": MX + "io_sem_release_bits_id",
        "acc_write": P + "accRAM_sram.io_fullWrite_0_valid",
        "acc_write_addr": P + "accRAM_sram.io_fullWrite_0_addr",
        "dma_acc_read": P + "io_acc_read_0_valid",
        "spad_write": P + "io_spad_write_0_valid",
        "spad_write_addr": P + "io_spad_write_0_addr",
        "spad_write_subbank": P + "io_spad_write_0_subBankIdx",
    }
    for r in range(n_rows):
        for b in ("load_reg_li", "load_reg2_li", "mul_regs", "update_reg", "mac"):
            s[f"pe{r}_{b}"] = MX + f"io_pe_ctrl_{r}_bits_{b}"
        s[f"pe{r}_valid"] = MX + f"io_pe_ctrl_{r}_valid"
    return s


def extract(vcd, n_rows, out_csv):
    sig = signals(n_rows)
    tr = v.trace(vcd, list(sig.values()))
    clk = tr[sig["clock"]]
    cols = {k: v.per_cycle(tr[name], clk) for k, name in sig.items() if k != "clock"}
    n = len(cols["issue_valid"])
    fire = [cols["issue_valid"][i][1] == 1 and cols["issue_ready"][i][1] == 1 for i in range(n)]
    t0 = fire.index(True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        keys = list(cols)
        w.writerow(["cycle", "issue"] + keys)
        for i in range(n):
            issue = FUNC.get(cols["func"][i][1], str(cols["func"][i][1])) if fire[i] else ""
            w.writerow([i - t0, issue] + [cols[k][i][1] for k in keys])
    return t0


if __name__ == "__main__":
    vcd_dir, out_dir = sys.argv[1], sys.argv[2]
    for name, rows in (("hadamard_4x4", 4), ("matmul_8x8", 8), ("transpose_8x8", 8)):
        t0 = extract(os.path.join(vcd_dir, name + ".vcd"), rows, os.path.join(out_dir, f"wave_{name}.csv"))
        print(name, "first issue at sampled cycle", t0)
