"""Per-cycle PE control grid for rows 0..n-1 around LOAD_STATIONARY -> TENSOR_MUL."""
import sys
sys.path.insert(0, sys.argv[2])
import vcdtools as v
n = int(sys.argv[3])
P = "TOP.TestDriver.testHarness.chiptop0.system.domain.fsa."
MX = P + "fsa.mxControl."
sig = {"clk": P + "clock", "state": P + "state", "mx_v": MX + "io_in_valid", "mx_r": MX + "io_in_ready",
       "mx_f": MX + "io_in_bits_header_func", "rd_v": P + "fsa.spRAM_sram.io_fullRead_0_valid",
       "rd_a": P + "fsa.spRAM_sram.io_fullRead_0_addr"}
bits = ("load_reg_li", "mac", "update_reg", "acc_ui")
for r in range(n):
    sig[f"v{r}"] = MX + f"io_pe_ctrl_{r}_valid"
    for b in bits:
        sig[f"{b}{r}"] = MX + f"io_pe_ctrl_{r}_bits_{b}"
hdr = {x for x, _, _ in v._header(sys.argv[1])}
sig = {k: s for k, s in sig.items() if s in hdr}
tr = v.trace(sys.argv[1], list(sig.values()))
pc = {k: dict(v.per_cycle(tr[s], tr[sig["clk"]])) for k, s in sig.items() if k != "clk"}
E = sorted(pc["state"]); first = next(i for i, e in enumerate(E) if pc["state"][e] == 1)
issues = [i for i in range(first, len(E)) if pc["mx_v"].get(E[i]) == 1 and pc["mx_r"].get(E[i]) == 1]
start = issues[0]
print("MX issues at", [(i - start, pc["mx_f"][E[i]]) for i in issues], "(cycle 0 = LOAD_STATIONARY issue)")
print("cyc  rd   " + "  ".join(f"row{r}[L M U A]" for r in range(n)))
for i in range(start, start + 3 * n + 10):
    e = E[i]
    rd = f"r{pc['rd_a'].get(e)}" if pc["rd_v"].get(e) == 1 else "  "
    cells = []
    for r in range(n):
        if pc.get(f"v{r}", {}).get(e) != 1:
            cells.append("   .   ".ljust(12)); continue
        cells.append(("[" + " ".join("1" if pc[f"{b}{r}"].get(e) == 1 else "." for b in bits) + "]").ljust(12))
    print(f"{i - start:3}  {rd:4} " + " ".join(cells))
