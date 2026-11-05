import sys, struct
sys.path.insert(0, sys.argv[2])
import vcdtools as v
P = "TOP.TestDriver.testHarness.chiptop0.system.domain.fsa."
A = P + "fsa.accRAM_sram.io_fullWrite_0_"
MX = P + "fsa.mxControl."
sig = {"clk": P + "clock", "state": P + "state", "v": A + "valid", "a": A + "addr",
       "mx_v": MX + "io_in_valid", "mx_r": MX + "io_in_ready", "cv": MX + "io_acc_ctrl_valid", "cc": MX + "io_acc_ctrl_bits_cmd"}
for d in range(4): sig[f"d{d}"] = A + f"data_{d}"
tr = v.trace(sys.argv[1], list(sig.values()))
pc = {k: dict(v.per_cycle(tr[s], tr[sig["clk"]])) for k, s in sig.items() if k != "clk"}
E = sorted(pc["state"]); first = next(i for i, e in enumerate(E) if pc["state"][e] == 1)
iss = [i for i in range(first, len(E)) if pc["mx_v"].get(E[i]) == 1 and pc["mx_r"].get(E[i]) == 1]
f32 = lambda x: struct.unpack("<f", struct.pack("<I", x))[0] if x is not None and x >= 0 else float("nan")
for i in range(iss[0], iss[0] + 40):
    e = E[i]
    if pc["v"].get(e) == 1 or pc["cv"].get(e) == 1:
        s = f"{i - iss[0]:3} "
        if pc["cv"].get(e) == 1: s += f"acc_ctrl cmd={pc['cc'].get(e)} "
        if pc["v"].get(e) == 1: s += f"WRITE row{pc['a'].get(e)} " + " ".join(f"{f32(pc[f'd{d}'].get(e)):9.4f}" for d in range(4))
        print(s)
