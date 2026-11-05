# AI assistance: written with Claude (Anthropic; Claude Code, Claude Opus 5/5.5),
# then reviewed, run and validated by Ben Colbourne. See the thesis's
# Statement on the Use of Generative AI.
"""Store-path timeline: per-port accumulator reads (data next cycle) and AXI AW/W."""
import sys, struct
sys.path.insert(0, sys.argv[2])
import vcdtools as v
P = "TOP.TestDriver.testHarness.chiptop0.system.domain.fsa."
F = P + "fsa."; D = P + "dma."
sig = {"clk": P + "clock", "state": P + "state",
       "aw": F + "accRAM_sram.io_fullWrite_0_valid", "awa": F + "accRAM_sram.io_fullWrite_0_addr",
       "mrel": F + "mxControl.io_sem_release_valid", "mrel_id": F + "mxControl.io_sem_release_bits_id"}
for p in range(4):
    for k in ("valid", "ready", "addr", "subBankIdx", "data_0", "data_1"):
        sig[f"r{p}_{k}"] = F + f"io_acc_read_{p}_{k}"
    for k in ("aw_valid", "aw_ready", "aw_bits_addr", "w_valid", "w_ready", "w_bits_data"):
        sig[f"x{p}_{k}"] = D + f"auto_out_{p}_{k}"
    sig[f"sq{p}_acq"] = D + ("storeQueue" if p == 0 else f"storeQueue_{p}") + ".io_semAcquire_valid"
tr = v.trace(sys.argv[1], list(sig.values()))
pc = {k: dict(v.per_cycle(tr[s], tr[sig["clk"]])) for k, s in sig.items() if k != "clk"}
E = sorted(pc["state"]); first = next(i for i, e in enumerate(E) if pc["state"][e] == 1)
g = lambda k, e: pc[k].get(e)
f32 = lambda x: struct.unpack("<f", struct.pack("<I", x))[0] if x is not None and x >= 0 else float("nan")
def w2(x):
    if x is None or x < 0: return "?"
    return f"{f32(x & 0xffffffff):.4f},{f32(x >> 32):.4f}"
last_aw = max(i for i in range(first, len(E)) if g("aw", E[i]) == 1)
for i in range(last_aw - 6, min(len(E), last_aw + 30)):
    e = E[i]; ev = []
    if g("aw", e) == 1: ev.append(f"ACC-WRITE row{g('awa', e)}")
    if g("mrel", e) == 1: ev.append(f"MX-REL sem{g('mrel_id', e)}")
    for p in range(4):
        if g(f"r{p}_valid", e) == 1:
            ok = g(f"r{p}_ready", e) == 1
            ev.append(f"P{p} accRD row{g(f'r{p}_addr', e)} sb{g(f'r{p}_subBankIdx', e)} {'fire' if ok else 'STALL'}")
        if i > 0 and g(f"r{p}_valid", E[i - 1]) == 1 and g(f"r{p}_ready", E[i - 1]) == 1:
            ev.append(f"P{p} data=[{f32(g(f'r{p}_data_0', e)):.4f},{f32(g(f'r{p}_data_1', e)):.4f}]")
        if g(f"x{p}_aw_valid", e) == 1 and g(f"x{p}_aw_ready", e) == 1:
            ev.append(f"P{p} AW {hex(g(f'x{p}_aw_bits_addr', e))}")
        if g(f"x{p}_w_valid", e) == 1 and g(f"x{p}_w_ready", e) == 1:
            ev.append(f"P{p} W [{w2(g(f'x{p}_w_bits_data', e))}]")
    if ev: print(f"{i - first:4}  " + " | ".join(ev))
