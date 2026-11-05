# AI assistance: written with Claude (Anthropic; Claude Code, Claude Opus 5/5.5),
# then reviewed, run and validated by Ben Colbourne. See the thesis's
# Statement on the Use of Generative AI.
"""Event timeline of one kernel: DMA spad writes, MX spad reads, semaphores, MX issue."""
import sys
sys.path.insert(0, sys.argv[2])
import vcdtools as v
P = "TOP.TestDriver.testHarness.chiptop0.system.domain.fsa."
F = P + "fsa."
MX = F + "mxControl."
sig = {"clk": P + "clock", "state": P + "state",
       "rd_v": F + "spRAM_sram.io_fullRead_0_valid", "rd_a": F + "spRAM_sram.io_fullRead_0_addr",
       "mx_v": MX + "io_in_valid", "mx_r": MX + "io_in_ready", "mx_f": MX + "io_in_bits_header_func",
       "mrel_v": MX + "io_sem_release_valid", "mrel_id": MX + "io_sem_release_bits_id",
       "drel0_v": P + "dma.io_semaphoreRelease_0_valid", "drel0_id": P + "dma.io_semaphoreRelease_0_bits_id",
       "drel1_v": P + "dma.io_semaphoreRelease_1_valid", "drel1_id": P + "dma.io_semaphoreRelease_1_bits_id"}
for a in range(3):
    for k in ("valid", "ready", "bits_id", "bits_value"):
        sig[f"acq{a}_{k}"] = P + f"semaphores.io_acquire_{a}_{k}"
for p in range(4):
    sig[f"w{p}_v"] = F + f"spRAM_sram.io_narrowWrite_{p}_valid"
    sig[f"w{p}_a"] = F + f"spRAM_sram.io_narrowWrite_{p}_addr"
    if p:
        sig[f"w{p}_r"] = F + f"spRAM_sram.io_narrowWrite_{p}_ready"
    sig[f"w{p}_sb"] = F + f"io_spad_write_{p}_subBankIdx"
    for d in range(4):
        sig[f"w{p}_d{d}"] = F + f"io_spad_write_{p}_data_{d}"
tr = v.trace(sys.argv[1], list(sig.values()))
clk = tr[sig["clk"]]
pc = {k: dict(v.per_cycle(tr[s], clk)) for k, s in sig.items() if k != "clk"}
E = sorted(pc["state"])
first = next(i for i, e in enumerate(E) if pc["state"][e] == 1)
g = lambda k, e: pc[k].get(e)
import struct
def h(x):
    if x is None or x < 0: return "?"
    return f"{struct.unpack('<e', struct.pack('<H', x))[0]:g}"
FUNC = {0: "LOAD_STATIONARY", 6: "TENSOR_MUL", 5: "TENSOR_TRANSPOSE"}
for i in range(first, len(E)):
    e = E[i]; c = i - first; ev = []
    for p in range(4):
        if g(f"w{p}_v", e) == 1 and (p == 0 or g(f"w{p}_r", e) == 1):
            ev.append(f"W{p} row{g(f'w{p}_a', e)} sb{g(f'w{p}_sb', e)} [{','.join(h(g(f'w{p}_d{d}', e)) for d in range(4))}]")
        elif g(f"w{p}_v", e) == 1:
            ev.append(f"W{p} STALL row{g(f'w{p}_a', e)}")
    if g("rd_v", e) == 1: ev.append(f"RD row{g('rd_a', e)}")
    if g("mx_v", e) == 1 and g("mx_r", e) == 1: ev.append(f"MX-ISSUE {FUNC.get(g('mx_f', e), g('mx_f', e))}")
    if g("mrel_v", e) == 1: ev.append(f"MX-REL sem{g('mrel_id', e)}")
    if g("drel0_v", e) == 1: ev.append(f"DMA-LD-REL sem{g('drel0_id', e)}")
    if g("drel1_v", e) == 1: ev.append(f"DMA-ST-REL sem{g('drel1_id', e)}")
    for a in range(3):
        if g(f"acq{a}_valid", e) == 1:
            ev.append(f"ACQ{a} sem{g(f'acq{a}_bits_id', e)}>={g(f'acq{a}_bits_value', e)} {'ok' if g(f'acq{a}_ready', e) == 1 else 'WAIT'}")
    if ev:
        print(f"{c:4}  " + " | ".join(ev))
    if pc["state"][e] != 1 and c > 0:
        break
