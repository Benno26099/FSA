# AI assistance: written with Claude (Anthropic; Claude Code, Claude Opus 5/5.5),
# then reviewed, run and validated by Ben Colbourne. See the thesis's
# Statement on the Use of Generative AI.
"""Per-cycle activity of one kernel from a VCD (Finding 11).

    ~/.local/share/thesis-plots/bin/python overhead_trace.py <vcd> <thesis/figure_tools dir>
"""
import sys
sys.path.insert(0, sys.argv[2])
import vcdtools as v
P = "TOP.TestDriver.testHarness.chiptop0.system.domain.fsa."
D = P + "dma."
sig = {
 "clk": P+"clock", "state": P+"state", "exec": P+"perfCnt_execTime",
 "mx_busy": P+"fsa.io_busy", "dma_active": D+"io_active", "dma_busy": D+"io_busy",
 "raw_v": P+"rawInstQueue.io_deq_valid", "raw_r": P+"rawInstQueue.io_deq_ready",
 "mx_iv": P+"fsa.io_inst_valid", "mx_ir": P+"fsa.io_inst_ready",
 "dma_iv": D+"io_inst_valid", "dma_ir": D+"io_inst_ready",
 "f_v": P+"decoder.io_outFence_valid", "f_r": P+"decoder.io_outFence_ready",
 "ar_v": D+"auto_out_0_ar_valid", "ar_r": D+"auto_out_0_ar_ready",
 "r_v": D+"auto_out_0_r_valid", "r_r": D+"auto_out_0_r_ready",
 "aw_v": D+"auto_out_0_aw_valid", "aw_r": D+"auto_out_0_aw_ready",
 "w_v": D+"auto_out_0_w_valid", "w_r": D+"auto_out_0_w_ready",
}
tr = v.trace(sys.argv[1], list(sig.values()))
clk = tr[sig["clk"]]
pc = {k: dict(v.per_cycle(tr[s], clk)) for k, s in sig.items() if k != "clk"}
edges = sorted(pc["state"])
act = [e for e in edges if pc["state"][e] == 1]
print("active cycles:", len(act), " execTime counter at end:", max(pc["exec"].values()))
t0 = act[0]
idx = {e: i for i, e in enumerate(edges)}
def f(k, e): return pc[k].get(e) == 1
ev = []
for c, e in enumerate(act):
    tags = []
    if f("raw_v", e) and f("raw_r", e): tags.append("raw-deq")
    if f("mx_iv", e) and f("mx_ir", e): tags.append("MX-issue")
    if f("dma_iv", e) and f("dma_ir", e): tags.append("DMA-issue")
    if f("f_v", e) and f("f_r", e): tags.append("FENCE-done")
    if f("ar_v", e) and f("ar_r", e): tags.append("AR")
    if f("r_v", e) and f("r_r", e): tags.append("R")
    if f("aw_v", e) and f("aw_r", e): tags.append("AW")
    if f("w_v", e) and f("w_r", e): tags.append("W")
    busy = ("mx " if f("mx_busy", e) else "") + ("dmaA " if f("dma_active", e) else "") + ("dmaB" if f("dma_busy", e) else "")
    ev.append((c, busy.strip(), " ".join(tags)))
# run-length print
prev = None; start = 0
for c, busy, tags in ev + [(len(ev), "END", "")]:
    key = (busy, tags)
    if key != prev:
        if prev is not None:
            print(f"{start:4}-{c-1:4} ({c-start:3})  busy[{prev[0]:<14}] {prev[1]}")
        prev, start = key, c
mxb = sum(f("mx_busy", e) for e in act); dma = sum(f("dma_active", e) for e in act)
both = sum(f("mx_busy", e) and f("dma_active", e) for e in act)
neither = sum(not f("mx_busy", e) and not f("dma_active", e) for e in act)
neither_b = sum(not f("mx_busy", e) and not f("dma_busy", e) for e in act)
print(f"mxBusy {mxb}  dmaActive {dma}  both {both}  neither(mx,dmaActive) {neither}  neither(mx,dmaBusy) {neither_b}")
