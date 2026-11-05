# AI assistance: written with Claude (Anthropic; Claude Code, Claude Opus 5/5.5),
# then reviewed, run and validated by Ben Colbourne. See the thesis's
# Statement on the Use of Generative AI.
"""WaveDrom timing diagrams from the per-cycle wave CSVs (extract_waves.py).

    ~/.local/share/thesis-plots/bin/python plot_wavedrom.py <data_dir> <out_dir>

Writes, per figure, the WaveJSON (<name>.json, hand-editable and re-renderable
with any WaveDrom tool), an SVG, a PDF for the thesis and a PNG preview.
Idle stretches longer than MAX_IDLE cycles are cut to one WaveDrom gap ('|');
the "cycles" lane labels each kept segment with its real cycle range.
"""
import csv
import json
import os
import sys

import cairosvg
import wavedrom

MAX_IDLE = 3
SHORT = {}   # full opcode names in the instruction lane


def load(path):
    rows = list(csv.DictReader(open(path)))
    return {k: [r[k] for r in rows] for k in rows[0]}


def kept_windows(d, windows, cols):
    keep, cyc = [], None
    for lo, hi in windows:
        k, cyc = kept_positions(d, lo, hi, cols)
        keep += ([None] if keep else []) + k
    return keep, cyc


def kept_positions(d, lo, hi, cols):
    """Indices to draw: every cycle near a change in any shown column; long idle runs dropped."""
    cyc = [int(c) for c in d["cycle"]]
    idx = [i for i, c in enumerate(cyc) if lo <= c <= hi]
    change = set()
    for col in cols:
        for a, b in zip(idx, idx[1:]):
            if d[col][a] != d[col][b]:
                change.update((a, b))
    keep, run = [], []
    for i in idx:
        near = any(abs(i - j) <= 1 for j in change) or i in (idx[0], idx[-1])
        if near:
            if len(run) > MAX_IDLE:
                keep.append(None)               # gap marker
            else:
                keep.extend(run)
            run = []
            keep.append(i)
        else:
            run.append(i)
    keep.extend(run if len(run) <= MAX_IDLE else [None])
    return keep, cyc


def bit_lane(name, vals, keep):
    wave, prev = "", None
    for i in keep:
        if i is None:
            wave += "|"
            continue
        v = "1" if vals[i] == "1" else "0"
        wave += "." if v == prev else v
        prev = v
    return {"name": name, "wave": wave}


def data_lane(name, labels, keep):
    """labels[i] = text while active, '' while idle (drawn as 'x')."""
    wave, data, prev = "", [], None
    for i in keep:
        if i is None:
            wave += "|"
            continue
        lab = labels[i]
        if lab == prev:
            wave += "."
        elif lab:
            wave += "="
            data.append(lab)
        else:
            wave += "x"
        prev = lab
    return {"name": name, "wave": wave, "data": data}


def cycle_lane(keep, cyc):
    labels, seg = {}, []
    for i in keep + [None]:
        if i is None:
            if seg:
                text = str(cyc[seg[0]]) if len(seg) == 1 else f"{cyc[seg[0]]} to {cyc[seg[-1]]}"
                for j in seg:
                    labels[j] = text
            seg = []
        else:
            seg.append(i)
    return data_lane("Cycles", [labels.get(i, "") for i in range(len(cyc))], keep)


def issue_labels(d, lo=None):
    """Executing instruction per cycle; blank before the first issue at or after cycle lo."""
    out, cur = [], ""
    for c, name in zip(d["cycle"], d["issue"]):
        if name and (lo is None or int(c) >= lo):
            cur = SHORT.get(name, name)
        out.append(cur)
    return out


def place_node(lane, keep, i, letter):
    pos = keep.index(i)
    node = lane.get("node", "")
    node += "." * (pos - len(node)) + letter
    lane["node"] = node


def render(spec, out, name):
    spec.setdefault("config", {"hscale": 1})
    with open(os.path.join(out, name + ".json"), "w") as f:
        json.dump(spec, f, indent=1)
    svg_path = os.path.join(out, name + ".svg")
    wavedrom.render(json.dumps(spec)).saveas(svg_path)
    cairosvg.svg2pdf(url=svg_path, write_to=os.path.join(out, name + ".pdf"))
    cairosvg.svg2png(url=svg_path, write_to=os.path.join(out, name + ".png"), scale=2)


def with_cycles(keep, d, cycles):
    """Add the given cycles to a kept-index list (re-deriving the gap markers), e.g. to
    give a short window enough width for its full opcode name."""
    idx = sorted({i for i in keep if i is not None} | {idx_of(d, c) for c in cycles})
    out = []
    for a, b in zip([None] + idx, idx):
        if a is not None and b != a + 1:
            out.append(None)
        out.append(b)
    return out


def idx_of(d, cycle):
    return d["cycle"].index(str(cycle))


def main(data_dir, out):
    os.makedirs(out, exist_ok=True)

    # Elementwise multiplication (ELEMENTWISE_MUL), 4x4
    d = load(os.path.join(data_dir, "wave_hadamard_4x4.csv"))
    cols = ["sp_read", "pe0_load_reg_li", "pe0_load_reg2_li", "pe0_mul_regs", "pe0_mac", "acc_write", "sem_release"]
    keep, cyc = kept_positions(d, -1, 113, cols + ["issue"])
    keep = with_cycles(keep, d, range(71, 77))   # room for "ELEMENTWISE_MUL"
    render({"signal": [
        cycle_lane(keep, cyc),
        data_lane("Instruction", issue_labels(d), keep),
        bit_lane("Scratchpad read", d["sp_read"], keep),
        bit_lane("load_reg_li (PE row 0)", d["pe0_load_reg_li"], keep),
        bit_lane("load_reg2_li (PE row 0)", d["pe0_load_reg2_li"], keep),
        bit_lane("mul_regs (PE row 0)", d["pe0_mul_regs"], keep),
        bit_lane("mac, drain (PE row 0)", d["pe0_mac"], keep),
        bit_lane("Accumulator write", d["acc_write"], keep),
        bit_lane("Semaphore release", d["sem_release"], keep),
    ], }, out, "wave_hadamard_4x4")

    # TENSOR_MULTIPLICATION, 8x8: semaphore release vs accumulator writes (release after the writes, and at rows-1)
    d = load(os.path.join(data_dir, "wave_matmul_8x8.csv"))
    old = ["1" if c == "41" else "0" for c in d["cycle"]]      # plan cycle rows-1 = 7 -> 33+1+7
    d["old_release"] = old
    d["acc_addr"] = [a if w == "1" else "" for a, w in zip(d["acc_write_addr"], d["acc_write"])]
    cols = ["pe0_mac", "acc_write", "sem_release", "old_release", "dma_acc_read"]
    keep, cyc = kept_positions(d, 32, 90, cols + ["issue"])
    old_l = bit_lane("Release at r \u2212 1 (scratchpad read done)", old, keep)
    new_l = bit_lane("Release at 2r + c \u2212 1 (result written)", d["sem_release"], keep)
    acc_l = data_lane("Accumulator write (row)", d["acc_addr"], keep)
    place_node(old_l, keep, idx_of(d, 41), "A")
    place_node(acc_l, keep, idx_of(d, 50), "B")
    place_node(acc_l, keep, idx_of(d, 57), "C")
    place_node(new_l, keep, idx_of(d, 57), "D")
    render({"signal": [
        cycle_lane(keep, cyc),
        data_lane("Instruction", issue_labels(d, 32), keep),
        bit_lane("mac (PE row 0)", d["pe0_mac"], keep),
        acc_l, old_l, new_l,
        bit_lane("DMA store reads accumulator", d["dma_acc_read"], keep),
    ], "edge": ["A~>B 9 cycles before the first write", "C-|D"],
        },
        out, "wave_matmul_release_8x8")

    # LOAD_STATIONARY, 8x8: SRAM read vs stationary-register writes
    keep, cyc = kept_positions(d, -1, 11, ["sp_read", "pe0_load_reg_li", "pe7_load_reg_li", "sem_release", "issue"])
    keep = [i for i in range(idx_of(d, -1), idx_of(d, 11) + 1)]        # short window: no gaps
    rd = bit_lane("Scratchpad read", d["sp_read"], keep)
    r0 = bit_lane("load_reg_li (PE row 0)", d["pe0_load_reg_li"], keep)
    place_node(rd, keep, idx_of(d, 8), "A")
    place_node(r0, keep, idx_of(d, 9), "B")
    render({"signal": [
        data_lane("Instruction", issue_labels(d), keep),
        {"name": "Clock", "wave": "p" + "." * (len(keep) - 1)},
        rd, r0,
        bit_lane("load_reg_li (PE row 7)", d["pe7_load_reg_li"], keep),
    ], "edge": ["A~>B 1 cycle"],
        "head": {"tick": -1}}, out, "wave_loadstationary_8x8")

    # TRANSPOSE_SRAM, 8x8: two sub-bank writes per row
    d = load(os.path.join(data_dir, "wave_transpose_8x8.csv"))
    for b in ("0", "1"):
        d[f"sb{b}"] = ["1" if w == "1" and s == b else "0" for w, s in zip(d["spad_write"], d["spad_write_subbank"])]
    d["row"] = [a if w == "1" else "" for a, w in zip(d["spad_write_addr"], d["spad_write"])]
    keep, cyc = kept_windows(d, [(-85, -60), (-19, -1)], ["sb0", "sb1"])
    render({"signal": [
        cycle_lane(keep, cyc),
        data_lane("Scratchpad row written", d["row"], keep),
        bit_lane("Sub-bank 0", d["sb0"], keep),
        bit_lane("Sub-bank 1", d["sb1"], keep),
    ]},
        out, "wave_transpose_subbanks_8x8")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
