# AI assistance: written with Claude (Anthropic; Claude Code, Claude Opus 5.5),
# then reviewed, run and validated by Ben Colbourne. See the thesis's
# Statement on the Use of Generative AI.
"""Print-size timing diagrams from the WaveJSON files written by plot_wavedrom.py.

    ~/.local/share/thesis-plots/bin/python plot_timing.py <wavejson_dir> <out_dir>

WaveDrom draws text at a fixed pixel size relative to each cycle's width, so a
figure 40-50 cycles wide prints at about 2 pt when scaled to the text width.
This script redraws the same WaveJSON (same lanes, values, gaps and arrows)
with matplotlib at the size it is printed, so lane names are 9 pt in the PDF (Latin Modern, matching the body text).
Include each PDF at its natural size (or at the width given in WIDTH_IN).

Supported WaveJSON subset (all that plot_wavedrom.py emits): bit lanes
('0', '1', '.', '|'), data lanes ('=', 'x', '.', '|' with "data"), a clock
lane ('p', '.'), "node" strings, "edge" entries of the forms "A~>B label"
and "A-|B", and head.tick.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon

TEXT_WIDTH_IN = 6.3          # A4 with 3 cm + 2 cm margins
WIDTH_IN = {                 # printed width of each figure, inches
    "wave_hadamard_4x4": TEXT_WIDTH_IN,
    "wave_matmul_release_8x8": TEXT_WIDTH_IN,
    "wave_transpose_subbanks_8x8": TEXT_WIDTH_IN,
    "wave_loadstationary_8x8": 0.75 * TEXT_WIDTH_IN,
}
# Edge labels that would collide with lane names at the default place (left of
# the arrow's start): offset from the start node in cycles / lanes, and alignment.
LABEL_AT = {"9 cycles before the first write": (0.6, -0.58, "left")}
LANE_IN = 0.27               # height of one lane, inches
NAME_PT, DATA_PT = 9, 8      # lane names, values inside data lanes
GAP_W = 0.6                  # width of a '|' gap column, in cycles
SLOPE = 0.12                 # transition width, in cycles
AMP = 0.32                   # half the signal height, in lane units
WRAP_AMP = 0.45              # half-height of a data lane whose labels wrap onto two lines
WRAP_BELOW = 7.5             # wrap a label (at an underscore) that would print below this size

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
for _f in sorted(os.listdir(FONT_DIR)):
    font_manager.fontManager.addfont(os.path.join(FONT_DIR, _f))
plt.rcParams.update({"font.family": ["Latin Modern Roman", "DejaVu Serif"],
                     "pdf.fonttype": 3, "svg.fonttype": "none"})


def columns(n_cols, waves):
    """x position of the left edge of each column; gap columns are narrower."""
    gap = [any(i < len(w) and w[i] == "|" for w in waves) for i in range(n_cols)]
    xs, x = [], 0.0
    for g in gap:
        xs.append(x)
        x += GAP_W if g else 1.0
    return xs + [x], gap


def expand(wave):
    """Per-column state: '0'/'1'/'x' or ('=', k) for the k-th data value, with '.' and '|' resolved."""
    out, prev, k = [], "x", -1
    for ch in wave:
        if ch in ".|":
            out.append(prev)
            continue
        if ch == "=":
            k += 1
            prev = ("=", k)
        elif ch in "01x":
            prev = ch
        out.append(prev)
    return out


def draw_bit(ax, xs, states, y):
    pts = []
    for i, s in enumerate(states):
        lvl = y + (AMP if s == "1" else -AMP)
        x0, x1 = xs[i], xs[i + 1]
        if pts and pts[-1][1] != lvl:
            pts.append((x0 + SLOPE, lvl))
        elif not pts:
            pts.append((x0, lvl))
        pts.append((x1, lvl))
    ax.plot(*zip(*pts), color="black", lw=0.8, solid_joinstyle="miter")


def draw_clock(ax, xs, y):
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        m = (x0 + x1) / 2
        ax.plot([x0, x0, m, m, x1], [y - AMP, y + AMP, y + AMP, y - AMP, y - AMP],
                color="black", lw=0.8)


def draw_data(ax, xs, states, data, y, gap):
    """Data lane. A label that would print below WRAP_BELOW pt on one line is split at an
    underscore onto two lines, and the whole lane's boxes are drawn taller (WRAP_AMP)."""
    segs, i = [], 0
    while i < len(states):
        j = i
        while j + 1 < len(states) and states[j + 1] == states[i]:
            j += 1
        segs.append((xs[i], xs[j + 1], states[i]))
        i = j + 1
    labels = {}
    for x0, x1, st in segs:
        if st != "x":
            labels[(x0, x1)] = data[st[1]] if st[1] < len(data) else ""
    wrap = {k for k, v in labels.items()
            if "_" in v and text_size(ax, v, k[1] - k[0] - 2 * SLOPE - 0.1) < WRAP_BELOW}
    amp = WRAP_AMP if wrap else AMP
    for x0, x1, st in segs:
        poly = [(x0, y), (x0 + SLOPE, y + amp), (x1 - SLOPE, y + amp),
                (x1, y), (x1 - SLOPE, y - amp), (x0 + SLOPE, y - amp)]
        if st == "x":
            ax.add_patch(Polygon(poly, closed=True, facecolor="none", edgecolor="0.55",
                                 hatch="////", lw=0.5))
            continue
        ax.add_patch(Polygon(poly, closed=True, facecolor="white", edgecolor="black", lw=0.8))
        label = labels[(x0, x1)]
        if (x0, x1) in wrap:
            k = min((m for m, ch in enumerate(label) if ch == "_"), key=lambda m: abs(m - len(label) / 2))
            label = label[:k + 1] + "\n" + label[k + 1:]
        # white background so a gap mark inside the box passes behind the label
        t = ax.text((x0 + x1) / 2, y, label, ha="center", va="center", fontsize=DATA_PT,
                    linespacing=1.0, zorder=3, bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
        fit_text(ax, t, x1 - x0 - 2 * SLOPE - 0.1)


def text_size(ax, label, width):
    """Font size fit_text would settle on for a one-line label."""
    t = ax.text(0, 0, label, fontsize=DATA_PT)
    fit_text(ax, t, width)
    size = t.get_fontsize()
    t.remove()
    return size


def fit_text(ax, t, width, min_pt=6.0):
    """Shrink a label until it fits within width (data units), down to min_pt."""
    r = ax.figure.canvas.get_renderer()
    inv = ax.transData.inverted()
    while t.get_fontsize() > min_pt:
        bb = t.get_window_extent(r)
        w = inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0]
        if w <= width:
            break
        t.set_fontsize(t.get_fontsize() - 0.25)


def draw_gaps(ax, xs, gap, n_lanes):
    for i, g in enumerate(gap):
        if not g:
            continue
        m = (xs[i] + xs[i + 1]) / 2
        for lane in range(n_lanes):
            y = -lane
            for dx in (-0.09, 0.09):
                ax.plot([m + dx - 0.07, m + dx + 0.07], [y - AMP - 0.12, y + AMP + 0.12],
                        color="black", lw=0.7)
            ax.add_patch(Polygon([(m - 0.09, y - 0.5), (m + 0.09, y - 0.5),
                                  (m + 0.09, y + 0.5), (m - 0.09, y + 0.5)],
                                 facecolor="white", edgecolor="none", zorder=1.5))


def node_positions(signal, xs):
    pos = {}
    for lane, s in enumerate(signal):
        for i, ch in enumerate(s.get("node", "")):
            if ch != ".":
                pos[ch] = (xs[i], -lane)
    return pos


def draw_edges(ax, edges, pos):
    for e in edges:
        spec, _, label = e.partition(" ")
        if "~>" in spec:
            a, b = spec.split("~>")
            ax.annotate("", xy=pos[b], xytext=pos[a],
                        arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=0.8,
                                        shrinkA=0, shrinkB=0, mutation_scale=7,
                                        connectionstyle="arc3,rad=-0.15"), zorder=4)
        elif "-|" in spec:
            a, b = spec.split("-|")
            (xa, ya), (xb, yb) = pos[a], pos[b]
            ax.plot([xa, xa, xb], [ya, yb, yb], color="tab:blue", lw=0.8, zorder=4)
        else:
            continue
        if label:   # beside the arrow's start, clear of the signal lines
            xa, ya = pos[a]
            dx, dy, ha = LABEL_AT.get(label, (-0.3, 0.0, "right"))
            ax.text(xa + dx, ya + dy, label, ha=ha, va="center",
                    fontsize=DATA_PT, color="tab:blue", zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.6))


def render(spec, name, out):
    signal = [s for s in spec["signal"] if isinstance(s, dict)]
    waves = [s.get("wave", "") for s in signal]
    n_cols = max(len(w) for w in waves)
    xs, gap = columns(n_cols, waves)
    width = WIDTH_IN.get(name, TEXT_WIDTH_IN)
    tick = spec.get("head", {}).get("tick")
    top = 0.9 if tick is not None else 0.6
    height = LANE_IN * (len(signal) - 1 + top + 0.6)

    fig = plt.figure(figsize=(width, height))
    # left margin wide enough for the longest lane name
    probe = fig.text(0, 0, max((s["name"] for s in signal), key=len), fontsize=NAME_PT)
    fig.canvas.draw()
    name_in = probe.get_window_extent().width / fig.dpi + 0.08
    probe.remove()
    ax = fig.add_axes([name_in / width, 0, 1 - name_in / width - 0.01, 1])
    ax.set_xlim(0, xs[-1])
    ax.set_ylim(-(len(signal) - 1) - 0.6, top)
    ax.axis("off")

    for i in range(len(xs)):
        if i < len(gap) and gap[i]:
            continue
        ax.axvline(xs[i], color="0.85", lw=0.4, ls=(0, (1, 2)), zorder=0)
    if tick is not None:
        for i in range(n_cols):
            ax.text((xs[i] + xs[i + 1]) / 2, 0.62, str(tick + i), ha="center", va="center",
                    fontsize=DATA_PT, color="0.35")

    for lane, s in enumerate(signal):
        y = -lane
        ax.text(-0.25, y, s["name"], ha="right", va="center", fontsize=NAME_PT,
                transform=ax.transData, clip_on=False)
        w = s.get("wave", "")
        if w.startswith("p"):
            draw_clock(ax, xs[:len(w) + 1], y)
        elif "=" in w:
            draw_data(ax, xs, expand(w.ljust(n_cols, ".")), s.get("data", []), y, gap)
        else:
            draw_bit(ax, xs, expand(w.ljust(n_cols, ".")), y)
    draw_gaps(ax, xs, gap, len(signal))
    draw_edges(ax, spec.get("edge", []), node_positions(signal, xs))

    fig.savefig(os.path.join(out, name + ".pdf"))
    fig.savefig(os.path.join(out, name + ".png"), dpi=200)
    plt.close(fig)


def main(src, out):
    os.makedirs(out, exist_ok=True)
    for f in sorted(os.listdir(src)):
        if f.endswith(".json"):
            render(json.load(open(os.path.join(src, f))), f[:-5], out)
            print("wrote", f[:-5])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
