# AI assistance: written with Claude (Anthropic; Claude Code, Claude Opus 5/5.5),
# then reviewed, run and validated by Ben Colbourne. See the thesis's
# Statement on the Use of Generative AI.
"""Minimal VCD reader for the thesis figures (no dependencies).

    signals(path)                      -> {full.hier.name: id}
    find(path, regex)                  -> [(name, id, width)] matching regex
    trace(path, names, clock=None)     -> {name: [(time, int_value)]} value changes
    per_cycle(changes, clk_changes)    -> value of a signal at each rising clock edge
"""
import re


def _header(path):
    scope, out = [], []
    with open(path, "r", errors="replace") as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "$scope":
                scope.append(tok[2])
            elif tok[0] == "$upscope":
                scope.pop()
            elif tok[0] == "$var":
                width, ident, name = int(tok[2]), tok[3], tok[4]
                out.append((".".join(scope + [name]), ident, width))
            elif tok[0] == "$enddefinitions":
                break
    return out


def find(path, regex):
    pat = re.compile(regex)
    return [v for v in _header(path) if pat.search(v[0])]


def trace(path, names):
    """Value changes for the given full names: {name: [(time, value)]}."""
    hdr = {n: i for n, i, _ in _header(path)}
    want = {}
    for n in names:
        want.setdefault(hdr[n], []).append(n)
    out = {n: [] for n in names}
    t = 0
    in_body = False
    with open(path, "r", errors="replace") as f:
        for line in f:
            if not in_body:
                if line.startswith("$enddefinitions"):
                    in_body = True
                continue
            c = line[0]
            if c == "#":
                t = int(line[1:])
            elif c in "01xzXZ":
                ident = line[1:].strip()
                if ident in want:
                    v = int(c) if c in "01" else -1
                    for n in want[ident]:
                        out[n].append((t, v))
            elif c == "b":
                val, ident = line[1:].split()
                if ident in want:
                    v = int(val, 2) if set(val) <= {"0", "1"} else -1
                    for n in want[ident]:
                        out[n].append((t, v))
    return out


def per_cycle(changes, clk_changes):
    """Sample a signal just before each rising clock edge -> [(edge_time, value)]."""
    edges = [t for (t, v), (_, pv) in zip(clk_changes[1:], clk_changes) if v == 1 and pv == 0]
    res, i, cur = [], 0, None
    for e in edges:
        while i < len(changes) and changes[i][0] < e:
            cur = changes[i][1]
            i += 1
        res.append((e, cur))
    return res
