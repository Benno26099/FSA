# AI assistance: written with Claude (Anthropic; Claude Code, Claude Opus 5/5.5),
# then reviewed, run and validated by Ben Colbourne. See the thesis's
# Statement on the Use of Generative AI.
"""Enumerate every einsum in the 2D class, canonicalise, group by index roles.

Class: 1-2 operands, operand rank 0..2, no letter repeated within an operand,
output = distinct letters drawn from the inputs, output rank <= 2.
"""
from itertools import permutations, product, combinations
from collections import defaultdict

L = "ijkl"

def canon(ins, out):
    m = {}
    for op in ins:
        for c in op:
            if c not in m:
                m[c] = L[len(m)]
    t = lambda s: "".join(m[c] for c in s)
    return ",".join(t(o) for o in ins) + "->" + t(out)

def strings(rank):
    return ["".join(p) for p in permutations(L, rank)]

def roles(ins, out):
    a = ins[0]; b = ins[1] if len(ins) == 2 else ""
    r = []
    for c in sorted(set(a) | set(b)):
        ina, inb, ino = c in a, c in b, c in out
        if len(ins) == 1:
            r.append("F" if ino else "R")
        elif ina and inb:
            r.append("E" if ino else "C")
        elif ina:
            r.append("Fa" if ino else "Ra")
        else:
            r.append("Fb" if ino else "Rb")
    return tuple(sorted(r))

eqs = set()
for nops in (1, 2):
    for ranks in product(range(3), repeat=nops):
        for ins in product(*(strings(r) for r in ranks)):
            letters = sorted(set("".join(ins)))
            for orank in range(min(2, len(letters)) + 1):
                for out in permutations(letters, orank):
                    eqs.add(canon(ins, "".join(out)))

groups = defaultdict(list)
for e in sorted(eqs):
    lhs, out = e.split("->")
    ins = lhs.split(",")
    key = (len(ins), tuple(len(o) for o in ins), len(out), roles(ins, out))
    groups[key].append(e)

if __name__ == "__main__":
  print(f"{len(eqs)} canonical equations, {len(groups)} role classes\n")
  for (n, rk, orank, r), es in sorted(groups.items()):
      print(f"{n} ops ranks={rk} out={orank} roles={'+'.join(r) or '-':14} {len(es):2}  {' '.join(es)}")
