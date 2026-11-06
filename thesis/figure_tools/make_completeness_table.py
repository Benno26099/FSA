# AI assistance: written with Claude (Anthropic; Claude Code, Claude Opus 5.5),
# then reviewed, run and validated by Ben Colbourne. See the thesis's
# Statement on the Use of Generative AI.
"""Completeness check behind tab:completeness: enumerate every (C, E, Fa, Fb) the rank limits allow,
map each through the planner rule order, and count the 137 class equations reaching it.

    cd python && .venv/bin/python3 ../thesis/figure_tools/make_completeness_table.py
"""
import sys, os, itertools, collections
sys.path.insert(0, os.getcwd())
from einsum import roles_helper
sys.path.insert(0, os.path.abspath("../thesis/figure_tools"))
from make_class_cases import rule_of
eqs=[l.split("\t")[0] for l in open("einsum_class_expected_passes.tsv") if l.strip() and not l.startswith("#") and "->" in l]
print("equations:",len(eqs))
def rule(c,e,fa,fb):
    if e==2: return "hadamard"
    if e==1 and c==1: return "diag"
    if e==1 and fa+fb==1: return "broadcast"
    if e==1: return "vec_hadamard"
    if c==2: return "trace"
    if fa<=1 and fb<=1: return "matmul"
    if fa==2 or fb==2: return "sI"
    return None
# counts per combo from real equations (two-operand)
cnt=collections.Counter(); unary=0; ex=collections.defaultdict(list)
for eq in eqs:
    lhs,out=eq.split("->"); ins=lhs.split(",")
    if len(ins)==1: unary+=1; continue
    roles=roles_helper(ins,out)
    red=["".join(ch for ch in op if not roles[ch].startswith("R")) for op in ins]
    v=list(roles_helper(red,out).values())
    k=(v.count("C"),v.count("E"),v.count("Fa"),v.count("Fb")); cnt[k]+=1; ex[k].append(eq)
    assert rule(*k)==rule_of(eq)[0], eq
print("unary eqs:",unary)
tot=0
for c,e,fa,fb in itertools.product(range(3),repeat=4):
    if c+e+fa>2 or c+e+fb>2 or e+fa+fb>2: continue
    r=rule(c,e,fa,fb); tot+=cnt[(c,e,fa,fb)]
    print(f"C={c} E={e} Fa={fa} Fb={fb} -> {r:13s} eqs={cnt[(c,e,fa,fb)]:3d} e.g. {ex[(c,e,fa,fb)][:2]}")
print("two-operand total", tot, "; combos with eqs not enumerated:", [k for k in cnt if not (k[0]+k[1]+k[2]<=2 and k[0]+k[1]+k[3]<=2 and k[1]+k[2]+k[3]<=2)])
