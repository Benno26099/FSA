# anti_transpose source experiment — results (2026-10-02)

Script: `antitranspose_experiment.py` (this directory). Run from
`generators/fsa/python` with `uv run python <script> --config <cfg> --out <dir>`.
Repo state: commit `c700418`, `-debug` simulators. Operands: X seed 42, S seed 100,
uniform [-1, 1) fp16. Each output was checked against 7 candidate formulas;
every variant matched **exactly one**, the predicted one.

| variant | stationary load | stream revInput | predicted | 4x4 | 8x8 | 16x16 |
|---|---|---|---|---|---|---|
| A | normal (stride +1) | True  | X @ anti_transpose(S) | PASS | PASS | PASS |
| B | normal (stride +1) | False | (X @ S.T)[:, ::-1]    | PASS | PASS | PASS |
| C | reversed (last row, stride -1) | True  | X @ S[:, ::-1].T | PASS | PASS | PASS |
| D | reversed (last row, stride -1) | False | X @ S.T          | PASS | PASS | PASS |

Conclusion: the two reversals in `anti_transpose` come from (1) the
LoadStationary shift chain and (2) the stream's revInput flag. Both are
controllable from the instruction stream with no RTL change. OutputDelayer
contributes no permutation (it only de-skews). The remaining transpose
(spad row -> PE column) is structural to the stationary load.
