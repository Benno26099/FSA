> AI assistance: written with Claude (Anthropic; Claude Code, Claude Opus 5/5.5), then reviewed and validated by Ben Colbourne. See the thesis's Statement on the Use of Generative AI.

# Coverage reconcile (Task 3c, parts 1–2) — 2026-10-11

`python/extract_einsum_cases.py` re-run against the role planner. Raw outputs:
`coverage_planner_sa{4,8,16,32}.txt` in this directory (the older
`coverage_sa4.txt` / `coverage_sa16.txt` are the pre-planner runs). Each run
reads all 168 benchmark.einsum.org instances, 1,615,342 pairwise steps, in
~35 s with `~/miniforge3/bin/python`.

## Correction: the old 90.1% was overstated — it was 78.1%

The old classifier (`canonical()`) matched the Hadamard patterns on index
*sets* only, so it counted `ij,ji->ij` (26,804 steps) and `ij,ji->` (12) as
executable by the old table. The old table had neither; `mx_einsum` would have
rejected them. Fixed in the script (B must match A's order). Corrected old
figure at N=16: **174,385 / 223,319 = 78.1%** of real-2D steps (was 201,201 =
90.1%). The thesis must not quote 90.1% any more.

## Headline numbers, N = 16

| | steps | share of all 1,615,342 |
|---|---|---|
| **in the 2D class** (planner executes) | **976,967** | 60.5% |
| — real 2D (both operands rank 2) | 223,289 | 13.8% |
| — mat-vec / scalar × matrix | 52,375 | 3.2% |
| — scalar/vector (trivial; report separately) | 701,303 | 43.4% |
| in class excluding scalar/vector | 275,664 | 17.1% |
| outside: an operand of rank > 2 | 618,103 | 38.3% |
| outside: rank-2 operands, rank-3+ output | 18,342 | 1.1% |
| outside: a dimension > N | 1,311 | 0.1% |
| outside: repeated index (diagonal) | 38 | 0.0% |

- **Real-2D coverage: old table 78.1% → planner 100%** (223,289 / 223,289).
  The old denominator (223,319) also held 30 scalar × matrix steps, which the
  planner reports under mat-vec/scaling (and covers).
- Everything the planner does not run is outside the 2D class by
  construction (rank-3+ operands or outputs, diagonals, oversized dims) — none
  is a 2D equation the engine lacks.
- **58 of the 137 canonical equations occur** in the benchmark. Most common:
  `i,->` 506,791; `ij,ij->ij` 127,301; `,->` 94,872; `i,i->` 92,337;
  `ij,ji->ij` 26,804; `ij,ik->jk` 23,674; `ij,j->i` 22,585; `ij,ij->j` 20,794.
  Rank-0 operands are common (`i,->`, `,->`), so scalar support (stage 4) matters
  for real workloads, not just for completeness.

## By SA size

| N | in class | excl. scalar/vector | real 2D | old table (corrected) |
|---|---|---|---|---|
| 4 | 976,244 | 274,988 | 223,039 | 174,150 |
| 8 | 976,787 | 275,503 | 223,222 | 174,321 |
| 16 | 976,967 | 275,664 | 223,289 | 174,385 |
| 32 | 977,129 | 275,821 | 223,429 | 174,523 |

4x4 relative to 32x32: 99.91% / 99.70% / 99.83% (in class / excl.
scalar-vector / real 2D). 4x4 vs 16x16 in class: 99.93% (the old "99.88%"
figure becomes 99.87% with the corrected classifier; same conclusion). The
suite still cannot drive the scaling study — dims are almost all 2.

## Per family (in class excl. scalar/vector, of all steps), N = 16

| family | covered / all | % |
|---|---|---|
| mc | 241,438 / 1,199,188 | 20.1 |
| qc | 18,209 / 61,740 | 29.5 |
| wmc | 13,912 / 329,271 | 4.2 |
| rnd | 1,534 / 9,093 | 16.9 |
| gm | 520 / 7,172 | 7.3 |
| str | 51 / 6,743 | 0.8 |
| lm | 0 / 2,135 | 0.0 |

## Part 3b — EVERY distinct benchmark case executed, all four sizes (2026-10-12)

The chip's work depends only on (equation, shapes), so the 976k–977k in-class
steps reduce to 165 / 448 / 582 / 740 distinct cases at N = 4 / 8 / 16 / 32
(`--mode extract --distinct`, `benchmark_distinct_saN.jsonl`, each record
carrying the number of steps it stands for). All were run on the matching
simulator with `run_cases.py`:

| SA | distinct cases | steps they stand for | passed | max abs err | max rel err |
|---|---|---|---|---|---|
| 4x4 | 165 | 976,244 | 165/165 | 6.7e-04 | 2.0e-03 |
| 8x8 | 448 | 976,787 | 448/448 | 2.0e-03 | 2.0e-03 |
| 16x16 | 582 | 976,967 | 582/582 | 2.0e-03 | 2.0e-03 |
| 32x32 | 740 | 977,129 | 740/740 | 2.0e-03 | 2.0e-03 |

**Claim this supports:** every distinct 2D contraction in benchmark.einsum.org
executes correctly on the chip, at every SA size — exhaustive, not sampled.
Weighted by steps: 45% one pass, 55% two (dominated by `i,->` = sum then
scale, 506,790 steps), 0.3% three. 32x32 run: 108 min. Supersedes Part 3 below
(the K=10 sample), kept for the record.

## Part 3 — benchmark steps executed on the chip (4x4), 2026-10-11

`extract_einsum_cases.py --mode extract --sa-size 4 --sample 10 --seed 42` ->
`benchmark_sample_k10.jsonl`: **1,097 real benchmark steps**, at most 10 per
(family, canonical equation) stratum, 147 strata, 54 canonical equations, 6
families (`lm` has no in-class steps). dims <= 4, so the same file runs at
every SA size. Run with `python/run_cases.py` (fp64 reference, `rtol=atol=1e-2`,
perf counters summed over a case's passes) -> `benchmark_sample_k10_4x4.csv`.
6 min at 4x4.

**1,097 / 1,097 pass.**

| group | cases | mean passes | max abs err | max rel err (normwise) |
|---|---|---|---|---|
| real 2D | 441 | 1.33 | 6.7e-04 | 2.0e-03 |
| mat-vec / scaling | 410 | 1.12 | 3.3e-04 | 3.9e-04 |
| scalar/vector | 246 | 1.32 | 2.0e-04 | 2.8e-04 |

Passes: 870 one-pass, 182 two-pass, 45 three-pass. `execTime` per pass (case total / passes):
210 cycles for 1,040 cases; 57 cases average 265 or 320 per pass — presumably
those with a transpose or Hadamard pass (not broken down per pass). Median `mxActive/execTime` 8% — these operands are tiny (mostly 2 or
2x2), so a pass is dominated by DMA and fixed overhead. Not yet run at 8x8 and
up.
