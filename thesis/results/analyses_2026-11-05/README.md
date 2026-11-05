<!-- AI assistance: computed by Claude (Anthropic; Claude Code, Claude Opus 5.5) at Ben Colbourne's request, 2026-11-05, from thesis/results/final_new. -->
# Analyses for the second review (2026-11-05)

## Per-rule error levels (all 4,525 executed cases, every suite, all four sizes; 0 failures)
Rule and pre-reduction count taken from the class suite by canonical equation (BLAS3 TN/TT mapped to the matrix product).

| Level | Rules | Cases | Max normwise rel. error |
|---|---|---|---|
| exact | broadcast, scalar-identity, vector-Hadamard (no pre-reduction); unary | 668 | 0 |
| fp32 | matrix product, diagonal, trace (no pre-reduction) | 2,794 | 1.40e-5 (2,777 <= 1e-6) |
| fp16, non-scalar output | Hadamard; anything with a pre-reduction | 904 | 8.6e-4 (all <= 1e-3) |
| fp16, scalar output | as above, scalar result | 159 | 1.33e-2 (near-cancelling references) |

The 17 fp32-level cases above 1e-6: 15 are BLAS3 at 32x32 with k = 18, 21, 27 (all 7-8 routines at
k = 18: 1.07e-5 to 1.40e-5; k = 21 SYMM 7.7e-6; k = 27: 3.3e-6 to 4.5e-6), and two trace cases at 1.0e-6.
NOTE: the thesis says BLAS3 errors are at most 2e-7 except one SYMM case (1.4e-5); that is inaccurate.

## Language-model family of the einsum benchmark (lm_*, 25 instances)
2,135 steps; 0 in the 2D class at 16x16 or 32x32 (none scalar/vector either).
2,083 (97.6%) have an operand of rank > 2 (typical operand ranks 2-6, output ranks 3-6);
52 (2.4%) have a dimension larger than the array (dimensions up to 3,500).
The family is language-model likelihood computations (batch_likelihood, first_last), not transformer layers.
