# Pre-registration v3: vary-one variance ratio, with a dimensionality rule that works

Supersedes v2's **dimensionality rule only**. v1 (`d1a0a88`) and v2 (`9faecb7`)
stay frozen as the record; nothing in either is edited. The estimator itself is
unchanged from v2.

## Results already obtained under v2, recorded as final

These were predicted in v2 and are **not re-tested here**. Re-running them under a
new document would be double-dipping.

**H3 (v2) — PASSED, decisively.** Variance ratios, design A over design C:

| model | top-8 ratios |
|---|---|
| grokked addition, p=113, test acc 0.9997 | 106.3, 63.3, 56.3, 51.3, 28.0, 27.1, 24.9, 22.1 |
| randomly initialised, 5 seeds | 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ~0.9 |

Maximum ratio across all five random seeds: **0.9895**. Random networks sit at the
theoretical no-structure value of 1.0 — a direction driven equally by both
designs. For contrast, unconstrained nonlinear DAS reaches 0.989 interchange
accuracy on a random network; this estimator reaches the floor exactly.

**H6 (v2) — PASSED.** overlap(S_A, S_B) = 1.0000 on the grokked model and 0.9995
to 0.9999 on all five random seeds. Present at initialisation, so the shared
operand subspace follows from the shared embedding matrix rather than from
training.

Saved at `experiments/results/vary_one_variance_ratio_v2.json`.

## Why v2's dimensionality rule is replaced

`MIN_GAP_FACTOR = 10` reports **0** result directions on a model whose spectrum
runs to 106x. It was calibrated on synthetics where the planted structure was a
clean two-dimensional block with a decade-wide gap; the real spectrum is graded,
so no gap reaches a decade and the rule returns a false negative.

The obvious repair — count directions above the random-network null — fails in the
opposite direction. The random maximum is 0.9895, and **114 of 128** grokked
directions exceed it.

Both fail because a variance *ratio* is scale-free: a direction carrying 1e-9 of
design A's variance can have a large ratio and mean nothing. Measured on the
grokked model:

| rank | ratio | variance share |
|---|---|---|
| 1-8 | 106 down to 22 | 0.098, 0.011, 0.267, 0.189, 0.012, 0.021, 0.025, 0.008 |
| 9 | 10.35 | 0.00044 |
| 10-81 | 6.9 down to 6.2 | 0.00017, flat |

Variance share collapses 18-fold at rank 9 and is flat thereafter. The top eight
directions carry 63% of design A's within-configuration variance; the other 106
that clear the null carry almost none.

## The v3 rule

A direction is **result-specific** when both hold:

1. variance ratio > 2.0, against a random-network null pinned at 1.0 with a
   measured maximum of 0.9895 across five seeds; and
2. share of design A's within-configuration variance > 1/d, the uniform value.

Neither reference point is tuned. The first comes from the random-network control
already run under v2; the second is uniformity. On the grokked model the rule
returns **7**.

That the rule is being fixed after seeing the grokked spectrum is why this is a
new pre-registration rather than an amendment, and why **no hypothesis resolved
under v2 is re-tested here**.

## Frozen scripts

| script | SHA-256 |
|---|---|
| `experiments/vary_one.py` | `976461e7ef7ead4cbe608b51caac95ef7d5355ab6f09c1243f38893420d8c5a9` |
| `experiments/test_vary_one.py` | `16f09840560ab5423fae2de46d1874db1eba1c594debdd415353806fcda64d41` |
| `experiments/pilot_vary_one_grokking.py` | `4670a074764d615a828afbaca329ec7c6d3b70a986a593daaa0525435dcc3ac4` |

## What has not been run

**Nothing causal.** Every number above is variance. No interchange intervention
has been performed in any recovered subspace. Until H1 resolves, this is a probe
result and is to be described as one.

## Hypotheses

All confirmatory. H1 is primary.

**H1 (primary) — the recovered directions mediate, not merely covary.**
Interchange interventions restricted to the 7 result-specific directions achieve
strict accuracy exceeding the upper bound of the task's empirical random-subspace
floor by more than 0.05. *Predicted: pass.*

*This is the hypothesis the method lives or dies on.* A subspace can carry 63% of
the variance and mediate nothing.

**H2 — dimensionality matches the Fourier account.** The count of 7 equals 2F
within one, with F the number of key frequencies measured independently by
`analyze_fourier_alignment`. *Predicted: F = 3 or 4, so 6 or 8 directions.* The
prediction is recorded before F is measured; a count matching only after F is
chosen to fit would be worthless.

**H3 — graded structure, not a block.** Strict interchange accuracy using the top
j directions increases monotonically in j up to 7 and plateaus after. *Predicted:
pass.* The spectrum is graded rather than a clean block, so the causal
contribution should be graded too; a step function would indicate the
dimensionality rule is cutting in the wrong place.

**H4 — convergence with DAS.** Grassmannian distance between the 7-direction
result subspace and the DAS subspace at matched k, fitted under
`das.mib_config("arithmetic")`, falls below the 2.5th percentile of the empirical
null. *Predicted: pass, with lower confidence than H1*, since Méloux et al. (ICLR
2025) show DAS does not pin a unique subspace.

**H5 — memorisation control.** On a non-grokked model (test accuracy below 0.95),
the rule returns zero result-specific directions while designs A and B still
recover input subspaces. *Predicted: pass.* Memorisation stores input-output pairs
without computing the sum.

**H6 — replication across seeds.** On five independently grokked addition models,
the rule returns between 4 and 12 directions on each. *Predicted: pass.*

## Decision rules

| outcome | consequence |
|---|---|
| H1 holds | With v2's H3 already passed, the method is an optimisation-free route to a causally effective subspace that returns the theoretical floor on a random network. Reported as corroboration of the DAS subspace independent of any behavioural objective. |
| H1 fails | The estimator finds variance without mediation. Reported as a probe, explicitly not an alignment method, and the paper claims nothing causal from it. H2-H6 become descriptive. |
| H2 count differs from 2F by more than one | Reported as a discrepancy with the Fourier account. 2F is not re-derived to fit. |
| H3 shows a step rather than a ramp | The dimensionality cut is in the wrong place. Report the interchange-versus-j curve and let it define the count instead of the rule. |
| H4 fails while H1 holds | Two methods, both admissible, find two different causally effective subspaces — Méloux's non-identifiability observed directly. Reported as such, not as a failure of either. |
| H5 fails | A memorising model has a result subspace, which would undermine the interpretation of v2's H3. Reported with equal prominence. |
| H6 gives counts outside 4-12 on any seed | The dimensionality is seed-dependent and no single number is reported; the distribution is. |

## Not varied

Operation, prime, hook point, shrinkage (1e-3), the ratio cut (2.0), the
variance-share cut (1/d), the centring rule. Changing any after seeing causal
results is a v4.
