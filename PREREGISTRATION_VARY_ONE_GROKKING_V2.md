# Pre-registration v2: vary-one variance ratio on grokked modular arithmetic

Supersedes `PREREGISTRATION_VARY_ONE_GROKKING.md` for the estimator. That
document stays frozen at commit `d1a0a88` as the record of what was predicted
before the pilot; nothing in it is edited.

**Frozen before the variance-ratio estimator has been run on any grokked model.**

## Why v1's estimator was replaced

The v1 pilot ran and reported, on the cached grokked addition model:

- **overlap(S_A, S_B) = 1.0000 at every k.** Designs A and B recover the
  identical subspace. This is structural rather than incidental: with
  `h(a,b) = f(a) + g(b) + q(a+b)`, design A spans `span(f) ⊕ span(q)` and design
  B spans `span(g) ⊕ span(q)`, so they coincide exactly when `span(f) = span(g)`
  — which holds because both operands are read through the same embedding
  matrix. **The intersection step in v1 is therefore a no-op**, and v1's gate 3
  ("intersection non-trivial") passed vacuously, since it is satisfied when the
  intersection is everything.
- overlap(S_A, S_C) = 0.48 at k=2, 0.26 at k=8, 0.81 at k=16. Partial overlap.
- `dim(result) = k` at k ≤ 8: the subtraction removed nothing, because at matched
  k the recovered S_C is not contained in the recovered S_A.

Subspace subtraction is the wrong estimator for a second reason the pilot could
not have shown. If the model encodes the result at the same Fourier frequencies
as the inputs, then at fixed `b0`,
`cos(w(a+b0)) = cos(wa)cos(wb0) - sin(wa)sin(wb0)`, so the result direction is a
phase-rotated copy of the input's own Fourier pair and the two are genuinely
inseparable. That regime must be *detectable*, not silently reported as a result
subspace.

## The v2 estimator

Directions where varying the result moves activations much more than varying only
the inputs does. With `Sigma_A` and `Sigma_C` the within-configuration covariances
of designs A and C, solve the generalized eigenproblem

    Sigma_A v = lambda Sigma_C v

by Cholesky whitening of `Sigma_C`, with shrinkage 1e-3 toward a scaled identity.
Leading eigenvectors are result-specific; `lambda` near 1 means a direction is
driven equally by both designs and so is shared.

No matched k, no cosine tolerance, no residual-norm threshold — the three
constants v1 had to fix in advance, all of which degrade precisely in the
partial-overlap regime the model turns out to occupy.

**Design B leaves the construction and becomes a measurement.** overlap(S_A, S_B)
now reports whether the model encodes both operands in one shared subspace. On
this model it is 1.0000, which is a finding about the model rather than a step in
the method.

**Dimensionality is read from the spectral gap**, at the largest multiplicative
drop, requiring at least `MIN_GAP_FACTOR = 10`.

## Calibration of MIN_GAP_FACTOR, done on known-answer synthetics only

| planted structure | correct answer | n found | gap at boundary |
|---|---|---|---|
| separate result subspace, noise 0.05 | 2 | 2.0 | 82.5x |
| separate result subspace, noise 0.02 | 2 | 2.0 | 542.5x |
| separate result subspace, noise 0.005 | 2 | 2.0 | 6078x |
| result shares the inputs' directions | 0 | 0.0 | 1.0x |
| no result term at all | 0 | 0.0 | 1.1x |

A decade sits roughly an order of magnitude clear of both regimes. It was chosen
from these five cases, none of which involves a trained model.

The estimator is unbiased: overlap between the leading eigenvectors and the
planted result subspace goes 0.846 (noise 0.20), 0.880, 0.903, 0.981, and
**exactly 1.0000 at zero noise**. At zero noise the non-result eigenvalues are
exactly 1.0, matching the theoretical prediction that a shared direction is
driven equally by both designs.

## Frozen scripts

| script | SHA-256 |
|---|---|
| `experiments/vary_one.py` | `fba420dd56717ed9f95ef6bb54a123781f4d2585e40a9c7a7ea6463cc2c03f18` |
| `experiments/test_vary_one.py` | `16f09840560ab5423fae2de46d1874db1eba1c594debdd415353806fcda64d41` |
| `experiments/pilot_vary_one_grokking.py` | `4670a074764d615a828afbaca329ec7c6d3b70a986a593daaa0525435dcc3ac4` |

## What has been run

The v1 pilot, whose numbers are quoted above and saved at
`experiments/results/pilot_vary_one_grokking.json`. The synthetic calibration
above. **The variance-ratio estimator has not been run on any trained model.**

## Hypotheses

Every hypothesis below is confirmatory. H1 is primary; the others are secondary
in ordering only, and each carries its own decision rule.

**H1 (primary) — a result subspace exists and is causally effective.** On the
grokked addition model, the variance-ratio estimator returns at least one
direction, and interchange interventions restricted to it achieve strict accuracy
exceeding the upper bound of the empirical random-subspace floor by more than
0.05. *Predicted: pass.*

**H2 — dimensionality follows the Fourier account.** The number of result
directions equals 2F, with F the number of key frequencies measured independently
by `analyze_fourier_alignment`. *Predicted: F between 2 and 4, so 4 to 8
directions.* Revised down from v1's prediction of F between 4 and 6, because the
v1 pilot's explained-variance curve for design A reached 0.917 by six components.

**H3 (the discriminating control) — vacuity does not transfer.** On five randomly
initialised models, the estimator returns **zero** result directions, or its
leading directions fail interchange within 0.05 of the floor. Designs A and B
still recover subspaces above the null, since token identity is present at
initialisation. *Predicted: pass.*

*Falsification, no partial credit:* a result subspace clearing the floor by more
than 0.05 on a random network means the method is as vacuous as unconstrained
nonlinear DAS. Withdrawn as a contribution, reported as a negative result, and
not reframed.

**H4 — convergence with DAS.** Grassmannian distance between the result subspace
and the DAS subspace (fitted under `das.mib_config("arithmetic")`) falls below the
2.5th percentile of the empirical null. *Predicted: pass, with lower confidence
than H1*, since Méloux et al. (ICLR 2025) show DAS does not pin a unique subspace.

**H5 — memorisation control.** On a non-grokked model, zero result directions
while designs A and B still recover input subspaces. *Predicted: pass.*

**H6 — the shared-operand finding replicates.** overlap(S_A, S_B) exceeds 0.95 on
all five grokked seeds. *Predicted: pass*, on the argument that a shared embedding
forces `span(f) = span(g)`.

## Decision rules

| outcome | consequence |
|---|---|
| H1 and H3 hold | Reported as an optimisation-free corroboration of the DAS subspace; the paper's vacuity claim localises to optimisation against a behavioural target. |
| H1 fails | The estimator finds variance but not mediation. Reported as a probe result, explicitly not an alignment method. |
| H3 fails | Method withdrawn, negative result reported as described above. |
| H2 gives a count not equal to 2F | Reported as a discrepancy with the Fourier account; 2F is not re-derived post hoc to fit. |
| H4 fails while H1 and H3 hold | Two admissible methods find two different causally effective subspaces — Méloux's non-identifiability observed directly, reported as such. |
| H5 or H6 fails | Reported. H6 failing would mean the v1 pilot's central observation does not replicate, which would require re-examining this document's premise. |

## Not varied

Operation, prime, hook point, shrinkage (1e-3), `MIN_GAP_FACTOR` (10.0),
`min_ratio` (2.0), the centring rule. Changing any of these after seeing results
on a trained model is a v3, not an amendment.
