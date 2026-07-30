# Pre-registration: vary-one PCA on grokked modular arithmetic

**Frozen before the pilot ran. No result from any grokked model exists.**

Design in `SPEC_VARY_ONE_GROKKING.md`. Method from Shai et al., *Transformers
learn factored representations*, §H.1.1.

## Frozen scripts

| script | SHA-256 |
|---|---|
| `experiments/vary_one.py` | `13798ba2a93ed7fb91491471747d7f05477fb826fa3864413e0d360480289eff` |
| `experiments/test_vary_one.py` | `39a415af9924780144b35a90854d8866d6504a5598ab395a16e526ea11f9a1cd` |
| `experiments/pilot_vary_one_grokking.py` | `4670a074764d615a828afbaca329ec7c6d3b70a986a593daaa0525435dcc3ac4` |

## What is already known, stated so it is not hidden

`test_vary_one.py` has been run, on **synthetic** activations with a planted
factored structure. That is implementation validation, not a result about any
model. It established:

- designs A, B and C each recover the span their construction implies (overlap
  0.9999 in every case);
- design C does not leak the result subspace (overlap 0.0000);
- `(S_A ∩ S_B) ⊖ S_C` recovers the planted result subspace at overlap 0.9999,
  with recovered dimension 2 against planted dimension 2;
- with no result term planted, the construction returns **zero** dimensions;
- within-configuration centring gives 0.9999 where global centring gives 0.6527,
  so the step the spec calls load-bearing is load-bearing.

**Nothing has been run on the grokked model.** The cached checkpoint
(`results/holonomy_analysis_v4/addition/`, p=113, grokked, test accuracy 0.9996)
has been loaded by no analysis in this project's vary-one line of work.

## Setting

Grokked modular addition, p = 113, d_model = 128, hook
`blocks.0.hook_resid_post`, k ∈ {2, 4, 8, 16}. The activation grid over all
113² input pairs is computed once; the three designs are different groupings of
it, so they see identical activations.

Comparisons use DAS fitted under **MIB's arithmetic baseline** configuration
(`das.mib_config("arithmetic")`: lr 1e-2, 1 epoch, batch 256, n_features 16), not
the library default, which no MIB baseline uses.

## Pilot gate predictions

**G1 — anisotropy.** Design A's top-4 explained variance exceeds ten times the
isotropic value 4/d = 0.031. *Predicted: pass.* Near-certain; it is a sanity
check on the pipeline rather than a test of the method, and is recorded so that
its passing is not later reported as evidence for anything.

**G2 — the designs separate.** Overlap between S_A and S_C exceeds the empirical
null's 97.5th percentile and stays below 0.95. *Predicted: pass, and this is the
genuinely uncertain gate.* A spans the `a` and result directions, C spans `a` and
`b`; they share `a`, so substantial overlap is expected. If overlap exceeds 0.95
the designs are not separating anything and the construction in §2 of the spec is
void.

**G3 — intersection non-empty.** `S_A ∩ S_B` has at least one direction at some
k. *Predicted: pass at k ≥ 8, uncertain at k = 2.*

If G2 fails the spec is revised before any further compute. G1 or G3 failing at
every k is also a stop.

## Hypotheses

**H1 (primary) — the result subspace is recoverable without optimisation.**
`(S_A ∩ S_B) ⊖ S_C` achieves strict interchange accuracy exceeding the upper
bound of the task's empirical random-subspace floor by more than 0.05, at some
k ≤ 16. *Predicted: pass.* The result is the causal variable that DAS already
locates on this task.

**H2 — convergence with DAS.** Grassmannian distance between the recovered result
subspace and the DAS subspace falls below the 2.5th percentile of the empirical
null. *Predicted: pass, with lower confidence than H1.* Méloux et al. (ICLR 2025)
show DAS does not pin a unique subspace, so a variance-defined subspace need not
land on the particular one DAS returned.

**H3 (the discriminating control) — vacuity does not transfer.** On randomly
initialised models, five seeds: designs A and B still recover subspaces above the
null, because token identity is present at initialisation, while
`(S_A ∩ S_B) ⊖ S_C` returns zero dimensions or strict interchange accuracy within
0.05 of the floor. *Predicted: pass.*

*Falsification, no partial credit:* if the result subspace clears the floor by
more than 0.05 on a random network, vary-one is as vacuous as unconstrained
nonlinear DAS, the method's central claim is withdrawn, and that is reported
rather than reframed. It would be the more important finding — vacuity would then
be a property of subspace selection in high dimensions generally rather than of
optimisation against a behavioural target.

**H4 — dimensionality follows the Fourier account.** Explained variance in the
result subspace concentrates in approximately 2F components, where F is the
number of key frequencies measured independently by `analyze_fourier_alignment`.
*Predicted: F between 4 and 6, so 8 to 12 components.* This departs from Shai et
al.'s roughly two dimensions per factor, because a Fourier-coded factor with F
frequencies occupies 2F dimensions; a result near 2 would contradict the Fourier
account rather than confirm the method.

**H5 — memorisation control.** On a non-grokked model (test accuracy below 0.95),
designs A and B still recover input subspaces while the result subspace is empty
or at floor. *Predicted: pass.* Memorisation stores input-output pairs without
computing the sum, so there should be no dedicated result representation.

## Decision rules, fixed in advance

| outcome | consequence |
|---|---|
| H1 and H3 hold | Vary-one is reported as an optimisation-free corroboration of the DAS subspace, and the paper's vacuity claim is localised to optimisation against a behavioural target. |
| H1 fails | The construction finds variance but not mediation. Reported as a probe result, explicitly not as an alignment method, and H2–H5 become uninterpretable. |
| H3 fails | The method is vacuous. Withdrawn as a contribution and reported as the negative result described above. |
| H2 fails while H1 and H3 hold | Two admissible methods find two different causally effective subspaces. That is Méloux's non-identifiability observed directly, and it is reported as such rather than as a failure of either method. |
| H4 gives ≈2 components | The Fourier account does not explain the recovered subspace. Report the discrepancy; do not reinterpret 2F post hoc. |
| H5 fails | A non-grokked model has a result subspace, which would mean the construction detects something other than computed structure. Treated as a failure of H3's logic and reported with it. |

## Not varied

Operation (addition only for the pilot), prime, hook point, the k grid, the
centring rule, the intersection tolerance (cos > 0.9) and the residual-norm
threshold (0.1) in `vary_one.py`. Changing any of these after seeing results is a
new pre-registration, not an amendment.

## Multiplicity

One primary hypothesis, H1. H3 is the control that determines whether H1 means
anything; it is labelled primary in the spec's §5 and is treated here as a gate
on interpretation rather than as a second chance at a positive result. H2, H4 and
H5 are secondary and exploratory, and no secondary result is reported as
confirmatory evidence for the method.
