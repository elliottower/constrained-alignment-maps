# Pre-registration: seeding Paper A's single-run tables

**Frozen** 2026-07-29, before the run. The tables named below currently report
single runs; no multi-seed results for them exist.

**Script:** `experiments/k1_vae_vs_das.py`
**SHA-256:** \TODO{fill after the seed argument is threaded, before launch}

## Why

Three tables carry Paper A's headline numbers and are single runs per cell:

1. six GPT-2 tasks, standard and strict IIA at $k = 1$;
2. cross-distribution splits on IOI (standard, held-out entities, held-out
   templates);
3. the four control ablations.

Single runs have already produced two claims in this project that did not survive
seeding. A $k = 8$ saturation point was explained by a Fourier-dimension argument
and turned out to be $0.806 \pm 0.570$ across seeds. A decline in unconstrained
nonlinear DAS at $k = 32$ was reported as evidence of convergence below the causal
variable and turned out to be noise. Both were removed. The remaining single-run
tables are exposed to the same failure.

## Design

Three seeds, varying alignment-map initialisation and training-batch sampling.
The base model (GPT-2) is fixed and pretrained, so unlike the arithmetic sweeps
there is no model seed: the only randomness is in the fitting. Evaluation pairs
are held fixed across seeds so that replicates differ in the map and not in the
data.

Reported as mean with a 95\% confidence interval from Student's $t$, with the
maximum over seeds beneath it, for every method. Three seeds gives a wide interval
by construction; that is the honest consequence of three seeds.

## Predictions

**H1 (primary).** The strict-IIA ordering on the six tasks is unchanged: the
structured VAE exceeds DAS on all six, and its advantage on subject--verb
agreement (currently $1.00$ against $0.00$) remains non-overlapping.

**H2 (primary).** The cross-distribution dissociation survives. On held-out
entities the structured VAE stays below $0.10$ and the unconstrained map stays
above $0.90$; on held-out templates the structured VAE stays above $0.90$.

**H3 (secondary).** Every control remains within $0.05$ of its task's
random-subspace floor.

**H4 (secondary).** Confidence intervals on the six-task table are narrower than
those on the arithmetic k-sweep, because the base model is fixed rather than
retrained per seed.

## Decision rule, fixed in advance

| outcome | action |
|---|---|
| H1 and H2 hold | Report with intervals. The claims stand as written. |
| H1 fails on any task (intervals overlap) | That task is reported as not separating the methods. If it fails on three or more of six, the six-task table is demoted from headline to supporting and the paper leads on the reconstruction result instead. |
| H2 fails | The cross-distribution section is withdrawn. It is currently described as the sharpest result after the random-network control, and it cannot carry that role on overlapping intervals. |
| H3 fails | A control sitting above its floor means the corresponding ablation does not rule out what it claims to. Report the specific control as inconclusive rather than adjusting the floor. |

Any table whose ordering does not survive is reported with intervals showing the
overlap, not dropped.

## What would invalidate the run

A seed failing to train (loss not decreasing, or IIA at chance for every method
including the structured VAE) is excluded and reported, not replaced.

## Not being varied

Layer, $k$, task set, evaluation pairs, hard-example criteria, all method
hyperparameters, the base model. Only fitting randomness.
