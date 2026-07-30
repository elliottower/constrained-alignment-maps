# Pre-registration: is DAS's subspace degeneracy in the model or in the fitting?

**Frozen** 2026-07-29, before the script has been run. No results from it exist.

**Script:** `experiments/degeneracy_decomposition.py`
**SHA-256:** `7ce4496de0e7cbe4cd05983f9d09b8284366fefd88b6ba1b6e3d9face7056d7c`

## Why

The existing degeneracy result trains ten grokked multiplication models and fits
DAS once to each, finding pairwise subspace overlap $\approx 0.008$ against a
chance value of $k/d = 0.016$. Every model reaches IIA between $0.94$ and $1.00$.

That design varies the model seed and the fit together, so two explanations are
indistinguishable:

- each grokked model has a canonical subspace, and different models select
  different ones (**model-level**);
- a single model does not pin a subspace, and DAS lands somewhere arbitrary on
  each fit (**fitting-level**).

The claim "DAS recovers a subspace, not the subspace" is true either way, but it
means different things, and a paper built on it has to say which.

## Design

Grokked modular multiplication, $p = 113$, $k = 2$, $d_\text{model} = 128$.
Six models (seeds 1000--1005), ten DAS fits where applicable.

| condition | varies | holds fixed | isolates |
|---|---|---|---|
| **A\_random** | fit seed | model | whether one model pins a subspace |
| **A\_pca** | fit seed | model, initialisation | batch-sampling sensitivity alone |
| **B** | model seed | one fit each | the published claim |
| **SVD** | model seed | no fitting at all | whether weight geometry is consistent |

`A_pca` exists as a control: the delta-PCA initialisation is a deterministic
function of the data, so fits sharing it can only differ through batch sampling.
If `A_pca` clusters tightly and `A_random` does not, the spread in `A_random` comes
from the initialisation and not from batch noise.

`SVD` takes the top-$k$ right singular subspace of $W_\text{in} W_\text{out}$ per
model. No optimisation is involved, so it measures the weights directly.

Reported for every condition: mean, minimum and maximum pairwise subspace overlap
(mean squared cosine of principal angles) and geodesic distance on $\Gr(k,d)$.
Chance overlap is $k/d = 0.0156$; maximum geodesic distance is
$\pi/\sqrt{2} \approx 2.22$ radians.

DAS uses random orthogonal initialisation throughout except in `A_pca`, matching
the reference implementation.

## Predictions

**H1 (primary).** `A_random` gives mean overlap below $0.10$. A single model does
not pin a subspace, and the degeneracy is at least partly in the optimisation
landscape rather than only in differences between models.

**H2 (primary).** `A_pca` gives mean overlap above $0.50$, confirming that fits
sharing a deterministic initialisation cluster and that `A_random`'s spread is
attributable to initialisation rather than batch sampling.

**H3 (secondary).** `SVD` gives mean overlap above $0.50$ — weight geometry is
consistent across models where fitted subspaces are not.

**H4 (secondary).** `B` reproduces the published result, mean overlap below
$0.05$.

## Decision rule, fixed in advance

| outcome | what the paper claims |
|---|---|
| H1 holds, H2 holds | Degeneracy is fitting-level. The claim becomes that DAS's objective has many near-equivalent optima on a single model, and reporting one is arbitrary. |
| H1 fails (`A_random` overlap $\geq 0.10$), H4 holds | Degeneracy is model-level. Each model has a canonical subspace; different models choose differently. The claim narrows to a statement about cross-model comparison, and the phrase "not the subspace" must be qualified. |
| H1 holds, H2 fails | The spread is batch sampling, not initialisation. Report as an optimisation-stability finding and do not attribute it to the landscape. |
| H3 holds | Report as "DAS is degenerate, weight geometry is not." This is the stronger version of the paper and the invariants section is built on it. |
| H3 fails | The invariants section rests on circle geometry and equivariance alone. Say so; do not present the weight-space result as an invariant. |

**Consequence for the weight-space comparison.** The manuscript reports SVD
reaching $\IIA = 0.999$ against DAS's $0.150$ at mismatched dimensions ($k=32$
against $k=8$). That comparison is not reportable and is not used. `SVD` here is
dimension-matched to the DAS conditions at $k = 2$.

## What would invalidate the run

Any model failing to grok (test accuracy below $0.95$) is excluded, and the
condition is reported with the reduced count rather than the model being replaced.
Fewer than four surviving models voids the run.

## Not being varied

Operation, prime, $k$, layer, hook site, DAS step count, learning rate, batch
size, the number of intervention pairs. Only what each condition's column states.
