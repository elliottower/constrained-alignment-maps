# Pre-registration: latent dimension needed to realize IOI's output position

Frozen before the sweep runs. The commit SHA of this file is the registration.

## Claim under test

At the attention heads MIB selects, under MIB's own causal-abstraction metric, a
nonlinear alignment map realizes the `output_position` abstraction at a smaller
latent dimension than linear Distributed Alignment Search.

## Setup, fixed

Everything except the map is MIB's, unmodified:

| element | value | source |
|---|---|---|
| task | indirect object identification, causal-variable track | `tasks/IOI_task/ioi_task.py` |
| model | GPT-2 small | `ioi_utils.get_model_config` |
| sites | attention heads (7,3), (7,9), (8,6), (8,10) | `ioi_baselines.py:29` defaults |
| target variable | `output_position` | `ioi_baselines.py:193` |
| metric | squared error against the target logit difference, via `ioi_utils.checker` | `ioi_utils.py` |
| training | `init_lr` 1.0, 2 epochs, `regularization_coefficient` 0.0 | `ioi_baselines.py:144` |
| data | `get_counterfactual_datasets`, filtered by `FilterExperiment` | MIB |
| coefficients | fit once by `ioi_learn_linear_params.py` with the heads above, then frozen across every arm | MIB |

`output_position` rather than `output_token` because `raw_output` has exactly one
parent, `output_token` (`ioi_task.py:94`). Interchanging position leaves the
generated text unchanged, so a text criterion scores a do-nothing map at 1.000 —
measured, not assumed. `logit_diff` is the only node with both variables as
parents, which is why MIB scores this variable through it.

The coefficients are fit before these predictions are frozen. They are MIB's
observation model, not an outcome of any arm, and the same fitted values are used
by every arm.

## Arms

| arm | map | trained on |
|---|---|---|
| `random` | orthogonal subspace, untrained | nothing |
| `das` | MIB's `SubspaceFeaturizer` | interchange |
| `lcp_vae_interchange` | label-conditional partitioned VAE, MIB `Featurizer` contract | interchange |

Swept at k = 1, 2, 4, 8, 16, 32. Head units are 64-dimensional, so k = 32 is half
the representation and k = 1 is a single direction.

## Metric direction and the reference points

The metric is a squared error: **lower is better, and it has no chance level.**
Every number is therefore read against two measured reference points computed in
the same run:

- `E_random`: the untrained arm at the same k. A map that has learned nothing.
- `E_das(32)`: linear DAS at the largest k. The strongest linear result available.

An arm "realizes the abstraction at k" when its error is at or below
`E_das(32) * 1.10` — within 10% of the best linear result — at that k.

## Predictions

**H1 (primary).** `lcp_vae_interchange` realizes the abstraction at a smaller k
than `das`. Writing k\*(arm) for the smallest swept k at which an arm realizes it:
k\*(`lcp_vae_interchange`) < k\*(`das`).

**H2.** Both trained arms beat the untrained control at every k:
`E_arm(k) < E_random(k)` for all k. An arm that does not is not measuring the
model.

**H3.** `das` improves monotonically in k, up to noise. A linear subspace cannot
lose capacity by gaining a dimension, so a non-monotone curve indicates a
training failure rather than a property of the map, and is reported as such.

**Not predicted.** Whether either arm succeeds at k = 1. The registered claim is
the ordering of k\*, not any absolute value. Both arms failing at k = 1 and
separating at k = 2 or 4 confirms H1.

## Decision rule, fixed in advance

| outcome | consequence |
|---|---|
| k\*(VAE) < k\*(DAS) | H1 holds. Reported as the dimension gap, with both curves shown. |
| k\*(VAE) = k\*(DAS) | No dimension advantage. Reported as a null. The nonlinear map is not more compressive on this task at these sites. |
| k\*(VAE) > k\*(DAS) | H1 fails in the opposite direction and is reported as such. |
| neither arm ever reaches `E_das(32) * 1.10` | The threshold is unreachable, meaning DAS at 32 is itself not a meaningful reference. The run is reported as uninformative and the metric is re-examined before anything is claimed. |
| H2 fails for an arm | That arm's numbers are withdrawn. An arm that cannot beat an untrained map is not measuring anything. |

No outcome here is a reason not to report. A null on H1 is a result about
nonlinear alignment maps and is publishable as one.

## Deliberately not varied

The heads, the target variable, the metric, the coefficients, the training
configuration, the filter, and the train/test split. Changing any of them after
seeing results would make the comparison unregistered.

The sites are MIB's selection, not ours. They are held fixed across arms so that
the only difference between `das` and `lcp_vae_interchange` is the map.

## Known limitation, recorded now

This design cannot measure vacuity — whether an expressive map imposes the
abstraction rather than locating it. That requires a criterion defined on a
network that cannot do the task, and MIB's squared-error metric has no chance
level. Vacuity is a separate experiment and no claim about it is registered here.

## Script

`experiments/mib_ioi_heads.py`, already written. A small-`size` smoke run on one
k precedes the sweep; it validates plumbing and its numbers are not results.
