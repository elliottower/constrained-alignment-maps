# Method registry

The single source of truth for what each method **is**, what it is **called in
code**, and what it is **called in the paper**. Built by reading the code, not
from memory.

Written because the code and the manuscript drifted apart. The manuscript renamed
the method; the code did not. That drift is not cosmetic — it is what let the
end-to-end versus non-end-to-end mismatch survive: the six-task table reports one
arm and the random-network control tested a different one, and nothing in the
naming made that visible.

**Rule going forward:** the paper never uses the string `pi_sae` or `pi-SAE`. The
code keeps its identifiers so existing result files stay readable. This table is
the mapping.

---

## Alignment maps

| code key | builder | trainer | label-conditional prior | causal/nuisance partition | 8x expansion | interchange term in loss | paper name |
|---|---|---|---|---|---|---|---|
| `random` | — | — | — | — | — | — | random subspace (noise floor) |
| `delta_pca` | — | `run_delta_pca` | — | — | — | no (training-free SVD) | delta-PCA |
| `das` | — | `train_das` | — | — | — | yes | **DAS (delta-PCA warm start)** — NOT standard DAS |
| `das_reference` | — | `das.train_das_mib` | — | — | — | yes | DAS (reference implementation, MIB) |
| `nldas` | — | `train_nldas` | — | — | — | yes | unconstrained nonlinear DAS |
| `nldas_recon` | — | `train_nldas(recon_weight=1.0)` | — | — | — | yes + reconstruction penalty | nonlinear DAS + reconstruction |
| `structured_vae` | `_build_structured_vae` | `train_vae_family` | **no** | yes | no | no | structured VAE (no label conditioning) |
| `pi_vae` | `_build_pi_vae` | `train_vae_family(use_pi_prior=True)` | yes | yes | no | no | label-conditional structured VAE, partitioned |
| `pi_sae` | `_build_pi_sae` | `train_vae_family(use_pi_prior=True, l1_coeff=1e-3)` | yes | yes | **yes** | **no** | **label-conditional structured VAE** — the method |
| `pi_sae_e2e` | `_build_pi_sae` | `train_pi_sae_e2e` | yes | yes | yes | **yes** | label-conditional structured VAE **with end-to-end interchange training** |
| `pi_plain_vae` | `build_pi_plain_vae` | — | yes | **no** | no | no | 2x2 cell: flat |
| `pi_plain_sae` | `build_pi_plain_sae` | — | yes | **no** | yes | no | 2x2 cell: flat + expansion |

**`pi_sae` and `pi_sae_e2e` share a builder.** They differ only in the trainer,
and therefore only in whether the interchange cross-entropy is in the loss. That
single difference is what the random-network control turns on.

### The 2x2 ablation in the manuscript

`tab:ablation` (main_v12:784-800) states that all four cells use the
label-conditional prior and vary partition and expansion. The cells are:

| | no partition | causal/nuisance partition |
|---|---|---|
| **no expansion** | `pi_plain_vae` | `pi_vae` |
| **8x expansion** | `pi_plain_sae` | `pi_sae` |

`structured_vae` is **not** a cell of this table — it has no label-conditional
prior and is a separate control.

---

## Controls (in `k1_vae_vs_das.py`)

| code key | what it removes |
|---|---|
| `c1_random_labels` | correct supervision |
| `c2_recon_only` | classification and interchange signal |
| `c3_untrained` | training |
| `c7_unconstrained` | causal/nuisance partition and supervision |
| `random_das` | any fitting (random subspace floor) |

---

## Which arms each script runs

| script | arms |
|---|---|
| `random_network_control.py` | random, delta_pca, das, das_reference, nldas, nldas_recon, structured_vae, pi_vae, pi_sae, pi_sae_e2e |
| `k1_vae_vs_das.py` | das, das_reference, nldas, vae, pi_vae, pi_sae, pi_sae_e2e, pi_plain_vae, pi_plain_sae, plus c1/c2/c3/c7 and random_das |

---

## Measured results, IOI at k=1 (2026-07-30, first corrected run)

From `experiments/results/random_network_control_e2e/`, 180 held-out pairs,
random-subspace floor 0.000 over five draws with zero spread.

| arm | pretrained | randomly initialised |
|---|---|---|
| `delta_pca` | 0.039 | 0.000 |
| `das` | 0.200 | 0.000 |
| `das_reference` | 0.000 (**degenerate, see below**) | 0.000 |
| `nldas` | 1.000 | 0.433 |
| `nldas_recon` | 0.917 | 0.650 |
| `structured_vae` | 0.294 | 0.000 |
| `pi_vae` | 0.356 | 0.000 |
| **`pi_sae`** | **1.000** | **0.000** |
| **`pi_sae_e2e`** | **1.000** | **0.961** |

Two things follow directly.

**The end-to-end arm is vacuous.** 0.961 on a network with no computation, on
held-out pairs, with an honest evaluation (argmax over the model's own logits).
It is not a code defect: the split is disjoint, and the same architecture without
the interchange term scores 0.000.

**The non-end-to-end arm passes and loses nothing on this task.** `pi_sae`
reaches 1.000 pretrained and 0.000 random. On indirect object identification the
end-to-end training was unnecessary and is exactly what broke the control.

**`das_reference` at 0.000 pretrained is a defect, not a result.** Its recorded
configuration is MIB's real IOI recipe (`lr=1.0, n_epochs=2, batch_size=1024`),
but this script's pair set is far smaller than 1024, so it trains for two
optimiser steps. Needs a batch size capped at dataset size before it means
anything.

---

## Naming drift to fix in the manuscript

The paper's method is `pi_sae`. Everywhere the manuscript says "label-conditional
structured VAE" without qualification, check whether the number came from
`pi_sae` or `pi_sae_e2e`. The six-task table (main_v12:846) is honestly labelled
`E2E` in its column headers; `paperA_constrained_alignment_v1.tex` is **not** —
it reports the same numbers as "structured VAE". That must be corrected before
either draft goes anywhere.
