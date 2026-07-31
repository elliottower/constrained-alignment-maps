# Pre-registration: reconstruction as the admissibility criterion for alignment maps

**Status: DESIGN FROZEN, SCRIPTS NOT YET WRITTEN, NOTHING RUN.**

This document freezes the design. It is **not binding** until every script named
below exists and its SHA-256 is filled in. Empty SHA fields mean the
corresponding experiment is unregistered and its results are exploratory.

Revision 2, after external review. Changes from revision 1: the absolute 0.30
vacuity threshold is replaced by per-task floors plus the relative criterion it
was taken from; the exclusion rule is closed against post-hoc reclassification;
fit counts raised; loss recovered added as the admissibility metric of record;
one primary hypothesis per experiment.

Revision 3, after second external review. The vacuity definition was a
conjunction where De Morgan requires a disjunction, which would have certified as
safe a map scoring up to 0.45 on a random network for indirect object
identification; floors are re-estimated with intervals, since three of them are
point estimates of exactly zero against which a 0.05 margin is undetectable;
Experiment 1's primary is now admissibility rather than supervision, so that
every primary bears on the criterion; H0.3's evaluation protocol is stated, since
it is compared against a hard-example sweep while H0.1 uses matched benchmark
settings; the λ grid is widened from five points to nine, because a rank
correlation over five points cannot support H2.2; and Experiment 0 now fits both
parametrisations in one run rather than replacing one with the other.

Revision 4, after the canonical implementation was built and tested locally.
Adds Experiment 0.5. Two findings from `experiments/test_das.py` forced it: the
parametrisation gap is a convergence artifact that closes by 1000 steps
(+0.055 at 250, −0.003 at 1000), and the paper trains DAS at 300 steps while
giving NL-DAS 5000 in the same table. Every DAS number and every vacuity claim
is confounded with optimiser budget until Experiment 0.5 completes, so it joins
the gate.

Revision 5, after reading MIB's source rather than inferring from behaviour.
**Revision 4's diagnosis was wrong, and so was revision 3's.** DAS here was
trained at 1e-3 against MIB's 1e-2, and both the parametrisation gap and the
step-budget concern were symptoms of that single error: under MIB's
configuration the parametrisation gap falls from +0.099 to +0.0002 and DAS
reaches 0.97 within fifty steps. Experiments 0 and 0.5 are merged into one gate
whose content is "run DAS as MIB runs it, using MIB's code". The standard arm now
imports `CausalAbstraction.neural.featurizers.SubspaceFeaturizer` from the
vendored checkout instead of reimplementing it, and
`experiments/test_das_matches_mib.py` verifies the equivalence by class identity,
by numerical agreement, and by reading MIB's `DEFAULT_CONFIG` at test time. The
manuscript's delta-PCA initialisation advantage becomes a hypothesis under test
(H0.4) rather than a background fact, because it too disappears at the corrected
learning rate.

The lesson recorded for later revisions: revisions 3 and 4 each diagnosed a
symptom confidently and designed an experiment around it, and reading the
reference implementation would have pre-empted both.

## The claim these experiments serve

Alignment maps for causal abstraction have been evaluated on one axis,
expressivity, which produces an apparent dilemma: linear maps are weak, and
unconstrained nonlinear maps are vacuous (Sutter et al., 2025). We claim the
relevant property is a second, independent one:

> An alignment map is admissible only if it reconstructs. A map that does not
> reconstruct hands the downstream layers a vector the model never produces, so
> the intervention measures the decoder rather than the model.

Under this account, Distributed Alignment Search (DAS) is safe because its
rotation is orthogonal and therefore reconstructs exactly, and linearity is
incidental to that safety. The four corners of (admissible x expressive) are
then the object of study, and the admissible-and-expressive corner is unoccupied.

The claim is a **necessary condition, not a sufficient one**. Reconstruction
failure is what makes vacuity possible. It does not guarantee it.

**The single primary endpoint for the paper is H2.1.** If it fails, the criterion
is withdrawn and no other result rescues it.

## Vacuity threshold, fixed here for every experiment below

Revision 1 used a flat bound of 0.30. That is wrong on this paper's own
evidence: the random-subspace floor ranges from 0.000 to 0.482 across these
tasks (main_v12, Table `tab:controls_main`), whose caption states that "a single
numeric bound across tasks would be uninformative". A flat 0.30 is below the
floor for subject--verb agreement (0.482) and greater-than (0.450), so a
correct method would be recorded as failing on two of six tasks.

The source rule in `PREREGISTRATION_RANDOM_NETWORK.md:54` is a conjunction of
two **safety** conditions: a map passes when IIA ≤ 0.30 **and** it is at least
0.50 below the unconstrained nonlinear arm. Vacuity is the negation of that
conjunction, which by De Morgan is a **disjunction**. Revision 2 wrote the
negation as a conjunction, which certified as safe any map that cleared its floor
while remaining far below the unconstrained arm — on indirect object
identification, floor 0.000 and unconstrained arm 0.989, a map scoring 0.45 on a
network with no computation would have passed.

A map is **vacuous on a given task and arm** when **either** of the following
holds:

1. interchange accuracy exceeds the upper bound of that task's random-subspace
   floor interval by more than 0.05, **or**
2. interchange accuracy is less than 0.50 below the unconstrained nonlinear arm
   on the same task and arm.

**Floors are recomputed with intervals in Experiment 0**, before any other
experiment launches. Three tasks currently measure a floor of exactly 0.000
(indirect object identification, hypernymy, addition), which is a point estimate
with no sampling spread behind it, so a 0.05 excess over it is not a detectable
quantity. Experiment 0 therefore re-estimates every floor over five fits and
reports mean with a 95% interval, and clause 1 above is evaluated against the
interval's **upper bound**, not the point estimate.

Current point values, superseded by Experiment 0 and listed for reference only:
indirect object identification 0.000, hypernymy 0.000, addition 0.000, capitals
0.105, gender bias 0.274, greater-than 0.450, subject--verb agreement 0.482.

## Admissibility metrics, fixed here for every experiment below

| id | metric | definition | role |
|---|---|---|---|
| A1 | round-trip relative error | `‖g(f(h)) − h‖² / ‖h‖²`, no intervention | reported, comparability with existing tables |
| A2 | no-op downstream divergence | KL between model logits on `h` and on `g(f(h))`, no intervention | reported |
| A3 | off-manifold distance of the intervened activation | Mahalanobis distance of `h'` to the activation distribution at the hook site | reported, required for additive arms |
| **A4** | **loss recovered** | **fraction of cross-entropy restored when `h` is replaced by `g(f(h))`, `(CE_ablate − CE_map) / (CE_ablate − CE_clean)`** | **admissibility metric of record** |

**A4 is the cut.** A map is **admissible** when loss recovered ≥ 0.95. Revision 1
set the cut on A2 at 0.05 nats while describing it as the sparse-autoencoder
literature's metric; those are different quantities, and the nat value had no
provenance. Loss recovered is the quantity that literature reports, so the cut is
placed there. The 0.95 value is a convention we fix in advance, not one inherited
from a citation, and is stated as such in the paper.

**A3 estimator, fixed in advance.** Sample covariance in 768 dimensions is
ill-conditioned at these sample sizes. A3 uses Ledoit-Wolf shrinkage covariance
estimated on held-out activations at the hook site, after projection onto the top
256 principal components of the activation distribution. Distances are
standardised against the same distance computed for genuine held-out activations,
so A3 is reported in standard deviations of the natural distribution.

**Why A3 is required.** The end-to-end arm intervenes additively,
`h' = h_b + dec(z_swap) − dec(z_orig)` (main_v12:830), which cancels
reconstruction error to first order. A1, A2 and A4 are therefore uninformative
for that arm: a decoder with large reconstruction error can still produce a
small, well-behaved delta, and can equally produce an arbitrary one. A3 measures
the object that actually reaches the downstream layers.

## Experiment 0 — DAS as MIB runs it (hard gate)

**SHA-256, frozen. Steps A-E are all wired; this experiment is REGISTERED.**

```
833578239789bbac2b4cf421e164a6eeb342584e2485491ef5bd3a1f54d5ac5b  experiments/das.py
b08543f8fcc783b3d588fc42223f62bb5e4613e07c06f8480b4513bbd24e561e  experiments/test_das.py
4aa43338b9ec39c29f10e15f7c7084cfc3fe4a137be2c3cd77cf93cf806bce6d  experiments/test_das_matches_mib.py
e0a72337b80dafbe5f97f1ee81c3c8374e1ac7b9c84738c33c1806407c1ac008  experiments/k1_vae_vs_das.py
8f0f17801bb0f8267fdebc1ffc11250bc329152e0a3bab6f8ec258df1a760dd1  experiments/random_network_control.py
```

Two launch-blocking defects were fixed before the first run, both in
`k1_vae_vs_das.py`:

- `output_dir` was applied verbatim to every task in a multi-task run, so six
  tasks wrote to one file and overwrote each other; omitting it instead
  overwrote the published results in place. It is now a prefix.
- `run_task`'s Modal timeout was 6 hours against a 5-fit budget, and the volume
  is committed once at the end of a task, so a timeout would have discarded that
  task entirely. Raised to 24 hours.

Results are written to
`/results/grassmannian_atlas/mib_reference_das_corrected/<task>/results.json`,
leaving the published `k1_pi_ablations/` tree untouched for comparison.

`random_network_control.py` now also carries the **end-to-end arm** (Experiment 3
below), ported from `k1_vae_vs_das.py::train_pi_sae_e2e` with this file's key
names. Experiments 0 and 3 therefore run in one job, which is what the primary
endpoint requires: the manuscript's headline numbers come from the end-to-end
arm, and until now no script in the repository contained both `e2e` and
`random_init`.

Two earlier freezes in this file were voided before any run, both from reading
MIB too shallowly:

1. The first encoded the library default (3 epochs, lr 1e-2) as MIB's settings.
   Every baseline overrides it; indirect object identification trains 2 epochs at
   **lr 1.0**.
2. The second assumed batch size 32 for indirect object identification. It is
   read from `get_model_config` in `baselines/ioi_baselines/ioi_utils.py` and is
   model dependent: **1024 for GPT-2**, 256 for Qwen, Llama and Gemma.

MIB's actual recipe for indirect object identification on GPT-2 is therefore
batch 1024, learning rate 1.0, two epochs — about **four optimiser steps** on a
two-thousand-pair split. Large batch, large rate, very few steps. This repo ran
300 steps at learning rate 1e-3 with batch 16.

The 400-pair training cap is consequently lifted for the reference arm: at batch
1024 it would have yielded two optimiser steps. The historical arm keeps the cap
so its published numbers stay reproducible, and both training-set sizes are
recorded per run (`das_reference_n_train`, `das_local_n_train`).

No value in `das.MIB_TASK_CONFIG` is assumed; `MIB_BATCH_SIZE_IS_ASSUMED` is
empty, and `test_das_matches_mib.py` asserts it stays empty.

Experiments 1-4 remain unregistered.

**Known deviations from MIB, recorded before the run.**

1. **Training-set cap.** MIB trains on the full counterfactual split; this
   project caps DAS training at 400 pairs (`train_pairs[:400]`). The cap is
   retained so the reference arm and the retained old-configuration arm see
   identical data, making their difference attributable to hyperparameters
   alone. Since MIB's budget is measured in epochs, the cap shortens it: two
   epochs at batch 32 over 400 pairs is 26 steps on indirect object
   identification. Whether to lift the cap is recorded here rather than decided
   silently mid-run.

2. **Batch size on indirect object identification is assumed, not read.** MIB's
   IOI baseline takes `batch_size` from `setup_pipeline`, which is model
   dependent and not a literal in their source. We use the library default of
   32 and flag it (`batch_size_assumed`), rather than presenting it as a value
   taken from their code.

3. **Five of six language tasks have no MIB baseline**, as above. They run the
   library default with `is_mib_baseline=False` recorded in results.

4. **Grokking tasks are passed their operation name**, so they fall back to the
   library default and are flagged. MIB's arithmetic baseline covers two-digit
   addition in a language model, which is a different task from modular
   arithmetic in a toy transformer; borrowing its configuration would claim an
   endorsement MIB has not given.

**No other experiment launches until this completes.** Every experiment below
reports DAS numbers and per-task floors, and both change under the corrected
configuration.

Revisions 1-4 split this into "port the parametrisation" (Experiment 0) and
"fix the step budget" (Experiment 0.5). Both diagnoses were wrong. They are
merged here.

### Why

DAS in this project was trained at **one tenth MIB's learning rate**, and the
symptoms were misread twice.

MIB's configuration, read from
`reference/MIB/.../CausalAbstraction/experiments/config.py` and
`pyvene_core.py::_train_intervention`:

Shared across every MIB task: `AdamW(lr, weight_decay=0)`, a constant schedule,
and task cross-entropy alone as the loss. Everything else is **per task**, and
each baseline overrides the library default, so `DEFAULT_CONFIG` is not what MIB
runs anywhere:

| source | epochs | init_lr | n_features |
|---|---|---|---|
| `baselines/ioi_baselines/ioi_baselines.py` | 2 | **1.0** | 32 |
| `baselines/arithmetic_baselines.py` | 1 | inherited 1e-2 | 16 |
| `baselines/ravel_baselines.py` | 1 | inherited | — |
| `baselines/ARC_baselines.py` | 2 | inherited | — |
| `baselines/simple_MCQA_baselines.py` | 8 | inherited | — |
| `CausalAbstraction/experiments/config.py` (default only) | 3 | 1e-2 | 32 |

Against this repo's previous settings — Adam, lr 1e-3, batch 16, 300 fixed steps
— the learning rate on indirect object identification was off by **a factor of
one thousand**. `init_lr = 1.0` appears exactly once in the benchmark, in the IOI
baseline, and is not a transcription error on our side: AdamW's per-step update
is bounded near `lr`, and the orthogonal parametrisation re-projects onto the
Stiefel manifold every step, so a large rate under a two-epoch budget is a
deliberate fast-convergence choice.

**MIB has no baseline for five of our six language tasks.** Only indirect object
identification is configured by the benchmark. Subject--verb agreement,
greater-than, gender bias, hypernymy and capitals fall back to the library
default, and `das.mib_config` records `is_mib_baseline=False` for them, which is
written into every result. The paper may therefore say it used MIB's
configuration **for indirect object identification**, and must not say it for the
other five.

Under MIB's configuration the two problems earlier revisions were built around
both vanish, measured on the planted-direction toy in `test_das.py`:

| | lr=1e-3, batch 16 | MIB: lr=1e-2, batch 32 |
|---|---|---|
| parametrisation gap at 250 steps | +0.0990 | **+0.0002** |
| standard arm at 250 steps | 0.886 | **0.9878** |
| alignment at 50 steps | 0.403 | **0.9698** |

So the parametrisation gap was a low-learning-rate symptom, and the step-budget
concern — including the 300-against-5000 asymmetry treated in revision 4 as the
most damaging finding in the project — was largely the same symptom. DAS reaches
0.97 within fifty steps when trained correctly. The correction is MIB's recipe,
not more compute.

**A published claim is now in doubt.** Under MIB's configuration the six
initialisation/parametrisation arms all land at 0.984-0.990, and delta-PCA
(0.9837) no longer beats standard random-orthogonal (0.9878). The manuscript's
k-sweep reports delta-PCA initialisation beating random initialisation at every
k (0.806 against 0.722 at k=8; 1.000 against 0.919 at k=16). That advantage is a
candidate low-learning-rate artifact and is tested here.

### Design

**Step A — run MIB's code, not a reimplementation.** The standard arm uses
`CausalAbstraction.neural.featurizers.SubspaceFeaturizer`, which wraps pyvene's
`LowRankRotateLayer` under `torch.nn.utils.parametrizations.orthogonal`, with
MIB's hyperparameters above. `experiments/test_das_matches_mib.py` verifies this
by class identity, by numerical agreement with MIB's featurize-swap-inverse
round trip (25 random cases to 1e-5, with a negative control confirming a wrong
projector fails), and by reading MIB's `DEFAULT_CONFIG` at test time so their
checkout drifting breaks the test rather than passing silently.

**Step B — re-run every DAS number** under that configuration, on the six GPT-2
tasks and the grokking operations, five fits per cell, with per-task
random-subspace floors re-estimated over the same five fits and reported as mean
with a 95% interval.

**Step C — convergence verification on real tasks.** The toy converges by fifty
steps; GPT-2 tasks are not assumed to. Using `train_das_snapshots`, one
trajectory per fit is evaluated at {50, 100, 189, 400, 800}, where 189 is MIB's
epoch budget on IOI. The convergence rule is unchanged: converged at step s when
accuracy at s is within 0.02 of accuracy at 2s, for two consecutive doublings.
A task not converged at MIB's budget is reported as such rather than being
quietly given more steps, since MIB's budget is what the benchmark uses.

**Step D — calibration against the benchmark harness.** MIB's implementation run
through MIB's own harness on indirect object identification, compared against our
invocation of it at matched k, layer and evaluation pairs.

**Step E — the old configuration, retained as an ablation.** The previous
settings (lr 1e-3, batch 16, 300 steps) and the initialisation variants
(delta-PCA, Gaussian) are kept as labelled ablation arms, run in the same job on
the same data and seeds. They are the paper's initialisation-and-budget
sensitivity result, and they are what published numbers are traceable to.

### Predictions

**H0.1 (primary).** Under MIB's configuration, DAS accuracy is at least 0.05
higher than under the old configuration at k=1 on at least four of the six GPT-2
tasks. The paper's DAS numbers were depressed by the learning rate.

**H0.2 (secondary).** Our invocation matches the MIB harness within 0.05 on
indirect object identification at matched settings.

**H0.3 (secondary).** At each task's converged step count, orthogonal and
per-step-QR parametrisations differ by less than 0.02, as on the toy.

**H0.4 (secondary).** The delta-PCA initialisation advantage does not survive:
under MIB's configuration, delta-PCA and random-orthogonal differ by less than
0.05 at every k on grokked modular addition.

**H0.5 (secondary).** DAS is converged at MIB's epoch budget on at least four of
six GPT-2 tasks, by the rule in Step C.

### Decision rule

| outcome | consequence |
|---|---|
| H0.1 holds | Every DAS number in the manuscript is replaced. Any phrasing implying DAS fails is removed, and the efficiency claim is re-derived from the corrected numbers rather than restated. |
| H0.1 fails | The learning rate was not what depressed DAS, and the old numbers stand on that point. Report the null and keep MIB's configuration anyway, since it is the reference. |
| H0.2 fails | Our invocation is not MIB's DAS despite the equivalence tests. Every DAS number comes from the MIB harness directly before any claim is made. |
| H0.3 fails | Parametrisation is a confound at the budgets used. Report both at every step count in an appendix rather than choosing one. |
| H0.4 holds | The manuscript's delta-PCA initialisation result is withdrawn as a low-learning-rate artifact, and the withdrawal is stated in the paper rather than the row being deleted. |
| H0.5 fails on three or more tasks | MIB's epoch budget does not converge on those tasks. Report DAS at both MIB's budget and the converged budget, and state that the benchmark's default under-trains them. |

**Reported regardless of outcome:** the full old-against-new comparison for every
DAS cell, with sign and magnitude. "We re-ran under the reference implementation
and it moved by X" is a credibility asset whether X is large or small, and it is
the honest record of a correction the project made itself.

### What would invalidate the run

A fit that fails to train (accuracy at its task's floor) is excluded and
reported, not replaced. Exclusion is decided on the pretrained arm alone, before
that fit's random-network number is computed, under the rule in Experiment 2.

## Experiment 1 — the sparse-autoencoder corner

**Script:** `experiments/sae_alignment_baseline.py` (does not exist)
**SHA-256:** _unfilled_

### Why

The 2x2 organising the paper asserts that a standard sparse autoencoder is
admissible but unsupervised. No unsupervised sparse-autoencoder baseline exists
in this repository; every `SAE` in `experiments/` is `PiSAE`, our own
architecture. The cell is asserted rather than measured, and it is the first cell
a reviewer will question.

### Design

A standard sparse autoencoder trained on the same layer-8 GPT-2 activations,
unsupervised, with no access to task labels during training. Two variants, since
the field has no single default: TopK and JumpReLU. Width and sparsity follow
SAELens defaults for GPT-2 small rather than being tuned here.

Interchange requires choosing which features to swap, and an unsupervised basis
provides no such choice. That gap is the point of the cell, so the selection rule
is fixed in advance and is deliberately generous: select the `m` features with
highest mutual information with the task label on the training split, `m` matched
to the causal-dimension budget of the structured model it is compared against.
Selection uses labels; training does not.

Tasks: the six GPT-2 tasks. Fixed model, **five fits** per variant per task.

### Predictions

Every experiment's primary hypothesis bears on the criterion, which is the
paper's spine. For this experiment that is admissibility. Whether supervision
separates the corners is an expressivity question and is secondary. Revision 2
had these the other way round.

**H1.1 (primary).** The sparse autoencoder is admissible: loss recovered ≥ 0.95
on all six tasks.

**H1.2 (secondary).** The sparse autoencoder underperforms the structured model
on strict interchange accuracy on at least four of six tasks, with
non-overlapping 95% intervals.

**H1.3 (secondary).** The sparse autoencoder exceeds linear DAS at `k = 1` on at
least three of six tasks.

H1.1 is expected to hold, since sparse autoencoders are trained on
reconstruction. A near-certain primary is the intended choice here: the cell is
asserted in the paper's central figure and must be measured, and a near-certain
prediction that fails is the most informative outcome available.

### Decision rule

| outcome | consequence |
|---|---|
| H1.1 holds | The admissible-but-unsupervised cell is measured rather than asserted. The 2x2 stands. |
| H1.1 fails | A standard sparse autoencoder is not admissible under our own metric of record. The 2x2 is wrong as drawn and is redrawn before submission. This is a substantive finding about sparse autoencoders and is reported as one, not as an inconvenience. |
| H1.2 fails | Supervision is not what distinguishes the corners. The contribution narrows to the admissibility criterion alone and the fourth-corner framing is dropped. |
| H1.3 fails | Reported without reinterpretation. It bears on expressivity, not on the criterion. |

## Experiment 2 — the reconstruction-vacuity spectrum

**Script:** `experiments/recon_vacuity_spectrum.py` (does not exist)
**SHA-256:** _unfilled_

### Why

The criterion is currently supported by two points plus a penalty sweep. Two
points do not establish a criterion. A necessary condition predicts an **empty
region**, which is stronger and more falsifiable than a correlation.

### Design

Maps spanning a constraint gradient, evaluated on both pretrained GPT-2 and a
randomly initialised network of the same architecture:

- linear DAS (exact reconstruction by construction)
- sparse autoencoder, both variants from Experiment 1
- structured model, with and without the label-conditional prior
- unconstrained nonlinear DAS at reconstruction penalties
  λ ∈ {0, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10}. Revision 2 used five values;
  a rank correlation over five points has a sampling distribution too wide to
  support H2.2, and the λ arm is the causal evidence for the criterion, so the
  grid is widened rather than the hypothesis weakened
- a deliberately inadmissible control: a nonlinear map with a `k`-dimensional
  bottleneck and no reconstruction term

**Two tasks, not one:** indirect object identification and subject--verb
agreement. Revision 1 used indirect object identification alone, which cannot
support a general claim about alignment maps. Subject--verb agreement is the
adversarial choice, since linear DAS scores 0.00 strict there and its floor
(0.482) is the highest of any task.

`k = 1`. **Ten fits per map**, giving the coverage an emptiness claim requires:
emptiness under sparse sampling is indistinguishable from not having looked.

Each fit contributes (A1, A2, A3, A4, interchange accuracy on both arms).

### Predictions

**H2.1 (primary, and the paper's single primary endpoint).** The upper-left
quadrant is empty: no map with loss recovered ≥ 0.95 is vacuous under the
two-part threshold above, on either task.

**H2.2 (secondary).** Across the nine-point λ arm, loss recovered and
random-network interchange accuracy are negatively rank-correlated, Spearman
ρ below −0.7, computed on per-λ means over ten fits.

**H2.3 (secondary).** The bottleneck control is inadmissible (loss recovered
< 0.95) and its random-network accuracy is **not** predicted to be high. This
tests the one-sidedness directly: inadmissible maps are permitted to be vacuous,
not required to be.

### Decision rule

| outcome | consequence |
|---|---|
| H2.1 holds | The criterion is reported as a necessary condition, with the empty-quadrant figure as the paper's primary evidence. |
| H2.1 fails: any admissible map is vacuous | The criterion is false as stated and is withdrawn. No reinterpretation is permitted. The paper reverts to reporting the reconstruction trade as an observation about two methods. |
| H2.2 fails while H2.1 holds | Report the empty quadrant without the correlation. A necessary condition does not require monotonicity and claiming it would overstate. |
| H2.3 shows the bottleneck control also vacuous | Report. The claim was always one-sided; this neither strengthens nor weakens it. |

### Exclusion rule, closed against post-hoc reclassification

A map that reconstructs well and scores high on the random network **is the
falsifier**. Revision 1's invalidation rule permitted excluding maps that "failed
to train", which would allow the falsifier to be reclassified as a training
failure after the fact.

Therefore: exclusion is decided **on the pretrained arm alone**, using a
threshold fixed now — a map is excluded when its pretrained-arm interchange
accuracy fails to exceed that task's random-subspace floor by more than 0.05. The
exclusion decision for every map is **recorded to disk before that map's
random-network accuracy is computed**, and the script enforces this ordering. No
map may be excluded after its random-network number exists.

## Experiment 3 — end-to-end training under the random-network control

**Script:** `experiments/random_network_control.py` (exists; requires an
end-to-end arm and the Experiment 0 rotation swap)
**SHA-256:** _unfilled, script must be modified first_

### Why

Every headline number in the six-task table is the end-to-end arm
(main_v12:846). The random-network control tests the non-end-to-end arm:
`random_network_control.py` contains no occurrence of `e2e`, `interchange`, or
`intervention_loss`, and no script in the repository contains both `e2e` and
`random_init`. The control does not cover the method producing the paper's
headline results.

This matters most where it is load-bearing: the end-to-end objective is the same
one unconstrained nonlinear DAS optimises (main_v12:826), and the paper's
argument is that constraints rather than objectives separate the methods.

### Design

Add the end-to-end arm to the existing control, unchanged in every other respect:
same tasks, same optimisation budgets, same evaluation. **Five fits.** Report A1
through A4 alongside interchange accuracy for every arm.

### Predictions

**H3.1 (primary).** The end-to-end arm is non-vacuous under the two-part
threshold, on every task.

**H3.2 (secondary).** A3 for the end-to-end arm is within two standard deviations
of the distance measured for genuine activations, confirming the additive
intervention keeps the intervened activation on the manifold.

### Decision rule

| outcome | consequence |
|---|---|
| H3.1 holds | The control covers the headline method. The claim strengthens: same objective as unconstrained nonlinear DAS, constraints alone remove the vacuity. |
| H3.1 fails | The headline method is vacuous under the test the paper uses to condemn the baseline. The six-task table cannot be reported as evidence for the method, and the paper is restructured around the non-end-to-end arm or withdrawn. There is no partial-credit reading. |
| H3.2 fails while H3.1 holds | The additive intervention produces off-manifold activations without producing vacuity. Report as an open question; do not claim the additive construction establishes admissibility. |

## Experiment 4 — dimension sweep on all six language tasks

**Script:** `experiments/k1_vae_vs_das.py` (exists; requires a `k` sweep)
**SHA-256:** _unfilled_

### Why

The paper's defensible claim is efficiency: a constrained map reaches the causal
variable in fewer dimensions than a linear one. That claim is measured on one
task. The other five report `k = 1` only, where a low DAS number cannot separate
"needs more dimensions" from "cannot represent this variable" — the difference
between the honest claim and an overclaim that DAS fails.

### Design

`k ∈ {1, 2, 4, 8, 16, 32}` on all six tasks, both scoring rules, **five fits**
per cell, hard-example filtering as already defined in Methods.

### Predictions

**H4.1 (primary).** On at least four of six tasks, DAS reaches within 0.05 of the
structured model's `k = 1` accuracy at some `k ≤ 32`. Linear alignment recovers
the variable given enough dimensions.

**H4.2 (secondary).** The dimension at which DAS matches the structured model's
`k = 1` accuracy is at least 4 on at least four of six tasks.

**H4.3 (secondary).** The strict-versus-standard gap for DAS narrows as `k`
increases on all six tasks.

### Decision rule

| outcome | consequence |
|---|---|
| H4.1 holds | The efficiency claim is supported across tasks; every "DAS fails" phrasing is replaced. |
| H4.1 fails on three or more tasks | DAS does not recover the variable at any tested dimension on those tasks. This is stronger than efficiency and must be verified against Experiment 0 before it is claimed. |
| H4.2 fails | The efficiency advantage is under one dimension-doubling and does not warrant the paper's emphasis. Report the comparison without the efficiency framing. |

## Multiplicity

Five experiments (0, 1, 2, 3, 4), one primary hypothesis each, one of which
(H2.1) is the paper's primary endpoint. Revision 4's two-primary exception for
Experiment 0.5 is gone with that experiment; Experiment 0 has a single primary,
H0.1, and four secondaries.

All hypotheses labelled secondary are exploratory: they
are reported with intervals, they inform discussion, and **no secondary result is
presented as confirmatory evidence for the criterion**. Where a primary
hypothesis is evaluated across six tasks, the per-task results are reported in
full and the hypothesis is judged on the stated count, fixed above, rather than
on whichever subset is favourable.

## Not registered here

**Identifiability.** Whether the label-conditional prior raises cross-fit
consistency is a separate claim, secondary to the criterion, and belongs in its
own pre-registration. The accuracy ablation already in hand (0.419 without the
prior against 0.429 with it, three seeds, gender bias) bears on accuracy and not
on identifiability, and must not be cited either way on that question.

**Subspace degeneracy.** `PREREGISTRATION_DEGENERACY.md` is retired. The claim is
published as Méloux et al., ICLR 2025, whose fourth failure mode is that one
algorithm can align with different subspaces, and
`experiments/subspace_overlap_null.py` showed the sub-chance and narrow-band
versions consistent with random subspaces. The result is cited, not reproduced.

## Order of execution

Experiment 0 is a hard gate; nothing else launches until the reference
parametrisation is in both scripts and the per-task floors are recomputed under
it, which requires wiring both Modal scripts to `experiments/das.py`. Experiment
3 gates the paper's headline table. Experiments 1 and 2 are the new structural
contributions and depend only on the Experiment 0 gate.

## Reporting protocol, fixed for all experiments above

Mean with a 95% confidence interval from Student's t as the primary figure, with
the maximum over fits reported alongside for every method. Nulls are empirical,
generated by the same code path as the statistic they calibrate, never analytic.
Any cell whose ordering does not survive is reported with overlapping intervals
shown, rather than dropped. Every count of excluded fits is reported.
