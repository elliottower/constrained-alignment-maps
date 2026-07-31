# Lab notebook

Consolidated record of decisions, killed claims, surviving claims, and open
contradictions. Written 2026-07-30 because the state of the project had become
impossible to hold in one head, and the same questions were being re-litigated.

Sources: the manuscript drafts, `PAPER_SPLIT_PLAN.md`, the pre-registrations, the
measured result files, and the working session of 2026-07-30. Entries marked
**[unverified]** come from memory of earlier sessions and have not been traced to
a file.

Companion documents:
- `METHOD_REGISTRY.md` — what each method is and what it is called
- `PAPER_REVISION_SPEC_V13.md` — what to change in the manuscript
- `PREREGISTRATION_RECONSTRUCTION_CRITERION.md` — the live experiment plan

---

## 1. Where the project stands in one paragraph

The thesis is that **constraint, not linearity, is what makes an alignment map
trustworthy**. Linear alignment is expensive but safe; unconstrained nonlinear
alignment is vacuous; a constrained generative map was supposed to thread the
needle. As of 2026-07-30 the first corrected run showed that the *end-to-end*
variant of that constrained map is itself vacuous (0.961 on a randomly
initialised network), while the non-end-to-end variant passes (0.000) and loses
nothing on indirect object identification (1.000 pretrained). The paper is
therefore alive but must be rebuilt on the non-end-to-end arm, and the reason for
vacuity has been relocated from *architecture* to *whether the interchange
objective is in the loss*.

---

## 2. Claims killed, and what killed them

| claim | killed by |
|---|---|
| k=8 Fourier-span story for DAS | seeding: 0.806 +/- 0.570 across seeds |
| nonlinear DAS "regression" at k=32 | it is a plateau: 0.939 -> 0.961 -> 0.956 |
| two-sided diversity-ratio account | rho_within = 1.068 on a random network at IIA 0.989 |
| capability gap on indirect object identification | DAS reaches 0.994 at k=16 under hard IIA |
| DAS subspaces "less aligned than chance" | empirical null: mean 0.01562, sd 0.01091, central 95% [0.00193, 0.04307]; observed 0.008 sits inside |
| "spectral simplex" narrow geodesic band | 98.8% of random pairs fall in the reported [1.93, 2.19] |
| subspace degeneracy as a contribution | Méloux et al. (fourth failure mode); Schiffman measures overlap 0.02-0.04 **and resolves it** with CCA ~0.99 |
| circle geometry at grokking as a contribution | Schiffman, with operator fits and eigenvalue spectra |
| "interventions go off-manifold and that matters" as a contribution | Grant, Han, Tartaglini & Potts, ICLR 2026 |
| "divergence predicts intervention quality" | Grant et al., OOD IIA on EMD, R^2 = 0.73 |
| "accuracy is purchased with reconstruction error" (general form) | Grant et al. Fig 3B is the same curve, and their *targeted* penalty escapes it (EMD 0.007 at IIA 0.9988) |
| "the method optimizes the same objective as NL-DAS, constraints differ" (main_v12:826) | **our own run**: give the constrained map that objective and it scores 0.961 on a random network |

---

## 3. Claims that survive

- **The random-network dissociation.** `pi_sae` 1.000 pretrained / 0.000 random,
  against unconstrained nonlinear DAS at 1.000 / 0.433. Nobody has run Sutter's
  test on a constrained map; Grant et al. cite Sutter as open and never test
  vacuity.
- **Dimension efficiency.** A nonlinear constrained map reaches the causal
  variable in far fewer dimensions than a linear one. No amount of divergence
  regularisation fixes this, because the CL loss addresses divergence and not
  expressivity.
- **Cross-distribution dissociation.** Held-out entities failing correctly at
  0.001 while the unconstrained map reports 1.000.
- **Group action, not linearity.** Affine (2a+3b+5) is linear and fails
  equivariance at 0% across four primes. Schiffman does modular addition only and
  does not touch this.
- **DAS false positives on memorising models.** Squaring and cubing reach IIA
  0.857 without grokking. Connects to Makelov et al.'s subspace illusion.
- **The atlas breadth.** 14 operations x 4 primes x 3 depths, against Schiffman's
  modular addition only. Single-run; usable as scope evidence, not as a headline.
- **Stochastic grokking.** Composite addition 5/10 seeds, power 0/10.
- **The diversity ratio failing.** An honest negative about our own metric.

---

## 4. Decisions taken, with reasons

**Paper structure**
- One paper, not two. The split was justified when the second paper had its own
  spine (degeneracy); that spine is now published elsewhere, so the material
  returns as evidence in Act 1.
- Title clause "When Does Linear Causal Abstraction Work?" survives v10-v12.
  Grassmannian subtitle to be dropped with the geometry sections.
- Three acts: linear is expensive (grokking evidence) -> unconstrained is vacuous
  (Sutter, reproduced) -> constrained threads it.
- Grokking is **evidence**, not subject.
- `paperB_grassmannian_grokking_v1.tex` deleted (recoverable via git).

**Reporting**
- Mean with confidence interval as the primary figure, maximum over seeds beside
  it. Best-of-N alone is a biased upper order statistic.
- Nulls are empirical, generated by the same code path as the statistic they
  calibrate. Never analytic. This rule exists because the analytic k/d chance
  value produced the false "sub-chance" claim.
- Pre-registration lives in an appendix, never the main text.

**Method**
- DAS runs MIB's own code (`CausalAbstraction.neural.featurizers.SubspaceFeaturizer`),
  with MIB's per-task hyperparameters, verified by
  `experiments/test_das_matches_mib.py`.
- Old configurations retained as labelled ablation arms rather than deleted, so
  published numbers stay traceable.

---

## 5. Open contradictions — resolve before drafting

**5.1 Does the expansion matter? Three sources disagree.**

| source | evidence | implication |
|---|---|---|
| main_v12 `tab:ablation`, IOI row | partitioned 0.95 against +expansion 0.98 | expansion nearly irrelevant on IOI |
| main_v12 text (771-778) | "the label-conditional VAE (structured prior, no expansion) achieves IIA ~ 0 across all operations" | expansion is essential on arithmetic |
| measured 2026-07-30, IOI | `pi_vae` **0.356** against `pi_sae` **1.000** | expansion is essential on IOI too |

`tab:ablation`'s caption states "Single runs per cell", so its 0.95 may be one
lucky fit. **The belief that "we only need pi_vae, not pi_sae" traces to that
single cell and is contradicted by the 2026-07-30 measurement.** Do not act on
either until the six-task run returns with seeds.

**5.2 Does the label-conditional prior do anything?**

Gender bias, three seeds: `pi_plain_vae` (no prior) 0.419 against `pi_vae` (with
prior) 0.429. The prior buys 0.01 on accuracy. It is the component that carries
the iVAE identifiability motivation, so the motivation is currently decorative.
Note the separate point that accuracy is the wrong axis for an identifiability
claim — that would need cross-fit consistency, which has never been measured.

**5.3 `das_reference` = 0.000 on pretrained IOI.**

A defect, not a result: MIB's IOI recipe is batch 1024, and this script's pair
set is smaller, so it trains for two optimiser steps. Needs the batch capped at
dataset size, then re-run.

**5.4 Does vacuity appear gradually?**

Untested. The interchange weight beta and the number of end-to-end epochs are
both unswept. If random-network accuracy rises with beta while pretrained
accuracy saturates early, there is a usable operating point and a finding that
generalises Sutter beyond unconstrained maps. This is the next pre-registration.

---

## 6. Defects found in the code, 2026-07-30

Recorded because several were load-bearing and none were visible from results.

1. **Three different DAS implementations** across the paper: Gaussian init with
   per-step QR, delta-PCA warm start, and random-orthogonal with an orthogonal
   parametrisation. "DAS" meant different things in different tables.
2. **DAS trained at 1e-3 where MIB's indirect-object-identification baseline uses
   1.0** — a factor of one thousand. Found only by reading MIB's baseline source
   rather than its library default.
3. **Batch size 32 assumed where MIB uses 1024** for GPT-2, read from
   `get_model_config`.
4. **MIB's `DEFAULT_CONFIG` is not what MIB runs anywhere** — every baseline
   overrides it, and they differ from each other.
5. **The random-network control had no end-to-end arm.** No script in the
   repository contained both `e2e` and `random_init`, so the control tested a
   different method from the one the paper reports.
6. **Noise floors were single draws**, so three tasks reported exactly 0.000 with
   no sampling spread behind them.
7. **`output_dir` was applied verbatim to every task** in a multi-task run, so six
   tasks would have written to one file; omitting it overwrote published results
   in place.
8. **`run_task` timeout was 6h against a 5-fit budget**, with a single volume
   commit at the end, so a timeout would have discarded a whole task.
9. **`data/mib` was absent from the repository**, so the six-task table has never
   been reproducible from this repo alone.
10. **`RotateLayer` at module scope** would have raised `NameError` on import,
    since torch is imported under a try/except in the Modal scripts.

---

## 7. Runs

| run | status | output |
|---|---|---|
| six-task corrected DAS (ioi, greater_than, sva, gender_bias, capitals, hypernymy), 5 seeds | **in flight**, 6 pods, app `ap-BWyoqajepf08WyIcvYuSZJ` | `/results/grassmannian_atlas/mib_reference_das_corrected/<task>/` |
| random-network control, pretrained, IOI k=1 | complete | `experiments/results/random_network_control_e2e/pretrained_ioi.json` |
| random-network control, random-init, IOI k=1 | complete | `experiments/results/random_network_control_e2e/random_init_ioi.json` |

---

## 8. Immediate next actions

1. Wait for the six-task run. It decides whether `pi_sae` without end-to-end
   training carries the other five tasks, and therefore how much restructuring
   the manuscript needs.
2. Fix `das_reference`'s batch degeneracy (5.3) and re-run.
3. Pre-register the beta / end-to-end-epoch dose-response (5.4).
4. Build the CL-regularised **nonlinear** DAS arm — applying Grant et al.'s loss
   to the map that is actually vacuous. Applying it to linear DAS tests nothing,
   since linear DAS was never vacuous.
5. Correct `paperA_constrained_alignment_v1.tex`, which reports end-to-end
   numbers labelled plainly as "structured VAE".
