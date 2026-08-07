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

- **The random-network dissociation.** `lcp_vae` 1.000 pretrained / 0.000 random,
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

## 4b. Rename and method change, 2026-07-31

Two things happened at once, deliberately: the identifiers changed **and** the
method changed. So the name a result file uses tells you which method produced
it, with no ambiguity.

**Any result file whose keys start with `pi_` is stale**: it was produced by the
old method, which carried an L1 penalty. Any file using the names below is
post-correction and L1-free. This is the point of doing both at once.

| old code key | new code key | name in the paper |
|---|---|---|
| `pi_sae` | **`lcp_vae`** | label-conditional partitioned VAE — the method |
| `pi_sae_e2e` | `lcp_vae_interchange` | the same map with interchange training |
| `pi_vae` | `lc_vae` | label-conditional VAE, no expansion |
| `pi_plain_vae` | `lc_flat_vae` | label-conditional, no partition |
| `pi_plain_sae` | `lc_flat_expanded` | label-conditional, no partition, expanded |
| `use_pi_prior` | `use_label_prior` | --- |

**Why the name changed.** "SAE" was wrong twice over. The method's sparsity
penalty was measured to be inert, and the component doing the work is the
eightfold expansion of the causal latent, which is not sparsity. Keeping the
name invited a comparison to sparse autoencoders that the method does not
support. The paper uses full words and no acronym.

**Why the method changed.** `l1_coeff` now defaults to `0.0` everywhere. The
parameter survives only so `sparsity_vs_overcompleteness_ablation.py` can still
measure what the penalty does. The method is therefore not sparse in any sense.

**Consequence.** Every number previously reported for `pi_sae` was measured with
`l1_coeff = 1e-3` and does not carry over. All `lcp_vae` numbers must be
regenerated, including the 1.000 pretrained / 0.000 random result on indirect
object identification that the paper's central comparison rests on.

Frozen pre-registrations were **not** edited; they retain the old identifiers,
which is correct, since they governed runs that used them. This table is how to
read them.

## 4c. Known defect: grokking-path VAE trains on evaluation activations

`random_network_control.py:1258` builds the VAE training set from **all test
activations** (its own comment says so), while the intervention pairs are drawn
from the same test indices. So on grokking tasks every VAE arm is fitted partly
on activations it is later scored on.

The indirect-object-identification path does not have this: line 1322 builds
from `train_pairs` only. All the random-network results reported so far are that
task, so they are unaffected.

**Superseded, not outstanding.** The corrected grokking run in section 9 uses
`experiments/hybrid_pilot.py`, which splits the pairs 70/30 and fits every map
on `train_pairs` only, scoring on held-out `eval_pairs`. DAS and both VAE arms
are fitted the same way there. So the clean measurement exists and the leaked
numbers are simply the older ones.

The practical consequence is that **no rerun is required for anything currently
claimed.** The affected figures are the pre-correction grokking numbers,
including the $\IIA = 1.000$ that earlier notes describe as a result in hand.
On held-out pairs `lcp_vae` reaches 0.000 at $k=1$, and an unconstrained
full-latent swap reaches only 0.298.

`random_network_control.py` itself is left unrepaired on purpose: it is the
superseded path, and fixing it would invite its numbers back into circulation.
Any grokking figure must come from `hybrid_pilot.py`.

## 5. Open contradictions — resolve before drafting

**5.1 Does the expansion matter? Three sources disagree.**

| source | evidence | implication |
|---|---|---|
| main_v12 `tab:ablation`, IOI row | partitioned 0.95 against +expansion 0.98 | expansion nearly irrelevant on IOI |
| main_v12 text (771-778) | "the label-conditional VAE (structured prior, no expansion) achieves IIA ~ 0 across all operations" | expansion is essential on arithmetic |
| measured 2026-07-30, IOI | `lc_vae` **0.356** against `lcp_vae` **1.000** | expansion is essential on IOI too |

`tab:ablation`'s caption states "Single runs per cell", so its 0.95 may be one
lucky fit. **The belief that "we only need lc_vae, not lcp_vae" traces to that
single cell and is contradicted by the 2026-07-30 measurement.** Do not act on
either until the six-task run returns with seeds.

**5.2 Does the label-conditional prior do anything?**

Gender bias, three seeds: `lc_flat_vae` (no prior) 0.419 against `lc_vae` (with
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

1. Wait for the six-task run. It decides whether `lcp_vae` without end-to-end
   training carries the other five tasks, and therefore how much restructuring
   the manuscript needs.
2. Fix `das_reference`'s batch degeneracy (5.3) and re-run.
3. Pre-register the beta / end-to-end-epoch dose-response (5.4).
4. Build the CL-regularised **nonlinear** DAS arm — applying Grant et al.'s loss
   to the map that is actually vacuous. Applying it to linear DAS tests nothing,
   since linear DAS was never vacuous.
5. Correct `paperA_constrained_alignment_v1.tex`, which reports end-to-end
   numbers labelled plainly as "structured VAE".

---

## 9. Grokking validity: the registered run, and its verdict

Ran to completion on 3 seeds, `addition`, $p=113$, $k=1$, 25000 grokking epochs.
Results in `results/grokking_validity/addition_seed{0,1,2}.json`, hypotheses in
`docs/PREREGISTRATION_GROKKING_VALIDITY.md`.

**H5, the falsifier, passes.** Test accuracy 0.9996 / 0.9991 / 0.9996, all above
the 0.95 threshold. All three models grokked, so the run is valid and every
hypothesis below is readable rather than void.

| arm | necessity | interchange ($k{=}1$) | full-latent swap | steer success |
|---|---|---|---|---|
| random subspace | $-0.035 \pm 0.272$ | 0.000 | -- | 0.000 |
| **DAS** | $\mathbf{+7.245 \pm 1.457}$ | $0.011 \pm 0.022$ | -- | $0.017 \pm 0.017$ |
| `lcp_vae` | $+1.640 \pm 0.914$ | 0.000 | $0.298 \pm 0.071$ | $0.015 \pm 0.013$ |
| `lcp_vae_empirical` | $+1.765 \pm 1.051$ | 0.000 | $0.298 \pm 0.071$ | $\mathbf{0.315 \pm 0.042}$ |
| `lcp_vae_interchange` | $+2.007 \pm 1.508$ | $0.002 \pm 0.004$ | $0.626 \pm 0.133$ | $0.019 \pm 0.019$ |

Mean $\pm$ 95% CI over 3 seeds. Clean logit difference $49.6 \pm 2.4$.

**H1 (primary) fails.** DAS is roughly four times more necessary than `lcp_vae`
($7.245$ against $1.640$), the reverse of the prediction. The cyclic variable was
chosen as the case where a linear map should have little to remove, and the
linear map removed more.

**H2 (primary) fails.** No arm reaches held-out operator $R^2 > 0.5$ on any
shift. Every value is negative, and `predicted_shift_accuracy` is exactly 0.000
for every arm, every shift, every seed. No map found the group action. `lcp_vae`
is the worst arm on shift 2 at $-23.4 \pm 18.2$.

**H3 (secondary) fails.** `lcp_vae` $0.015 \pm 0.013$ against DAS
$0.017 \pm 0.017$; indistinguishable, and both at the floor.

**H4 (secondary) fails, and is unreadable regardless.** It predicted
`lcp_vae_interchange` below `lcp_vae` on held-out $R^2$; measured, it is above
($-1.06$ against $-2.30$ on shift 1). Since neither arm found an operator at all,
the ordering between two failures carries no information and should not be
reported as a result.

**The registered consequence, quoted:** "DAS localises at least as well even
where the variable is cyclic. The method has then shown no advantage on any task
measured, and the paper is a negative result about generative alignment maps
together with the finding that interchange accuracy does not measure
localisation."

### Two things this run surfaced that were not registered

**The prior is misplaced here too, and steering is the one non-floor number.**
`lcp_vae_empirical` steers at $0.315 \pm 0.042$ where `lcp_vae` reaches
$0.015 \pm 0.013$ from the same fitted model -- a factor of twenty from changing
only the target source. This replicates the indirect-object finding (§4 of
`SALVAGE_PLAN.md`) on a second task with a different variable type, which makes
"the label-conditional prior does not live where the encoder puts things" a
two-task result rather than a one-task anomaly. It is unregistered here and must
be reported as such.

**The registration's stated premise did not survive the leakage fix.** The
document motivates the run by "the generative map reaches $\IIA = 1.000$ at
$k = 1$ where linear DAS needs $k = 8$ or more". Measured here on held-out pairs,
`lcp_vae` reaches 0.000 at $k = 1$, and even an unconstrained full-latent swap
reaches only $0.298$. The likely explanation is the grokking-path train/eval
leakage recorded in §4c, which the earlier number predates. **This is a
hypothesis, not a finding** -- confirming it requires re-running the old
configuration against the fixed split. Until that is done, no earlier grokking
interchange number should be quoted anywhere.

---

## 10. Steering: the rank sweep, the algebra arms, and what the specificity control actually did

Indirect object identification, layer 10, 3 seeds, 60 scored pairs per arm.
`results/steering/seed{0,1,2}.json`. The two `additive_push` arms are seed 0 only;
seeds 1--2 were still in flight when this was written.

| arm | steer | LM-loss rise | norm. logit diff | still original |
|---|---|---|---|---|
| random subspace | $0.000 \pm 0.000$ | $0.002 \pm 0.004$ | $-0.998$ | $1.000$ |
| DAS $k{=}1$ | $0.006 \pm 0.011$ | $0.008 \pm 0.009$ | $-0.853$ | $0.900$ |
| DAS $k{=}8$ | $0.211 \pm 0.022$ | $0.026 \pm 0.018$ | $-0.108$ | $0.322$ |
| **DAS $k{=}16$** | $\mathbf{0.650 \pm 0.075}$ | $\mathbf{0.072 \pm 0.045}$ | $+0.428$ | $0.050$ |
| DAS $k{=}32$ | $0.928 \pm 0.071$ | $0.177 \pm 0.101$ | $+0.754$ | $0.000$ |
| $\Delta$-PCA $k{=}16$ | $0.778 \pm 0.058$ | $0.161 \pm 0.055$ | $+0.256$ | $0.133$ |
| $\Delta$-PCA $k{=}32$ | $1.000 \pm 0.000$ | $0.328 \pm 0.026$ | $+0.536$ | $0.000$ |
| `lcp_vae` | $0.000 \pm 0.000$ | $0.022 \pm 0.033$ | $-0.618$ | $0.511$ |
| `lcp_vae_empirical` | $0.939 \pm 0.061$ | $0.795 \pm 0.272$ | $+0.458$ | $0.039$ |
| `lcp_vae_interchange` | $0.017 \pm 0.000$ | $0.050 \pm 0.062$ | $-0.274$ | $0.278$ |
| additive push $\alpha{=}1$ | $0.000$ | $0.062$ | $-0.922$ | $1.000$ |
| additive push $\alpha{=}4$ | $0.050$ | $1.265$ | $-0.688$ | $0.850$ |

### The specificity control was not defective. It discriminates.

`SALVAGE_PLAN.md` §2 withdrew the 0.993 steering result on the grounds that the
control was unpassable -- it steers generic prompts toward a name they have no
slot for, so any working edit looks destructive. **The rank sweep refutes that.**
DAS at $k = 16$ steers at $0.650$ while raising language-modelling loss by
$0.072$ nats, inside the registered $0.10$ threshold. The control is passable, and
one arm passes it.

What the control separates is cost per unit of steering. At comparable success,
the damage differs by an order of magnitude:

| | steer | LM-loss rise |
|---|---|---|
| DAS $k{=}16$ | 0.650 | **0.072** |
| $\Delta$-PCA $k{=}16$ | 0.778 | 0.161 |
| `lcp_vae_empirical` | 0.939 | 0.795 |

`lcp_vae_empirical` does not steer better in the sense that matters. It pushes
harder, and the generic-prompt loss records the collateral. This removes the
second of the two positive claims in `SALVAGE_PLAN.md`.

### Registered verdicts

**`PREREGISTRATION_STEERING.md`**

- **H1 fails as written.** Not every non-random arm clears the floor by 0.10:
  DAS $k{=}1$ reaches $0.006$ and `lcp_vae` $0.000$. The rule's stated rationale
  -- "steering along a random subspace works as well as along a fitted one" --
  does **not** describe what happened, since the random arm scored $0.000$ and one
  arm scored $0.939$. The literal verdict is a failure; the rationale anticipated a
  different failure mode. Recorded rather than reinterpreted.
- **H2a (space) holds** at $k = 1$: `lcp_vae_empirical` $0.939$ against DAS
  $0.006$. It does not survive the rank sweep, since DAS at $k = 32$ reaches
  $0.928$.
- **H2b (prior) fails** by $-0.939$, the same fitted model under two target
  sources. The label-conditional prior contributes nothing beyond an empirical
  mean, now on two tasks.
- **H3 fails for four arms**, whose steering numbers are withdrawn by the
  registered rule: DAS $k{=}32$ ($+0.177$), $\Delta$-PCA $k{=}32$ ($+0.328$),
  `lcp_vae_empirical` ($+0.795$), push $\alpha{=}4$ ($+1.265$).
- **H4 fails.** `lcp_vae_interchange` sits *closer* to 1.0 than `lcp_vae`
  ($1.274$ against $1.618$ in absolute deviation), the reverse of the prediction.

**`PREREGISTRATION_STEERING_ALGEBRA.md`**

- **H1.1 fails.** $\Delta$-PCA beats DAS at 2 of 6 ranks, below the registered 4.
  Consequence as registered: fitting earns its cost, and $\Delta$-PCA is a
  baseline. The crossover is unregistered but worth keeping: DAS wins at
  $k \leq 8$, $\Delta$-PCA at $k \geq 16$.
- **H1.2 could not be read.** The `steering` function records no interchange
  accuracy, so the metric the hypothesis is about was never written. This is a
  registration defect -- the prediction was made about a quantity the script does
  not produce -- and it is reported as unmeasured, not as a null.
- **H2.1 fails on both clauses.** Set-arm idempotence error was to stay below
  $0.01$; the three linear arms are exactly $0.0000$, but the generative arms
  reach $0.136$ and $0.159$. The push at $\alpha = 1$ gives $0.059$, below the
  $0.10$ the hypothesis required. **The registered consequence is that all of
  Part 2 is withdrawn.** Honoured.
  The registration's error is now visible: it classed the generative map as a
  coordinate-setting arm, but its operation
  $h + \mathrm{dec}(\mu(y), z_n) - \mathrm{dec}(z_c, z_n)$ is not a projection and
  has no reason to be idempotent. Non-idempotence there is a property of the
  operation, not a bug.
- **H2.2 fails** ($|\Delta| = 0.050$), and is a floor effect: the push scores
  $0.000$ and $0.050$, so there is no working operation whose coefficient
  sensitivity could be measured.
- **H2.3 fails in the opposite direction.** The push shows *higher* selectivity
  ($2.94$) than every set-arm ($1.96$--$2.89$).

### Where this leaves the steering claim

Nothing survives as a positive claim for the method. The one arm that both steers
above $0.5$ and passes the specificity control is DAS at $k = 16$, a linear map.
The honest summary is that steering behaves like localisation did: the generative
map needs a larger edit to achieve the same effect, and the controls charge it for
that.

---

## 11. Abstention: the gate works, and it is detecting the wrong thing

3 seeds, GPT-2, layer 10, `vae_k = 1` / hidden 256 (the steering architecture),
`das_k = 16`. `results/abstain/seed{0,1,2}.json`, hypotheses in
`docs/PREREGISTRATION_ABSTENTION.md` + Amendment 1.

| arm | steer | LM rise, generic | LM rise, near |
|---|---|---|---|
| DAS $k{=}16$ | $0.789 \pm 0.013$ | $0.048 \pm 0.035$ | $0.271 \pm 0.541$ |
| `lcp_vae_empirical` | $0.981 \pm 0.019$ | $0.664 \pm 0.181$ | $-0.654 \pm 0.219$ |
| **gated** | $0.830 \pm 0.038$ | $\mathbf{0.000 \pm 0.000}$ | $0.000 \pm 0.000$ |
| norm-matched | $0.515 \pm 0.076$ | $0.137 \pm 0.007$ | $-0.655 \pm 0.133$ |
| gated + norm-matched | $0.426 \pm 0.052$ | $0.000$ | $0.000$ |

Reconstruction error: task $0.193 \pm 0.004$, near $0.628 \pm 0.016$, generic
$0.746 \pm 0.009$, threshold $\tau = 0.220 \pm 0.006$.

**H1 holds decisively.** AUROC separating ordinary text from task text is
$1.000 \pm 0.000$. The signal exists.

**H2 fails on both clauses, which is the registered failure mode.** The AUROC gap
between the near and generic sets is $0.000 \pm 0.000$ against a required
$\geq 0.15$, and the near-set abstention rate is $1.000$ against a required
$< 0.90$. The gate abstains on every ordinary sentence that ends in a name, so it
separates "is an IOI prompt" from "is not", and **not** "can be edited here" from
"cannot".

Registered consequence, applied: this is reported as **template detection** and
explicitly not as selective steering. Had the near set been left out, the same run
would have shown AUROC 1.000 and zero damage at 85% of steering retained, and
would have read as the capability claim.

**H3 fails on its steering clause.** Gating drives the language-modelling rise to
exactly $0.000$, but retains $0.845 \pm 0.028$ of ungated steering against a
required $0.90$. The margin is small and the registered bar is the registered bar.

**H4 holds.** Norm-matching leaves the rise at $0.137 \pm 0.007$, above the 0.10
threshold, while cutting steering roughly in half ($0.524 \pm 0.068$ of ungated).
The damage is not purely edit magnitude, so scaling is not the missing knob.

**H5 holds, and is the mechanism.** The generative edit grows off-distribution --
relative norm $0.355 \to 0.550$ from task text to ordinary text -- while DAS's
*shrinks*, $0.257 \to 0.181$. A projection's edit is bounded by its subspace; a
decoder difference is not.

### Two things worth recording that were not registered

**The reconstruction signal is graded, and the threshold discards the gradation.**
Near-set error ($0.628$) sits between task ($0.193$) and generic ($0.746$), so the
map does register that name-final sentences are closer to its training
distribution than arbitrary text. The registered $\tau$, the 95th percentile of
training error, is far below both, so both abstain completely. Whether a higher
threshold separates near from generic is a real question and **any answer
obtained by moving $\tau$ now is post-hoc**; it requires its own registration.

**The near set is a flawed control in one direction.** `lcp_vae_empirical` shows
a *negative* language-modelling rise on it ($-0.654$): steering toward a name
makes name prediction easier on sentences that end in names. So the near set
rewards the edit rather than penalising it, and cannot serve as a specificity
control by itself. It works for its registered purpose -- testing what the gate
discriminates -- and should not be reused as a damage measure.

---

## 12. Composition on RAVEL: no positive claim, and one error of mine

GPT-2, layer 8, entity's last token, 3 seeds, `k = 16`.
`results/composition/seed{0,1,2}.json`. Hypotheses in
`docs/PREREGISTRATION_STRUCTURAL_CAPABILITIES.md` Part 3.

| attribute | values | cause DAS | cause VAE | isolation DAS | isolation VAE |
|---|---|---|---|---|---|
| Continent | 4 | $0.174 \pm 0.057$ | $\mathbf{0.244 \pm 0.047}$ | $\mathbf{0.766 \pm 0.010}$ | $0.647 \pm 0.027$ |
| Country | 11 | $0.130 \pm 0.028$ | $\mathbf{0.315 \pm 0.010}$ | $\mathbf{0.637 \pm 0.029}$ | $0.489 \pm 0.032$ |
| Language | 9 | $0.300 \pm 0.114$ | $\mathbf{0.378 \pm 0.148}$ | $\mathbf{0.580 \pm 0.018}$ | $0.518 \pm 0.077$ |

No-op cause: $0.000$, $0.000$, $0.096$. Random subspace: $0.000$, $0.003$, $0.101$.

**An error to record.** During the run I reported Country as having 85 values and
the generative map's advantage as growing with the number of values -- 1.4x at 6
values, 2.8x at 85. The value counts were read off the raw entity file rather
than off the data the run used. After filtering to entities GPT-2 answers
correctly, capping at 800 entities and requiring eight per value, the counts are
4, 11 and 9. The measured ratios are 1.40, 2.42 and 1.26, which are **not**
monotone in the number of values. The rank-efficiency story from
`SALVAGE_PLAN.md` §2 gets no support here and the claim is withdrawn.

**H3.2 holds on all three attributes.** Margins over the no-op are $+0.174$ /
$+0.244$, $+0.130$ / $+0.315$, $+0.204$ / $+0.282$, all above the registered
$0.10$. Isolation is therefore readable, which is what the earlier RAVEL attempt
could not achieve. Moving the intervention to the entity's last token, as the
necessity profile indicated, is what changed.

**H3.1 is unreadable, and points the wrong way.** It required cause matched
within $0.05$; the gaps are $0.070$, $0.185$ and $0.078$. What is visible is that
the generative map causes more and isolates worse on **every** attribute. That is
the same pattern as the steering specificity control: a larger edit buys success
and pays for it in the side-effect measure. Distinguishing "worse map" from
"stronger edit" needs the cause-isolation curve from a rank sweep, which is a new
registration rather than a reinterpretation of this one.

**H3.3 fails, and the threshold was badly chosen.** Registered as mean cosine
above $0.30$ between DAS's per-attribute subspaces. Measured $0.233$, $0.212$,
$0.307$ -- one of three clears it.

The threshold was set without computing the null first, which was a mistake.
Two independent rank-16 subspaces in 768 dimensions give mean cosine
$0.121 \pm 0.001$ over 200 draws (95th percentile $0.130$). Every measured pair is
$1.7$ to $2.5$ times chance, so the subspaces do overlap well above random, most
strongly for Country and Language -- the pair where one attribute largely
determines the other. The registered verdict stands as a failure and the null is
reported beside it so the numbers mean something.

**Language is the least trustworthy attribute** on two counts. Its no-op scores
$0.096$, because `tok_of` compares only the first token of a value and RAVEL's
language values share prefixes, so a target can collide with the base answer
without any edit. Its seeds also spread widely: DAS scored $0.239$, $0.245$ and
$0.417$, a range larger than the DAS-VAE gap it is meant to resolve.

**Registered consequence, applied:** H3.1 fails, so capability 3 is withdrawn.
Its unreadability does not rescue it.

**Not to be reused without fixing:** the cause and isolation scoring here is
hand-rolled. Anything reported against MIB's baselines must run through their
`ravel.py` scoring, since the house rule is to use benchmark code directly.

---

## 13. Capability 4: the structured prior fails its own hypothesis and fixes a different problem

GPT-2, layer 10, 18 name values, **5 held out of training entirely**, 3 seeds.
`results/unseen/seed{0,1,2}.json`. Part 1 of
`PREREGISTRATION_STRUCTURAL_CAPABILITIES.md`.

| arm | steer | LM rise |
|---|---|---|
| `das_unseen` | **undefined** | -- |
| `das_seen` | $0.018 \pm 0.018$ | $0.021 \pm 0.010$ |
| `free_embedding_unseen` | $0.000 \pm 0.000$ | $0.045$ |
| `structured_prior_unseen` | $0.000 \pm 0.000$ | $0.044$ |
| `free_embedding_seen` | $0.021 \pm 0.021$ | $0.081$ |
| **`structured_prior_seen`** | $\mathbf{0.714 \pm 0.062}$ | $0.332$ |
| `free_embedding_seen_empirical` | $0.982 \pm 0.025$ | $0.668$ |
| `structured_prior_seen_empirical` | $0.986 \pm 0.019$ | $0.535$ |

**H1.1 fails outright.** Both priors steer held-out values at exactly $0.000$.
Tying $\mu(y)$ to the token embedding gives an unseen value a target, and the
target is useless: $W$ is fitted only on seen classes and nothing forces it to
extrapolate, while the decoder has never rendered an activation for those values.
Registered consequence applied: **the unseen-value capability is withdrawn
permanently.**

DAS is recorded as *undefined* rather than $0.000$, since it has no target to be
asked for. That distinction is the one honest thing this experiment establishes
about the linear map.

**H1.2 fails, in the favourable direction, and this is the real finding.** It
required the structured prior to stay within $0.10$ of the free table on *seen*
values -- a clause written to catch the structured prior **damaging** ordinary
performance. Measured, it is better by $0.692$: $0.714$ against $0.021$, a
thirty-fold improvement in prior-target steering.

**H1.3 holds.** The prior-against-empirical gap narrows from $-0.961$ to
$-0.272$. The structured prior lands much closer to where the encoder actually
puts things.

So the H2b problem recorded in `SALVAGE_PLAN.md` §4 -- "the label-conditional
prior does not live where the encoder puts things" -- is **substantially a
consequence of the free lookup table**, not of the idea of a prior. That was
measured twice as a defect of the method and is largely an artifact of one
architectural choice. It is unregistered as a positive claim and must be reported
as such.

---

## 14. Ripple consistency: the isolation failures are substantially coherent

RAVEL, GPT-2, layer 8, 3 seeds, 9 runs. `results/consistency/`.
`docs/PREREGISTRATION_RIPPLE_CONSISTENCY.md`. All figures are conditional on the
co-attribute having changed; "null" is consistency with a randomly drawn
*different* value of the set attribute, on the same items.

| set | ripple into | arm | consistent \| changed | null | margin |
|---|---|---|---|---|---|
| Continent | Country | DAS | $0.212$ | $0.013$ | $+0.198 \pm 0.041$ |
| Continent | Country | **VAE** | $\mathbf{0.404}$ | $0.045$ | $\mathbf{+0.359 \pm 0.050}$ |
| Continent | Language | DAS | $0.925$ | $0.821$ | $+0.103 \pm 0.102$ |
| Continent | Language | VAE | $0.978$ | $0.819$ | $+0.159 \pm 0.240$ |
| Country | Continent | DAS | $0.062$ | $0.089$ | $-0.027 \pm 0.087$ |
| Country | Continent | VAE | $0.093$ | $0.067$ | $+0.026 \pm 0.032$ |
| Country | Language | DAS | $0.359$ | $0.231$ | $+0.128 \pm 0.069$ |
| Country | Language | **VAE** | $\mathbf{0.429}$ | $0.226$ | $\mathbf{+0.204 \pm 0.095}$ |
| Language | Country | DAS | $0.483$ | $0.014$ | $+0.469 \pm 0.087$ |
| Language | Country | **VAE** | $\mathbf{0.625}$ | $0.013$ | $\mathbf{+0.612 \pm 0.073}$ |

**H2 holds in every pair without exception.** The generative map's
consistency-over-chance margin exceeds DAS's on all six, including the two where
both are near zero. A manifold-respecting edit propagates more coherently than a
minimal one, and this is the first prediction in the project that held
everywhere.

**H3 holds decisively.** The registered bar was that at least one pair show over
$0.25$ of the generative map's isolation failures being coherent. Measured:
$0.404$ (Continent$\to$Country), $0.429$ (Country$\to$Language), $0.625$
(Language$\to$Country), $0.978$ (Continent$\to$Language). **A large share of what
RAVEL scores as isolation failure is correct propagation.**

**H1 holds on two of four primary pairs, and the registration was ambiguous.**
It required a margin of at least $0.10$ but never said whether that is per-pair
or aggregate. Per-pair: Continent$\to$Country $+0.359$ and Country$\to$Language
$+0.204$ clear it; Continent$\to$Language $+0.159$ has a confidence interval
spanning zero; Country$\to$Continent $+0.026$ fails. Recorded as ambiguous rather
than resolved in the direction that suits us; any future version must fix the
aggregation rule in advance.

**H4 holds where it can be read.** The random subspace margin is below $0.05$ on
three of six pairs and noisy elsewhere -- it changes so little ($2$--$8\%$ of
items) that "given changed" is computed over a handful, giving intervals up to
$\pm 0.4$. Not evidence against, but not evidence for.

### An asymmetry that was not predicted

Propagation runs from **coarse to fine and not back**. Setting the continent
moves the country to a continent-consistent one at $0.404$ against $0.045$
chance. Setting the country leaves the continent at chance, $0.093$ against
$0.067$, even though that is the *tighter* test -- each country has exactly one
continent, and the null controls for the set size.

So the edit propagates when the set value **constrains** the dependent attribute
loosely, and fails when the dependent attribute is entailed exactly. No account
of this yet. It should not be smoothed over: an edit that sets a city's country
to Japan and leaves its continent unchanged is incoherent, and the generative map
does that.

---

## 15. What the "isolation failures" actually are

Re-run of §14 with per-item outcome tokens logged (`*_outcomes`), same seeds and
maps. `results/consistency/`. This corrects the account in §14.

**§14's "coarse to fine and not back" is wrong.** Regrouped by the attribute
*receiving* the ripple, coherence is a property of the receiver and has nothing
to do with entailment direction:

| receiver | set | VAE margin | DAS margin |
|---|---|---|---|
| Continent | Country | $+0.026$ | $-0.027$ |
| Continent | Language | $+0.056$ | $+0.038$ |
| Country | Continent | $+0.359$ | $+0.198$ |
| Country | Language | $+0.612$ | $+0.469$ |
| Language | Continent | $+0.159$ | $+0.103$ |
| Language | Country | $+0.204$ | $+0.128$ |

Country receives from everything; Continent receives from nothing. The
entailment account in §14 was pattern-matching on one pair and is withdrawn.

### The Continent readout is not mis-propagating, it is breaking

Fraction of changed answers that are a valid value of the receiving attribute,
and the fraction that are a token of the attribute we **set**:

| set | read | valid | contaminated by the set value |
|---|---|---|---|
| Country | **Continent** | $0.404$ | $\mathbf{0.444}$ (DAS $0.480$) |
| Language | **Continent** | $0.353$ | $0.017$ |
| Continent | Country | $0.733$ | $0.017$ |
| Country | Language | $0.951$ | $0.038$ |
| Language | Country | $0.891$ | $0.005$ |
| Continent | Language | $1.000$ | $0.000$ |

Set the country to Japan, ask which continent the city is in, and **45% of the
time the model answers "Japan"**. It is not failing to infer Asia; it is
answering a different question. Two distinct failures, neither about entailment:

- **from Country: capture.** The written value answers the continent prompt.
- **from Language: degeneration.** Contamination is only $0.017$, but validity is
  $0.353$ and the most common answer is `' the'` -- the model falls out into a
  function word.

Consistency came out *below* uniform over continents for exactly this reason: the
answer is usually not a continent.

### It is not edit magnitude

The same country edit contaminates Continent at $0.444$ and Language at $0.038$.
One edit, one magnitude, two readouts, a twelvefold difference. The explanation is
answer-space overlap: a country name is a plausible completion of "the city of X
is in the ---" and not of "the language spoken in X is ---".

Both arms show it (DAS $0.480$, VAE $0.444$), so it is a property of the model
and the prompt, not of either map.

### What survives

Four cells have contamination $\leq 0.04$ and validity $0.73$--$1.00$, and in all
four the generative map's coherence margin exceeds DAS's:

| | VAE | DAS |
|---|---|---|
| Language $\to$ Country | $+0.612$ | $+0.469$ |
| Continent $\to$ Country | $+0.359$ | $+0.198$ |
| Country $\to$ Language | $+0.204$ | $+0.128$ |
| Continent $\to$ Language | $+0.159$ | $+0.103$ |

### The sharper metric critique

RAVEL's isolation assigns one number to three different behaviours: a coherent
update, a readout captured by the written value, and a degeneration to `' the'`.
The no-op argument in `BENCHMARK_FACTS.md` §1 says the metric *rewards inaction*;
this says it *cannot distinguish* the failure modes it does score. The second is
the stronger criticism, because it does not depend on anyone choosing a
degenerate solution.
