# Statistical protocol

Rules that apply to every pre-registration and results table in this repository.

Each rule exists because it was violated in a document that was subsequently
withdrawn, and each carries the measurement that establishes it. They are written
down so the same errors are not re-derived. The measurements come from a parallel
line of work on grokked modular arithmetic; the statistics transfer, the results
do not, and none of that data appears here.

---

## 1. Bootstrap over the unit of manipulation, never over derived pairs

With `n` fits, the `n(n-1)/2` pairwise comparisons are a U-statistic: each fit
appears in `n-1` pairs, so the pairs are not independent.

**Measured:** at `n = 20`, variance inflation of the pairwise mean is 9.9–11.5x,
giving `n_eff ≈ 17–19`, not 190. A naive Student's t interval over the 190 pairs
is **3.1–3.4x too narrow**.

Resample fits, recompute all pairs within each resample. Drop duplicate-index
self-pairs: resampling 20 fits inserts `E[190/20] = 9.5` self-pairs at overlap
1.0 into each mean, about 5% of the pairs, inflating both arms and shrinking any
difference by roughly 5%. With that correction the fit-level bootstrap is
approximately valid — type-I error 0.040 at a nominal 0.05.

## 2. No tail quantiles from small samples

A 97.5th percentile of 190 dependent values is the **5th largest**. After
retention losses at 12 fits it is the 2nd largest of 66, and two fits landing
together flips it.

**Measured consequence:** a pre-registered endpoint built on a 97.5th percentile
was shown by simulation to be a step function with a cliff, and to *invert* — it
withdrew the paper's title in precisely the world where the hypothesis was most
true. Use means and differences of means.

## 3. Clopper-Pearson for proportions, and size the sample first

Interchange accuracy is a proportion. Report exact Clopper-Pearson intervals.

**Measured:** the lower bound on 200/200 is **0.9851**, so "1.000" from 200 pairs
carries no information beyond "at least 0.985" and cannot separate 1.000 from
0.99.

The opposite error is equally real. One revision replaced an under-powered `n`
with "all valid held-out pairs", which was **80,011,692 pairs** — 74 days of
compute. Fix `n` by power calculation and state it.

## 4. Nulls must contain the nuisance factor under test

A null lacking the property being tested cannot calibrate it. This error was
made three times in different forms.

- **The null must match the question, and two questions get confused here.**
  For *"what accuracy does chance produce on this task"*, a random subspace is
  the correct baseline, and the random-subspace floors used in
  `PREREGISTRATION_RECONSTRUCTION_CRITERION.md` are appropriate. For *"is this
  fitted subspace unusual"*, a random subspace is the wrong null, because random
  subspaces are not causally sufficient and the fitted one is; there the null is
  other subspaces that are also sufficient. An earlier draft of this rule
  collapsed the two and would have condemned a correct baseline.
- **Haar nulls cannot calibrate angle thresholds.** Under Haar-random 10-frames
  in R^128, the count of principal angles below 0.25 rad is a point mass at zero
  (max 0 over 20,000 pairs). Use a **matched-overlap** null: generate pairs at
  the observed overlap through the same code path. Doing so reversed a
  conclusion — an angle reported as "unusually distant" was below the *minimum*
  of 300 matched draws, i.e. unusually close.
- **Shared initialisation needs a placebo.** An anchoring experiment measured
  Δ = +0.143 for a meaningful anchor. A **placebo anchor** — one arbitrary random
  frame, carrying no information — produced Δ = +0.153, reproducing 107% of the
  effect. The difference-in-differences was −0.010, CI [−0.020, −0.000]. The
  entire effect was shared-trajectory correlation.

If a manipulation has `n = 1` (one anchor, one initialisation, one model),
inference at the fit level is pseudoreplication. Resample the manipulation.

## 5. Verify that a metric tracks the quantity of interest

**Measured:** a shared-direction count was used as the headline statistic. In the
same data, a control subspace with interchange accuracy **0.005** shared *three*
directions with the target, while a subspace with accuracy **1.000** shared
*zero*. The metric ran anti-correlated with the property it was meant to
evidence.

Before a metric becomes an endpoint, show it separates cases whose answers are
already known. Report threshold sensitivity: the headline above held only for a
threshold window 0.032 rad wide.

## 6. Every failure row must delete a sentence

For each hypothesis, write the sentence that would be removed from the abstract
if it fails. If no sentence would be removed, it is descriptive and must be
labelled so.

**Measured:** in one withdrawn document, exactly **1 of 11** failure outcomes
retracted anything. One hypothesis failing was written up as evidence *for* the
thesis.

## 7. Predict at least one failure

A document in which every prediction is "pass" is a plan for confirmation. State
which predictions are expected to fail and why. A prediction already determined
by data in hand is not a risky prediction — check before labelling it as one.

## 8. Freezing means hashing before the first fit

A document is frozen when every script exists, its SHA-256 is recorded, and the
hash is in `PREREG_sha256.txt`. Writing "frozen" in the header is not freezing.

**Record:** four consecutive documents in the parallel line of work carried
unfilled SHA fields and self-declared non-binding status. None was binding.

## 9. Separate train and test, including for anything derived from data

**Measured, in a line of work that has since been discarded**
(`subspace_agreement.py`, `N_PAIRS = 200`): an alignment map was fitted and
scored on the same 200 pairs, so its accuracy was in-sample while the arms
compared against it were not. Held-out accuracy was 0.995, not 1.000.

**This defect is not present in the scripts that produce the paper's tables.**
`k1_vae_vs_das.py:1328`, `random_network_control.py:944` and
`k1_hard_mode.py:578` all split disjointly before fitting. The rule stands as a
rule; the attribution is recorded here so it is not read as a finding about the
current pipeline.

The subtler version: a document stated that a derived quantity used "the training
half only", when three of its six inputs used no data and two used the full
activation grid, which contains every held-out input.

## 10. One confirmatory endpoint per experiment

Name it. Correct across experiments (Holm is sufficient). Everything else is
descriptive, carries no decision rule, and is reported with an interval.

Attaching sentence-deletion rules to a "descriptive" quantity makes it
confirmatory in operation; either promote it and correct for it, or remove the
rule.
