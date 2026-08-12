# Does effective write rank measure a causal variable's dimensionality?

**Status:** DRAFT — not frozen until committed. The commit SHA of this file is the registration.

Sections use the [OSF Preregistration](https://osf.io/prereg/) question titles verbatim. A
question that does not apply is answered **N/A** with the reason, never deleted.

## Research questions or hypotheses

Interchange methods report `k`, the dimension of the subspace they intervene on. `k` is a
hyperparameter someone swept: MIB uses 16 for one variable, 32 for another, and half the residual
stream for a third, with no argument that these are properties of the variables. Distributed
Alignment Search treats `(N, |Y_0|, ..., |Y_k|)` as "discrete hyperparameters" and reports a max
over them; the strings "cardinal" and "identifiab" do not appear in that paper.

Effective write rank — the participation ratio of the intervention deltas' singular values — is a
measurement rather than a setting. It returns exactly `k` for an orthogonal subspace map,
verified at 1.00, 3.87, and 14.75 for `k` = 1, 4, 16. The question is whether it tracks the
variable being localized or merely the map doing the localizing.

**H1 (primary, ordering).** With the model, layer, map, and training recipe held fixed, effective
write rank is ordered by the target variable's cardinality:

    rank(ones_carry, c=2) < rank(ones_out, c=10)

**H2 (scale).** The gap is substantial rather than incidental: `rank(ones_out)` exceeds
`rank(ones_carry)` by at least 2.0.

**H3 (null of interest).** If both variables return ranks within 1.0 of each other, effective
write rank is a property of the map's architecture, not of the variable, and no claim that it
measures dimensionality survives.

The prediction worth staking is H1, and it is genuinely at risk. Distributed Alignment Search
assigns 256 dimensions to a *binary* variable in one experiment and reports a binary variable
reaching only 0.88 at `k=1`, rising to 1.00 at `k=8` (Table 1). If dimensionality were
`c-1`, neither would happen. "Four values need three dimensions" is our inference, not an
established result.

## Foreknowledge of data or evidence

**No arithmetic-task result exists.** An attempt on Qwen2.5-0.5B failed before training with
`num_samples=0`: the model cannot do two-digit addition, so MIB's correctness filter removed every
example. Nothing was measured. MIB runs this task on Gemma-2 and Llama-3.1 only, which is
consistent with that failure.

**One rank measurement exists for a different task and is the reason for this experiment.** On
MIB's multiple-choice task, target `answer_pointer` (4 values), the same map reports 2.93 on
Qwen2.5-0.5B (16 seeds) and 2.54 on Gemma-2-2B (1 seed), despite residual widths of 896 and 2304.

**A control already undercuts the optimistic reading.** On Gemma at layer 16 the map scored
sensitivity **0.000** — it localized nothing — and still reported ranks of 2.69, 2.93, and 3.97
across three configurations. A number that appears when the map fails is not yet a measurement of
anything, which is what H3 exists to settle.

## Explanation of foreknowledge and managing unintended influences

The 2.93 figure creates an obvious pull toward reading any arithmetic result near 1 and near 9 as
confirmation. H2 fixes a numeric threshold in advance for that reason, and H3 names the outcome
that kills the claim rather than leaving it to interpretation.

Rank is measured **only where the map works**, defined below, because the Gemma layer-16 control
shows the quantity is reported regardless of success. A rank taken from a failed run is excluded
before it can be averaged into a favourable mean.

## Study type

Experimental. An interchange intervention manipulates the model's internal state.

## Intention for causal interpretation

Yes, and narrowly: whether a measurement of a map's write channel reflects a property of the
causal variable it was trained to localize.

## Blinding of experimental treatments

N/A — no human judgement enters the measurement.

## Additional blinding during research or analysis

The model, layer, map, recipe, both variables, the success threshold, and every decision rule are
fixed here before either variable is run.

## Study design

MIB's two-digit addition task, unmodified: their counterfactual datasets, causal model, token
positions, filter, and checker.

| held fixed | value |
|---|---|
| model | Gemma-2-2B |
| layer | 19 |
| map | label-conditional partitioned VAE, expansion 8 |
| recipe | reconstruction + label-conditional KL + classifier at alpha 10, 500 epochs, Adam 1e-3 |
| nominal k | 1 |
| hidden width, nuisance width | 1024, 32 |

| varied | values | cardinality |
|---|---|---|
| target variable | `ones_carry` | 2 |
| | `ones_out` | 10 |

Layer 19 is where this map reaches sensitivity 1.000 on Gemma for the multiple-choice task. It was
chosen before this experiment and is not re-selected here. If the map fails at layer 19 on
arithmetic, that is reported as a failure of the design, not repaired by moving the layer.

## Randomization

Seeds 0, 1, 2 per variable. A seed varies map initialization and training only; MIB fixes the
dataset and split.

## Data collection procedures

MIB's `get_counterfactual_datasets` and `FilterExperiment`, unmodified. Only the pretrained
condition is run: a randomly initialized network cannot do the task, so a rank measured there
would describe the map alone.

## Data collection procedures - File upload

N/A — the benchmark's published datasets are used as-is.

## Sample size

Three seeds per variable, two variables, one model, one layer. Six runs.

## Sample size rationale

Three seeds distinguishes an effect from a draw and is what the budget supports. It cannot resolve
rank differences below about 0.5, and H2's threshold of 2.0 is set well above that.

## Starting and stopping rules

Fixed epoch counts, no early stopping, no inspection of held-out results during training.

## Manipulated variables

The target causal variable, and only that. Everything else in the table above is held fixed.

## Measured variables

**Effective write rank.** Participation ratio `(sum s_i^2)^2 / sum s_i^4` of the singular values
of the intervention deltas `h' - h`, over held-out pairs. Calibrated: returns 1.00, 3.87, 14.75
for an orthogonal subspace map at `k` = 1, 4, 16.

**Sensitivity.** MIB's accuracy on the counterfactual datasets, used only as the inclusion gate.

## Measured variables - File upload

N/A — definitions are complete above; implementations are committed alongside.

## Indices

None.

## Indices - File upload

N/A.

## Statistical models

None. Mean and standard deviation of rank over the included seeds, reported with the seed count
and the number excluded.

## Statistical models - File upload

N/A.

## Transformations

None.

## Inference criteria

| hypothesis | holds when |
|---|---|
| H1 | mean rank for `ones_out` exceeds mean rank for `ones_carry`, and the gap exceeds the pooled seed standard deviation |
| H2 | that gap is at least 2.0 |
| H3 | the two mean ranks differ by less than 1.0 |

H1 and H3 cannot both hold. If neither holds — a gap between 1.0 and the pooled standard
deviation — the result is reported as inconclusive at this sample size, and no claim is made
either way.

## Data inclusion and exclusion

**A run contributes its rank only if its sensitivity is at least 0.5.** Rank from a map that
failed to localize the variable describes the architecture, which is the confound this experiment
exists to rule out. Excluded runs are reported with their sensitivity and rank, so the exclusion
is visible rather than silent.

**If fewer than two seeds per variable clear the gate, the experiment is reported as failed** and
no rank comparison is made. This is a live possibility: MIB does not report this task on this
model at nominal `k`=1.

## Missing data

A run that crashes is rerun with the same seed and the rerun disclosed. A variable whose runs all
fail the gate is reported as unmeasurable here, not substituted.

## Other planned analysis

Exploratory and labelled as such: rank for `tens_out` (10 values, but with `ones_carry` as a
parent, so it may behave unlike `ones_out`), and comparison against the multiple-choice
`answer_pointer` figure of 2.54 on the same model. Neither is registered, because both cross tasks
and the task is not held fixed.

## Context and additional information

A null here is worth as much as a positive. If rank is architectural, then a measurement we have
been treating as informative is not, and several claims elsewhere in this project rest on it. If
rank tracks cardinality, the quantity every interchange paper currently reports as a swept
hyperparameter has a measurable counterpart.

## Log

- Created. Not frozen until committed; the commit SHA is the registration.
