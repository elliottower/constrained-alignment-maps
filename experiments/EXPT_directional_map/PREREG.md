# Does a rank-constrained nonlinear map localize a causal variable better than a rotation?

**Status:** DRAFT — not frozen until committed. The commit SHA of this file is the registration.

Sections use the [OSF Preregistration](https://osf.io/prereg/) question titles verbatim. A
question that does not apply is answered **N/A** with the reason, never deleted.

## Research questions or hypotheses

Distributed Alignment Search writes along `k` fixed directions by an amount linear in the
activation. Every nonlinear alternative we have measured buys accuracy by widening the write
channel instead of by reading better, which makes a comparison at equal `k` meaningless: `k`
counts coordinates in each map's own system, and a decoder's coordinate is an input-dependent
direction while a rotation's is a fixed one.

The map registered here holds the write channel fixed and frees only the readout:

    h' = h + sum_j [alpha_j(z_src) - alpha_j(z_base)] d_j

with `d_1..d_k` learned directions and `alpha_j` a nonlinear function of the latent. Distributed
Alignment Search is the special case where `alpha` is linear, so this is a strict generalization
at identical write capacity.

**H1 (primary).** At equal effective write rank, the directional map's interchange accuracy
exceeds linear Distributed Alignment Search on the multiple-choice task.

**H2.** The directional map's specificity is at or above that of the unconstrained
encoder/decoder map, because it cannot write outside a `k`-dimensional span.

**H3.** On a randomly initialized network the directional map scores at the measured floor. A
rank-`k` write is a stated capacity bound, and the one map we have tested without such a bound —
an exactly invertible flow — reached 0.380 there where every bounded map reached 0.000.

**H4 (gauge).** The directional map's explicit directions recur across seeds: mean matched
cosine exceeds the within-span null. A rotation cannot do this — its basis is arbitrary, measured
at pairwise overlap 0.008 across ten seeds in prior work — so a positive result is a property no
rotation-based method can have, and a negative one retires the feature-level claim entirely.

The prediction worth staking is H1 *conditional on* the write-rank check: a win at equal capacity
is a statement about reading, and it is the only version of the comparison that survives the
objection that the decoder did the work.

## Foreknowledge of data or evidence

**The directional map has never been run.** It was implemented immediately before this document
and only its structural properties were verified: exact round trip (1.2e-07) and effective write
rank equal to `k` (participation ratio 1.00 at k=1, 2.97 at k=4, matching a rotation's 1.00 and
3.87). Those are properties of the construction, not outcomes, and are stated here rather than
predicted.

**Substantial results already exist for the comparison arms and are not registerable.** On Qwen2.5-0.5B,
multiple-choice, layer 18, k=1, seed 0: the encoder/decoder map scores sensitivity 1.000 and
specificity 0.808; Distributed Alignment Search scores 0.280 and 1.000; the untrained floor is
0.000. Gemma-2-2B has a working site at layer 16 and a dead one at layer 13. Specificity was
never measured in any prior work on this method, in either repository, under any name; the 0.808
is its first measurement.

**No effective write rank has been measured for any trained map.** The instrument was added after
those runs and is in flight for the encoder/decoder arm. Its value there is unknown at the time
of writing and is the quantity H1 conditions on.

**The gauge analysis has not been run.** Directions are being logged; the scoring script exists
and has been validated only on synthetic cases with known answers.

## Explanation of foreknowledge and managing unintended influences

Knowing the encoder/decoder arm reaches 1.000 creates an obvious pull toward tuning the
directional map until it matches. Two commitments block it. The directional map is trained with
the *same* four-term objective and the *same* hyperparameters as the encoder/decoder arm — the
recipe is inherited, not searched — so any difference between the two arms is the write
constraint and nothing else. And H1's comparison is against Distributed Alignment Search at
matched write rank, a number nobody has seen.

The layer is fixed to a site chosen before this map existed, by a sweep run for a different arm.

## Study type

Experimental. An interchange intervention is a manipulation of the model's internal state.

## Intention for causal interpretation

Yes, in a restricted sense. The claim is about which map recovers a causal variable the benchmark
defines, not about discovering a new causal structure in the model.

## Blinding of experimental treatments

N/A — no human judgement enters the measurement.

## Additional blinding during research or analysis

The arms, the site, the training recipe, the metrics, and every decision rule below are fixed
here before the directional map is run once.

## Study design

MIB's causal-variable track, multiple-choice question answering, target variable
`answer_pointer` (`XOrder` in their Table 3c). Everything except the maps is theirs: the
counterfactual datasets, the causal model, the token positions, the filter, and the checker.

| arm | write channel | trained on |
|---|---|---|
| `random` | untrained orthogonal subspace | nothing |
| `das` | k directions of a learned rotation | interchange |
| `lcp_vae` | decoder output, unconstrained | reconstruction, label-conditional prior, classifier |
| `directional` | k learned directions, nonlinear coefficients | same objective as `lcp_vae` |

Models: Qwen2.5-0.5B at layer 18, Gemma-2-2B at layer 16. Both sites were selected by sweeps run
before this map existed. Conditions: pretrained, and the same architecture with weights from
`AutoModelForCausalLM.from_config`.

## Randomization

Seeds 0, 1, 2 per arm per condition. A seed varies map initialization and training only; MIB
fixes the dataset and split upstream.

## Data collection procedures

MIB's `get_counterfactual_datasets` and `FilterExperiment`, unmodified. The pretrained condition
keeps only examples the model answers correctly; a randomly initialized network answers nothing
correctly, so no filter applies there and pair counts are reported per condition.

## Data collection procedures - File upload

N/A — no new data is collected; the benchmark's datasets are used as published.

## Sample size

Three seeds per arm per condition per model. Roughly 300 training examples, MIB's own figure for
this task, and their held-out test split for evaluation.

## Sample size rationale

Three seeds is the minimum that distinguishes an effect from a draw, and it is what the compute
budget supports across two models and four arms. It is not powered to resolve differences smaller
than about 0.1 in accuracy, and no such difference is claimed.

## Starting and stopping rules

Fixed epoch counts, no early stopping, no monitoring of the test split during training. Every arm
runs to completion regardless of intermediate values.

## Manipulated variables

The alignment map's write channel: unconstrained decoder output versus a rank-`k` span versus a
rotation's `k` fixed directions. Nominal `k` = 1 throughout, with k=16 — MIB's own published
dimensionality for this variable — as the baseline reference point.

## Measured variables

**Sensitivity.** MIB's accuracy on the counterfactual datasets where the target variable changes.

**Specificity.** MIB's accuracy on `randomLetter`, where the variable does *not* change and the
correct behavior is to leave the output alone. Reported separately, never pooled: an untrained map
scores 1.000 here by doing nothing, so a pooled mean rewards inaction.

**Effective write rank.** Participation ratio of the singular values of the intervention deltas.
Calibrated: it returns 1.00, 3.87, and 14.75 for a rotation at k = 1, 4, 16.

**Matched cosine across seeds**, against a within-span null. The null is a random basis of the
seed's own span, not random directions — two maps that find the same subspace with arbitrary
bases score 0.59 against a random-direction floor of 0.055, so the random floor would pass a
result that means nothing.

## Measured variables - File upload

N/A — definitions above are complete and the implementations are committed alongside this file.

## Indices

None. Sensitivity and specificity are reported separately and never combined into a single score.

## Indices - File upload

N/A.

## Statistical models

None. Means and standard deviations over three seeds, reported with the seed count. No test is
performed, and no claim rests on one.

## Statistical models - File upload

N/A.

## Transformations

None.

## Inference criteria

| hypothesis | holds when |
|---|---|
| H1 | directional sensitivity exceeds `das` by more than the pooled seed standard deviation, *and* the two arms' effective write ranks agree within 20% |
| H2 | directional specificity is at or above `lcp_vae`'s, seed means |
| H3 | directional sensitivity on the randomly initialized network is within one seed standard deviation of the `random` arm in the same condition |
| H4 | mean matched cosine exceeds the within-span null by more than two seed standard deviations plus 0.05 |

**H1 is void if the write ranks disagree.** A win at unequal capacity is the result this design
exists to avoid claiming, and it is reported as uninterpretable rather than as a win.

## Data inclusion and exclusion

No exclusions. Every seed that completes is reported. A seed that crashes is rerun with the same
seed and the rerun disclosed.

## Missing data

An arm that fails to train is reported as failed, not dropped. If Gemma fails at layer 16 the
model is reported as unresolved rather than substituted with another layer, since a site chosen
after seeing a failure is not the registered site.

## Other planned analysis

Exploratory and labelled as such: per-direction interchange ablation — zero one direction, re-run
the intervention, record the drop — which is available only because the directions are explicit.
No prediction is registered for it.

## Context and additional information

This registers a methods claim, not a discovery about a model. The negative outcomes are
publishable: H1 failing means a rank-constrained nonlinear readout buys nothing over a linear one,
which is a result about the whole family of nonlinear alignment maps. H4 failing retires the
feature-level interpretation for sparse causal latents, which several drafts in these repositories
currently assume.

## Log

- Created. Not frozen until committed; the commit SHA is the registration.
