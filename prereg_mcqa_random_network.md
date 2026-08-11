# Pre-registration: alignment maps on MIB's multiple-choice task, across models

Frozen before any MCQA result exists. The commit SHA of this file is the
registration.

## Why this task

MIB benchmarks multiple-choice question answering on Qwen2.5-0.5B, Gemma-2-2B and
Llama-3.1-8B, and publishes per-model baselines (Table 3c of arXiv 2504.13151).
Their indirect-object task is GPT-2 only (Table 3d), so no configuration of it can
support a claim about more than one model.

The task is also structurally cleaner. Its causal model has
`raw_output <- answer <- answer_pointer` (`simple_MCQA.py:37`), so both target
variables reach the generated text and plain accuracy measures them. The
indirect-object model has `raw_output <- output_token` only, which is why
interchanging `output_position` there is invisible to any text criterion and
requires a fitted logit-difference model.

## Setup, fixed

Everything except the maps is MIB's, at their settings
(`baselines/simple_MCQA_baselines.py:73,109`): `LMPipeline` with no `max_length`,
float16, `training_epoch` 8, `n_features` 16, `regularization_coefficient` 0.0,
`init_lr` inherited from `DEFAULT_CONFIG` (1e-2), MIB's single-token-position
indexer, MIB's checker (`expected in output_text`), MIB's `FilterExperiment`.

Target variable `answer_pointer`, which is XOrder in their Table 3c.

The randomly initialized condition is `AutoModelForCausalLM.from_config`, Hugging
Face's own initializer, handed to `LMPipeline` as a model object. The pretrained
condition keeps only examples the model answers correctly; a randomly initialized
network answers nothing correctly, so no filter applies and pair counts are
reported per condition.

## Arms

| arm | nonlinear | interchange-trained |
|---|---|---|
| `random` | no | no |
| `das` | no | yes |
| `lcp_vae` | yes | no |
| `lcp_vae_interchange` | yes | yes |
| `nldas` | yes | yes |

## A correction carried into this design

Two claims made earlier in this project were wrong and are corrected here before
any measurement.

First, that MIB's output-matching criterion cannot detect a vacuous map. It can
in principle: a map writing into the residual stream can steer the output token,
including in a network that cannot do the task. What was measured on the
indirect-object task is that no arm *did* so — every arm scored exactly 0.000 on
the randomly initialized network under output matching, while the same class of
map reached 0.807 under a two-way logit comparison.

That difference is the point registered below. Vacuity is a property of the
criterion as much as of the map.

## Predictions

**H0 (reproduction).** On the pretrained model, `das` reaches within 15 points of
MIB's published DAS result for this task, model and variable. This is a check
that our harness reproduces theirs; failing it blocks every other reading.

**H1 (the conjunction).** On the randomly initialized network, under MIB's
output-matching criterion, every arm scores at the measured floor. The floor is
the `random` arm in the same condition.

**H2 (criterion dependence, primary).** On the randomly initialized network,
under a two-way criterion — counterfactual answer symbol outranking the base
answer symbol, chance 0.5 — at least one nonlinear interchange-trained arm
(`lcp_vae_interchange` or `nldas`) exceeds 0.6, while `das` and `lcp_vae` do not.

**H3 (pretrained ordering).** On the pretrained model, both interchange-trained
nonlinear arms score at or above `das`, and `lcp_vae` scores at the floor. A map
never trained on interchange should not move the answer.

**H4 (cross-model).** Whatever pattern holds on Qwen2.5-0.5B holds on at least one
further model from MIB's list for this task.

## Decision rule, fixed in advance

| outcome | consequence |
|---|---|
| H0 fails | The harness does not reproduce MIB. Nothing else is reported until traced. |
| H1 and H2 both hold | The headline result: an expressive map is vacuous under a lenient criterion and not under a strict one, on a benchmark task with published baselines. |
| H1 holds, H2 fails | No vacuity under either criterion here. Reported as a null, and the earlier indirect-object finding is reported as task-specific. |
| H1 fails | Some arm beats the floor on a network that cannot do the task, under the strict criterion. That is a stronger vacuity result and is reported as the finding. |
| H3 fails for `lcp_vae` | The reconstruction-only arm is moving the answer without interchange training, which would indicate leakage; blocks reporting until traced. |
| H4 fails | The result is reported as model-specific, with the count stated in the abstract. |

No outcome here is a reason not to report.

## Not varied

MIB's settings above, the arms, the target variable, and the train/test split.
The layer is chosen once per model by a sweep on the pretrained condition and
then held fixed across arms and conditions; the sweep is reported.

## Script

`experiments/mcqa_random_network.py`. A small-`size` smoke run precedes the sweep
and its numbers are not results.
