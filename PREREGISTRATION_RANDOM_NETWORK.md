# Pre-registration: the randomly initialised network control

**Frozen** 2026-07-28. Written before any result from this script existed: two
earlier launches were cancelled and wrote nothing, and both output directories
(`random_network_control/random_init/k1`, `random_network_control/pretrained/k1`)
are empty.

**Script:** `experiments/random_network_control.py`
**SHA-256:** `887ff8f2693e20319375430cf79320af3277881e0596a5bea1e2cc28d12e8675`

## Why this is being run

\citet{sutter2025nonlinear} show that unconstrained nonlinear alignment maps
reach 100% interchange intervention accuracy on **randomly initialised** language
models, which is the sharpest form of their claim that unrestricted causal
abstraction is uninformative. The paper argues that adding generative constraints
to the same interchange objective removes this failure mode.

That argument has not been tested against their actual setup. The existing
"untrained" control is an untrained *probe* on a trained model, which is a
different claim. No experiment in this line of work runs a constrained alignment
map on a random network.

This is the experiment most able to falsify the paper.

## Design

Two arms, identical in every respect except the base model's weights:

- **pretrained** — GPT-2 as released
- **random** — GPT-2 with every parameter re-initialised (`randomize_weights_`:
  normal with the config's `initializer_range` for tensors of rank $> 1$, zeros
  for the rest), preserving architecture, tokenizer, and hook names

Task: indirect object identification, layer 10, $k = 1$. Methods: NL-DAS and the
label-conditional structured VAE, plus DAS and delta-PCA as they appear in the
existing comparison. Reported: interchange accuracy and diversity ratio $\rho$.

**Target definition on the random arm.** A random model is correct on nothing, so
pairs cannot be filtered on the model already answering correctly. The
counterfactual target is set by the IOI task rather than by the model's outputs
(`require_correct=False`). This is the only coherent choice, and it is what makes
Sutter et al.'s result meaningful: the alignment map is asked to make a network
that cannot do the task produce algorithm-consistent outputs.

## Predictions

**Primary (H1).** On the random arm, NL-DAS reaches IIA $\geq 0.90$.
This reproduces Sutter et al. in our setup and validates the harness. If it
fails, the harness is not testing what we think and no other result from this
run is interpretable.

**Primary (H2).** On the random arm, the label-conditional structured VAE reaches
IIA $\leq 0.30$, and in any case at least $0.50$ below NL-DAS on the same arm.
There is no causal structure in a random network to recover, and the
reconstruction term should prevent the encoder from fabricating one.

**Secondary (H3).** On the random arm the structured VAE's diversity ratio stays
comparable to its pretrained value, showing that low accuracy comes from the
absence of a recoverable variable rather than from a collapsed decoder.

**Secondary (H4).** On the pretrained arm both methods reproduce the existing
IOI numbers within $0.05$ (NL-DAS $\approx 1.000$ at $\rho \approx 0.05$;
structured VAE $\approx 1.000$ at $\rho \approx 0.83$).

## Decision rule, fixed in advance

| Outcome | Action in the paper |
|---|---|
| H1 and H2 both hold | The constraints exclude the failure mode Sutter et al. identify. Report as the paper's most direct evidence, in the Results section, with the prediction attributed to this pre-registration. |
| H1 holds, **H2 fails** (structured VAE also reaches high accuracy on a random network) | **The paper's central claim is not supported.** The constrained map is vacuous by Sutter et al.'s own test. Report it, withdraw the claim that constraints exclude vacuity, and reframe the paper around what the diversity ratio and distribution-shift results still show. Do not submit the current framing. |
| H1 fails | Harness problem. Diagnose before interpreting anything else; report nothing from this run. |
| H2 holds but H3 fails (low accuracy with collapsed $\rho$) | Low accuracy may reflect a degenerate encoder rather than absent structure. Report as ambiguous rather than as support. |

The failure case is stated first because it is the one that matters. This
experiment is worth running only if we are willing to report it against
ourselves.

## Not being varied

Layer, $k$, task, pair count, training epochs, all method hyperparameters. The
random arm differs from the pretrained arm in the base weights and in the pair
filter, and in nothing else.

---

## Outcome, first attempt (recorded 2026-07-28, after unblinding)

Run: IOI, layer 10, $k = 1$, NL-DAS trained for 200 steps (script default).
Results at `results/random_network_control/{pretrained,random_init}_ioi.json`.

| Method | Pretrained IIA | Pretrained $\rho$ | Random IIA | Random $\rho$ |
|---|---|---|---|---|
| DAS | 0.194 | 1.001 | 0.000 | 1.000 |
| NL-DAS | 1.000 | 0.037 | 0.722 | 1.585 |
| NL-DAS + recon | 0.850 | 0.248 | 0.433 | 1.172 |
| Structured VAE | 1.000 | 0.838 | 0.000 | 0.672 |

**H4 confirmed.** The pretrained arm reproduced the existing IOI numbers within
$0.05$ on both methods, from a fresh run on a separate app.

**H1 failed.** NL-DAS reached $0.722$ on the random arm against a threshold of
$0.90$. Per the decision rule fixed above, **nothing else from this run is
interpretable**, and no result from it is reported in the paper.

H2 would have passed ($0.000$, and $0.722$ below NL-DAS against a required gap of
$0.50$). It is not being claimed, because the harness-validation hypothesis it
depends on did not hold. Recording this explicitly: the favourable result was
available and was not used.

**Diagnosis.** NL-DAS ran for 200 optimisation steps. Reaching high IIA on a
network with no structure to exploit is an optimisation result, and 200 steps is
plausibly insufficient. Its diversity ratio on the random arm is $1.585$, against
$0.037$ on the pretrained arm; a value above 1 means intervened activations are
more dispersed than natural ones, which is not the lookup-table signature and is
consistent with an unconverged map.

**Amendment.** The random arm will be re-run with NL-DAS at 2000 and 5000 steps,
with H1 evaluated as a convergence check across that sweep. All other design
choices, hypotheses, and decision rules above are unchanged. If NL-DAS plateaus
below $0.90$ at higher step counts, that is a finding about the harness and must
be reported as such before the constrained-versus-unconstrained comparison is
interpreted.

## Amendment 1 — NL-DAS convergence sweep (2026-07-28, before results)

`nldas_steps` threaded through the entry point so it can be varied. No other
change to the experiment.

**Script SHA-256 after amendment:**
`3123689540df759d8d4cd1fce58ebd64780ef0042cf17efd6a1f7bfdd06a2101`

Runs: random arm at `nldas_steps` $\in \{2000, 5000\}$, pretrained arm at 2000
as a matched reference. Output directories are tagged by step count so nothing
overwrites the first attempt.

**H1 is now evaluated as a convergence check.** It passes if NL-DAS reaches
IIA $\geq 0.90$ on the random arm at either step count. If NL-DAS plateaus below
$0.90$ at 5000 steps, that is reported as a limitation of this harness, and the
constrained-versus-unconstrained comparison on the random arm stays uninterpreted.

H2, H3, H4 and the decision table are unchanged.

**Step-count disagreement.** If NL-DAS converges at one step count and not the
other, the higher step count governs, and the discrepancy is itself reported as
an optimisation-stability finding. This is fixed now so the call is not made
after seeing which way it went.

**The $0.30$ threshold is one criterion, not two.** A reviewer may argue that the
structured VAE achieves high accuracy by routing the answer through its
supervised causal latent rather than through the model's computation, which would
be the same lookup-table failure it attributes to NL-DAS. The random arm tests
this directly, because a randomly initialised network contains no causal
structure and any accuracy above chance there is leakage.

This is the same measurement H2 already specifies, read two ways. On the random
arm: IIA $> 0.30$ means label leakage and the corresponding claim is withdrawn
regardless of what the pretrained arm shows; IIA $\leq 0.30$ means H2 holds and
there was no structure to recover. There is no outcome on which the two readings
disagree, and we are not treating a single number as independent support for two
arguments.

## Amendment 2 — diversity ratio: grouped metric added (2026-07-28, before re-run)

**Provenance.** This change was prompted by a code audit comparing every displayed
equation in the manuscript against its implementing function, not by a result
looking wrong. The audit was triggered by the definition mismatch itself. The
$\rho = 8.05$ value on the random arm was already recorded before the audit and is
not the reason for the change; recording this because a metric redefined after
seeing an anomalous number would otherwise be indistinguishable from one redefined
to obtain a better one.

**The mismatch.** The manuscript defines
$\rho = \mathbb{E}_{y_s}[\mathrm{std}(h'_{y_s})] / \mathbb{E}_{y_s}[\mathrm{std}(h_{y_s})]$,
grouped by source label. `compute_diversity_ratio` computes a global standard
deviation over all evaluation examples with no grouping. The stated interpretation
("$\rho \approx 0$ indicates a lookup table, all intervened activations for the
same $y_s$ collapse to a single point") is the within-label reading and does not
follow from the global quantity.

**Why the grouped version is primary.** A lookup table maps every base sharing a
source label to one activation. It therefore has zero within-label spread and
undiminished across-label spread. The global metric sums both, so its numerator is
inflated by exactly the variation a lookup table preserves, making it prone to
false negatives for the failure mode it is meant to detect.

**Consequence for results already recorded.** Global $\rho$ remains valid as a
conservative measure. NL-DAS reaching $\rho = 0.05$ on a metric biased toward
missing collapse is stronger evidence than the grouped value would be, not weaker.
Previously recorded global values are retained and reported as secondary.

**Change.** `compute_diversity_grouped` is added, returning `rho_within` together
with its numerator (`iv_std_within`), denominator (`nat_std_within`), and the
number of label groups meeting the minimum size. Both metrics are reported for
every method. The global function is unchanged. No training procedure, objective,
architecture, or evaluation protocol is altered: this is an evaluation-time metric
only.

**Minimum group size, fixed now.** Within-label standard deviation is undefined
for singleton groups and unstable for very small ones. Labels with fewer than
five evaluation examples are excluded from both numerator and denominator, and
each result reports `n_groups_kept` alongside `n_groups_dropped`. Tasks where
most labels are dropped yield an uninformative `rho_within`: capitals has 182
classes over 190 examples and is expected to fail this check outright, in which
case the global metric governs for that task and we say so rather than reporting
a ratio computed from a handful of groups.

**Degenerate-denominator prediction, fixed now.** On the random arm the per-label
denominator may itself be degenerate, since a network that cannot perform the
task gives source labels little meaning. The comparison is **per method and per
arm**: for each method, if `nat_std_within` on the random arm falls below one
quarter of that same method's `nat_std_within` on the pretrained arm,
`rho_within` is reported as uninformative for that method on that arm and the
global metric governs there. Per-method is specified because pretrained $\rho$
differs by more than an order of magnitude across methods (NL-DAS $0.05$ against
the structured VAE $0.838$), so an aggregate threshold would mean different
absolute values depending on which method it was read against.

**Persistence, added now.** Raw intervened activations, natural activations, and
source labels are written to disk for every method call. Any future metric
question is then a re-score rather than a re-run. This has no effect on the
experiment and is recorded here only so the change is on file.

**Script SHA-256 after Amendment 2:**
`f47b40924bf02a0b48a620c51b00636b82831fe518945eeaa30e750a31ef612e`

The 200-step and 2000-step runs already recorded were executed under the previous
SHAs (`887ff8f2...` and `3123689540df...` respectively) and are not re-attributed.
The 5000-step run currently in flight was launched under `3123689540df...`; if its
H1 verdict stands, it is reported with global $\rho$ only, and grouped $\rho$ comes
from the re-run.

## Outcome, convergence sweep (recorded 2026-07-28, after unblinding)

Results at `results/random_network_control/`. Global $\rho$ only; the run
predates Amendment 2, so grouped $\rho$ is not available for these numbers.

| NL-DAS steps | Arm | NL-DAS IIA | NL-DAS $\rho$ | Structured VAE IIA | Structured VAE $\rho$ |
|---|---|---|---|---|---|
| 200 | random | 0.722 | 1.585 | 0.000 | 0.672 |
| 2000 | random | 0.906 | 8.050 | 0.000 | 0.705 |
| 5000 | random | **0.950** | 3.881 | **0.000** | 0.712 |
| 200 | pretrained | 1.000 | 0.037 | 1.000 | 0.838 |
| 2000 | pretrained | 1.000 | 0.055 | 1.000 | 0.834 |

**H1 confirmed.** NL-DAS reaches $0.950$ at 5000 steps and $0.906$ at 2000,
both above the $0.90$ threshold. The two step counts agree, so the
higher-governs rule is not invoked. Accuracy is monotone in step count
($0.722 \to 0.906 \to 0.950$), supporting the recorded diagnosis that the
first attempt was undertrained rather than mis-specified.

**H2 confirmed.** The structured VAE reaches $0.000$ on the random arm at every
step count, against a threshold of $\leq 0.30$, and is $0.950$ below NL-DAS
against a required gap of $0.50$.

**H3 confirmed.** Structured VAE $\rho$ on the random arm is $0.712$ against
$0.834$ pretrained. The near-zero accuracy is not accompanied by a collapsed
decoder, so it reflects absent structure rather than encoder degeneracy.

**H4 confirmed.** The matched pretrained arm reproduces the reported IOI numbers
at both step counts.

**Reading, per the decision table fixed in advance.** The constraints exclude the
failure mode \citet{sutter2025nonlinear} identify. This is also the pre-committed
answer to the leakage objection: a randomly initialised network contains no
causal structure, so accuracy above chance there is leakage, and the structured
VAE scores zero. Had it scored above $0.30$ the claim would have been withdrawn.

**Unresolved and reported as such.** NL-DAS $\rho$ exceeds 1 on the random arm at
every step count ($1.585$, $8.050$, $3.881$), and NL-DAS+recon reaches $1.995$ on
the *pretrained* arm at 2000 steps, so the effect is not confined to random
networks. The manuscript currently interprets only $\rho \to 0$. A reading for
$\rho \gg 1$ is required before publication and is not attempted here.

## Amendment 3 — two diagnostic runs, neither confirmatory (2026-07-28, before results)

**H1 through H4 are settled** on the sweep reported above and are not revisited.
Both runs described here investigate the unexplained $\rho \gg 1$ pattern. Neither
re-tests an IIA threshold, and no outcome of either changes the status of any
hypothesis. This is stated so that a reader encountering three random-arm runs
knows which is authoritative: the 5000-step run above.

**Run A: is $\rho = 8.05$ reproducible?** On the random arm, global $\rho$ for
NL-DAS is non-monotonic in optimisation steps: $1.585$, $8.050$, $3.881$ at 200,
2000 and 5000. Every cell is a single run, so a peak at 2000 and noise around a
settled value are not currently distinguishable. Adding step counts does not
separate them; replicates at a fixed step count do. Three seeds at 2000 steps.

Fixed in advance: if the three seeds span a range narrower than $2.0$ in $\rho$,
the value is treated as reproducible and the non-monotonicity is reported as a
property of the optimisation. If the range exceeds $2.0$, $\rho$ on the random arm
is reported as unstable across seeds and no step-count trend is claimed from the
three-point sweep.

**Run B: grouped $\rho$.** Re-run under the Amendment 2 script so `rho_within`
is available with its numerator and denominator separated. This may dissolve the
$\rho \gg 1$ pattern, because within-label spread cannot be inflated by
across-label variation the way the global metric can. If it does not, the paper
reports $\rho \gg 1$ as unexplained rather than supplying a mechanism after the
fact.

**Neither run is a condition for reporting the random-network result.** If both
fail or are inconclusive, the section is written with $\rho \gg 1$ stated as an
open observation.

### Amendment 3 implementation note (2026-07-28, before launch)

Run A requires seed control, which the IOI path did not have: seeds existed only
for grokking model training. A `map_seed` argument now sets the torch seed before
alignment-map initialisation and training. Pair generation retains its own fixed
generator (`random.Random(42)`), so replicates differ in the map and not in the
evaluation data, which is what makes them controlled replicates rather than
different experiments. Output directories are tagged by seed.

No hypothesis, threshold, or decision rule changes. Run A is three seeds at 2000
steps on the random arm; Run B is the 5000-step random arm and its matched
pretrained arm, both under the Amendment 2 script so grouped rho is recorded.

**Script SHA-256 for Runs A and B:**
`d0a904d1d5677c3d897538585aeea256190150cdd9c481d6b1db5969a3d2309e`
