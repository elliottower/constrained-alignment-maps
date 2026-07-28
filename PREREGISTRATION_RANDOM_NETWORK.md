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
