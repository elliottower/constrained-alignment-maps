# Revision spec: constrained_interchange_training_v3.tex -> v4

Written 2026-07-30. **No edits applied.**

**Supersedes `PAPER_REVISION_SPEC_V13.md`**, which targeted
`main_v12_blinded.tex`. That was the wrong base. `constrained_interchange_training_v3.tex`
is more recent in intent, a quarter the length, already calibrated, already names
the method consistently ("the constrained map", never "pi-SAE"), and already
contains the section that today's run resolves. Work from it.

Line numbers refer to `paper/constrained_interchange_training_v3.tex`. Copy to
`_v4.tex` and edit the copy.

---

## 0. BLOCKER — resolve before touching anything else

**The draft contradicts itself about what its own method is.**

The abstract and introduction say the two maps share the interchange objective:

> "two nonlinear alignment maps trained on the same interchange objective,
> differing in whether that objective is optimised subject to a variational
> autoencoder's reconstruction and regularisation terms" (abstract)

> "The constrained map **adds the interchange term** to a variational
> autoencoder's reconstruction, regularisation, and supervision terms, so both
> maximise interchange accuracy on identical data. Holding the objective fixed
> isolates the constraints as a group" (intro, 88-95)

But `tab:shift` (246-249) lists two distinct rows:

```
Structured VAE            1.000  0.83 | 0.001  0.78 | 0.963  0.51
  + interchange loss      1.000  0.83 | 0.001  0.80 | 0.993  0.53
```

So either the headline "Structured VAE" column throughout is the **non**-interchange-trained
arm — in which case the abstract and intro misdescribe the paper's central
comparison — or it is interchange-trained and the second row is redundant.

**This is not a wording problem. It decides whether the paper survives.**

Measured 2026-07-30, indirect object identification, k=1, 180 held-out pairs,
random-subspace floor 0.000 over five draws:

| arm | pretrained | randomly initialised |
|---|---|---|
| `pi_sae` (ELBO only, no interchange term) | 1.000 | **0.000** |
| `pi_sae_e2e` (interchange term added) | 1.000 | **0.961** |

- **If the paper's method is the non-interchange arm**: it passes the control,
  §`sec:randomnet` is filled with a positive result, and the abstract and
  introduction must be rewritten, because "both optimise the same interchange
  objective" is then false and the controlled comparison it advertises does not
  exist.
- **If the paper's method is the interchange-trained arm**: the draft's own
  pre-stated condition fires — "If the structured VAE also reaches high accuracy
  on the random arm, the reading given in \S\ref{sec:vacuity} is not supported
  and the paper needs rewriting around that" (320-327) — and the paper is
  rebuilt around the finding below.

**Action:** determine, from the code path that produced each table, which arm
each "Structured VAE" number came from. Until that is settled, do not edit prose.

---

## 1. The finding that reframes the paper either way

`pi_sae` and `pi_sae_e2e` share a builder (`_build_pi_sae`). Identical
architecture, identical data, identical seed. The only difference is whether the
interchange cross-entropy is in the loss. One scores 0.000 on a network with no
computation; the other 0.961.

The non-interchange arm cannot be made vacuous by training longer, because its
loss contains no term rewarding counterfactual outputs through the model.

So the axis is not expressivity, and it is not constraint. It is **whether the
map is trained on the metric being reported**. That extends
\citet{sutter2025nonlinear} rather than patching it, and it is a stronger claim
than the draft currently makes.

It also costs the draft its stated design. "Holding the objective fixed isolates
the constraints as a group" is exactly what the result refutes: hold the
objective fixed and the constraints do not save you.

---

## 2. Number audit

| table | status | action |
|---|---|---|
| `tab:vacuity` (119-146) | single runs, caption admits it. IOI k=1 Structured VAE 1.000 and rho_NL 0.05 both reproduce today (1.000, 0.04) | seed it; re-run the DAS column under MIB's configuration |
| `tab:ksweep` (166-183) | single runs. **DAS reaches 1.000 at k=8 here, against 0.722 at k=8 and 0.919 at k=16 in main_v12's sweep.** Two drafts, incompatible numbers for the same experiment | trace which run produced each before either is reported |
| `tab:nlp` (197-219) | six-task run in flight covers this | needs arm labels: is this column interchange-trained or not? |
| `tab:shift` (236-256) | single runs; the only table that labels arms correctly | seed it; keep the two-row structure, it is the right one |
| `tab:ablation` (282-300) | single runs, **three empty cells** (`\TODO{--}`, Plain VAE on Add/Mult/Quartic), plus the sparsity confound | run `experiments/sparsity_vs_overcompleteness_ablation.py`; fill the holes |
| `sec:randomnet` (317-327) | `\TODO{RUNNING}` | **fillable now** from `experiments/results/random_network_control_e2e/` |

The draft is already honest about the k-sweep discrepancy's cousin: limitations
(371) notes DAS on indirect object identification appears at both 0.183 and 0.19
because values come from separate runs. The `tab:ksweep` conflict is larger and
needs resolving rather than noting.

---

## 3. Literature that must be added

Three papers landed after this draft. None is cited.

**Grant, Han, Tartaglini & Potts, ICLR 2026** (arXiv:2511.04638). Divergent
representations from causal interventions, from DAS's own group. They prove
interventions go off-manifold, separate harmless (behavioural null-space) from
pernicious divergence, and mitigate with a Counterfactual Latent loss reaching
EMD 0.007 at IIA 0.9988. **Consequence:** the lookup-table mechanism in the
introduction (79-86) is their diagnosis, and the draft must cite it there rather
than present it as its own. The `NL-DAS+r` reconstruction-penalty result becomes
a statement about *generic* penalties, since their targeted penalty escapes the
trade.

**Schiffman, ICML 2026 MI Workshop** (arXiv:2602.22600). Invariant algorithmic
cores; modular addition cores crystallise at grokking; projector overlap
0.02-0.04 across independent models with CCA ~0.99. **Consequence:** cite for
grokking geometry; do not make geometric claims about modular addition.

**Méloux, Maniu, Portet & Peyrard** (arXiv:2502.20914). Non-identifiability;
one algorithm aligns with different subspaces. **Verify the venue** — Grant et
al.'s bibliography says a 2024 MI workshop, other sources say ICLR 2025.

---

## 4. Naming

The draft already does this correctly: "the constrained map" (24 uses),
"structured VAE", never "pi-SAE". Keep it.

For provenance, introduce a **new code key** so post-correction results are
self-evidently fresh:

- code: `lcp_vae` (label-conditional partitioned VAE), replacing `pi_sae`
- code: `lcp_vae_interchange`, replacing `pi_sae_e2e`
- old keys stay readable so historical result files remain interpretable

Paper prose keeps full words, no acronym, per the standing rule against invented
acronyms.

**Drop the L_1 term from the method** if the sparsity ablation confirms it is
inert — the draft's own TODO (309-315) already instructs this. Without it the
method is three components (label-conditional prior, causal/nuisance partition,
overcomplete causal latent), each doing measurable work, and it stops inviting
the sparse-autoencoder comparison.

---

## 5. Structure

The draft is already close to the right shape for a machine-learning conference.
Keep the section order. Two changes:

- **`sec:randomnet` moves up**, immediately after `sec:vacuity`. It is the
  control that makes the rest credible and the only claim here that nobody else
  has published. It should not sit sixth.
- **`tab:ablation` and the sparsity ablation move to an appendix**, with a
  one-line summary in the main text. They are supporting detail, not the claim.

Main text: introduction -> accuracy and diversity -> randomly initialised model
-> distribution shift -> subspace dimension -> language model tasks ->
discussion. Appendix: constraint ablation, sparsity, per-task detail,
hyperparameters, pre-registrations.

---

## 6. What to run

1. **Nothing** for `sec:randomnet` — data exists, fill it.
2. **Seeds** on `tab:vacuity`, `tab:ksweep`, `tab:shift`. Every headline table is
   single-run and this project has lost two claims to seeding already.
3. **Arithmetic DAS under MIB's configuration.** The run in flight is GPT-2 only.
4. **The sparsity ablation**, to settle the L_1 question and the three empty
   ablation cells.
5. **NL-DAS + Counterfactual Latent loss.** Grant et al.'s penalty applied to the
   map that is actually vacuous. Applying it to linear DAS tests nothing, since
   linear DAS was never vacuous. If CL-regularised NL-DAS stays vacuous, the
   constrained map does something a penalty cannot; if it does not, the claim
   narrows to dimension efficiency, which no linear method reaches regardless.

---

## 7. Order of work

1. Resolve §0. Nothing else is safe until it is settled.
2. Fill `sec:randomnet` from existing data.
3. Trace the `tab:ksweep` DAS discrepancy.
4. Wait for the six-task run; update `tab:nlp` with arm labels.
5. Launch seeds, the sparsity ablation, and the CL arm.
6. Add citations, rewrite abstract, introduction and discussion **last** — their
   framing depends on §0 and on item 5.
