# Pre-registration: does the $L_1$ penalty do anything?

**Frozen** 2026-07-28. Written before any result from this script existed: an
earlier launch was cancelled during the first cell and wrote nothing, and the
output directory `sparsity_vs_overcompleteness_ablation/k1` is empty.

**Script:** `experiments/sparsity_vs_overcompleteness_ablation.py`
**SHA-256:** `61af4dced184b1b47f6500dc47499700f8bc44497bcdb589134309eea1cb91b7`

## Why this is being run

The paper describes its alignment map as combining a label-conditional prior, a
causal/nuisance latent partition, and an eightfold-expanded causal latent under
an $L_1$ penalty. The claim that sparsity matters rests on the published 2×2
ablation, in which the non-expanded model reaches IIA $= 0.00$ on grokked
arithmetic and the full model reaches $1.00$.

That comparison is confounded. `build_pi_vae` takes no expansion factor while
`build_pi_sae` uses eight, so the two conditions differ in **expansion and $L_1$
simultaneously**. The implemented coefficient is $\lambda = 10^{-3}$ applied to
`mu_c.abs().mean()`, which is weak enough that the expansion could account for
the entire effect.

## Design

Grokked modular addition, $p = 113$, $k = 1$, all other settings identical to the
run that produced the published numbers. The prior and the causal/nuisance
partition are held fixed. Two factors are crossed:

- expansion factor $\in \{1, 8\}$
- $\lambda \in \{0,\; 10^{-3},\; 1\}$

$\lambda = 10^{-3}$ is the implemented value; $\lambda = 1$ is the value the
paper's methods section previously claimed; $\lambda = 0$ removes sparsity.

## Predictions

**Primary (H1).** Expansion 8 with $\lambda = 0$ reaches IIA $\geq 0.95$.
This is the outcome we consider more likely. It would mean the $L_1$ term is
inert and the expansion carries the effect attributed to sparsity.

**Secondary (H2).** Expansion 1 with $\lambda = 10^{-3}$ reaches IIA $\leq 0.10$,
matching the published non-expanded result.

**Tertiary (H3).** $\lambda = 1$ degrades IIA relative to $\lambda = 10^{-3}$ at
expansion 8, since a penalty a thousand times larger should suppress the causal
latent.

## Decision rule, fixed in advance

| Outcome | Action in the paper |
|---|---|
| H1 holds (expansion 8, $\lambda = 0$ reaches $\geq 0.95$) | Remove $L_1$ from the method description and from the mechanism account. Describe the causal latent as overcomplete, not sparse. Report this ablation. |
| H1 fails (expansion 8, $\lambda = 0$ falls below $0.95$) | Sparsity is load-bearing at $\lambda = 10^{-3}$. Keep the current description and report the ablation as support. |
| Expansion 1 with any $\lambda$ reaches $\geq 0.95$ | Expansion is not required either. Both terms come out of the mechanism account and the constraint story rests on prior and partition alone. |

The paper reports this ablation whichever way it resolves. The result changes
what the method is called and what mechanism is claimed, so it is not a null
result that can be dropped.

## What would invalidate the run

Test accuracy on the grokked model below $0.99$, or a cell failing to train.
Either voids that cell rather than being reported as a low IIA.

## Not being varied

Prior type, partition, $\alpha$, learning rate, epochs, hidden width, $k$, the
operation, the modulus, the seed handling. Single runs per cell, as in the
published ablation; run-to-run variation is not characterised here and the
writeup will say so.

---

## Outcome (recorded 2026-07-28, after unblinding)

Run: grokked modular addition, $p = 113$, $k = 1$, test accuracy $0.9996$.
Results at `results/sparsity_ablation_addition.json`.

| Expansion | $\lambda$ | IIA | $\rho$ |
|---|---|---|---|
| 1 | 0 | 0.017 | 0.837 |
| 1 | $10^{-3}$ | 0.033 | 0.912 |
| 1 | 1.0 | 0.017 | 0.876 |
| 8 | 0 | **1.000** | 0.987 |
| 8 | $10^{-3}$ | 1.000 | 0.989 |
| 8 | 1.0 | 1.000 | 0.994 |

**H1 confirmed.** Expansion 8 with $\lambda = 0$ reached $1.000$ against a
threshold of $0.95$. The $L_1$ penalty is not responsible for the effect.

**H2 confirmed.** Expansion 1 with $\lambda = 10^{-3}$ reached $0.033$ against a
threshold of $0.10$.

**H3 refuted.** $\lambda = 1$ was predicted to degrade IIA relative to
$\lambda = 10^{-3}$ at expansion 8. It did not: IIA is $1.000$ at all three
values of $\lambda$, and $0.017$–$0.033$ at expansion 1 for all three. $\lambda$
has no measurable effect anywhere in the grid, across three orders of magnitude.
This is a stronger result than H1 anticipated — the penalty is inert rather than
merely weak.

**Action taken**, per the decision rule fixed above: sparsity removed from the
method description and from the mechanism account; the causal latent is described
as overcomplete. The $\lambda$ term remains in the reported loss because it was
present in the runs, annotated as ablated with no effect.
