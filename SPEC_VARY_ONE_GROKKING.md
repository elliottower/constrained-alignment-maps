# SPEC — VARY-ONE PCA ON GROKKED MODULAR ARITHMETIC

Port of `SPEC_VARY_ONE_PCA.md` from indirect object identification to grokked
modular arithmetic. Same method, better setting. This document supersedes that
one for execution; keep both, since the language-task version is still the
generalisation test.

Method from Shai et al., *Transformers learn factored representations*, §H.1.1.

---

## 1. Why arithmetic rather than a language task

Four reasons, in descending order of importance.

**The vary-one design is exhaustive rather than sampled.** For $(a+b) \bmod p$
at $p = 113$, holding $b$ fixed and sweeping $a$ covers the input space
completely: $C = 113$ configurations by $M = 113$ realisations, every cell
populated, no template sampling and no confounds from lexical choice. Shai et al.
needed synthetic generalised hidden Markov models to obtain this. Modular
arithmetic supplies it for free, which is the whole reason their method ports
cleanly here and awkwardly to IOI.

**The design risk that could have sunk the IOI version becomes a built-in
control.** `SPEC_VARY_ONE_PCA.md` §3 correctly worries that token identity is
encoded at random initialisation, so vary-one finds a subspace in an untrained
network and the contrast with DAS collapses. In arithmetic the variables have
*different computational status* and can be separated by construction:

| variable | represented at random init? | role |
|---|---|---|
| $a$ (input token) | yes, it is an embedding | positive control — the method must find it |
| $b$ (input token) | yes | positive control |
| $(a+b) \bmod p$ (the result) | **no** — requires computation | the actual test |

So the confound stops being a threat and becomes a validation: vary-one *should*
recover $a$ in a random network, and *should not* recover the result. A method
that finds both, or neither, has failed in a diagnosable way.

**There is analytic ground truth.** Grokked modular addition computes through
Fourier features: the residual stream carries $\cos(2\pi k a/p)$ and
$\sin(2\pi k a/p)$ for a small set of key frequencies. The recovered subspace can
therefore be scored against a *known* answer, not only against what DAS found.
No language task offers this. `analyze_fourier_alignment` in
`experiments/k1_vae_vs_das.py` already computes the correlation.

**The empirical null already exists.** §2.2 of the original spec requires a
random-subspace baseline. `experiments/subspace_overlap_null.py` provides it, and
provides it *empirically* rather than as the analytic $k/d$ — which matters,
because the analytic value is what produced this project's retracted
"below chance" claim.

---

## 2. The three designs, and how they isolate the result

The result is a *function* of the inputs, so strict vary-one does not apply to
it: there is no configuration of "everything else" to hold fixed while the result
moves. This is the one real adaptation the port requires, and arithmetic makes it
exact.

| design | held fixed | varied | subspace recovered |
|---|---|---|---|
| **A** | $b = b_0$ | $a$ over all $p$ | $S_a \oplus S_{\text{result}}$ — both move together |
| **B** | $a = a_0$ | $b$ over all $p$ | $S_b \oplus S_{\text{result}}$ |
| **C** | $a + b \equiv s$ | $a$, with $b = s - a$ | $S_a \oplus S_b$ — the result is **constant** |

Design C is the key construction and it exists only because addition is a group
operation: sweeping $a$ with $b = s - a$ moves both inputs while pinning the
result. The result subspace is then what A and B share and C lacks:

$$S_{\text{result}} \;\approx\; (S_A \cap S_B) \ominus S_C .$$

Intersections and complements of subspaces are read off principal angles, which
`degeneracy_decomposition.py:289` already computes. Report the principal angles
themselves, not only a scalar distance, since the claim is about which directions
are shared.

**Predicted contrast, and it is sharp.** On a grokked model, the result subspace
recovered this way should align with the Fourier directions
$\cos(2\pi k(a{+}b)/p), \sin(2\pi k(a{+}b)/p)$. On a randomly initialised model,
$S_A \cap S_B$ should be empty beyond the null, because nothing computes the sum.

---

## 3. Pilot gate — run before anything else

Ten minutes, one grokked addition model at $p = 113$, one hook point.

1. Run design A. Confirm the recovered subspace exceeds the empirical null.
   *If it does not, the method has not ported and everything below is void.*
2. Run design C. Confirm it also exceeds the null, and that it is **not** the
   same subspace as A — if $d_{\mathrm{Gr}}(S_A, S_C)$ is at null levels, the
   designs are not separating anything and §2's construction fails.
3. Compute $S_A \cap S_B$ and check it is non-trivial.

Gate: all three pass, or the spec is revised before any compute is spent.

---

## 4. What to measure

### 4.1 Recovery

Designs A, B, C at $k \in \{1, 2, 4, 8, 16\}$, five grokked seeds. Mean-centre
**within configuration** before pooling — centring globally destroys the result,
and it is the step most likely to be got wrong on a re-implementation. Report
explained variance per component so it is visible whether the subspace is
genuinely low-dimensional.

### 4.2 Geometric comparison, against four references

For every recovered subspace, on $\mathrm{Gr}(k, 128)$:

- the **DAS** subspace at matched $k$, fitted with `experiments/das.py` under
  MIB's configuration;
- the **weight-space** subspace from `ff_svd_subspace`;
- the **Fourier** ground-truth subspace;
- the **empirical null** from `subspace_overlap_null.py`.

Four references rather than one. Agreement with any single reference is weak
evidence; agreement with DAS *and* Fourier *and* weight space, against the null,
is triangulation.

### 4.3 Causal test

Variance is not mediation. Every recovered subspace goes through the existing
interchange harness and reports **strict IIA** beside the DAS and structured-VAE
columns. A subspace that carries variance but fails interchange is a probe result
and must be labelled as one.

### 4.4 Admissibility

Vary-one is **admissible by construction** under
`PREREGISTRATION_RECONSTRUCTION_CRITERION.md`: projection onto the subspace with
the residual carried is exactly MIB's featurizer form, $f, \varepsilon =
\mathrm{feat}(x)$ and $\mathrm{inv}(f, \varepsilon) = fr + \varepsilon$, which
reconstructs to numerical precision. Report loss recovered anyway — a metric that
must return 1.0 is a check on the implementation.

---

## 5. Pre-registration

Freeze with SHA before running §4, per the existing pattern.

**H1 (primary) — the result subspace is recoverable without optimisation.** On
grokked models, $(S_A \cap S_B) \ominus S_C$ achieves strict IIA above its task's
empirical floor by more than 0.05, at some $k \le 16$.

**H2 — convergence with DAS.** $d_{\mathrm{Gr}}$ between the recovered result
subspace and the DAS subspace falls below the 2.5th percentile of the empirical
null. Two routes, one optimisation-based and one not, landing in the same place.

**H3 — the discriminating control.** On randomly initialised models, designs A
and B still recover subspaces above the null (token identity is present at
initialisation), while the result subspace $(S_A \cap S_B) \ominus S_C$ does not,
and its strict IIA stays within 0.05 of the floor.

*Falsification:* if the result subspace clears the floor on a random network,
vary-one is as vacuous as unconstrained nonlinear DAS. The method's entire
selling point is gone and that gets reported, not reframed. It would also be the
more important finding — vacuity would be a property of subspace selection in
high dimensions generally rather than of optimisation.

**H4 — dimensionality matches the Fourier account.** Explained variance in the
result subspace concentrates in $2F$ components, where $F$ is the number of key
frequencies the model uses, measured independently by `analyze_fourier_alignment`
rather than fitted here. This is sharper than the original spec's "$\le 4$
components", because $F$ is measurable in advance.

**H5 — negative control.** Vary-one on a **non-grokked** model (memorising, test
accuracy below 0.95) recovers no result subspace above the null, while still
recovering $a$ and $b$. Memorisation stores input-output pairs without computing
the sum, so the result should have no dedicated subspace.

Fix before freezing: alpha, correction across the design × $k$ grid, the number
of interchange trials, and the floor definition — which is the empirical
random-subspace floor with an interval, per the reconstruction-criterion
pre-registration, not an analytic chance value.

---

## 6. Where this lands

Not a third competing method. **An optimisation-free check on whether the
subspace DAS finds is real.**

The paper's current claim is that constraining the alignment map removes vacuity.
Vary-one adds a second axis — how the subspace is *selected*, by optimisation
against behaviour or by measurement of variance — orthogonal to whether the map
reconstructs. Both DAS and vary-one are admissible. They differ only in
selection. So if they converge on trained models and vary-one is null on random
ones, DAS's subspace is independently corroborated by a route that cannot overfit
a behavioural target, and the vacuity result is localised precisely to
optimisation.

That is a stronger and more specific claim than the paper currently makes, and on
this task it comes with analytic ground truth to check it against.

---

## 7. What this does not do

- It does not port to natural language on its own. Design C depends on the
  operation being a group action; there is no "hold the answer fixed while
  varying the inputs" construction for IOI. `SPEC_VARY_ONE_PCA.md` remains the
  generalisation test and should be attempted only after H1–H3 resolve here.
- It does not establish uniqueness. Recovering *a* causally effective subspace
  without optimisation does not show it is *the* mechanism — Méloux et al. (ICLR
  2025) applies to this method as much as to DAS.
- It does not replace the model-wide factorisation work, which has no task and no
  controlled variable.
