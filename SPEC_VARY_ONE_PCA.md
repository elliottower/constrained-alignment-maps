# SPEC — VARY-ONE PCA AS A THIRD ALIGNMENT METHOD

Add a **training-free** subspace-identification method to the existing
DAS vs. structured-VAE comparison, and run it through the same harness and the
same pre-registered random-network control.

Motivation: DAS finds a causal subspace by *optimizing* an alignment map against
a behavioural target, which is why unconstrained nonlinear DAS reaches 0.989 on a
randomly initialised network — the search does the work. Vary-one PCA finds a
subspace by *measuring where activations move when you move a variable*. There is
no optimisation, so there is nothing to overfit. If it recovers DAS's subspace on
the real model and returns nothing on the random one, the vacuity problem has a
constructive answer rather than only a regularisation-based one.

Method adapted from Shai et al., *Transformers learn factored representations*
(ICML 2026 Mech Interp Workshop), §H.1.1 "vary-one" analysis. Their setting is
synthetic GHMMs with known factors; the port to real tasks works because task
datasets are already vary-one designs.

---

## 1. The method, precisely

For a task $T$ with controlled variables $V = \{v_1, \ldots, v_N\}$ (IOI:
indirect-object identity, subject identity, template) and a hook point $h$
(layer $\ell$, token position $p$):

**For each target variable $v$:**

1. Sample $C$ **configurations** of the non-target variables
   $V \setminus \{v\}$, held fixed within a configuration.
2. Within each configuration $c$, generate $M$ **realisations** varying only $v$.
3. Run the model, collect residual-stream activations
   $a^{(c,m)} \in \mathbb{R}^{d}$ at hook point $h$.
4. **Mean-centre within configuration:**
   $\tilde{a}^{(c,m)} = a^{(c,m)} - \frac{1}{M}\sum_{m'} a^{(c,m')}$.
   This is the step that makes the method work — it removes all
   between-configuration variance, leaving only variation attributable to $v$.
   Centring globally instead of per-configuration destroys the result.
5. Pool $\{\tilde{a}^{(c,m)}\}$ across all $C \times M$ and run PCA.
6. Top-$k$ principal components span the candidate subspace $S_v^{(k)}$.

Expect roughly **2 useful dimensions per variable**. In Shai et al. the
orthogonal structure lives in the first two PCs only; summed factor dimensions
reach 14 against a union of 12, so orthogonality is approximate.

---

## 2. What the experiment must do

### 2.1 Recover subspaces (no model training)

Run §1 for each IOI variable at the hook points the existing DAS runs already
use. Reuse the hook-point configuration from `experiments/k1_vae_vs_das.py` so
the comparison is like-for-like. Sweep $k \in \{1, 2, 4, 8\}$.

Report explained variance per component, so it is visible whether the subspace is
genuinely low-dimensional or PCA is slicing noise.

### 2.2 Compare geometrically against DAS and the VAE

For each variable and each $k$, compute:

- **Principal angles** between $S_v^{(k)}$ and the DAS-learned subspace at
  matched $k$; and against the structured-VAE subspace.
- **Grassmannian geodesic distance** $d_{\mathrm{Gr}}$ on $\mathrm{Gr}(k,d)$.
  Machinery already exists in the factorization repo's Grassmannian analysis
  code — reuse it rather than reimplementing.
- A **random-subspace baseline**: geodesic distance between $S_v^{(k)}$ and $k$
  random $k$-dimensional subspaces, to establish what "close" means at this $d$.
  Without this baseline the distances are uninterpretable — in high $d$ random
  subspaces are far apart, so any agreement looks impressive.

### 2.3 Test causality, not just variance

**This is the step that makes it an alignment method rather than a probe.**

A subspace found by variance is not thereby causal. Run interchange
interventions restricted to $S_v^{(k)}$ — project the source activation onto
$S_v$, substitute into the base run, measure interchange intervention accuracy —
using the *existing* IIA harness so the numbers are directly comparable to the
DAS and VAE columns.

Report both standard IIA and the **strict IIA** already used in the paper
(argmax actually flips), since the DAS-shifts-mass-without-flipping result is
one of the paper's findings and the same failure mode may appear here.

### 2.4 The random-network control

Repeat §2.1 and §2.3 on a randomly initialised GPT-2, five seeds, matching the
existing pre-registered protocol in `PREREGISTRATION_RANDOM_NETWORK.md`.

**Prediction:** vary-one PCA recovers a subspace (PCA always returns something),
but interchange interventions in it produce IIA at chance, against unconstrained
nonlinear DAS at 0.989.

---

## 3. The design risk that could sink it — read before running

**Token identity is encoded at random initialisation.** Varying the
indirect-object token changes the input embedding, so activations move even in an
untrained network, and vary-one PCA will happily find that subspace. Interchange
interventions in it might then *partially work*, because swapping a token
representation does swap the token — without the network computing anything.

If that happens, the clean contrast with DAS collapses and the control is
uninformative.

**Mitigations, in order of preference:**

1. **Choose a variable that requires computation.** For IOI, target the
   *inhibition* or *name-mover output* variable rather than raw IO token
   identity. A random network cannot compute "which name should be suppressed";
   it can trivially represent "which token is here."
2. **Hook late.** Take activations at a layer after the circuit would have run,
   where surface token identity has been transformed. Early-layer residual stream
   is mostly embedding.
3. **Add a shuffled-label control.** Recompute vary-one with the variable's labels
   permuted across configurations. Any IIA surviving that is surface artifact.

**Run mitigation 1 as a ten-minute pilot before committing to the full protocol.**
If vary-one on the trained model recovers nothing above the random-subspace
baseline for a computed variable, the method does not port and the design fails
cheaply.

---

## 4. Pre-registration

Freeze before running any of §2. Follow the existing SHA-freeze pattern.

**H1 — recovery.** On the trained model, $S_v^{(k)}$ achieves strict IIA
significantly above the random-subspace baseline, for at least one variable at
some $k \le 8$.

**H2 — agreement.** $d_{\mathrm{Gr}}(S_v^{\text{vary-one}}, S_v^{\text{DAS}})$ is
smaller than the random-subspace baseline distance. Two optimisation-free and
optimisation-based routes landing in the same place is the convergence claim.

**H3 — the one that matters.** On randomly initialised networks, vary-one strict
IIA is at chance (predicted $< 0.05$), while unconstrained nonlinear DAS is high
(previously 0.989). **Falsification:** if vary-one exceeds 0.5 on random
networks, it is as vacuous as DAS, the method's central selling point is gone,
and this must be reported rather than reframed.

**H4 — dimensionality.** Explained variance concentrates in $\le 4$ components,
consistent with Shai et al.'s ~2 dimensions per factor. A subspace needing $k=8$
to work is not the factored structure the theory predicts.

State decision criteria numerically before freezing: alpha, correction across the
variable × $k$ grid, and what counts as "at chance" given the number of
interchange trials.

---

## 5. What this does not do

- **It does not replace DST for model-wide factorization.** Vary-one requires
  controlled variation of a known variable. The parcellation work needs factors
  over all 144 heads with no task and no labels; vary-one cannot produce those.
  This is a task-specific method.
- **It does not test gauge covariance.** Separate question, separate spec.
- **It does not establish that the subspace is *the* mechanism** — only that a
  training-free procedure finds a causally effective subspace. Uniqueness is a
  further claim requiring the alternatives to be ruled out.

---

## 6. Where it lands in the paper

Paper A currently contrasts two methods: unconstrained nonlinear DAS (vacuous on
random networks) and the structured VAE (not vacuous). Vary-one PCA is a third
column, and a stronger one if H3 holds, because it is immune by *construction*
rather than by *regularisation* — there is no objective to constrain.

That also sharpens the paper's thesis. The current claim is "constraining the
alignment map fixes vacuity." With vary-one it becomes "vacuity comes from
optimising against a behavioural target at all; two independent escapes exist —
constrain the map, or don't optimise." That is a stronger and more general claim,
and it costs one more column in the existing tables.

If H3 fails, the finding is also worth reporting: it would mean vacuity is a
property of subspace-selection in high dimensions generally, not of optimisation
specifically — which is a bigger problem for the field and a different paper.
