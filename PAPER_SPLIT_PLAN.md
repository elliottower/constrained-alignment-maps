# Two papers, split 2026-07-29

`main_v12_blinded.tex` currently holds both. It reads as unwieldy because it is
two arguments sharing a document. They separate cleanly, and the split is
asymmetric: Paper A is nearly finished and every result in it survived the
seeding pass; Paper B needs work but contains one genuinely strong finding.

**The motivation for A does not require grokking.** Sutter et al. showed
unconstrained nonlinear maps are vacuous; linear DAS scores $0.00$ strict IIA on
subject--verb agreement; therefore a constrained nonlinear map, validated on
their random-network test. Modular arithmetic appears nowhere in that argument.
Grokking is how the phenomenon was found, not why the method is justified.

---

## Paper A — constrained alignment maps on GPT-2

Real model, established baseline, one claim. This is the strong one.

### Spine

Interchange accuracy alone does not distinguish a recovered causal variable from
a lookup table. Constraining a nonlinear alignment map with a variational
autoencoder's reconstruction and regularisation terms produces a map that
survives the tests an unconstrained one fails.

### Results, all present and holding

| Result | Evidence | State |
|---|---|---|
| Strict-IIA gap, six MIB tasks at $k=1$ | DAS $0.00$--$0.52$, structured VAE $\geq 0.90$ on five of six | single run |
| DAS shifts mass without flipping argmax | SVA: $0.92$ standard, $0.00$ strict | single run |
| Hard-IIA sweep on IOI, $k \in \{1..32\}$ | VAE dominates to $k=8$; DAS reaches $0.994$ at $k=16$ | **5-fold CV** |
| Cross-distribution dissociation | held-out entities VAE $0.001$ / NL-DAS $1.000$; held-out templates $0.993$ / $0.537$ | single run |
| Random-network control | VAE $0.000$, NL-DAS $0.989$ | **5 runs, pre-registered** |
| Diversity ratio detects collapse | NL-DAS on pretrained IOI: IIA $1.000$ at $\rho = 0.055$ | replicated |
| Diversity ratio misses converged vacuity | random arm, $\rho_\text{within} = 1.068$ at IIA $0.989$ | **reported against ourselves** |
| Controls sit at per-task chance | floors from $0.000$ to $0.482$ | recomputed |
| Capacity is not the explanation | VAE has fewer parameters than NL-DAS | needs recount |

### Attribution, non-negotiable

The random-network test is \citet{sutter2025nonlinear}'s, stated in the section
title and first sentence. Our addition is applying it to a constrained map, which
they did not examine. The diversity ratio is ours, and its failure at convergence
is reported as our own.

### Outstanding

1. **Seeds on the six-task table and the cross-distribution table.** These are
   single-run and they are the headline numbers. The k-sweep pass showed single
   runs generate claims that do not survive.
2. Parameter recount ($414$K vs $545$K depending on config).
3. An account of $\rho \gg 1$, or an explicit statement that there is none.
   NL-DAS+recon reaches $2.249$ on the *pretrained* arm at IIA $0.944$.

### Venue

TMLR. No length limit, and the criterion is whether claims are supported, which
suits a paper whose central move is reporting what its own metric failed to do.

---

## Paper B — Grassmannian structure in grokked models

Thinner, and one section is doing most of the work.

### The strongest result, which should be the spine

**Subspace identity is degenerate.** Ten seeds of grokked multiplication all
reach IIA $0.94$--$1.00$ and produce near-orthogonal DAS subspaces: pairwise
overlap $\approx 0.008$ against a chance value of $k/d = 0.016$. The recovered
subspaces are *less* aligned than random. Rotating a solution by more than a
radian on $\Gr(k,d)$ barely moves IIA.

DAS does not identify a unique object. The paper should be built around that,
not around the atlas.

### Supporting material

- **The atlas.** 14 operations, four primes, three depths. Grokking rate scales
  with data ($8\%$ at $p=53$, $71\%$ at $p=211$); depth is non-monotonic, with
  two layers unlocking operations that four layers lose.
- **Stochastic grokking.** Composite addition groks at 5 of 10 seeds, power at 0
  of 10. Single-condition classifications are unreliable near the boundary.
- **Memorization artifacts.** Squaring and cubing reach IIA $0.857$ at $k=2$
  without grokking. High accuracy without structure.
- **Group action, not linearity.** Affine ($2a + 3b + 5$) is linear and fails
  equivariance at $0\%$ across four primes, because under $a \mapsto a+g$ the
  output shifts by $2g$. Operation-linearity and causal-variable-linearity are
  different properties.
- **The seeded k-sweep.** DAS becomes reliable at $k=16$; NL-DAS plateaus near
  $0.95$ and never recovers the variable, including at $k=64$.
- **Weight-space SVD.** IIA $0.999$ with no optimisation against DAS's $0.150$ —
  but not dimension-matched ($k=32$ vs $k=8$), so unusable as written.

### Outstanding

1. **Dimension-matched SVD comparison.** The current numbers cannot be reported.
2. **Seeds on the atlas table.** Grokking status is binary per cell and the
   stochastic result shows cells near the boundary flip across seeds.
3. Decide whether this is a TMLR paper or a workshop paper. As it stands, one
   strong section plus survey suggests the latter unless the gauge result is
   built out.

### Not included

The structured VAE. Paper B is about what DAS finds and fails to find in a
controlled setting. Introducing a method makes it a different paper.

---

## Dead, and not to be revived

- **The $k=8$ Fourier-span story.** Claimed DAS saturates at $k=8$ because four
  key frequencies at two components each span eight dimensions. That cell is
  $0.806 \pm 0.570$ across seeds; DAS becomes reliable at $k=16$. The mechanism
  was fitted to a single run that landed on a round number.
- **The NL-DAS regression at $k=32$.** Seeded values are $0.939$, $0.961$,
  $0.956$ at $k=16, 32, 64$. A plateau, not a decline.
- **The two-sided $\rho$ account.** $\rho \gg 1$ as a second failure mode does
  not hold: NL-DAS reaches $\rho_\text{within} = 1.068$ on a random network at
  IIA $0.989$, which the metric reads as faithful.
- **A capability gap on IOI.** Under hard IIA, DAS reaches $0.994$ at $k=16$ and
  slightly exceeds the VAE thereafter. The claim is efficiency, not capability,
  in both settings.

---

## Order

Paper A first. It is nearly written, its results held up under the scrutiny that
shrank everything else, and it does not depend on Paper B existing.
