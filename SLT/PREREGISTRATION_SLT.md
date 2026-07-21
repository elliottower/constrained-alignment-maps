# Pre-Registration: Loss-Landscape Geometry Meets Representation Geometry in Grokking

**Working title:** *Three Probes of the Grokking Phase Transition: Learning Coefficient,
Grassmannian Distance, and Causal-Abstraction Dimensionality*

**Date:** 2026-07-20
**Author:** Elliot Tower
**Status:** Pre-registration (frozen predictions before running LLC/pi-SAE at checkpoints)
**Repo:** `causal-geometry-grokking`
**Companion paper:** *When Does Linear Causal Abstraction Work? Mapping the Boundary
on the Grassmannian* (TMLR, under review)

---

## 0. Overview

We already have a TMLR paper showing that **grokking governs whether Grassmannian
causal variables form**, with a natural experiment: **composite_addition at p=113 groks
in 5/10 seeds, power at p=113 groks in 0/10 seeds**. Same architecture, same
hyperparameters, opposite outcomes by seed.

This project adds one axis: the **loss-landscape** view from singular learning theory
(SLT). We measure the **local learning coefficient (LLC)** at a small number of
checkpoints on those same stochastic grokkers and ask whether the SLT basin transition
lines up with (a) the Grassmannian subspace snapping into place and (b) the pi-SAE
causal variable becoming compressible.

**Three independent probes** of the same phase transition: weight-space (LLC),
representation-space (Grassmannian), causal-abstraction-space (pi-SAE). If they converge,
the cheaper forward-pass probes serve as proxies for the expensive SGLD estimator. If
they don't, the temporal ordering is itself the finding.

No prior work connects SLT to causal abstraction or pi-SAE compression.

---

## 1. What we already have

| Asset | Detail |
|---|---|
| Operation atlas | 14 ops x 4 primes x depths 1/2/4 |
| Stochastic grokkers | composite_addition p=113: 5/10 grok; power p=113: 0/10 grok |
| Seed-level data | Grokked: 53 (ep 8078), 137 (5420), 256 (6196), 500 (26254), 2024 (9042). Non-grokked: 7 (never), 19 (28390 partial), 42 (8334 partial), 101 (9099 partial), 777 (never) |
| pi-SAE code | Full impl; IIA >= 0.90 at k=1 on GPT-2 language tasks |
| Grassmannian tooling | Geodesic distance via principal angles; grassmann_to_final at ~9 epochs for 82 atlas entries |
| Holonomy final models | Saved weights + VAEs at k=2/4/8/16 for 4 ops |

**Critical gap:** Stochastic grokkers did not save intermediate checkpoints. Must retrain
with checkpoint saving (~10 min/run on GPU).

---

## 2. Hypotheses (frozen, falsifiable)

Each hypothesis has a prediction, null, and **confound that would kill it**.

### H1: LLC drops across the grokking transition

Grok seeds: LLC(pre) > LLC(post). Non-grok seeds: LLC stays high/flat.

**Prediction:** Non-overlapping interquartile ranges between 5 grokked and 5 non-grokked
composite_addition seeds at final checkpoint.

**Kill condition:** LLC drop tracks **weight-norm growth** or **loss value**, not basin
geometry. Partial out weight norm and train/val loss; the drop must exceed what
norm/loss predict.

### H2: Grassmannian distance drops at the same checkpoint as LLC

grassmann_to_final collapses at/after the LLC transition for grok seeds; no collapse
for non-grok.

**Prediction:** Grassmannian distance and LLC transitions localize to the same
checkpoint window for all grok seeds.

**Kill condition:** Subspace distance tracks **loss velocity** (generic drift). Include
loss velocity as covariate; alignment must survive.

**Strong version:** If transitions coincide, Grassmannian distance is a forward-pass-only
proxy for LLC phase transitions. Highest-value result for Timaeus (SGLD is their
bottleneck).

### H3: pi-SAE compression onsets at the transition

k=1 IIA poor pre-grok, improves at/after transition for grok seeds; stays low for
non-grok.

**Prediction:** IIA_grokked > 0.70 at k=1 post-grok. IIA_non_grokked < 0.30 at k=1
at all checkpoints.

**Kill condition:** Compressibility is a deterministic function of **val accuracy**. Report
IIA vs val acc; the claim requires that compression is an independent signal, not a
proxy for accuracy.

### H4: Three probes converge on transition location

LLC drop, Grassmannian collapse, and pi-SAE compression onset localize to the same
checkpoint window for grok seeds. None fire for non-grok seeds.

### H5: Never-grok control matches non-grokked composite_addition

Power mod 113 (0/10 grok) will show LLC and pi-SAE patterns indistinguishable from
non-grokked composite_addition seeds.

**Prediction:** LLC_power and IIA_power at final checkpoint fall within the range of
non-grokked composite_addition seeds.

### H6: LLC across the difficulty spectrum (exploratory)

Timaeus's grokking paper: one operation on a quadratic network. Their multi-problem
post found LLC did NOT resolve per-operation stages. Our atlas has a difficulty spectrum
(8%-71% grokking rates) they couldn't detect.

**Prediction:** Either (a) LLC resolves the spectrum (lower-grok-rate ops sit in
higher-LLC basins longer) or (b) the spectrum lives at a level LLC misses. Either is
publishable.

---

## 3. Falsification criteria

| Hypothesis | Falsified if |
|---|---|
| H1 | LLC IQRs overlap after partialling out weight norm and loss |
| H2 | Grassmannian distance decrease explained entirely by loss velocity |
| H3 | k=1 IIA is a monotone function of val accuracy (no independent signal) |
| H3 | Any grokked seed has IIA < 0.50 at k=1 post-grok |
| H4 | Transition locations disagree by more than 3000 epochs across probes |
| H5 | Power LLC falls outside range of non-grokked composite_addition |

---

## 4. Experimental design (laddered by cost)

### Phase 0 — Estimator gate (FIRST; ~1-2 GPU-hrs)

**Nothing downstream is trustworthy until this passes.**

- [ ] Confirm devinterp SGLD runs on HookedTransformer (1-layer, d_model=128)
- [ ] LLC stable under SGLD sweep: step size, chain length, num chains. Report mean +/- std
- [ ] Sanity: LLC(memorized) > LLC(grokked) on holonomy final models (zero retraining)
- **Gate:** If LLC not stable to usable tolerance, STOP and report estimator instability

### Phase A — Zero-retraining check (~1 afternoon)

Run LLC + pi-SAE on **holonomy final models** (division, squaring, subtraction,
multiplication). Do grokked models have lower LLC AND better k=1 pi-SAE compression
than non-grokked? Tests core H1/H3 at near-zero cost.

### Phase B — Minimal trajectory (~2-3 GPU-hrs incl. retraining)

Retrain **6 seeds of composite_addition p=113** (3 grok: 137, 256, 53; 3 non-grok:
7, 42, 777) + **3 seeds of power** (42, 137, 256). Save 3 checkpoints each.

At each of **27 checkpoints** compute:
- LLC (SGLD, settings from Phase 0)
- Grassmannian distance to seed's final subspace
- pi-SAE k=1 IIA, MSE, diversity ratio
- Covariates: train loss, val loss, val acc, weight norm, loss velocity

**Decision rule:** If H1-H4 separate on 6 seeds, proceed to Phase C. Otherwise pivot.

### Phase C — Scale-up (only if B works; ~10-30 GPU-hrs)

- Full 10-seed composite_addition + full 10-seed power control
- Fine-grained trajectory (every 1000 epochs) on 2 seeds for temporal ordering (H5→H6)
- LLC on final models across full 14-op atlas for difficulty spectrum analysis

### Cost summary

| Phase | Compute | Depends on |
|---|---|---|
| 0 (gate) | ~1-2 GPU-hrs | — |
| A (zero-retrain) | ~1-2 GPU-hrs | Phase 0 pass |
| B (minimal traj) | ~2-3 GPU-hrs | Phase 0 pass |
| C (scale-up) | ~10-30 GPU-hrs | Phase B separating |
| **Ceiling** | **~55 GPU-hrs (~$55-110)** | all pass |

---

## 5. Method

### Models

1-layer HookedTransformer (TransformerLens):
- d_model=128, n_heads=4, d_head=32, d_mlp=512
- act_fn=relu, no LayerNorm
- Training: AdamW (lr=1e-3, wd=1.0, betas=(0.9, 0.98))
- 30,000 epochs

### Checkpoints (3 per run)

1. **Pre-grok (epoch 1000):** All seeds memorizing
2. **Transition (seed-specific):** For grok seeds, epoch where test loss first < 1.0.
   For non-grok seeds, epoch 8000 (median of grok transitions)
3. **Post-grok (epoch 30,000):** Final

### Measurements

**LLC:** devinterp SGLD, n_chains=5, n_draws=500 (calibrate in Phase 0).
Cross-entropy loss on modular arithmetic task.

**pi-SAE:** Structured pi-SAE, k=1 primary (k=2/4 supplementary), 400 training steps,
blocks.0.hook_resid_post last position. Report: IIA, MSE, diversity ratio.

**Grassmannian:** Top-10 PCA of activations, geodesic distance on Gr(10, 128) to
epoch 30,000 subspace.

### Data

- Train/test split: 30% train
- Data seed: 598

---

## 6. Analysis plan

1. Phase 0 gate first; abort if LLC unstable
2. Phase A on holonomy models; check cross-operation separation
3. Phase B: retrain 9 runs with checkpoints, compute all probes at 27 checkpoints
4. Test H1-H2 with Mann-Whitney U (n=3 vs n=3 in Phase B; n=5 vs n=5 in Phase C)
5. Test H3-H4 with within-seed pre/post comparisons
6. Test convergence (H4): report transition checkpoint offsets across three probes
7. Confound regressions: LLC on {weight norm, train loss, val loss, loss velocity}
8. Figures:
   - Fig 1: LLC and pi-SAE IIA at 3 checkpoints, one line per seed, colored by grok status
   - Fig 2: Scatterplot of LLC vs pi-SAE IIA pooled
   - Fig 3: Grassmannian distance vs pi-SAE IIA

### Exploratory (not pre-registered)

- Fine-grained trajectory (every 1000 epochs) for temporal ordering
- Per-head rLLC if devinterp supports it
- pi-SAE dimensionality curve (k=1,2,4,8)
- Comparison to Timaeus grokking paper LLC values
- LLC across full 14-op difficulty spectrum

---

## 7. Power analysis

- Phase B (3 vs 3): 80% power to detect d > 2.5 at alpha=0.05
- Phase C (5 vs 5): 80% power to detect d > 1.7 at alpha=0.05
- Spearman (n=27 Phase B): 80% power for |rho| > 0.49
- Predictions (non-overlapping IQRs, |rho| > 0.60) exceed these thresholds

---

## 8. Framing and pitch

**Abstract sentence:** Grokking is a transition between an LLC basin and a Grassmannian
point; the loss-landscape complexity, the representation subspace, and the
causal-abstraction dimensionality all reorganize at the same seed-specific checkpoint,
giving a forward-pass proxy for the SGLD-estimated learning coefficient.

**Timaeus pitch:**
- They did 1 operation on a quadratic network; we have 14 operations on real
  transformers with a difficulty spectrum their multi-problem post couldn't detect
- The 5/10 vs 0/10 stochastic split holds architecture and scale fixed — cleanest
  possible control for isolating basin selection
- pi-SAE as an SLT probe is genuinely novel — nobody has connected causal-abstraction
  dimensionality to the learning coefficient
- Deliverable: 4-page workshop draft with result in hand + CV

---

## 9. Registration

- Pre-registration committed before any LLC or pi-SAE measurements
- Training script: `SLT/train_with_checkpoints.py` (to be written)
- Analysis script: `SLT/analyze_slt_pisae.py` (to be written)
- Code SHA recorded at commit time
