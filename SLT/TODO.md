# SLT + pi-SAE Grokking Paper — TODO

## Overview

Three probes of the grokking phase transition: LLC (weight-space), Grassmannian distance
(representation-space), pi-SAE compression (causal-abstraction-space). Tested on the
stochastic grokkers from the TMLR paper (composite_addition 5/10 grok, power 0/10 grok).

See PREREGISTRATION_SLT.md for frozen hypotheses and falsification criteria.

## Laddered phases (spend only what each gate justifies)

### Phase 0 — Estimator gate (~1-2 GPU-hrs)
- [ ] Install devinterp
- [ ] SGLD stability sweep on 1-layer HookedTransformer (chain length, num chains, step size)
- [ ] Sanity check: LLC(squaring) > LLC(multiplication) on holonomy final models
- [ ] **GATE:** If LLC not stable, STOP. Report estimator instability finding.

### Phase A — Zero-retraining check (~1 afternoon)
- [ ] LLC on 4 holonomy final models (division, squaring, subtraction, multiplication)
- [ ] pi-SAE k=1 on same 4 models
- [ ] Check: grokked models lower LLC AND better compression?
- [ ] **GATE:** If no separation, pivot before retraining.

### Phase B — Minimal trajectory (~2-3 GPU-hrs)
- [ ] Modify training script to save state_dict at 3 epochs (pre/transition/post)
- [ ] Retrain 6 composite_addition seeds (grok: 137, 256, 53; non-grok: 7, 42, 777)
- [ ] Retrain 3 power seeds (42, 137, 256) as never-grok control
- [ ] LLC + pi-SAE + Grassmannian distance at 27 checkpoints
- [ ] Covariates: train loss, val loss, val acc, weight norm, loss velocity
- [ ] **GATE:** If H1-H4 don't separate, pivot. Don't fund Phase C.

### Phase C — Scale-up (only if B works; ~10-30 GPU-hrs)
- [ ] Full 10-seed composite_addition + 10-seed power
- [ ] Fine-grained trajectory (every 1000 epochs) on 2 seeds
- [ ] LLC across full 14-op atlas for difficulty spectrum (H6)
- [ ] Write up

## Immediate next actions
1. Commit pre-registration (SHA-freeze)
2. Phase 0: install devinterp, run stability sweep on holonomy models
3. Phase A: LLC + pi-SAE on holonomy finals
4. Only then: modify training script for Phase B

## Target
- Workshop paper (4 pages) or TMLR extension
- Send to Jesse Hoogland / Timaeus with CV as Fellows Program application
