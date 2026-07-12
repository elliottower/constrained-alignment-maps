# Causal Geometry of Grokking

Code for "When Does Linear Causal Abstraction Work? Mapping the Boundary on the Grassmannian."

## Key findings

1. **Three-class partition**: 14 modular arithmetic operations split into Always Grassmannian (7), Stochastic (2), and Never Grassmannian (5) — the boundary is sharp and governed by grokking
2. **Stochastic grokking**: same operation, same hyperparameters, opposite outcomes from random initialization alone — Grassmannian variables appear if and only if the model generalizes
3. **Linear DAS returns zero IIA** on grokked modular addition at k <= 16, confirming the causal variable is fundamentally nonlinear (lives on S^1, not a linear subspace)
4. **NL-DAS is vacuous**: unconstrained nonlinear featurizers achieve perfect IIA by learning degenerate encoder-decoders (diversity ratio ~ 0)
5. **Structured pi-SAE recovers nonlinear causal variables**: pi-VAE + causal/nuisance split + L1 sparsity — neither component alone suffices
6. **Intrinsic dimension**: pi-SAE saturates at k=2 (the true dimensionality), while DAS climbs linearly without converging
7. **GPT-2 language tasks**: structured pi-SAE achieves IIA = 0.98 on IOI, works on gender bias, greater-than, hypernymy, SVA, capitals
8. **Cross-task transfer**: VAE trained on one IOI template transfers to unseen templates (IIA 0.82–0.96)
9. **E2E training**: end-to-end intervention loss eliminates the gap between additive and replacement interventions (hypernymy IIA 0.58 -> 0.97)

## Experiments

| Script | What it tests |
|--------|---------------|
| `experiments/grassmannian_geometry.py` | Core atlas: DAS k-sweeps, equivariance, circle geometry for 14 operations |
| `experiments/grokking_das_emergence.py` | DAS emergence during grokking training trajectory |
| `experiments/structured_vae_atlas.py` | Structured VAE across all 14 operations |
| `experiments/sparse_structured_vae.py` | Sparse VAE variants (L1, JumpReLU, TopK) |
| `experiments/sparse_das_grokking.py` | Sparse DAS on grokking tasks |
| `experiments/k1_vae_vs_das.py` | Head-to-head DAS vs pi-SAE at k=1 |
| `experiments/k1_hard_mode.py` | Hard-example IIA with continuous metrics |
| `experiments/multi_seed_stability.py` | 10-seed stability for stochastic operations |
| `experiments/cross_task_validation.py` | Cross-template transfer, persistent homology, sheaf consistency |
| `experiments/cyclic_and_jensen_validation.py` | Cyclic group equivariance, Jensen DoubleIO/TripleIO transfer |
| `experiments/ioi_subtask_transfer.py` | 8x8 transfer matrix across MIB IOI subtask counterfactuals |
| `experiments/ioi_subtask_transfer_baselines.py` | Baselines: random, per-subtask, joint, NL-DAS |
| `experiments/factorized_das_grokking.py` | Factorized DAS on grokking tasks |
| `experiments/nonlinear_dsi.py` | Nonlinear DSI experiments |
| `experiments/task_reference_baselines.py` | Canonical ground truth baselines for all tasks |
| `experiments/generate_figures.py` | Generate all figures from cached results |

## Results

Pre-computed results in `results/` and `experiments/results/`:

| Directory | Contents |
|-----------|----------|
| `results/grassmannian_atlas/` | Atlas results, factorized DAS, sparse DAS, VAE, multi-seed, cross-task |
| `experiments/results/feature_analysis/` | Per-feature ablation IIA drops, Fourier alignment |
| `experiments/results/k1_pi_ablations*/` | k=1 pi-SAE ablations across all tasks and layers |
| `experiments/results/cross_task/` | Cross-task transfer matrices |
| `experiments/results/e2e_and_additive/` | E2E vs additive intervention comparison |
| `experiments/results/gender_bias_e2e/` | Gender bias E2E results |
| `experiments/results/multi_seed/` | Multi-seed stability (addition, power) |

## Setup

```bash
pip install torch transformer-lens transformers einops matplotlib tqdm datasets

# Run locally (CPU, slow)
python experiments/grassmannian_geometry.py

# Run on Modal GPU (recommended)
modal run --detach experiments/grassmannian_geometry.py
```

## License

MIT
