# When Does Linear Causal Abstraction Work?

Mapping the boundary between linear and nonlinear causal variables in neural networks. We construct an atlas of 14 modular arithmetic operations spanning the Grassmannian boundary, show that grokking is the mechanism that determines which side you land on, and demonstrate that unconstrained nonlinear methods (NL-DAS) achieve perfect IIA vacuously via lookup tables while a structured pi-SAE recovers genuine nonlinear causal variables.

## Slides

- **Grokking geometry** — [PDF](slides/grokking/grokking_geometry_v1.pdf) / [source](slides/grokking/grokking_geometry_v1.tex)
- **Structured pi-SAE** — [PDF](slides/structured_sae/structured_pi_sae_v1.pdf) / [source](slides/structured_sae/structured_pi_sae_v1.tex)

## Paper

Draft paper in `paper/main.tex`. Working title: "When Does Linear Causal Abstraction Work? Mapping the Boundary on the Grassmannian."

## Key findings

1. **Three-class partition**: 14 operations split into Always Grassmannian (7), Stochastic (2), and Never Grassmannian (5) --- the boundary is sharp and governed by grokking
2. **Stochastic grokking**: same operation, same hyperparameters, opposite outcomes from random initialization alone --- Grassmannian variables appear if and only if the model generalizes
3. **Linear DAS returns zero IIA** on grokked modular addition at k <= 16, confirming the causal variable is fundamentally nonlinear (lives on S^1, not a linear subspace)
4. **NL-DAS is vacuous**: unconstrained nonlinear featurizers achieve perfect IIA by learning degenerate encoder-decoders (diversity ratio ~ 0)
5. **Structured pi-SAE recovers nonlinear causal variables**: pi-VAE + causal/nuisance split + L1 sparsity --- neither component alone suffices. IIA = 1.0 on all grokked operations, IIA ~ 0 on non-grokked
6. **Intrinsic dimension**: pi-SAE saturates at k=2 (the true dimensionality), while DAS climbs linearly without converging
7. **GPT-2 language tasks**: structured pi-SAE achieves IIA = 0.98 on IOI (vs NL-DAS 1.0 vacuously), works on gender bias, greater-than, hypernymy, SVA, capitals
8. **Cross-task transfer**: VAE trained on one IOI template transfers to unseen templates (IIA 0.82--0.96) and across MIB subtask counterfactuals
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
| `experiments/generate_figures.py` | Generate all paper figures from cached results |
| `experiments/generate_slide_figures.py` | Generate slide-specific figures |

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

Additional large results (atlas.jsonl, grokking_nonlinear_hunt/) are stored on the Modal `fc-results` volume.

## Setup

```bash
pip install torch transformer-lens transformers einops matplotlib tqdm datasets

# Run locally (CPU, slow)
python experiments/grassmannian_geometry.py

# Run on Modal GPU (recommended)
modal run --detach experiments/grassmannian_geometry.py
```

Most experiments were run on Modal A100 GPUs.

## Citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20819725.svg)](https://doi.org/10.5281/zenodo.20819725)

```bibtex
@software{tower2026grassmannian,
  author    = {Tower, Elliot},
  title     = {When Does Linear Causal Abstraction Work? Mapping the Boundary on the Grassmannian},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v0.1.0},
  doi       = {10.5281/zenodo.20819725},
  url       = {https://doi.org/10.5281/zenodo.20819725}
}
```

## License

MIT
