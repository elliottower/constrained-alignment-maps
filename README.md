# Constrained alignment maps

What interchange intervention accuracy does and does not establish about
alignment maps for causal abstraction.

Started 2026-07-30 as a clean rebuild. The predecessor repository accumulated
several parallel lines of work whose results became difficult to attribute; this
one carries only what feeds the current paper. Nothing here is a finished result.

## The question

Distributed Alignment Search learns a linear map identifying a subspace whose
interchange behaviour matches a high-level causal variable.
[Sutter et al. (2025)](https://arxiv.org/abs/2507.08802) showed that removing the
linearity constraint makes the framework uninformative: sufficiently expressive
maps reach 100% interchange accuracy on randomly initialised networks.

This work asks what a *constrained* nonlinear map does under the same tests, and
what separates maps that accuracy alone cannot.

## Current state

The measurement that organises the work, from
`experiments/results/random_network_control_e2e/` — indirect object
identification, k = 1, 180 held-out pairs, random-subspace floor 0.000 over five
draws:

| alignment map | pretrained GPT-2 | randomly initialised |
|---|---|---|
| linear DAS | 0.200 | 0.000 |
| unconstrained nonlinear DAS | 1.000 | 0.433 |
| structured VAE, no interchange term | 1.000 | **0.000** |
| structured VAE + interchange term | 1.000 | **0.961** |

The last two rows share a builder. Same architecture, same data, same seed. The
only difference is whether the interchange cross-entropy is in the loss.

So the axis that produces vacuity is not expressivity and not constraint. It is
whether the map is trained on the metric being reported. A map that never
optimises interchange cannot be made vacuous by training it longer, because
nothing in its objective rewards producing counterfactual outputs.

## Layout

```
paper/          the draft under revision
experiments/    the scripts that produce its tables
docs/           pre-registrations, frozen before their runs
scripts/        dependency fetching
```

- `METHOD_REGISTRY.md` — what each method is, what it is called in code, and what
  it is called in the paper. Read this before interpreting any result file: the
  predecessor repository had three different architectures sharing the name
  "structured VAE".
- `LAB_NOTEBOOK.md` — decisions, killed claims, surviving claims, open
  contradictions, and the defects found in the code. Kept deliberately, including
  the parts that did not work.
- `PAPER_REVISION_SPEC_constrained_interchange_v4.md` — what to change in the
  draft and why.

## Reproducing

```bash
./scripts/fetch_deps.sh          # MIB submodule, reference implementations, papers
uv run python experiments/test_das.py
uv run python experiments/test_das_matches_mib.py
```

The second verifies that Distributed Alignment Search here runs
[MIB's](https://github.com/aaronmueller/MIB) own featurizer with MIB's per-task
hyperparameters, by class identity, by numerical agreement with their
featurise-swap-inverse round trip, and by reading their configuration at test
time so their checkout drifting breaks the test rather than passing quietly.

GPU runs go through Modal and are detached; see the pre-registrations in `docs/`
for what each one is committed to measuring before it runs.

## Status of claims

Every table in the current draft is single-run. Two claims in this project have
already died to seeding, and one headline arm failed its own pre-registered
control on 2026-07-30. Treat reported numbers as provisional until the
pre-registered runs complete.
