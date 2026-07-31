"""Canonical Distributed Alignment Search, shared by every experiment.

Before this module there were 23 `train_das*` definitions across `experiments/`,
differing in initialisation and in how orthogonality was maintained. The training
objective was identical in all of them (cross-entropy on the counterfactual
token), so no result was computed with a wrong objective, but "DAS" meant
different things in different tables:

    hardcoded delta-PCA warm start   random_network_control, ravel_pi_sae,
                                     sparsity_vs_overcompleteness_ablation
    hardcoded Gaussian + per-step QR nine scripts, including k1_vae_vs_das
    hardcoded random orthogonal      method_comparison_unified
    parameterised (default random)   degeneracy_decomposition, multiseed_ksweep

This module makes initialisation and parametrisation explicit arguments so the
choice is reported rather than inherited. `STANDARD` is the reference
implementation used by the MIB benchmark: random orthogonal initialisation kept
on the Stiefel manifold by `torch.nn.utils.parametrizations.orthogonal`,
reproducing pyvene's `LowRankRotateLayer(init_orth=True)`. Everything else is an
ablation arm and must be labelled as one.

Torch is imported under a try/except and every nn.Module is defined inside a
function, matching the convention in the Modal scripts that import this module:
they are imported locally to launch, on machines without torch installed.
"""

from __future__ import annotations

import os
import random
import sys

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except (ImportError, AttributeError):
    pass

# MIB's causal-variable track is vendored at reference/MIB. Its featurizers
# module imports only torch and pyvene, so it can be used directly rather than
# reimplemented. The STANDARD arm below therefore runs MIB's own code, which is
# what licenses the paper to say so.
# Candidates in order: an explicit override, the Modal mount point, then the
# repo layout used locally. Modal containers do not have the repo tree, so the
# mount path has to be tried before the relative one.
_MIB_CANDIDATES = [
    os.environ.get("MIB_TRACK_PATH"),
    "/mib_track",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "reference", "MIB", "MIB-causal-variable-track"),
]
_MIB_PATH = None
for _cand in _MIB_CANDIDATES:
    if _MIB_PATH is None and _cand and os.path.isdir(os.path.join(_cand, "CausalAbstraction")):
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        _MIB_PATH = _cand

try:
    from CausalAbstraction.neural.featurizers import SubspaceFeaturizer
    MIB_AVAILABLE = True
except (ImportError, AttributeError):
    SubspaceFeaturizer = None
    MIB_AVAILABLE = False

# MIB's training configuration. Shared across every task:
#
#   optimizer      torch.optim.AdamW(params, lr=init_lr, weight_decay=0)
#   scheduler      "constant" (a no-op on the learning rate)
#   loss           task cross-entropy alone; the regularisation term in
#                  _train_intervention applies only to mask interventions (DBM)
#
# Everything else is PER TASK. The library defaults in
# CausalAbstraction/experiments/config.py are overridden by each baseline, so
# DEFAULT_CONFIG is not what MIB actually runs:
#
#   baselines/ioi_baselines/ioi_baselines.py   epoch 2, init_lr 1.0,  n_features 32
#   baselines/arithmetic_baselines.py          epoch 1, init_lr -,    n_features 16, batch 256
#   baselines/ravel_baselines.py               epoch 1
#   baselines/ARC_baselines.py                 epoch 2
#   baselines/simple_MCQA_baselines.py         epoch 8
#   CausalAbstraction/experiments/config.py    epoch 3, init_lr 1e-2, batch 32  (default only)
#
# init_lr = 1.0 appears exactly once in the benchmark, in the IOI baseline. It is
# not a typo on our side: AdamW's per-step update is bounded near lr, and the
# orthogonal parametrisation re-projects onto the Stiefel manifold each step, so
# a large rate with a two-epoch budget is a deliberate fast-convergence choice.
MIB_SHARED = {
    "weight_decay": 0.0,
    "optimizer": "adamw",
    "scheduler": "constant",
}

# Batch size for the IOI baseline comes from get_model_config in
# baselines/ioi_baselines/ioi_utils.py and is MODEL dependent:
#   gpt2 1024, qwen 256, llama 256, gemma 256.
# This project uses GPT-2, so 1024.
MIB_IOI_BATCH_BY_MODEL = {"gpt2": 1024, "qwen": 256, "llama": 256, "gemma": 256}

MIB_TASK_CONFIG = {
    # Task keys follow this project's names; values follow MIB's baselines.
    # IOI on GPT-2: batch 1024 at lr 1.0 for 2 epochs is roughly four optimiser
    # steps on a two-thousand-pair split. Large batch, large rate, few steps.
    "ioi": {"lr": 1.0, "n_epochs": 2, "batch_size": 1024, "n_features": 32},
    "arithmetic": {"lr": 1e-2, "n_epochs": 1, "batch_size": 256, "n_features": 16},
    "ravel": {"lr": 1e-2, "n_epochs": 1, "batch_size": 32, "n_features": 32},
    "arc": {"lr": 1e-2, "n_epochs": 2, "batch_size": 32, "n_features": 32},
    "mcqa": {"lr": 1e-2, "n_epochs": 8, "batch_size": 32, "n_features": 32},
    # Tasks MIB has no baseline for fall back to the library default, and the
    # fallback is recorded in results rather than passing as if it were MIB's.
    "_default": {"lr": 1e-2, "n_epochs": 3, "batch_size": 32, "n_features": 32},
}

# Every value above is read from MIB's source. Nothing is assumed.
MIB_BATCH_SIZE_IS_ASSUMED = set()


def mib_config(task=None):
    """MIB's configuration for a task, falling back to the library default.

    Returns a dict with lr, n_epochs, batch_size, n_features, the shared
    optimiser settings, and `is_mib_baseline` recording whether MIB actually has
    a baseline for this task or the library default was substituted.
    """
    key = (task or "").lower()
    cfg = dict(MIB_TASK_CONFIG.get(key, MIB_TASK_CONFIG["_default"]))
    cfg.update(MIB_SHARED)
    cfg["task"] = key or None
    cfg["is_mib_baseline"] = key in MIB_TASK_CONFIG and key != "_default"
    cfg["batch_size_assumed"] = key in MIB_BATCH_SIZE_IS_ASSUMED
    return cfg


# Retained so existing call sites keep working; it is the library default, NOT
# what MIB runs on any specific task. Prefer mib_config(task).
MIB_CONFIG = dict(MIB_TASK_CONFIG["_default"], **MIB_SHARED)

# Initialisation of the rotation.
#   random_orthogonal  nn.init.orthogonal_ -- the reference implementation
#   gaussian_qr        N(0, 0.02^2), orthonormalised by the first QR
#   delta_pca          top-k right singular vectors of the intervention deltas,
#                      a warm start that is NOT standard DAS
INIT_CHOICES = ("random_orthogonal", "gaussian_qr", "delta_pca")

# How orthogonality is maintained during optimisation.
#   orthogonal   torch.nn.utils.parametrizations.orthogonal -- the reference
#   qr           torch.linalg.qr re-applied every step
PARAM_CHOICES = ("orthogonal", "qr")

# The reference configuration. Any deviation is an ablation and is labelled.
STANDARD = {"init": "random_orthogonal", "parametrization": "orthogonal"}


def _init_weight(train_data, d_model, k, device, init, key_base_act, key_src_act):
    """Return the (d_model, k) starting frame for the requested initialisation."""
    if init == "delta_pca":
        deltas = torch.stack([d[key_src_act] - d[key_base_act] for d in train_data])
        _, _, Vh = torch.linalg.svd(deltas, full_matrices=False)
        return Vh[:k].T.clone().to(device)
    if init == "gaussian_qr":
        return torch.randn(d_model, k, device=device) * 0.02
    if init == "random_orthogonal":
        w = torch.empty(d_model, k, device=device)
        nn.init.orthogonal_(w)
        return w
    raise ValueError(f"init must be one of {INIT_CHOICES}, got {init!r}")


def _fit(model, train_data, hook_name, device, k, snapshot_steps, lr, batch_size,
         init, parametrization, key_base_act, key_src_act, key_base_toks, key_target):
    """Fit the rotation, returning {step: Q} at each requested step count.

    Snapshots come from ONE trajectory rather than one fit per step count. That
    is both six times cheaper for a six-point convergence sweep and a better
    convergence estimate, since consecutive points share initialisation and batch
    order and differ only in how long the optimiser ran.
    """
    if parametrization not in PARAM_CHOICES:
        raise ValueError(f"parametrization must be one of {PARAM_CHOICES}, "
                         f"got {parametrization!r}")

    d_model = train_data[0][key_base_act].shape[0]
    w0 = _init_weight(train_data, d_model, k, device, init, key_base_act, key_src_act)

    if parametrization == "orthogonal":
        # MIB's own featurizer, imported from reference/MIB. Both branches are
        # constructors MIB supports: shape= uses LowRankRotateLayer(init_orth=True),
        # rotation_subspace= uses init_orth=False followed by a data copy, which is
        # how MIB itself loads a non-default rotation.
        if not MIB_AVAILABLE:
            raise ImportError(
                "The standard DAS arm runs MIB's SubspaceFeaturizer, which needs "
                "pyvene. Install pyvene, or pass parametrization='qr' to use the "
                "ablation implementation in this file.")
        if init == "random_orthogonal":
            feat = SubspaceFeaturizer(shape=(d_model, k), trainable=True)
        else:
            feat = SubspaceFeaturizer(rotation_subspace=w0.cpu(), trainable=True)
        feat.featurizer.to(device)
        params = list(feat.featurizer.parameters())
        get_q = lambda: feat.featurizer.rotate.weight  # noqa: E731 - on the Stiefel manifold
    else:
        # Ablation arm, our code: per-step QR re-orthonormalisation.
        A = nn.Parameter(w0.clone())
        params = [A]
        get_q = lambda: torch.linalg.qr(A)[0]  # noqa: E731 - re-orthonormalised each step

    # MIB uses AdamW with weight_decay=0 and a constant learning-rate schedule.
    # A constant schedule leaves the rate untouched, so it is omitted rather than
    # reproduced through a transformers scheduler object.
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=MIB_CONFIG["weight_decay"])
    wanted = sorted(set(snapshot_steps))
    snapshots = {}

    for step in range(1, max(wanted) + 1):
        optimizer.zero_grad()
        Q = get_q()
        proj = Q @ Q.T
        batch = random.sample(train_data, min(batch_size, len(train_data)))
        loss = torch.tensor(0.0, device=device)
        for d in batch:
            delta = proj @ (d[key_src_act] - d[key_base_act])

            def make_hook(_iv):
                def hk(act, hook=None):
                    new = act.clone()
                    new[0, -1, :] += _iv
                    return new
                return hk

            logits = model.run_with_hooks(
                d[key_base_toks], fwd_hooks=[(hook_name, make_hook(delta))]
            )[0, -1, :]
            loss = loss - F.log_softmax(logits, dim=-1)[d[key_target]]
        (loss / len(batch)).backward()
        optimizer.step()

        if step in wanted:
            with torch.no_grad():
                snapshots[step] = get_q().detach().clone()

    return snapshots


def steps_for_epochs(n_examples, batch_size=None, n_epochs=None):
    """Optimiser steps corresponding to MIB's epoch-based budget."""
    batch_size = batch_size or MIB_CONFIG["batch_size"]
    n_epochs = n_epochs or MIB_CONFIG["n_epochs"]
    return n_epochs * -(-n_examples // batch_size)


def train_das_mib(model, train_data, hook_name, device, k=1, task=None,
                  return_config=False, **kwargs):
    """DAS as MIB runs it for `task`: their featurizer, their hyperparameters.

    `task` selects the per-task configuration, because MIB's baselines override
    the library defaults and differ from each other (IOI trains 2 epochs at
    lr=1.0; arithmetic trains 1 epoch at lr=1e-2). Omitting `task` falls back to
    the library default and records that it did, so a run cannot silently claim
    MIB's settings for a task MIB has no baseline for.
    """
    cfg = mib_config(task)
    n_steps = steps_for_epochs(len(train_data), cfg["batch_size"], cfg["n_epochs"])
    Q = train_das(model, train_data, hook_name, device, k=k, n_steps=n_steps,
                  lr=cfg["lr"], batch_size=cfg["batch_size"],
                  init="random_orthogonal", parametrization="orthogonal",
                  **kwargs)
    if return_config:
        return Q, dict(cfg, n_steps=n_steps, n_examples=len(train_data))
    return Q


def train_das(model, train_data, hook_name, device, k=1, n_steps=300,
              lr=MIB_CONFIG["lr"], batch_size=MIB_CONFIG["batch_size"],
              init="random_orthogonal", parametrization="orthogonal",
              key_base_act="base_act", key_src_act="src_act",
              key_base_toks="base_toks", key_target="src_label"):
    """Fit a k-dimensional alignment subspace by interchange intervention.

    Returns the orthonormal frame Q with shape (d_model, k). Callers wanting the
    projector use Q @ Q.T.

    The intervention is applied additively, h' = h_base + P (h_src - h_base),
    which is algebraically identical to the replacement form
    h' = h_base - P h_base + P h_src used by some callers.

    Key names are arguments because callers disagree: some use `src_act` and
    `src_label`, others `source_act` and `src_id`.
    """
    return _fit(model, train_data, hook_name, device, k, [n_steps], lr, batch_size,
                init, parametrization, key_base_act, key_src_act,
                key_base_toks, key_target)[n_steps]


def train_das_snapshots(model, train_data, hook_name, device, k=1,
                        snapshot_steps=(100, 250, 500, 1000, 2000, 4000),
                        lr=MIB_CONFIG["lr"], batch_size=MIB_CONFIG["batch_size"],
                        init="random_orthogonal",
                        parametrization="orthogonal", key_base_act="base_act",
                        key_src_act="src_act", key_base_toks="base_toks",
                        key_target="src_label"):
    """Step A of Experiment 0.5: one trajectory, evaluated at several budgets.

    Returns {step: Q}. The caller evaluates each Q and applies the convergence
    rule in PREREGISTRATION_RECONSTRUCTION_CRITERION.md: converged at step s when
    accuracy at s is within 0.02 of accuracy at 2s, for two consecutive doublings.
    """
    return _fit(model, train_data, hook_name, device, k, snapshot_steps, lr,
                batch_size, init, parametrization, key_base_act, key_src_act,
                key_base_toks, key_target)


def converged_step(acc_by_step, tol=0.02, n_doublings=2):
    """Smallest step s meeting the pre-registered convergence rule.

    Returns None when no step qualifies, which the pre-registration requires be
    reported as "not converged within budget tested" rather than silently
    replaced by the highest observed value.
    """
    steps = sorted(acc_by_step)
    for i, s in enumerate(steps):
        ok, cur = True, i
        for _ in range(n_doublings):
            nxt = next((t for t in steps[cur + 1:] if t >= 2 * steps[cur]), None)
            if nxt is None or abs(acc_by_step[nxt] - acc_by_step[steps[cur]]) > tol:
                ok = False
                break
            cur = steps.index(nxt)
        if ok:
            return s
    return None


def random_subspace(d_model, k, device):
    """A uniformly random orthonormal k-frame, for noise-floor estimation."""
    Q, _ = torch.linalg.qr(torch.randn(d_model, k, device=device))
    return Q


# Student's t 95% critical values, keyed by sample count. Same table as
# experiments/aggregate_seeds.py so intervals are computed identically.
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
       9: 2.306, 10: 2.262}


def mean_ci95(xs):
    """Mean and 95% Student-t half-width. Half-width is 0.0 for a single value."""
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    se = (var / n) ** 0.5
    return m, T95.get(n, 1.96) * se
