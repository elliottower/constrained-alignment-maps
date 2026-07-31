"""Equivalence tests: our DAS runs MIB's code, with MIB's hyperparameters.

The paper intends to state that DAS was run using the original MIB
implementation. These tests are what make that statement checkable rather than
asserted. They compare against the vendored MIB source at reference/MIB, so they
fail loudly if that checkout is updated and drifts from what we assume.

Three things are verified:

1. the standard arm's rotation IS MIB's SubspaceFeaturizer wrapping pyvene's
   LowRankRotateLayer, by class identity rather than by resemblance;
2. our interchange arithmetic produces bit-comparable output to MIB's
   featurizer/inverse-featurizer round trip, which is what their
   FeatureInterchangeIntervention computes;
3. our hyperparameters equal MIB's DEFAULT_CONFIG, read from their file at test
   time so that a change on their side breaks the test instead of passing
   silently.

Run: uv run python experiments/test_das_matches_mib.py
"""

from __future__ import annotations

import torch

import das

from CausalAbstraction.neural.featurizers import (
    SubspaceFeaturizer,
    SubspaceFeaturizerModule,
    SubspaceInverseFeaturizerModule,
)
from CausalAbstraction.experiments.config import DEFAULT_CONFIG
import pyvene as pv


def test_standard_arm_is_mib_code():
    """Class identity, not resemblance: these objects must come from MIB."""
    assert das.MIB_AVAILABLE, "MIB featurizer did not import; pyvene missing?"
    assert das.SubspaceFeaturizer is SubspaceFeaturizer

    feat = SubspaceFeaturizer(shape=(32, 4), trainable=True)
    assert isinstance(feat.featurizer, SubspaceFeaturizerModule)
    assert isinstance(feat.inverse_featurizer, SubspaceInverseFeaturizerModule)

    # The rotation underneath is pyvene's, orthogonally parametrised.
    rotate = feat.featurizer.rotate
    assert type(rotate).__name__ == "ParametrizedLowRankRotateLayer", type(rotate).__name__
    assert isinstance(rotate, pv.models.layers.LowRankRotateLayer)
    assert torch.nn.utils.parametrize.is_parametrized(rotate, "weight")
    print(f"  featurizer={type(feat).__module__}.{type(feat).__name__}")
    print(f"  rotation={type(rotate).__mro__[1].__module__}.LowRankRotateLayer, parametrised")


def test_interchange_matches_mib_arithmetic():
    """Our delta form must equal MIB's featurize -> swap -> inverse round trip.

    MIB's FeatureInterchangeIntervention with subspaces=None computes
        f_base, base_err = featurizer(base); f_src, _ = featurizer(source)
        out = inverse_featurizer(f_src, base_err)
    We compute base + P (source - base) with P = Q Q^T. These agree only if the
    projector is symmetric and the residual is carried correctly, so this is a
    real check rather than a restatement.
    """
    torch_d, k = 64, 3
    for _ in range(25):
        feat = SubspaceFeaturizer(shape=(torch_d, k), trainable=True)
        base = torch.randn(torch_d)
        src = torch.randn(torch_d)

        f_base, base_err = feat.featurizer(base)
        f_src, _ = feat.featurizer(src)
        mib_out = feat.inverse_featurizer(f_src, base_err)

        Q = feat.featurizer.rotate.weight
        ours = base + (Q @ Q.T) @ (src - base)

        assert torch.allclose(mib_out, ours, atol=1e-5), (
            f"max abs diff {(mib_out - ours).abs().max().item():.3e}")
    print(f"  25/25 random cases agree to 1e-5 (d={torch_d}, k={k})")


def test_interchange_is_not_trivially_equal():
    """Guard: the previous test must be able to fail.

    A wrong projector has to produce a different answer, or the agreement above
    would be meaningless.
    """
    d, k = 64, 3
    feat = SubspaceFeaturizer(shape=(d, k), trainable=True)
    base, src = torch.randn(d), torch.randn(d)
    f_base, base_err = feat.featurizer(base)
    f_src, _ = feat.featurizer(src)
    mib_out = feat.inverse_featurizer(f_src, base_err)

    wrong_Q = torch.linalg.qr(torch.randn(d, k))[0]
    wrong = base + (wrong_Q @ wrong_Q.T) @ (src - base)
    diff = (mib_out - wrong).abs().max().item()
    assert diff > 1e-3, "a wrong projector matched, so the equivalence test is vacuous"
    print(f"  wrong projector differs by {diff:.3f}, so the check has teeth")


def _baseline_config(rel_path):
    """Extract the literal config dict from one of MIB's baseline scripts.

    Read from their source at test time rather than copied, so that an update to
    the vendored checkout breaks this test instead of passing silently.
    """
    import ast
    import os

    path = os.path.join(das._MIB_PATH, rel_path)
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
        if "training_epoch" not in keys:
            continue
        out = {}
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                out[k.value] = v.value
        return out
    raise AssertionError(f"no config dict with training_epoch found in {rel_path}")


def test_library_default_is_not_what_mib_runs():
    """The library default is overridden by every baseline; guard against using it.

    DEFAULT_CONFIG says 3 epochs at lr 1e-2. No baseline uses that combination,
    so treating it as "MIB's settings" would be wrong for every task.
    """
    ioi = _baseline_config("baselines/ioi_baselines/ioi_baselines.py")
    assert ioi["training_epoch"] != DEFAULT_CONFIG["training_epoch"], (
        "IOI baseline no longer overrides the default; re-check das.MIB_TASK_CONFIG")
    print(f"  DEFAULT_CONFIG: epochs={DEFAULT_CONFIG['training_epoch']} "
          f"lr={DEFAULT_CONFIG['init_lr']}")
    print(f"  IOI baseline:   epochs={ioi['training_epoch']} lr={ioi['init_lr']} "
          f"n_features={ioi['n_features']}")


def test_per_task_hyperparameters_match_mib_baselines():
    """Our per-task configuration must equal MIB's baselines, read from source."""
    ioi = _baseline_config("baselines/ioi_baselines/ioi_baselines.py")
    ours = das.mib_config("ioi")
    assert ours["lr"] == ioi["init_lr"], (ours["lr"], ioi["init_lr"])
    assert ours["n_epochs"] == ioi["training_epoch"], (ours["n_epochs"], ioi["training_epoch"])
    assert ours["n_features"] == ioi["n_features"], (ours["n_features"], ioi["n_features"])
    assert ours["is_mib_baseline"] is True
    print(f"  ioi:        lr={ours['lr']} epochs={ours['n_epochs']} "
          f"n_features={ours['n_features']} -- matches baseline")

    arith = _baseline_config("baselines/arithmetic_baselines.py")
    oursa = das.mib_config("arithmetic")
    assert oursa["n_epochs"] == arith["training_epoch"], (oursa, arith)
    assert oursa["n_features"] == arith["n_features"], (oursa, arith)
    # arithmetic sets no init_lr, so it inherits the library default
    assert "init_lr" not in arith
    assert oursa["lr"] == DEFAULT_CONFIG["init_lr"]
    print(f"  arithmetic: lr={oursa['lr']} (inherited) epochs={oursa['n_epochs']} "
          f"n_features={oursa['n_features']} -- matches baseline")

    fallback = das.mib_config("some_task_mib_has_no_baseline_for")
    assert fallback["is_mib_baseline"] is False, (
        "an unknown task must be flagged as not-a-MIB-baseline, so results "
        "cannot claim MIB's settings for a task MIB never configured")
    print(f"  unknown task falls back to library default and is flagged: "
          f"is_mib_baseline={fallback['is_mib_baseline']}")


def test_ioi_batch_size_is_read_not_assumed():
    """IOI batch size comes from get_model_config, and is model dependent.

    Read the literal out of MIB's ioi_utils.py so this cannot drift back into an
    assumption. GPT-2 is 1024; the other supported models are 256.
    """
    import ast
    import os

    path = os.path.join(das._MIB_PATH, "baselines/ioi_baselines/ioi_utils.py")
    tree = ast.parse(open(path).read())
    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
        if "gpt2" not in keys:
            continue
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and isinstance(v, ast.Dict):
                inner = {ik.value: iv.value for ik, iv in zip(v.keys, v.values)
                         if isinstance(ik, ast.Constant) and isinstance(iv, ast.Constant)}
                if "batch_size" in inner:
                    found[k.value] = inner["batch_size"]
    assert found, "could not find the model_configs dict in ioi_utils.py"
    assert found["gpt2"] == 1024, found
    assert das.MIB_IOI_BATCH_BY_MODEL == found, (das.MIB_IOI_BATCH_BY_MODEL, found)
    assert das.mib_config("ioi")["batch_size"] == found["gpt2"]
    assert not das.MIB_BATCH_SIZE_IS_ASSUMED, (
        "nothing should be flagged as assumed now that batch size is read")
    print(f"  get_model_config batch sizes: {found}")
    print(f"  ioi on gpt2 -> batch {das.mib_config('ioi')['batch_size']}, "
          f"assumed set is empty")


def test_ioi_budget_is_a_handful_of_steps():
    """MIB's IOI recipe is large batch, large rate, very few steps."""
    cfg = das.mib_config("ioi")
    for n in (1000, 2000, 4000):
        steps = das.steps_for_epochs(n, cfg["batch_size"], cfg["n_epochs"])
        print(f"  {n} pairs -> {steps} optimiser steps "
              f"(batch {cfg['batch_size']}, {cfg['n_epochs']} epochs, lr {cfg['lr']})")
    assert das.steps_for_epochs(2000, cfg["batch_size"], cfg["n_epochs"]) == 4


def test_ioi_learning_rate_is_actually_one():
    """lr=1.0 is surprising enough that it gets its own assertion."""
    ioi = _baseline_config("baselines/ioi_baselines/ioi_baselines.py")
    assert ioi["init_lr"] == 1.0, ioi["init_lr"]
    assert das.mib_config("ioi")["lr"] == 1.0
    print(f"  MIB's IOI baseline trains at init_lr=1.0; this repo used 1e-3 "
          f"(a factor of {1.0 / 1e-3:.0f})")


def test_optimizer_matches_mib():
    """MIB uses AdamW with weight_decay=0; verify the fitted optimiser agrees."""
    assert das.MIB_CONFIG["optimizer"] == "adamw"
    assert das.MIB_CONFIG["weight_decay"] == 0.0

    seen = {}
    real_adamw = torch.optim.AdamW

    def spy(params, lr=None, weight_decay=None, **kw):
        seen["lr"] = lr
        seen["weight_decay"] = weight_decay
        return real_adamw(params, lr=lr, weight_decay=weight_decay, **kw)

    import test_das as td
    torch.optim.AdamW = spy
    try:
        model, pairs, _ = td.build_toy()
        das.train_das(model, pairs, "hook", "cpu", k=1, n_steps=2)
    finally:
        torch.optim.AdamW = real_adamw

    assert seen["lr"] == DEFAULT_CONFIG["init_lr"], seen
    assert seen["weight_decay"] == 0.0, seen
    print(f"  optimiser AdamW(lr={seen['lr']}, weight_decay={seen['weight_decay']})")


def test_epoch_budget_matches_mib_formula():
    """Budget is epochs over the dataset, not a fixed step count."""
    for n, expected in ((1994, 3 * 63), (32, 3 * 1), (33, 3 * 2), (100, 3 * 4)):
        got = das.steps_for_epochs(n)
        assert got == expected, (n, got, expected)
    print(f"  1994 examples -> {das.steps_for_epochs(1994)} steps "
          f"(3 epochs @ batch 32); the repo previously used a flat 300")


if __name__ == "__main__":
    print("test_standard_arm_is_mib_code");         test_standard_arm_is_mib_code()
    print("test_interchange_matches_mib_arithmetic"); test_interchange_matches_mib_arithmetic()
    print("test_interchange_is_not_trivially_equal"); test_interchange_is_not_trivially_equal()
    print("test_library_default_is_not_what_mib_runs")
    test_library_default_is_not_what_mib_runs()
    print("test_per_task_hyperparameters_match_mib_baselines")
    test_per_task_hyperparameters_match_mib_baselines()
    print("test_ioi_batch_size_is_read_not_assumed")
    test_ioi_batch_size_is_read_not_assumed()
    print("test_ioi_budget_is_a_handful_of_steps")
    test_ioi_budget_is_a_handful_of_steps()
    print("test_ioi_learning_rate_is_actually_one")
    test_ioi_learning_rate_is_actually_one()
    print("test_optimizer_matches_mib");             test_optimizer_matches_mib()
    print("test_epoch_budget_matches_mib_formula");  test_epoch_budget_matches_mib_formula()
    print("\nALL PASS")
