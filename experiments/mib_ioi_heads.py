"""Latent dimension needed to realize IOI's output_position, on MIB's own sites.

The claim this measures: at the attention heads MIB selects, under MIB's own
causal-abstraction metric, a nonlinear map realizes the abstraction at k = 1
while linear Distributed Alignment Search needs a substantially larger subspace.

Everything except the map is MIB's. The heads are fixed benchmark plumbing, not
part of the contribution; the coefficients are MIB's observation model, fit once
and frozen across every arm so the comparison is like for like.

Why `output_position` and why the logit-difference metric. In MIB's IOI causal
model `raw_output` has exactly one parent, `output_token` (`ioi_task.py:94`), so
interchanging `output_position` leaves the generated text unchanged and any
text-based criterion scores a do-nothing map at 1.000 — measured. `logit_diff` is
the only node with both variables as parents, which is what makes position
observable, and MIB's `checker` scores squared error against it.

    modal run experiments/modal_mib_ioi_heads.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

MIB_ROOT = Path(__file__).resolve().parent.parent / "reference" / "MIB" / "MIB-causal-variable-track"
for _p in (MIB_ROOT, MIB_ROOT / "CausalAbstraction", MIB_ROOT / "baselines" / "ioi_baselines"):
    sys.path.insert(0, str(_p))

import qwen_compat  # noqa: F401  pyvene reads Qwen2Config.head_dim for head units
from CausalAbstraction.experiments.attention_head_experiment import PatchAttentionHeads
from CausalAbstraction.experiments.filter_experiment import FilterExperiment
from CausalAbstraction.experiments.pyvene_core import _train_intervention
from CausalAbstraction.neural.featurizers import SubspaceFeaturizer
from ioi_utils import (checker, filter_checker, ioi_loss_and_metric_fn,
                       setup_pipeline)
from tasks.IOI_task.ioi_task import (get_causal_model, get_counterfactual_datasets,
                                     get_token_positions)

from mib_featurizers import LCPVAEFeaturizer

TARGET = "output_position"
# MIB's own defaults for gpt2 (baselines/ioi_baselines/ioi_baselines.py:29).
HEADS = [(7, 3), (7, 9), (8, 6), (8, 10)]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def head_size(pipeline):
    cfg = pipeline.model.config
    return getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)


def build(pipeline, causal_model, token_positions, featurizers, k, batch_size, name):
    # IOI's baseline config. `init_lr` of 1.0 belongs to these 64-dim head units,
    # not to a d_model-wide rotation (ioi_baselines.py:144).
    config = {
        "method_name": name,
        "batch_size": batch_size,
        "evaluation_batch_size": batch_size,
        "training_epoch": 2,
        "init_lr": 1.0,
        "n_features": k,
        "regularization_coefficient": 0.0,
        "output_scores": True,
        "check_raw": True,
        "shuffle": True,
        "temperature_schedule": (1.0, 0.01),
        # For the attention-head path MIB passes the loss through the config
        # rather than as a constructor argument (ioi_baselines.py:155).
        "loss_and_metric_fn": ioi_loss_and_metric_fn,
    }
    exp = PatchAttentionHeads(
        pipeline=pipeline,
        causal_model=causal_model,
        layer_head_list=HEADS,
        token_positions=token_positions,
        checker=lambda logits, params: checker(logits, params, pipeline),
        featurizers=featurizers,
        config=config,
    )
    # The base class reads `self.loss_and_metric_fn`; the head experiment does not
    # take it as a named argument. Set it explicitly rather than relying on which
    # channel it arrives through, and fail loudly if the attribute moves.
    exp.loss_and_metric_fn = ioi_loss_and_metric_fn
    assert getattr(exp, "loss_and_metric_fn", None) is ioi_loss_and_metric_fn
    return exp


def mean_score(results):
    per, all_scores = {}, []
    for name, value in results.get("dataset", {}).items():
        for unit in value.get("model_unit", {}).values():
            s = unit.get(TARGET, {}).get("scores")
            if s:
                per[name] = sum(s) / len(s)
                all_scores += s
    return {"mib_squared_error": sum(all_scores) / len(all_scores) if all_scores else None,
            "n_examples": len(all_scores), "per_dataset": per}


def run(model_key, ks, linear_params, out_path, size=None, vae_hidden=128,
        on_checkpoint=None):
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline, batch_size = setup_pipeline(model_key, device)
    pipeline.tokenizer.padding_side = "left"
    d_head = head_size(pipeline)
    log(f"{model_key}: head_dim {d_head}, heads {HEADS}, batch {batch_size}")

    causal_model = get_causal_model(linear_params)
    datasets = get_counterfactual_datasets(hf=True, size=size)
    token_positions = get_token_positions(pipeline, causal_model)

    filt = FilterExperiment(pipeline, causal_model, filter_checker)
    datasets = filt.filter(datasets, verbose=True, batch_size=batch_size)
    train = {k: v for k, v in datasets.items() if "train" in k}
    test = {k: v for k, v in datasets.items()
            if "test" in k and "private" not in k}
    log(f"  {len(train)} train / {len(test)} test datasets")

    results = {"model_key": model_key, "target_variable": TARGET, "heads": HEADS,
               "d_head": d_head, "linear_params": linear_params, "by_k": {}}

    for k in ks:
        results["by_k"][str(k)] = {}
        for arm in ("random", "das", "lcp_vae_interchange"):
            log(f"  k={k} arm={arm}")
            t_arm = time.time()
            pos_id = token_positions[0].id

            if arm == "random":
                fz = {(l, h, pos_id): SubspaceFeaturizer(shape=(d_head, k),
                                                         trainable=False, id="random")
                      for l, h in HEADS}
                exp = build(pipeline, causal_model, token_positions, fz, k, batch_size, arm)
            elif arm == "das":
                exp = build(pipeline, causal_model, token_positions, None, k, batch_size, arm)
                exp.train_interventions(train, [TARGET], method="DAS", verbose=True)
            else:
                fz = {(l, h, pos_id): LCPVAEFeaturizer(
                          d_input=d_head, z_causal_dim=k, z_nuisance_dim=k,
                          hidden_dim=vae_hidden, n_classes=2, id=arm)
                      for l, h in HEADS}
                exp = build(pipeline, causal_model, token_positions, fz, k, batch_size, arm)
                for units in exp.model_units_lists:
                    for group in units:
                        for unit in group:
                            unit.set_feature_indices(list(range(k)))
                labelled = []
                for ds in train.values():
                    labelled += causal_model.label_counterfactual_data(ds, [TARGET])
                for units in exp.model_units_lists:
                    _train_intervention(pipeline, units, labelled, "interchange",
                                        exp.config, ioi_loss_and_metric_fn)

            raw = exp.perform_interventions(test, verbose=True,
                                            target_variables_list=[[TARGET]])
            results["by_k"][str(k)][arm] = {**mean_score(raw),
                                            "elapsed_seconds": time.time() - t_arm}
            log(f"    {arm} k={k}: MIB squared error "
                f"{results['by_k'][str(k)][arm]['mib_squared_error']}")

            # Checkpoint after every arm: a k-sweep is hours.
            results["elapsed_seconds"] = time.time() - t0
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            if on_checkpoint is not None:
                on_checkpoint()

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", default="gpt2")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--vae-hidden", type=int, default=128)
    ap.add_argument("--linear-params", required=True,
                    help="JSON file or dict with bias, token_coeff, position_coeff")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if os.path.isfile(args.linear_params):
        with open(args.linear_params) as f:
            params = json.load(f)
        params = params.get(args.model_key, params)
    else:
        params = json.loads(args.linear_params)

    run(args.model_key, args.ks, params, args.out, args.size, args.vae_hidden)


if __name__ == "__main__":
    main()
