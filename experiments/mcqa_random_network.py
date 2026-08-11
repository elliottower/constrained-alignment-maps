"""Alignment maps on MIB's multiple-choice question answering task.

MIB runs this task on Qwen2.5-0.5B, Gemma-2-2B, and Llama-3.1-8B, and publishes
per-model baselines (Table 3c), so a cross-model claim here is checkable against
numbers we did not produce. Their IOI track is GPT-2 only, which is why it cannot
carry a "not just GPT-2" result no matter how it is configured.

The task is also structurally cleaner than IOI. Its causal model has
`raw_output` <- `answer` <- `answer_pointer` (`simple_MCQA.py:37`), so both target
variables propagate to the generated text and plain accuracy measures them. IOI's
`raw_output` has only `output_token` as a parent, which is why interchanging
`output_position` there is invisible to any text criterion and needs a fitted
logit-difference model instead.

Everything except the maps is MIB's, at their settings
(`baselines/simple_MCQA_baselines.py:73,109`): no `max_length` on the pipeline,
float16, `training_epoch` 8, `n_features` 16, `init_lr` from `DEFAULT_CONFIG`
(1e-2), and MIB's own single-token-position indexer.

    uv run python experiments/mcqa_random_network.py --model-key qwen --layer 12 \
        --out results/mcqa/qwen_L12.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

MIB_ROOT = Path(__file__).resolve().parent.parent / "reference" / "MIB" / "MIB-causal-variable-track"
for _p in (MIB_ROOT, MIB_ROOT / "CausalAbstraction", MIB_ROOT / "baselines"):
    sys.path.insert(0, str(_p))

import qwen_compat  # noqa: F401
from CausalAbstraction.experiments.filter_experiment import FilterExperiment
from CausalAbstraction.experiments.pyvene_core import _collect_features, _train_intervention
from CausalAbstraction.experiments.residual_stream_experiment import (
    LM_loss_and_metric_fn, PatchResidualStream)
from CausalAbstraction.neural.featurizers import Featurizer, SubspaceFeaturizer
from CausalAbstraction.neural.pipeline import LMPipeline
from tasks.simple_MCQA.simple_MCQA import (get_causal_model,
                                           get_counterfactual_datasets,
                                           get_token_positions)

from mib_featurizers import LCPVAEFeaturizer, NonlinearFeaturizer

# MIB's own model list for this task (simple_MCQA_baselines.py:61).
MODELS = {
    "qwen": "Qwen/Qwen2.5-0.5B",
    "gemma": "google/gemma-2-2b",
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}
# `answer_pointer` is XOrder and `answer` is OAnswer in MIB's Table 3c.
TARGET = "answer_pointer"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def checker(output_text, expected):
    """MIB's MCQA checker (simple_MCQA_baselines.py:42)."""
    return expected in output_text


def build_pipeline(model_key, random_init, device, batch_size):
    path = MODELS[model_key]
    if random_init:
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(path))
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        pipeline = LMPipeline(model, max_new_tokens=1, device=device, dtype=torch.float16)
    else:
        pipeline = LMPipeline(path, max_new_tokens=1, device=device, dtype=torch.float16)
    pipeline.tokenizer.padding_side = "left"
    return pipeline


def build_experiment(pipeline, causal_model, layer, token_positions, featurizer,
                     batch_size, k, name):
    featurizers = ({(layer, token_positions[0].id): featurizer}
                   if featurizer is not None else None)
    return PatchResidualStream(
        pipeline=pipeline,
        causal_model=causal_model,
        layers=[layer],
        token_positions=token_positions,
        checker=checker,
        featurizers=featurizers,
        config={
            "method_name": name,
            "batch_size": batch_size,
            "evaluation_batch_size": batch_size,
            # MIB's MCQA settings; `init_lr` is absent there so DEFAULT_CONFIG's
            # 1e-2 applies (simple_MCQA_baselines.py:109).
            "training_epoch": 8,
            "n_features": k,
            "regularization_coefficient": 0.0,
            "output_scores": False,
            "check_raw": False,
            "shuffle": True,
        },
    )


def train_reconstruction(featurizer, experiment, train_datasets, epochs, device):
    """Fit the map on activations alone, with no interchange signal."""
    units = experiment.model_units_lists[0]
    saved = units[0][0].featurizer
    units[0][0].set_featurizer(Featurizer())
    acts = []
    for ds in train_datasets.values():
        collected = _collect_features(ds, experiment.pipeline, units,
                                      experiment.config, collect_counterfactuals=False)
        acts.append(collected[0][0])
    units[0][0].set_featurizer(saved)

    x = torch.cat(acts).to(device).float()
    log(f"    {tuple(x.shape)} activations for reconstruction training")
    core = featurizer.core.to(device).float()
    opt = torch.optim.Adam(core.parameters(), lr=1e-3)
    n = x.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 256):
            batch = x[perm[i:i + 256]]
            recon, _, mu_c, lv_c, mu_n, lv_n = core(batch)
            kl = -0.5 * torch.mean(1 + lv_c - mu_c.pow(2) - lv_c.exp()) \
                 - 0.5 * torch.mean(1 + lv_n - mu_n.pow(2) - lv_n.exp())
            loss = torch.nn.functional.mse_loss(recon, batch) + 1e-3 * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
    return featurizer


def score(results):
    per, all_scores = {}, []
    for name, value in results.get("dataset", {}).items():
        for unit in value.get("model_unit", {}).values():
            s = unit.get(TARGET, {}).get("scores")
            if s:
                per[name] = sum(s) / len(s)
                all_scores += s
    return {"accuracy": sum(all_scores) / len(all_scores) if all_scores else None,
            "n_examples": len(all_scores), "per_dataset_accuracy": per}


def run(model_key, random_init, layer, k, arms, size, device, out_path,
        batch_size=64, vae_hidden=512, vae_epochs=100, on_checkpoint=None):
    t0 = time.time()
    condition = "random_init" if random_init else "pretrained"
    log(f"MCQA {model_key} / {condition} / layer {layer} / k={k}")

    pipeline = build_pipeline(model_key, random_init, device, batch_size)
    d_model = pipeline.model.config.hidden_size
    log(f"  d_model {d_model}, {pipeline.model.config.num_hidden_layers} layers")

    causal_model = get_causal_model()
    datasets = get_counterfactual_datasets(hf=True, size=size)
    token_positions = get_token_positions(pipeline, causal_model)

    if not random_init:
        datasets = FilterExperiment(pipeline, causal_model, checker).filter(
            datasets, verbose=True, batch_size=batch_size)
    train = {a: b for a, b in datasets.items() if "train" in a}
    test = {a: b for a, b in datasets.items() if "test" in a and "private" not in a}
    log(f"  {len(train)} train / {len(test)} test datasets")

    results = {"task": "mcqa", "model_key": model_key, "model": MODELS[model_key],
               "condition": condition, "layer": layer, "k": k,
               "target_variable": TARGET, "d_model": d_model, "methods": {}}

    for arm in arms:
        log(f"  arm: {arm}")
        t_arm = time.time()
        if arm == "random":
            fz = SubspaceFeaturizer(shape=(d_model, k), trainable=False, id="random")
            exp = build_experiment(pipeline, causal_model, layer, token_positions,
                                   fz, batch_size, k, arm)
        elif arm == "das":
            exp = build_experiment(pipeline, causal_model, layer, token_positions,
                                   None, batch_size, k, arm)
            exp.train_interventions(train, [TARGET], method="DAS", verbose=True)
        elif arm in ("lcp_vae", "lcp_vae_interchange", "nldas"):
            if arm == "nldas":
                fz = NonlinearFeaturizer(d_input=d_model, n_features=k,
                                         hidden_dim=vae_hidden, id=arm)
            else:
                fz = LCPVAEFeaturizer(d_input=d_model, z_causal_dim=k,
                                      z_nuisance_dim=k, hidden_dim=vae_hidden,
                                      n_classes=4, id=arm)
            exp = build_experiment(pipeline, causal_model, layer, token_positions,
                                   fz, batch_size, k, arm)
            exp.model_units_lists[0][0][0].set_feature_indices(fz.causal_indices)
            if arm == "lcp_vae":
                train_reconstruction(fz, exp, train, vae_epochs, device)
            else:
                labelled = []
                for ds in train.values():
                    labelled += causal_model.label_counterfactual_data(ds, [TARGET])
                _train_intervention(pipeline, exp.model_units_lists[0], labelled,
                                    "interchange", exp.config, LM_loss_and_metric_fn)
        else:
            raise ValueError(f"unknown arm: {arm}")

        raw = exp.perform_interventions(test, verbose=True,
                                        target_variables_list=[[TARGET]])
        results["methods"][arm] = {**score(raw), "elapsed_seconds": time.time() - t_arm}
        log(f"    {arm}: accuracy {results['methods'][arm]['accuracy']} "
            f"over {results['methods'][arm]['n_examples']} examples")

        results["elapsed_seconds"] = time.time() - t0
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        if on_checkpoint is not None:
            on_checkpoint()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", default="qwen", choices=list(MODELS))
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+",
                    default=["random", "das", "lcp_vae", "lcp_vae_interchange", "nldas"])
    args = ap.parse_args()
    run(args.model_key, args.random_init, args.layer, args.k, args.arms,
        args.size, args.device, args.out, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
