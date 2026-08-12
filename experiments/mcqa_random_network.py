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
(1e-2), and MIB's own three token positions (`correct_symbol`,
`correct_symbol_period`, `last_token`), each carrying its own copy of the map.

    uv run python experiments/mcqa_random_network.py --model-key qwen --layer 12 \
        --out results/mcqa/qwen_L12.json
"""

import argparse
import json
import random
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
from tasks.simple_MCQA import simple_MCQA as _mcqa
from tasks.two_digit_addition_task import arithmetic as _arith
from tasks.IOI_task import ioi_task as _ioi

from mib_featurizers import (DirectionalFeaturizer, FlowFeaturizer,
                             LCPVAEFeaturizer, NonlinearFeaturizer)

# MIB's own model list for this task (simple_MCQA_baselines.py:61).
MODELS = {
    "qwen": "Qwen/Qwen2.5-0.5B",
    "gemma": "google/gemma-2-2b",
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}
# `answer_pointer` is XOrder and `answer` is OAnswer in MIB's Table 3c.
TARGET = "answer_pointer"

# Variables of known cardinality, for calibrating the write-rank measurement.
# A variable taking c values needs c-1 dimensions for its values to sit in
# general position, so a rank that tracks c-1 across variables is measuring the
# variable; a rank that is constant is measuring the map.
TASKS = {
    "mcqa": {"module": _mcqa, "targets": {"answer_pointer": 4, "answer": 4}},
    "arithmetic": {"module": _arith,
                   "targets": {"ones_carry": 2, "ones_out": 10, "tens_out": 10}},
    # `output_token` is a name drawn from the task's pool, so its cardinality is
    # read off the data rather than declared. `output_position` is excluded here:
    # `raw_output` has only `output_token` as a parent (`ioi_task.py:94`), so
    # interchanging position is invisible to a text criterion and to the loss
    # that reads its labels from it.
    "ioi": {"module": _ioi, "targets": {"output_token": None},
            "model_args": {"bias": 0.0, "token_coeff": 0.0, "position_coeff": 0.0},
            "space_prefix": True},
}


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


def build_experiment(pipeline, causal_model, layer, token_positions, make_featurizer,
                     batch_size, k, name, das_epochs=8):
    """One experiment over every token position MIB defines for this task.

    `get_token_positions` returns three (`simple_MCQA.py`): `correct_symbol`,
    `correct_symbol_period`, and `last_token`. Keying a featurizer dict on only
    the first attaches the map to a mid-prompt choice label and leaves the other
    positions with default identity featurizers — including `last_token`, where
    the model actually predicts. That is silent: nothing errors, training loss
    sits at log(vocab), and every arm reports zero.

    `make_featurizer` is a factory so each position gets its own parameters,
    which is what MIB's per-unit featurizer dict expects.
    """
    featurizers = None
    if make_featurizer is not None:
        featurizers = {(layer, tp.id): make_featurizer() for tp in token_positions}
        # `PatchResidualStream` silently substitutes a default identity Featurizer
        # for any key it cannot find or that holds None, which reads downstream as
        # "the map does nothing" rather than as an error.
        assert all(v is not None for v in featurizers.values()), \
            "featurizer factory returned None"
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
            "training_epoch": das_epochs,
            "n_features": k,
            "regularization_coefficient": 0.0,
            "output_scores": False,
            "check_raw": False,
            "shuffle": True,
        },
    )


def _collect_acts_and_labels(experiment, causal_model, train_datasets, units):
    """Activations at this unit, with the causal-variable label for each.

    `_collect_features` walks each dataset in order, so running the causal model
    over the same datasets in the same order aligns labels to rows.
    """
    saved = units[0][0].featurizer
    units[0][0].set_featurizer(Featurizer())
    acts, labels = [], []
    for ds in train_datasets.values():
        collected = _collect_features(ds, experiment.pipeline, units,
                                      experiment.config, collect_counterfactuals=False)
        acts.append(collected[0][0])
        labels += [causal_model.run_forward(ex["input"])[TARGET] for ex in ds]
    units[0][0].set_featurizer(saved)
    x = torch.cat(acts)
    # Values may be strings (a name) or integers (a digit); encode either way, and
    # let the observed set define the cardinality rather than declaring it.
    vocab = {v: i for i, v in enumerate(sorted({str(v) for v in labels}))}
    y = torch.tensor([vocab[str(v)] for v in labels[:x.shape[0]]], dtype=torch.long)
    return x, y, len(vocab)


def train_pi_sae(featurizer, experiment, causal_model, train_datasets, device,
                 units, epochs=500, batch_size=128, lr=1e-3, alpha=10.0,
                 l1_coeff=0.0):
    """The recipe that produced 1.000 at k=1 on GPT-2 indirect object identification.

    Ported from `train_vae_family` in the original repo. Four terms at the weights
    used there:

        loss = recon + kl_c + kl_n + alpha * ce   (+ l1_coeff * |mu_c|)

    `kl_c` is against a *label-conditional* prior, one learned mean and variance
    per class, not against a standard normal. `ce` is a classifier on the causal
    block at weight 10. Dropping either one, as an earlier version here did,
    leaves the map at the random-subspace floor.

    There is deliberately no interchange term. The arm carrying it scored 0.961 on
    a randomly initialized network where this one scored 0.000, so the interchange
    objective is what the vacuity result turns on.
    """
    x, y, _ = _collect_acts_and_labels(experiment, causal_model, train_datasets, units)
    x, y = x.to(device).float(), y.to(device)
    core = featurizer.core.to(device).float()
    opt = torch.optim.Adam(core.parameters(), lr=lr)
    n = x.shape[0]
    log(f"    fitting on {tuple(x.shape)} activations, {int(y.max()) + 1} classes")
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        last = None
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = x[idx], y[idx]
            x_r, logits, mu_c, lv_c, mu_n, lv_n = core(xb)
            recon = torch.nn.functional.mse_loss(x_r, xb)
            p_mu, p_lv = core.prior_mu(yb), core.prior_logvar(yb)
            kl_c = -0.5 * (1 + lv_c - p_lv
                           - ((mu_c - p_mu).pow(2) + lv_c.exp()) / p_lv.exp()).mean()
            kl_n = -0.5 * (1 + lv_n - mu_n.pow(2) - lv_n.exp()).mean()
            ce = torch.nn.functional.cross_entropy(logits, yb)
            loss = recon + kl_c + kl_n + alpha * ce
            if l1_coeff > 0:
                loss = loss + l1_coeff * mu_c.abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = (recon.item(), kl_c.item(), ce.item())
        if epoch % 100 == 0 or epoch == epochs - 1:
            log(f"    epoch {epoch}: recon {last[0]:.4f} kl_c {last[1]:.4f} ce {last[2]:.4f}")
    return featurizer


def effective_write_rank(deltas):
    """How many independent directions a map actually writes.

    `k` counts dimensions in each map's own coordinate system, and those systems
    are not comparable: a rotation's k-th coordinate is a fixed direction in
    activation space, while a decoder's is an input-dependent one. This measures
    the thing `k` is supposed to stand for. For an orthogonal subspace map it
    equals k by construction, which is what makes it a fair common axis.

    Reported as the participation ratio of the delta spectrum,
    (sum s_i^2)^2 / sum s_i^4, alongside the rank capturing 95% of variance.
    """
    d = deltas - deltas.mean(0, keepdim=True)
    s = torch.linalg.svdvals(d.float())
    p2 = s.pow(2)
    participation = (p2.sum() ** 2 / p2.pow(2).sum()).item()
    cum = torch.cumsum(p2, 0) / p2.sum()
    rank95 = int((cum < 0.95).sum().item()) + 1
    return {"participation_ratio": participation, "rank95": rank95,
            "n_deltas": int(deltas.shape[0])}


def collect_deltas(featurizer, x_base, x_src, device):
    """The activation deltas a map writes, base -> source, over held-out pairs."""
    with torch.no_grad():
        xb = x_base.to(device).float()
        xs = x_src.to(device).float()
        fb, eb = featurizer.featurize(xb)
        fs, _ = featurizer.featurize(xs)
        f_iv = fb.clone()
        idx = featurizer.causal_indices
        f_iv[:, idx] = fs[:, idx]
        return (featurizer.inverse_featurize(f_iv, eb) - xb).cpu()


def causal_feature_directions(featurizer, x, device, eps=1.0):
    """The activation-space direction each causal latent writes.

    A finite difference rather than a decoder weight column, because the decoder
    is nonlinear and no single column is the direction it writes. Averaged over
    inputs so the result is a property of the map, not of one example.

    These are what makes the gauge question answerable: DAS recovers a subspace
    whose basis is arbitrary (ten seeds, pairwise overlap 0.008), because a
    rotation has no privileged basis. If an L1-penalized causal latent selects a
    canonical basis instead, these directions agree across seeds; if they do not,
    its features are as arbitrary as DAS's and no feature story is warranted.
    """
    core = featurizer.core.to(device).float()
    with torch.no_grad():
        z = core.encode_mean(x.to(device).float())
        base = core.decoder(z)
        dirs = []
        for j in range(core.z_sparse):
            z_j = z.clone()
            z_j[:, j] += eps
            dirs.append(((core.decoder(z_j) - base) / eps).mean(0))
    return torch.stack(dirs).cpu()


def score(results):
    per, all_scores = {}, []
    for name, value in results.get("dataset", {}).items():
        for unit_key, unit in value.get("model_unit", {}).items():
            s = unit.get(TARGET, {}).get("scores")
            if s:
                pos = unit.get("metadata", {}).get("position", "?")
                per[f"{name}@{pos}"] = sum(s) / len(s)
                all_scores += s
    return {"accuracy": sum(all_scores) / len(all_scores) if all_scores else None,
            "n_examples": len(all_scores), "per_dataset_accuracy": per}


def run(model_key, random_init, layer, k, arms, size, device, out_path,
        batch_size=64, vae_hidden=256, vae_epochs=500, recon_lr=1e-3,
        flow_layers=4, flow_lr=1e-3, flow_epochs=20,
        vae_lr=1e-3, vae_ix_epochs=30, expansion=8, alpha=10.0,
        l1_coeff=0.0, prototype_write=False, z_nuisance=None, seed=0,
        das_epochs=8, task="mcqa", target_variable=TARGET,
        on_checkpoint=None):
    t0 = time.time()
    # Seeds map initialization and training only; the dataset and split are fixed
    # upstream by MIB, so a seed varies the map and nothing else.
    global TARGET
    TARGET = target_variable
    random.seed(seed)
    torch.manual_seed(seed)
    condition = "random_init" if random_init else "pretrained"
    log(f"MCQA {model_key} / {condition} / layer {layer} / k={k} / seed={seed}")

    pipeline = build_pipeline(model_key, random_init, device, batch_size)
    d_model = pipeline.model.config.hidden_size
    log(f"  d_model {d_model}, {pipeline.model.config.num_hidden_layers} layers")

    spec = TASKS[task]
    mod = spec["module"]
    declared = spec["targets"][TARGET]
    causal_model = (mod.get_causal_model(spec["model_args"])
                    if "model_args" in spec else mod.get_causal_model())
    if spec.get("space_prefix"):
        # The model emits " Mary" while the causal model returns "Mary", and the
        # loss compares token ids, so the bare form points training at a token the
        # model never produces.
        causal_model.mechanisms["raw_output"] = lambda output_token: " " + output_token
    datasets = mod.get_counterfactual_datasets(hf=True, size=size)
    token_positions = mod.get_token_positions(pipeline, causal_model)

    if not random_init:
        datasets = FilterExperiment(pipeline, causal_model, checker).filter(
            datasets, verbose=True, batch_size=batch_size)
    # Cardinality is read off the data, not declared: a value set can be smaller
    # than the causal model's nominal one, and `output_token` has no fixed size.
    n_classes = len({str(causal_model.run_forward(ex["input"])[TARGET])
                     for a, b in datasets.items() if "train" in a for ex in b})
    log(f"  target {TARGET}: {n_classes} distinct values observed"
        + (f" (declared {declared})" if declared else ""))
    train = {a: b for a, b in datasets.items() if "train" in a}
    test = {a: b for a, b in datasets.items() if "test" in a and "private" not in a}
    log(f"  {len(train)} train / {len(test)} test datasets")

    results = {"task": task, "model_key": model_key, "model": MODELS[model_key],
               "condition": condition, "layer": layer, "k": k, "seed": seed,
               "target_variable": TARGET, "n_classes": n_classes, "declared_cardinality": declared, "d_model": d_model, "methods": {}}

    for arm in arms:
        log(f"  arm: {arm}")
        t_arm = time.time()
        if arm == "random":
            exp = build_experiment(
                pipeline, causal_model, layer, token_positions,
                lambda: SubspaceFeaturizer(shape=(d_model, k), trainable=False,
                                           id="random"),
                batch_size, k, arm, das_epochs)
        elif arm == "das":
            exp = build_experiment(pipeline, causal_model, layer, token_positions,
                                   None, batch_size, k, arm, das_epochs)
            exp.train_interventions(train, [TARGET], method="DAS", verbose=True)
        elif arm == "flow_interchange":
            # Invertible by construction, so there is nothing to reconstruct and
            # no pretraining phase: the only objective is interchange, exactly as
            # for DAS. `n_features` is d_model because the flow is a bijection;
            # `k` selects how many coordinates the interchange writes.
            exp = build_experiment(
                pipeline, causal_model, layer, token_positions,
                lambda: FlowFeaturizer(d_input=d_model, k=k,
                                       hidden_dim=vae_hidden, n_layers=flow_layers,
                                       id=arm),
                batch_size, k, arm)
            for units in exp.model_units_lists:
                for group in units:
                    for unit in group:
                        unit.set_feature_indices(list(range(k)))
            labelled = []
            for ds in train.values():
                labelled += causal_model.label_counterfactual_data(ds, [TARGET])
            # MIB's 1e-2 is tuned for an orthogonal rotation, which has no free
            # scale. A coupling flow has hundreds of thousands of unconstrained
            # parameters and needs its own rate.
            saved_lr = exp.config["init_lr"]
            exp.config["init_lr"] = flow_lr
            exp.config["training_epoch"] = flow_epochs
            for units in exp.model_units_lists:
                _train_intervention(pipeline, units, labelled, "interchange",
                                    exp.config, LM_loss_and_metric_fn)
            exp.config["init_lr"] = saved_lr
        elif arm == "directional":
            # Rank-k write, nonlinear readout. Trained on the same four-term
            # objective as `lcp_vae`, so the only difference between the two arms
            # is what the decoder is allowed to write.
            exp = build_experiment(
                pipeline, causal_model, layer, token_positions,
                lambda: DirectionalFeaturizer(d_input=d_model, k=k,
                                              hidden_dim=vae_hidden, n_classes=n_classes,
                                              expansion_factor=expansion, id=arm),
                batch_size, k, arm, das_epochs)
            for units in exp.model_units_lists:
                for group in units:
                    for unit in group:
                        unit.set_feature_indices(
                            units[0][0].featurizer.causal_indices)
            for units in exp.model_units_lists:
                train_pi_sae(units[0][0].featurizer, exp, causal_model, train,
                             device, units, epochs=vae_epochs, lr=recon_lr,
                             alpha=alpha, l1_coeff=l1_coeff)
        elif arm in ("lcp_vae", "lcp_vae_interchange", "nldas"):
            def make():
                if arm == "nldas":
                    return NonlinearFeaturizer(d_input=d_model, n_features=k,
                                               hidden_dim=vae_hidden, id=arm)
                # Sizing from the original builder: nuisance = max(4k, 4),
                # hidden 256, causal block widened by `expansion`.
                return LCPVAEFeaturizer(d_input=d_model, z_causal_dim=k,
                                        z_nuisance_dim=z_nuisance or max(4 * k, 4),
                                        hidden_dim=vae_hidden, n_classes=n_classes,
                                        expansion_factor=expansion,
                                        prototype_write=prototype_write, id=arm)

            exp = build_experiment(pipeline, causal_model, layer, token_positions,
                                   make, batch_size, k, arm, das_epochs)
            for units in exp.model_units_lists:
                for group in units:
                    for unit in group:
                        unit.set_feature_indices(
                            units[0][0].featurizer.causal_indices)
            if arm in ("lcp_vae", "lcp_vae_interchange"):
                for units in exp.model_units_lists:
                    train_pi_sae(units[0][0].featurizer, exp, causal_model, train,
                                 device, units, epochs=vae_epochs, lr=recon_lr,
                                 alpha=alpha, l1_coeff=l1_coeff)
            # Reconstruction pretraining runs above for both VAE arms. DAS's
            # inverse is the transpose of an orthogonal rotation and costs nothing
            # to learn; the VAE must learn a decoder, and learning one from the
            # interchange loss alone left it writing garbage — at k <= 4 it scored
            # 0.000 on sensitivity *and* specificity, which is destroying the
            # output rather than failing to find the variable.
            if arm != "lcp_vae":
                labelled = []
                for ds in train.values():
                    labelled += causal_model.label_counterfactual_data(ds, [TARGET])
                saved_lr = exp.config["init_lr"]
                saved_ep = exp.config["training_epoch"]
                exp.config["init_lr"] = vae_lr
                exp.config["training_epoch"] = vae_ix_epochs
                for units in exp.model_units_lists:
                    _train_intervention(pipeline, units, labelled, "interchange",
                                        exp.config, LM_loss_and_metric_fn)
                exp.config["init_lr"] = saved_lr
                exp.config["training_epoch"] = saved_ep
        else:
            raise ValueError(f"unknown arm: {arm}")

        raw = exp.perform_interventions(test, verbose=True,
                                        target_variables_list=[[TARGET]])
        results["methods"][arm] = {**score(raw), "elapsed_seconds": time.time() - t_arm}
        if arm in ("lcp_vae", "lcp_vae_interchange", "directional"):
            units = exp.model_units_lists[-1]
            acts, _, _ = _collect_acts_and_labels(exp, causal_model, train, units)
            fz = units[0][0].featurizer
            dirs = causal_feature_directions(fz, acts[:256], device)
            results["methods"][arm]["causal_directions"] = dirs.tolist()
            half = acts.shape[0] // 2
            deltas = collect_deltas(fz, acts[:half], acts[half:2 * half], device)
            results["methods"][arm]["write_rank"] = effective_write_rank(deltas)
            if hasattr(fz.core, "unit_directions"):
                # Explicit vectors: no probing needed, and directly comparable
                # across seeds without a nonlinear decoder in the way.
                results["methods"][arm]["explicit_directions"] = \
                    fz.core.unit_directions().detach().cpu().tolist()
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
