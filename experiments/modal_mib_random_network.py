"""Modal wrapper for the MIB random-network control.

Holds no experiment logic. `mib_random_network.py` runs unchanged locally; this
file supplies the image, the GPU, the volume, and the ordering.

Both conditions run the same IOI task and are scored on interchange accuracy,
which needs nothing fit in advance.

    modal run experiments/modal_mib_random_network.py --model-key qwen --layer 15
"""

import modal

# Pinned to what the local smoke test ran on, except torch, which is the CUDA
# build. Ranges have cost this project five failed launches in one session.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "pyvene==0.1.8",
        "scikit-learn==1.9.0",
        "tensorboard==2.20.0",
        "seaborn==0.13.2",
        "matplotlib==3.9.2",
        "datasets==3.1.0",
        "accelerate==1.1.1",
        "pandas==2.2.3",
        "networkx==3.4.2",
        "tqdm==4.67.1",
    )
    .add_local_dir("experiments", "/root/experiments")
    .add_local_dir("reference/MIB/MIB-causal-variable-track",
                   "/root/reference/MIB/MIB-causal-variable-track")
)

app = modal.App("mib-random-network", image=image)
results_vol = modal.Volume.from_name("fc-results", create_if_missing=True)

RESULTS = "/results/mib_random_network"
# The fitter's own defaults for gpt2 are the name-mover heads; the qwen pair in
# MIB's README is a usage example, not a fitted setting, and produced R^2 = 3e-5.
FIT_HEADS = {"gpt2": ["(7,3)", "(7,9)", "(8,6)", "(8,10)"]}
# Qwen2.5-0.5B and GPT-2 are both under 4GB in fp16 with batches of cached
# activations. Nothing here needs a bigger card.
GPU = "L4"
TIMEOUT = 86400


@app.function(gpu=GPU, timeout=TIMEOUT, volumes={"/results": results_vol},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def fit_linear_params(model_key: str = "gpt2"):
    """MIB's own coefficient fitter. Fit once, then frozen across every arm."""
    import json, os, subprocess, sys
    out = f"{RESULTS}/linear_params/{model_key}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    mib = "/root/reference/MIB/MIB-causal-variable-track"
    env = dict(os.environ)
    env["PYTHONPATH"] = (f"/root/experiments:{mib}:{mib}/CausalAbstraction:"
                         f"{mib}/baselines/ioi_baselines")
    cmd = [sys.executable, "baselines/ioi_baselines/ioi_learn_linear_params.py",
           "--model", model_key, "--output_file", out]
    if model_key in FIT_HEADS:
        cmd += ["--heads_list", *FIT_HEADS[model_key]]
    subprocess.run(cmd, cwd=mib, env=env, check=True)
    results_vol.commit()
    with open(out) as f:
        params = json.load(f)
    inner = params.get(model_key, params)
    print(f"fitted {model_key}: {inner}", flush=True)
    # R^2 near zero means the coefficients carry no position signal and the
    # target logit difference degenerates to a constant.
    if float(inner.get("score", 0.0)) < 0.05:
        print(f"WARNING: degenerate fit, R^2={inner.get('score')}", flush=True)
    return inner


@app.function(gpu=GPU, timeout=TIMEOUT, volumes={"/results": results_vol},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def run_condition(model_key: str, random_init: bool, layer: int, k: int,
                  vae_hidden: int, vae_epochs: int, arms: list, size,
                  target_variable: str = "output_token"):
    import sys
    sys.path.insert(0, "/root/experiments")
    # The runner resolves MIB relative to its own parent, which the image mirrors.
    from mib_random_network import run

    condition = "random_init" if random_init else "pretrained"
    out = f"{RESULTS}/{model_key}_{condition}_L{layer}_k{k}_{target_variable}.json"
    # The two-way criterion needs a variable whose expected answer changes under
    # interchange. `raw_output`'s only parent is `output_token` (ioi_task.py:94),
    # so interchanging `output_position` leaves it fixed and every pair is
    # excluded for want of counterfactual signal.
    return run(model_key, random_init, layer, k, vae_hidden, vae_epochs, arms,
               size, "cuda", out, on_checkpoint=results_vol.commit,
               target_variable=target_variable)


@app.function(gpu=GPU, timeout=TIMEOUT, volumes={"/results": results_vol},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def sweep_layers(model_key: str, k: int, size):
    """DAS and the floor arm at every layer, on the pretrained model.

    MIB's `residual_stream_baselines` runs `range(start, end)` and reads results
    best-per-layer. A single hand-picked layer can sit before the site where the
    variable is written and read as total failure.
    """
    import gc, json, os, sys, time
    import torch
    sys.path.insert(0, "/root/experiments")
    from mib_random_network import build_pipeline, run

    pipeline, _ = build_pipeline(model_key, False, "cuda")
    n_layers = pipeline.model.config.num_hidden_layers
    del pipeline

    out = f"{RESULTS}/sweep_{model_key}_k{k}.json"
    done = {}
    if os.path.exists(out):
        with open(out) as f:
            done = json.load(f)
    for layer in range(n_layers):
        if str(layer) in done:
            continue
        t0 = time.time()
        # Batch capped: MIB's gpt2 default of 1024 with 8-epoch training does not
        # fit an L4. Batch size changes memory and step granularity, not the math.
        res = run(model_key, False, layer, k, 512, 100, ["random", "das"],
                  size, "cuda", f"/tmp/sweep_{model_key}_L{layer}.json",
                  batch_size=64)
        done[str(layer)] = {
            arm: {"accuracy": m["accuracy"],
                  "per_dataset_accuracy": m["per_dataset_accuracy"]}
            for arm, m in res["methods"].items()
        }
        done[str(layer)]["elapsed_seconds"] = time.time() - t0
        summary = "  ".join(f"{a}={done[str(layer)][a]['accuracy']:.3f}" for a in ("random", "das"))
        # Checkpoint per layer: a sweep is hours and the container can vanish.
        with open(out, "w") as f:
            json.dump(done, f, indent=2)
        results_vol.commit()
        # `run` builds a model per layer; without this the caching allocator holds
        # every previous layer's blocks and the second layer OOMs.
        del res
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[layer {layer}] {summary}", flush=True)
    return done


@app.function(gpu=GPU, timeout=TIMEOUT, volumes={"/results": results_vol},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def ioi_heads_sweep(model_key: str, ks: list, linear_params: dict, size, vae_hidden: int):
    """Registered latent-dimension sweep at MIB's attention heads.

    Design frozen in `prereg_mib_ioi_latent_dimension.md`.
    """
    import sys
    sys.path.insert(0, "/root/experiments")
    from mib_ioi_heads import run

    out = f"{RESULTS}/ioi_heads_{model_key}.json"
    return run(model_key, ks, linear_params, out, size=size,
               vae_hidden=vae_hidden, on_checkpoint=results_vol.commit)


@app.function(gpu=GPU, timeout=TIMEOUT, volumes={"/results": results_vol},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def mcqa_condition(model_key: str, random_init: bool, layer: int, k: int,
                   arms: list, size, batch_size: int, vae_hidden: int = 256,
                   vae_epochs: int = 500, recon_lr: float = 1e-3,
                   flow_layers: int = 4, flow_lr: float = 1e-3,
                   flow_epochs: int = 20, vae_lr: float = 1e-3,
                   vae_ix_epochs: int = 30, expansion: int = 8,
                   alpha: float = 10.0, l1_coeff: float = 0.0,
                   prototype_write: bool = False, z_nuisance: int = 0,
                   seed: int = 0, das_epochs: int = 8,
                   task: str = "mcqa", target_variable: str = "answer_pointer"):
    """MIB's multiple-choice task. Design frozen in prereg_mcqa_random_network.md."""
    import sys
    sys.path.insert(0, "/root/experiments")
    from mcqa_random_network import run

    condition = "random_init" if random_init else "pretrained"
    tag = f"h{vae_hidden}_e{vae_epochs}_lr{recon_lr}_fl{flow_layers}_x{expansion}_a{alpha}_l1{l1_coeff}_p{int(prototype_write)}_zn{z_nuisance}_s{seed}_de{das_epochs}"
    out = f"{RESULTS}/{task}-{target_variable}_{model_key}_{condition}_L{layer}_k{k}_{tag}.json"
    return run(model_key, random_init, layer, k, arms, size, "cuda", out,
               batch_size=batch_size, vae_hidden=vae_hidden,
               vae_epochs=vae_epochs, recon_lr=recon_lr,
               flow_layers=flow_layers, flow_lr=flow_lr,
               flow_epochs=flow_epochs, vae_lr=vae_lr,
               vae_ix_epochs=vae_ix_epochs, expansion=expansion,
               alpha=alpha, l1_coeff=l1_coeff,
               prototype_write=prototype_write,
               z_nuisance=z_nuisance or None, seed=seed,
               das_epochs=das_epochs, task=task,
               target_variable=target_variable,
               on_checkpoint=results_vol.commit)


@app.function(gpu=GPU, timeout=TIMEOUT, volumes={"/results": results_vol},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def mib_mcqa_baseline(quick: bool = False):
    """MIB's own MCQA baseline script, unmodified.

    The reproduction check: if their script produces their published numbers here
    and ours does not, the difference is in our code and is diffable. If theirs
    also fails, the cause is environmental and nothing downstream is trustworthy.
    """
    import os, subprocess, sys
    mib = "/root/reference/MIB/MIB-causal-variable-track"
    env = dict(os.environ)
    env["PYTHONPATH"] = f"/root/experiments:{mib}:{mib}/CausalAbstraction:{mib}/baselines"
    out = f"{RESULTS}/mib_mcqa_baseline"
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "baselines/simple_MCQA_baselines.py",
           "--skip_gemma", "--skip_llama", "--skip_answer",
           "--methods", "DAS", "full_vector",
           "--batch_size", "32", "--eval_batch_size", "64",
           "--results_dir", out, "--model_dir", "/tmp/mcqa_models"]
    if quick:
        cmd.append("--quick_test")
    subprocess.run(cmd, cwd=mib, env=env, check=True)
    results_vol.commit()
    return out


@app.local_entrypoint()
def repro(quick: bool = False):
    mib_mcqa_baseline.spawn(quick)
    print("spawned MIB's own MCQA baseline (Qwen, answer_pointer, DAS + full_vector)",
          flush=True)


@app.local_entrypoint()
def mcqa(model_key: str = "qwen", layer: int = 12, k: int = 16, size: int = None,
         batch_size: int = 64, vae_hidden: int = 256, vae_epochs: int = 500,
         recon_lr: float = 1e-3, flow_layers: int = 4,
         flow_lr: float = 1e-3, flow_epochs: int = 20,
         vae_lr: float = 1e-3, vae_ix_epochs: int = 30,
         expansion: int = 8, alpha: float = 10.0, l1_coeff: float = 0.0,
         prototype_write: bool = False, z_nuisance: int = 0, seeds: str = "0",
         das_epochs: int = 8, task: str = "mcqa",
         target_variable: str = "answer_pointer",
         pretrained_only: bool = False,
         arms: str = "random,das,lcp_vae,lcp_vae_interchange,nldas"):
    arm_list = arms.split(",")
    conditions = (False,) if pretrained_only else (False, True)
    for sd in [int(x) for x in seeds.split(",")]:
        for random_init in conditions:
            mcqa_condition.spawn(model_key, random_init, layer, k, arm_list, size,
                                 batch_size, vae_hidden, vae_epochs, recon_lr,
                                 flow_layers, flow_lr, flow_epochs, vae_lr,
                                 vae_ix_epochs, expansion, alpha, l1_coeff,
                                 prototype_write, z_nuisance, sd, das_epochs,
                                 task, target_variable)
    print(f"spawned MCQA {model_key} L{layer} k{k}, both conditions", flush=True)


@app.local_entrypoint()
def heads(model_key: str = "gpt2", ks: str = "1,2,4,8,16,32",
          size: int = None, vae_hidden: int = 128):
    params = fit_linear_params.remote(model_key)
    print(f"coefficients (frozen for every arm): {params}", flush=True)
    ioi_heads_sweep.spawn(model_key, [int(x) for x in ks.split(",")],
                          params, size, vae_hidden)
    print(f"spawned k-sweep for {model_key}; results in {RESULTS}", flush=True)


@app.local_entrypoint()
def main(model_key: str = "qwen", layer: int = 15, k: int = 32,
         vae_hidden: int = 256, vae_epochs: int = 500, size: int = None,
         arms: str = "random,das,lcp_vae,lcp_vae_interchange,nldas",
         sweep: bool = False, target_variable: str = "output_token"):
    if sweep:
        sweep_layers.spawn(model_key, k, size)
        print(f"spawned layer sweep for {model_key}; results in {RESULTS}", flush=True)
        return
    arm_list = arms.split(",")
    handles = [
        run_condition.spawn(model_key, random_init, layer, k, vae_hidden,
                            vae_epochs, arm_list, size, target_variable)
        for random_init in (False, True)
    ]
    print(f"spawned {len(handles)} conditions at layer {layer}; "
          f"results land in {RESULTS}", flush=True)
