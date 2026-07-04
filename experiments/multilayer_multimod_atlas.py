"""Multi-layer + multi-modulus Grassmannian atlas — path to Validated (Tier 5).

Addresses two gaps in the MAS:
  E4 (cross-model): tests 2-layer and 4-layer grokking models alongside 1-layer
  M6 (invariance): tests p=53, p=97, p=211 alongside p=113

For each (n_layers, modulus, operation, seed):
  1. Train transformer for enough epochs to allow grokking
  2. Record grokking status (test CE < 0.1)
  3. Fit DAS at k=2 and k=4 on the last layer's residual stream
  4. Compute equivariance (additive shift a -> a+1)
  5. Random-subspace control equivariance

Core 14 operations from the paper's three-class partition:
  Always-Grassmannian (8):  addition, multiplication, division, subtraction,
                            max_ab, min_ab, abs_diff, affine
  Stochastic (2):           composite_addition, power
  Never-Grassmannian (4):   squaring, cubing, sum_of_squares, bitwise_xor

Grid: 3 depths x 4 moduli x 14 ops x 1 seed = 168 configs (core sweep)
      + 3 depths x 1 modulus(113) x 2 stochastic_ops x 10 seeds = 60 (stochastic)
      = 228 containers total

Usage:
    modal run --detach experiments/multilayer_multimod_atlas.py
    modal run --detach experiments/multilayer_multimod_atlas.py --mode stochastic
    modal run --detach experiments/multilayer_multimod_atlas.py --mode core
"""
from __future__ import annotations

import json
import math
import os
import time
import traceback
from datetime import datetime, timezone

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "numpy==1.26.4",
        "setuptools<71",
    )
    .pip_install(
        "transformer-lens==2.11.0",
        "transformers==4.46.3",
        "einops>=0.8",
        "tqdm",
        "matplotlib",
    )
)

app = modal.App("multilayer-multimod-atlas", image=image)
results_vol = modal.Volume.from_name("fc-results", create_if_missing=True)

FRAC_TRAIN = 0.3
DATA_SEED = 598
GROK_THRESHOLD = 0.1

ALWAYS_OPS = [
    "addition", "multiplication", "division", "subtraction",
    "max_ab", "min_ab", "abs_diff", "affine",
]
STOCHASTIC_OPS = ["composite_addition", "power"]
NEVER_OPS = ["squaring", "cubing", "sum_of_squares", "bitwise_xor"]
ALL_OPS = ALWAYS_OPS + STOCHASTIC_OPS + NEVER_OPS

UNARY_OPS = {"squaring", "cubing"}
EXCLUDE_ZERO_OPS = {"multiplication", "division", "power"}

MODULI = [53, 97, 113, 211]
DEPTHS = [1, 2, 4]

EPOCH_DEFAULTS = {
    "addition": 25000, "multiplication": 40000, "division": 40000,
    "subtraction": 25000, "max_ab": 25000, "min_ab": 25000,
    "abs_diff": 25000, "affine": 25000,
    "composite_addition": 30000, "power": 30000,
    "squaring": 80000, "cubing": 25000,
    "sum_of_squares": 25000, "bitwise_xor": 25000,
}

STOCHASTIC_SEEDS = [42, 137, 2024, 7, 19, 53, 101, 256, 500, 777]


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@app.function(
    gpu="A100",
    timeout=86400,
    volumes={"/results": results_vol},
)
def run_single_config(operation: str, p: int, n_layers: int, seed: int) -> dict:
    """Train grokking model + DAS + equivariance for one config."""
    try:
        return _run_single_config_inner(operation, p, n_layers, seed)
    except Exception as e:
        tb = traceback.format_exc()
        error_result = {
            "operation": operation, "p": p, "n_layers": n_layers, "seed": seed,
            "status": "error", "error": f"{type(e).__name__}: {e}",
            "traceback": tb[-3000:], "timestamp": utc_ts(),
        }
        result_dir = f"/results/grassmannian_atlas/multilayer_multimod/{operation}/p{p}_L{n_layers}_s{seed}"
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, "error.json"), "w") as f:
            json.dump(error_result, f, indent=2)
        results_vol.commit()
        print(f"[{utc_ts()}] ERROR saved: {operation} p={p} L={n_layers} s={seed}: {e}")
        return error_result


def _run_single_config_inner(operation: str, p: int, n_layers: int, seed: int) -> dict:
    import einops
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from tqdm import tqdm
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    device = "cuda"

    print(f"[{utc_ts()}] Starting {operation} p={p} layers={n_layers} seed={seed}")

    class VanillaQ(nn.Module):
        def __init__(self, d_site: int, k: int, device="cpu"):
            super().__init__()
            self.R = nn.Parameter(torch.randn(d_site, k, device=device) * 0.02)

        def forward(self):
            Q, _ = torch.linalg.qr(self.R)
            return Q

    def compute_labels(a_vec, b_vec):
        if operation == "addition":
            return (a_vec + b_vec) % p
        elif operation == "subtraction":
            return (a_vec - b_vec) % p
        elif operation == "multiplication":
            return (a_vec * b_vec) % p
        elif operation == "division":
            result = torch.zeros_like(a_vec)
            for i in range(len(a_vec)):
                a, b = a_vec[i].item(), b_vec[i].item()
                if b == 0:
                    result[i] = 0
                else:
                    result[i] = (a * pow(int(b), p - 2, p)) % p
            return result
        elif operation == "squaring":
            return (a_vec * a_vec) % p
        elif operation == "cubing":
            return (a_vec * a_vec * a_vec) % p
        elif operation == "max_ab":
            return torch.maximum(a_vec % p, b_vec % p)
        elif operation == "min_ab":
            return torch.minimum(a_vec % p, b_vec % p)
        elif operation == "abs_diff":
            return ((a_vec - b_vec) % p).abs() % p
        elif operation == "sum_of_squares":
            return (a_vec * a_vec + b_vec * b_vec) % p
        elif operation == "power":
            result = torch.zeros_like(a_vec)
            for i in range(len(a_vec)):
                a, b = a_vec[i].item(), b_vec[i].item()
                result[i] = pow(int(a), int(b), p) if a > 0 else 0
            return result
        elif operation == "bitwise_xor":
            return a_vec ^ b_vec
        elif operation == "affine":
            return (2 * a_vec + 3 * b_vec + 1) % p
        elif operation == "composite_addition":
            p_mod = min(91, p - 1) if p > 91 else p
            return (a_vec + b_vec) % p_mod
        else:
            raise ValueError(f"Unknown operation: {operation}")

    is_unary = operation in UNARY_OPS
    exclude_zero = operation in EXCLUDE_ZERO_OPS
    frac_train = 0.5 if is_unary else FRAC_TRAIN

    if is_unary:
        a_vec = torch.arange(1, p)
        b_vec = torch.zeros(len(a_vec), dtype=torch.long)
    elif exclude_zero:
        a_vals = torch.arange(1, p)
        b_vals = torch.arange(1, p)
        a_vec = einops.repeat(a_vals, "i -> (i j)", j=len(b_vals))
        b_vec = einops.repeat(b_vals, "j -> (i j)", i=len(a_vals))
    else:
        a_vec = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
        b_vec = einops.repeat(torch.arange(p), "j -> (i j)", i=p)

    eq_vec = torch.full_like(a_vec, p)
    dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
    labels = compute_labels(a_vec, b_vec).long().to(device)

    if operation == "bitwise_xor":
        n_classes = int(labels.max().item()) + 1
    elif operation == "composite_addition":
        p_mod = min(91, p - 1) if p > 91 else p
        n_classes = max(p, p_mod + 1)
    else:
        n_classes = p

    rng = torch.Generator()
    rng.manual_seed(0 if is_unary else DATA_SEED)
    indices = torch.randperm(len(dataset), generator=rng)
    cutoff = int(len(dataset) * frac_train)
    train_idx, test_idx = indices[:cutoff], indices[cutoff:]
    train_data, train_labels = dataset[train_idx], labels[train_idx]
    test_data, test_labels = dataset[test_idx], labels[test_idx]

    d_model = 128
    n_epochs = EPOCH_DEFAULTS.get(operation, 25000)
    if n_layers > 1:
        n_epochs = int(n_epochs * 1.5)

    cfg = HookedTransformerConfig(
        n_layers=n_layers, n_heads=4, d_model=d_model, d_head=32, d_mlp=512,
        act_fn="relu", normalization_type=None,
        d_vocab=p + 1, d_vocab_out=n_classes, n_ctx=3,
        init_weights=True, device=device, seed=seed,
    )
    model = HookedTransformer(cfg)
    for name, param in model.named_parameters():
        if "b_" in name:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98),
    )

    def loss_fn(logits, lab):
        if len(logits.shape) == 3:
            logits = logits[:, -1]
        logits = logits.to(torch.float64)
        log_probs = logits.log_softmax(dim=-1)
        correct_log_probs = log_probs.gather(dim=-1, index=lab[:, None])[:, 0]
        return -correct_log_probs.mean()

    train_losses = []
    test_losses = []
    test_accs = []
    grok_epoch = None

    print(f"[{utc_ts()}] Training {n_epochs} epochs (n_layers={n_layers}, p={p})...")
    for epoch in tqdm(range(n_epochs), desc=f"{operation}/p{p}/L{n_layers}/s{seed}"):
        train_logits = model(train_data)
        train_loss = loss_fn(train_logits, train_labels)
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.inference_mode():
            test_logits = model(test_data)
            test_loss = loss_fn(test_logits, test_labels)
            test_preds = test_logits[:, -1].argmax(dim=-1)
            test_acc = (test_preds == test_labels).float().mean().item()

        tl = train_loss.item()
        tel = test_loss.item()
        train_losses.append(tl)
        test_losses.append(tel)
        test_accs.append(test_acc)

        if grok_epoch is None and tel < GROK_THRESHOLD:
            grok_epoch = epoch

        if (epoch + 1) % 5000 == 0:
            print(f"  [{utc_ts()}] Epoch {epoch+1}: train={tl:.4f} test={tel:.4f} acc={test_acc:.4f}")

    grokked = test_losses[-1] < GROK_THRESHOLD
    final_test_loss = test_losses[-1]
    final_test_acc = test_accs[-1]

    print(f"[{utc_ts()}] Training done. grokked={grokked}, test_loss={final_test_loss:.4f}")

    last_layer = n_layers - 1
    hook_name = f"blocks.{last_layer}.hook_resid_post"

    @torch.no_grad()
    def cache_das_pairs(n_pairs=300):
        pairs = []
        n = len(train_data)
        perm = torch.randperm(n)
        for step in range(0, n - 1, 2):
            i, j = perm[step].item(), perm[step + 1].item()
            if train_labels[i] != train_labels[j]:
                tok_i = train_data[i].unsqueeze(0)
                tok_j = train_data[j].unsqueeze(0)
                _, c_i = model.run_with_cache(tok_i, names_filter=[hook_name])
                _, c_j = model.run_with_cache(tok_j, names_filter=[hook_name])
                ba = c_i[hook_name][0, -1, :].clone()
                sa = c_j[hook_name][0, -1, :].clone()
                si = train_labels[j].item()
                pairs.append((tok_i, ba, sa, si))
                if len(pairs) >= n_pairs:
                    break
        return pairs

    def train_das(k, n_steps=400):
        cached = cache_das_pairs(300)
        if len(cached) < 10:
            return 0.0, None
        n_train = int(len(cached) * 0.75)
        cached_train = cached[:n_train]
        cached_eval = cached[n_train:]

        param = VanillaQ(d_model, k, device=device)
        opt = torch.optim.Adam([param.R], lr=1e-3)

        for step in range(n_steps):
            opt.zero_grad()
            Q = param()
            proj = Q @ Q.T
            batch_idx = torch.randint(0, len(cached_train), (min(16, len(cached_train)),))
            loss = torch.tensor(0.0, device=device)
            for idx in batch_idx:
                bt, ba, sa, si = cached_train[idx]
                iv = ba - ba @ proj + sa @ proj

                def hook_fn(act, hook, iv_vec=iv):
                    new = act.clone()
                    new[0, -1, :] = iv_vec
                    return new

                logits = model.run_with_hooks(
                    bt, fwd_hooks=[(hook_name, hook_fn)]
                )[0, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                loss = loss - log_probs[si]
            loss = loss / len(batch_idx)
            loss.backward()
            opt.step()

        Q = param().detach()
        proj = Q @ Q.T
        correct = 0
        with torch.no_grad():
            for bt, ba, sa, si in cached_eval:
                iv = ba - ba @ proj + sa @ proj

                def hook_fn(act, hook, iv_vec=iv):
                    new = act.clone()
                    new[0, -1, :] = iv_vec
                    return new

                logits_iv = model.run_with_hooks(
                    bt, fwd_hooks=[(hook_name, hook_fn)]
                )[0, -1, :]
                logits_base = model(bt)[0, -1, :]
                lp_iv = F.log_softmax(logits_iv, dim=-1)
                lp_base = F.log_softmax(logits_base, dim=-1)
                if lp_iv[si] > lp_base[si]:
                    correct += 1
        iia = correct / max(len(cached_eval), 1)
        return iia, Q

    def compute_equivariance(Q, n_test=200):
        if Q is None:
            return 0.0
        proj = Q @ Q.T
        shifts = [1, 2, 5]
        equivariant_count = 0
        total_tested = 0

        with torch.no_grad():
            rng = torch.Generator(device="cpu")
            rng.manual_seed(42)
            test_indices = torch.randperm(len(test_data), generator=rng)[:n_test]

            label_to_indices = {}
            for i in range(len(train_data)):
                lab = train_labels[i].item()
                if lab not in label_to_indices:
                    label_to_indices[lab] = []
                label_to_indices[lab].append(i)

            for idx in test_indices:
                tokens = test_data[idx].unsqueeze(0)
                _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
                act = cache[hook_name][0, -1, :]
                orig_pred = model(tokens)[0, -1, :].argmax().item()
                true_label = test_labels[idx].item()
                if orig_pred != true_label:
                    continue

                in_sub = act @ proj
                complement = act - in_sub

                for shift in shifts:
                    target_answer = (true_label + shift) % p
                    if target_answer not in label_to_indices:
                        continue

                    candidates = label_to_indices[target_answer]
                    best_match = None
                    best_dist = float('inf')
                    for ci in candidates[:50]:
                        s_tokens = train_data[ci].unsqueeze(0)
                        _, s_cache = model.run_with_cache(s_tokens, names_filter=[hook_name])
                        s_act = s_cache[hook_name][0, -1, :]
                        s_in_sub = s_act @ proj
                        dist = (s_in_sub - in_sub).norm().item()
                        if dist < best_dist:
                            best_dist = dist
                            best_match = s_in_sub

                    if best_match is None:
                        continue

                    shifted_act = best_match + complement

                    def hook_fn(act, hook, sv=shifted_act):
                        new = act.clone()
                        new[0, -1, :] = sv
                        return new

                    shifted_logits = model.run_with_hooks(
                        tokens, fwd_hooks=[(hook_name, hook_fn)]
                    )[0, -1, :]
                    shifted_pred = shifted_logits.argmax().item()

                    total_tested += 1
                    if shifted_pred == target_answer:
                        equivariant_count += 1

        return equivariant_count / max(total_tested, 1)

    def compute_random_equivariance(k):
        Q_rand = torch.randn(d_model, k, device=device)
        Q_rand, _ = torch.linalg.qr(Q_rand)
        return compute_equivariance(Q_rand, n_test=100)

    das_results = {}
    for k_val in [2, 4]:
        print(f"[{utc_ts()}] DAS k={k_val}...")
        iia, Q = train_das(k_val, n_steps=400)
        equiv = compute_equivariance(Q)
        rand_equiv = compute_random_equivariance(k_val)
        das_results[f"das_k{k_val}"] = {
            "iia": float(iia),
            "equivariance": float(equiv),
            "random_equiv": float(rand_equiv),
        }
        print(f"  k={k_val}: IIA={iia:.3f}, equiv={equiv:.3f}, rand={rand_equiv:.3f}")

    result = {
        "operation": operation,
        "p": p,
        "n_layers": n_layers,
        "seed": seed,
        "grokked": grokked,
        "final_test_loss": float(final_test_loss),
        "final_test_acc": float(final_test_acc),
        "n_epochs": n_epochs,
        "grok_epoch": grok_epoch,
        "das_k2": das_results.get("das_k2", {}),
        "das_k4": das_results.get("das_k4", {}),
        "das_layer": last_layer,
        "hook_name": hook_name,
        "d_model": d_model,
        "timestamp": utc_ts(),
    }

    result_dir = f"/results/grassmannian_atlas/multilayer_multimod/{operation}/p{p}_L{n_layers}_s{seed}"
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)

    subsample = 100
    with open(os.path.join(result_dir, "loss_curves.json"), "w") as f:
        json.dump({
            "epochs": list(range(0, n_epochs, subsample)),
            "train_loss": [train_losses[i] for i in range(0, n_epochs, subsample)],
            "test_loss": [test_losses[i] for i in range(0, n_epochs, subsample)],
            "test_acc": [test_accs[i] for i in range(0, n_epochs, subsample)],
        }, f)

    results_vol.commit()
    print(f"[{utc_ts()}] Done: {operation} p={p} L={n_layers} s={seed}")
    return result


@app.local_entrypoint()
def main(mode: str = "moduli"):
    t0 = time.time()

    configs = []

    if mode == "moduli":
        new_moduli = [p for p in MODULI if p != 113]
        for p in new_moduli:
            for op in ALL_OPS:
                if op == "composite_addition" and p <= 91:
                    continue
                configs.append((op, p, 1, 42))

    elif mode == "depth":
        for n_layers in [2, 4]:
            for op in ALL_OPS:
                configs.append((op, 113, n_layers, 42))

    elif mode == "depth-pilot":
        for n_layers in [2, 4]:
            for op in ["addition", "multiplication", "squaring", "composite_addition"]:
                configs.append((op, 113, n_layers, 42))

    elif mode == "depth-remaining":
        done_4layer = {"addition", "multiplication", "squaring", "composite_addition"}
        for op in ALL_OPS:
            if op not in done_4layer:
                configs.append((op, 113, 4, 42))

    elif mode == "stochastic":
        for n_layers in DEPTHS:
            for op in STOCHASTIC_OPS:
                for seed in STOCHASTIC_SEEDS:
                    configs.append((op, 113, n_layers, seed))

    elif mode == "fill-gaps":
        configs = [
            ("bitwise_xor", 113, 2, 42),
            ("abs_diff", 113, 4, 42),
            ("affine", 113, 4, 42),
            ("bitwise_xor", 113, 4, 42),
            ("cubing", 113, 4, 42),
            ("division", 113, 4, 42),
            ("max_ab", 113, 4, 42),
            ("min_ab", 113, 4, 42),
            ("power", 113, 4, 42),
            ("subtraction", 113, 4, 42),
            ("sum_of_squares", 113, 4, 42),
        ]

    elif mode == "all":
        for n_layers in DEPTHS:
            for p in MODULI:
                for op in ALL_OPS:
                    if op == "composite_addition" and p <= 91:
                        continue
                    configs.append((op, p, n_layers, 42))
        for n_layers in DEPTHS:
            for op in STOCHASTIC_OPS:
                for seed in STOCHASTIC_SEEDS:
                    key = (op, 113, n_layers, seed)
                    if key not in [(c[0], c[1], c[2], c[3]) for c in configs]:
                        configs.append(key)

    configs = list(dict.fromkeys(configs))

    print(f"Multilayer + Multimodulus Grassmannian Atlas")
    print(f"  Mode: {mode}")
    print(f"  Total configs: {len(configs)}")
    print()

    handles = []
    for op, p, n_layers, seed in configs:
        h = run_single_config.spawn(operation=op, p=p, n_layers=n_layers, seed=seed)
        handles.append((op, p, n_layers, seed, h))
        print(f"  Spawned {op} p={p} L={n_layers} s={seed}")

    print(f"\n{len(handles)} containers spawned. Collecting results...\n")

    results = []
    for op, p, n_layers, seed, h in handles:
        try:
            result = h.get()
        except Exception as e:
            tb = traceback.format_exc()
            result = {
                "operation": op, "p": p, "n_layers": n_layers, "seed": seed,
                "grokked": False, "status": "error",
                "error": f"{type(e).__name__}: {e}", "traceback": tb[-2000:],
            }
        results.append(result)

        grokked = result.get("grokked", False)
        iia_k2 = result.get("das_k2", {}).get("iia", -1)
        eq_k2 = result.get("das_k2", {}).get("equivariance", -1)
        status = "ERROR" if result.get("status") == "error" else ("GROK" if grokked else "no")
        print(f"  {op:22s} p={p:3d} L={n_layers} s={seed:4d}  {status:5s}  "
              f"IIA_k2={iia_k2:.3f}  eq_k2={eq_k2:.3f}")

    summary_path = "experiments/results/multilayer_multimod_atlas_summary.jsonl"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    total = time.time() - t0

    print(f"\n{'='*90}")
    print(f"Grassmannian Atlas — Summary")
    print(f"{'='*90}")

    seen_moduli = sorted(set(r.get("p") for r in results if "p" in r))
    for p in seen_moduli:
        print(f"\n--- p={p} ---")
        p_results = [r for r in results if r.get("p") == p]
        for category, ops in [("Always", ALWAYS_OPS), ("Stochastic", STOCHASTIC_OPS), ("Never", NEVER_OPS)]:
            cat_results = [r for r in p_results if r.get("operation") in ops]
            if not cat_results:
                continue
            n_grokked = sum(1 for r in cat_results if r.get("grokked", False))
            grokked_iia = [r["das_k2"]["iia"] for r in cat_results if r.get("grokked") and "das_k2" in r]
            grokked_eq = [r["das_k2"]["equivariance"] for r in cat_results if r.get("grokked") and "das_k2" in r]
            print(f"  {category:12s}: {n_grokked}/{len(cat_results)} grokked", end="")
            if grokked_iia:
                print(f"  mean_IIA={sum(grokked_iia)/len(grokked_iia):.3f}", end="")
            if grokked_eq:
                print(f"  mean_eq={sum(grokked_eq)/len(grokked_eq):.3f}", end="")
            print()

    print(f"\nTotal wall time: {total:.0f}s")
    print(f"Results: {summary_path}")
