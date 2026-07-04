"""Multi-seed grokking stability — Do stochastic operations grok reliably?

Trains grokking models with 10 different random seeds for each of 2 "stochastic"
operations (Power and Composite Addition), then runs DAS + equivariance evaluation
on each to measure the grokking rate and the conditional relationship between
grokking and Grassmannian structure.

For each (operation, seed):
  1. Train 1-layer transformer for 30k epochs
  2. Record train/test loss curves, classify as grokked (test CE < 0.1)
  3. Fit DAS at k=2 and k=4 (400 steps each)
  4. Compute equivariance (additive shift a -> a+1, check rotation in DAS projection)
  5. Random-subspace control equivariance

20 total tasks (2 ops x 10 seeds), each on its own A100.

Usage:
    modal run --detach experiments/batch6_atlas/06_21_2026_UPDATE/multi_seed_stability.py
"""
from __future__ import annotations

import json
import math
import os
import time
import traceback
from datetime import datetime, timezone

import modal

# ── Modal setup ──────────────────────────────────────────────────────────────

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

app = modal.App("multi-seed-grokking-stability", image=image)
results_vol = modal.Volume.from_name("fc-results", create_if_missing=True)

P = 113
FRAC_TRAIN = 0.3
DATA_SEED = 598
GROK_THRESHOLD = 0.1
N_EPOCHS = 30000
SEEDS = [42, 137, 2024, 7, 19, 53, 101, 256, 500, 777]

OPERATIONS = {
    "power": {
        "op_fn_str": "pow(int(a), int(b), p) if a > 0 else 0",
        "p_data": 113,
        "p_mod": 113,
        "exclude_zero_a": True,
    },
    "composite_addition": {
        "op_fn_str": "(a + b) % 91",
        "p_data": 113,
        "p_mod": 91,
        "exclude_zero_a": False,
    },
}


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Remote function: one (operation, seed) combo ─────────────────────────────

@app.function(
    gpu="A100",
    timeout=86400,
    volumes={"/results": results_vol},
)
def run_single_seed(operation: str, seed: int) -> dict:
    """Train grokking model + DAS + equivariance for one (operation, seed)."""
    import einops
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from tqdm import tqdm
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    device = "cuda"
    op_cfg = OPERATIONS[operation]
    p_data = op_cfg["p_data"]
    p_mod = op_cfg["p_mod"]

    print(f"[{utc_ts()}] Starting {operation} seed={seed}")

    # ── VanillaQ (inlined from factor_das_kernel.py) ─────────────────────

    class VanillaQ(nn.Module):
        def __init__(self, d_site: int, k: int, device="cpu"):
            super().__init__()
            self.R = nn.Parameter(torch.randn(d_site, k, device=device) * 0.02)
            self._d, self._k = d_site, k

        def forward(self):
            Q, _ = torch.linalg.qr(self.R)
            return Q

    # ── Operation function ───────────────────────────────────────────────

    if operation == "power":
        def op_fn(a, b, p):
            return pow(int(a), int(b), p) if a > 0 else 0
    elif operation == "composite_addition":
        def op_fn(a, b, p):
            return (int(a) + int(b)) % 91
    else:
        raise ValueError(f"Unknown operation: {operation}")

    # ── Data generation ──────────────────────────────────────────────────

    def build_data():
        if op_cfg["exclude_zero_a"]:
            a_vals = torch.arange(1, p_data)
            b_vals = torch.arange(1, p_data)
        else:
            a_vals = torch.arange(p_data)
            b_vals = torch.arange(p_data)
        a_vec = einops.repeat(a_vals, "i -> (i j)", j=len(b_vals))
        b_vec = einops.repeat(b_vals, "j -> (i j)", i=len(a_vals))
        eq_vec = torch.full_like(a_vec, p_data)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = torch.tensor(
            [op_fn(a.item(), b.item(), p_data) for a, b in zip(a_vec, b_vec)]
        ).to(device)

        torch.manual_seed(DATA_SEED)
        n_total = len(a_vec)
        indices = torch.randperm(n_total)
        cutoff = int(n_total * FRAC_TRAIN)
        return dataset, labels, indices[:cutoff], indices[cutoff:]

    dataset, labels, train_idx, test_idx = build_data()
    train_data, train_labels = dataset[train_idx], labels[train_idx]
    test_data, test_labels = dataset[test_idx], labels[test_idx]

    n_classes = max(p_data, p_mod + 1)

    # ── Model ────────────────────────────────────────────────────────────

    cfg = HookedTransformerConfig(
        n_layers=1, n_heads=4, d_model=128, d_head=32, d_mlp=512,
        act_fn="relu", normalization_type=None,
        d_vocab=p_data + 1, d_vocab_out=n_classes, n_ctx=3,
        init_weights=True, device=device, seed=seed,
    )
    model = HookedTransformer(cfg)
    for name, param in model.named_parameters():
        if "b_" in name:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98),
    )

    # ── Training ─────────────────────────────────────────────────────────

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

    print(f"[{utc_ts()}] Training {N_EPOCHS} epochs...")
    for epoch in tqdm(range(N_EPOCHS), desc=f"{operation}/s{seed}"):
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
    final_train_loss = train_losses[-1]

    print(f"[{utc_ts()}] Training done. grokked={grokked}, final_test_loss={final_test_loss:.4f}")

    # ── DAS + Equivariance ───────────────────────────────────────────────

    hook_name = "blocks.0.hook_resid_post"
    d_model = 128

    @torch.no_grad()
    def cache_das_pairs(n_pairs=300):
        """Cache counterfactual pairs with different labels."""
        pairs = []
        n = len(train_data)
        # Shuffle indices to get diverse pairs
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
        """Train DAS at subspace dimension k. Returns (IIA, Q)."""
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

        # Eval IIA
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

    def compute_equivariance(Q):
        """Compute equivariance fraction under additive shift a -> a+1 mod p_data."""
        if Q is None:
            return 0.0
        k = Q.shape[1]
        expected_angle = 2 * math.pi / p_data
        tolerance = 0.1 * expected_angle

        equivariant_count = 0
        total = 0

        with torch.no_grad():
            # Use a subset of test inputs
            n_check = min(500, len(test_data))
            for idx in range(n_check):
                a_val = test_data[idx, 0].item()
                b_val = test_data[idx, 1].item()
                a_shifted = (a_val + 1) % p_data

                tok_orig = test_data[idx].unsqueeze(0)
                tok_shifted = tok_orig.clone()
                tok_shifted[0, 0] = a_shifted

                _, cache_orig = model.run_with_cache(tok_orig, names_filter=[hook_name])
                _, cache_shifted = model.run_with_cache(tok_shifted, names_filter=[hook_name])

                h_orig = cache_orig[hook_name][0, -1, :]
                h_shifted = cache_shifted[hook_name][0, -1, :]

                z_orig = Q.T @ h_orig
                z_shifted = Q.T @ h_shifted

                if k >= 2:
                    # Measure angle of rotation in first 2 components
                    z1 = z_orig[:2]
                    z2 = z_shifted[:2]
                    if z1.norm() < 1e-8 or z2.norm() < 1e-8:
                        continue
                    z1_n = z1 / z1.norm()
                    z2_n = z2 / z2.norm()
                    cos_angle = torch.clamp(z1_n @ z2_n, -1.0, 1.0)
                    # Use atan2 for signed angle
                    sin_angle = z1_n[0] * z2_n[1] - z1_n[1] * z2_n[0]
                    angle = torch.atan2(sin_angle, cos_angle).abs().item()
                    if abs(angle - expected_angle) < tolerance:
                        equivariant_count += 1
                total += 1

        return equivariant_count / max(total, 1)

    def compute_random_equivariance(k):
        """Equivariance with a random orthonormal subspace (control)."""
        Q_rand = torch.randn(d_model, k, device=device)
        Q_rand, _ = torch.linalg.qr(Q_rand)
        return compute_equivariance(Q_rand)

    # Run DAS for k=2 and k=4
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

    # ── Save results ─────────────────────────────────────────────────────

    result = {
        "operation": operation,
        "seed": seed,
        "grokked": grokked,
        "final_test_loss": float(final_test_loss),
        "final_test_acc": float(final_test_acc),
        "n_epochs": N_EPOCHS,
        "das_k2": das_results.get("das_k2", {"iia": 0.0, "equivariance": 0.0, "random_equiv": 0.0}),
        "das_k4": das_results.get("das_k4", {"iia": 0.0, "equivariance": 0.0, "random_equiv": 0.0}),
        "train_loss_final": float(final_train_loss),
        "grok_epoch": grok_epoch,
        "timestamp": utc_ts(),
        "p_data": p_data,
        "p_mod": p_mod,
    }

    result_dir = f"/results/grassmannian_atlas/multi_seed/{operation}/seed_{seed}"
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, "results.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{utc_ts()}] Saved to {result_path}")

    # Also save loss curves (compact)
    curves_path = os.path.join(result_dir, "loss_curves.json")
    # Subsample to every 100 epochs for storage
    subsample = 100
    with open(curves_path, "w") as f:
        json.dump({
            "epochs": list(range(0, N_EPOCHS, subsample)),
            "train_loss": [train_losses[i] for i in range(0, N_EPOCHS, subsample)],
            "test_loss": [test_losses[i] for i in range(0, N_EPOCHS, subsample)],
            "test_acc": [test_accs[i] for i in range(0, N_EPOCHS, subsample)],
        }, f)

    results_vol.commit()
    print(f"[{utc_ts()}] Done: {operation} seed={seed}")
    return result


# ── Local entrypoint ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    t0 = time.time()

    print(f"Multi-Seed Grokking Stability Experiment")
    print(f"  Operations: {list(OPERATIONS.keys())}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total tasks: {len(OPERATIONS) * len(SEEDS)}")
    print()

    handles = []
    for operation in OPERATIONS:
        for seed in SEEDS:
            h = run_single_seed.spawn(operation=operation, seed=seed)
            handles.append((operation, seed, h))
            print(f"  Spawned {operation} seed={seed}")

    print(f"\n{len(handles)} containers spawned. Collecting results...\n")

    results = []
    for operation, seed, h in handles:
        try:
            result = h.get()
        except Exception as e:
            tb = traceback.format_exc()
            result = {
                "operation": operation,
                "seed": seed,
                "grokked": False,
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "traceback": tb[-2000:],
            }
        results.append(result)

        grokked = result.get("grokked", False)
        grok_ep = result.get("grok_epoch")
        test_loss = result.get("final_test_loss", -1)
        iia_k2 = result.get("das_k2", {}).get("iia", -1)
        iia_k4 = result.get("das_k4", {}).get("iia", -1)
        eq_k2 = result.get("das_k2", {}).get("equivariance", -1)
        print(f"  {operation:25s} seed={seed:4d}  grokked={str(grokked):5s}  "
              f"test_loss={test_loss:7.4f}  IIA_k2={iia_k2:.3f}  IIA_k4={iia_k4:.3f}  "
              f"eq_k2={eq_k2:.3f}  grok_ep={grok_ep}")

    # Save summary
    summary_path = "experiments/results/multi_seed_stability_summary.jsonl"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    total = time.time() - t0

    # Print summary table
    print(f"\n{'='*80}")
    print(f"Multi-Seed Grokking Stability — Summary")
    print(f"{'='*80}")

    for op_name in OPERATIONS:
        op_results = [r for r in results if r.get("operation") == op_name]
        n_grokked = sum(1 for r in op_results if r.get("grokked", False))
        grok_epochs = [r["grok_epoch"] for r in op_results if r.get("grok_epoch") is not None]

        print(f"\n{op_name} (p_data={OPERATIONS[op_name]['p_data']}, p_mod={OPERATIONS[op_name]['p_mod']}):")
        print(f"  Grokking rate: {n_grokked}/{len(op_results)}")
        if grok_epochs:
            print(f"  Grok epoch: mean={sum(grok_epochs)/len(grok_epochs):.0f}, "
                  f"min={min(grok_epochs)}, max={max(grok_epochs)}")

        # DAS IIA stats (grokked only)
        grokked_results = [r for r in op_results if r.get("grokked", False)]
        ungrokked_results = [r for r in op_results if not r.get("grokked", False)]

        if grokked_results:
            iia_k2 = [r["das_k2"]["iia"] for r in grokked_results if "das_k2" in r]
            iia_k4 = [r["das_k4"]["iia"] for r in grokked_results if "das_k4" in r]
            eq_k2 = [r["das_k2"]["equivariance"] for r in grokked_results if "das_k2" in r]
            rand_k2 = [r["das_k2"]["random_equiv"] for r in grokked_results if "das_k2" in r]

            if iia_k2:
                print(f"  Grokked seeds — DAS k=2 IIA: {sum(iia_k2)/len(iia_k2):.3f} "
                      f"(range {min(iia_k2):.3f}-{max(iia_k2):.3f})")
            if iia_k4:
                print(f"  Grokked seeds — DAS k=4 IIA: {sum(iia_k4)/len(iia_k4):.3f} "
                      f"(range {min(iia_k4):.3f}-{max(iia_k4):.3f})")
            if eq_k2:
                print(f"  Grokked seeds — Equivariance k=2: {sum(eq_k2)/len(eq_k2):.3f} "
                      f"(random: {sum(rand_k2)/len(rand_k2):.3f})")

        if ungrokked_results:
            iia_k2 = [r["das_k2"]["iia"] for r in ungrokked_results if "das_k2" in r]
            eq_k2 = [r["das_k2"]["equivariance"] for r in ungrokked_results if "das_k2" in r]
            if iia_k2:
                print(f"  Ungrokked seeds — DAS k=2 IIA: {sum(iia_k2)/len(iia_k2):.3f}")
            if eq_k2:
                print(f"  Ungrokked seeds — Equivariance k=2: {sum(eq_k2)/len(eq_k2):.3f}")

    print(f"\nTotal wall time: {total:.0f}s")
    print(f"Results: {summary_path}")
