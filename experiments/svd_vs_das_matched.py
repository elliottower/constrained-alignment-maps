"""Dimension-matched SVD vs DAS comparison.

Addresses the reviewer concern that SVD top-2 vs DAS k=2 is an unfair comparison
because SVD captures variance, not causal structure, and may need more dimensions.

For each grokking operation:
  1. Train model to grokking
  2. At the same intervention site (blocks.0.hook_resid_post, last token):
     - Compute SVD of W_in (MLP input weight matrix) → top-k right singular vectors
     - Train DAS at dimension k
     - Evaluate IIA for both at k = 2, 8, 32
  3. Also compare: random subspace baseline at each k

This is a fair comparison because both methods get the SAME number of dimensions
and intervene at the SAME site. If SVD catches up at high k, that's interesting.
If DAS still wins, the causal structure matters even when dimensions match.

Usage:
    modal run --detach experiments/svd_vs_das_matched.py
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
import traceback
from datetime import datetime, timezone

import modal

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from tqdm import tqdm
    from transformer_lens import HookedTransformer, HookedTransformerConfig
except (ImportError, AttributeError):
    pass

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy==1.26.4", "setuptools<71")
    .pip_install(
        "transformer-lens==2.11.0", "transformers==4.46.3",
        "einops>=0.8", "matplotlib", "tqdm",
    )
)

app = modal.App("svd-vs-das-matched", image=image)
results_vol = modal.Volume.from_name("fc-results", create_if_missing=True)

FRAC_TRAIN = 0.3
K_VALUES = [2, 8, 32]
DAS_STEPS = 200
DAS_LR = 1e-3
DAS_BATCH = 16
N_PAIRS = 300

OPERATIONS = {
    "multiplication": {"p": 113, "n_epochs": 40000},
    "composite_addition": {"p": 91, "n_epochs": 15000},
    "division": {"p": 113, "n_epochs": 40000},
    "polynomial": {"p": 113, "n_epochs": 60000},
}


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@app.function(gpu="A100", timeout=86400, volumes={"/results": results_vol})
def run_single_operation(operation: str) -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from tqdm import tqdm
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    device = "cuda"
    op_cfg = OPERATIONS[operation]
    p = op_cfg["p"]
    n_epochs = op_cfg["n_epochs"]
    layer = 0
    hook_name = f"blocks.{layer}.hook_resid_post"

    print(f"[{utc_ts()}] Starting {operation} (p={p}, epochs={n_epochs})")

    # ── VanillaQ ────────────────────────────────────────────────────────

    class VanillaQ(nn.Module):
        def __init__(self, d_site, k, device="cpu"):
            super().__init__()
            self.R = nn.Parameter(torch.randn(d_site, k, device=device) * 0.02)

        def forward(self):
            Q, _ = torch.linalg.qr(self.R)
            return Q

    # ── Data ────────────────────────────────────────────────────────────

    if operation == "multiplication":
        a_vals = torch.arange(1, p)
        b_vals = torch.arange(1, p)
        a_vec = a_vals.repeat_interleave(len(b_vals))
        b_vec = b_vals.repeat(len(a_vals))
        eq_vec = torch.full_like(a_vec, p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = ((a_vec * b_vec) % p).to(device)
    elif operation == "composite_addition":
        a_vals = torch.arange(p)
        b_vals = torch.arange(p)
        a_vec = a_vals.repeat_interleave(len(b_vals))
        b_vec = b_vals.repeat(len(a_vals))
        eq_vec = torch.full_like(a_vec, p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = ((a_vec + b_vec) % p).to(device)
    elif operation == "division":
        a_vals = torch.arange(1, p)
        b_vals = torch.arange(1, p)
        a_vec = a_vals.repeat_interleave(len(b_vals))
        b_vec = b_vals.repeat(len(a_vals))
        eq_vec = torch.full_like(a_vec, p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        b_inv = torch.tensor([pow(int(b.item()), p - 2, p) for b in b_vec])
        labels = ((a_vec * b_inv) % p).to(device)
    elif operation == "polynomial":
        a_vals = torch.arange(p)
        b_vals = torch.arange(p)
        a_vec = a_vals.repeat_interleave(len(b_vals))
        b_vec = b_vals.repeat(len(a_vals))
        eq_vec = torch.full_like(a_vec, p)
        dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
        labels = ((a_vec * a_vec + b_vec) % p).to(device)
    else:
        raise ValueError(f"Unknown operation: {operation}")

    # ── Train/test split ────────────────────────────────────────────────

    rng = torch.Generator()
    rng.manual_seed(0)
    perm = torch.randperm(len(dataset), generator=rng)
    frac = 0.5 if operation in ("squaring", "cubing") else FRAC_TRAIN
    n_train = int(len(dataset) * frac)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    train_data = dataset[train_idx]
    train_labels = labels[train_idx]
    test_data = dataset[test_idx]
    test_labels = labels[test_idx]

    # ── Train grokking model ────────────────────────────────────────────

    d_model = 128
    cfg = HookedTransformerConfig(
        n_layers=1, n_heads=4, d_model=d_model, d_head=32, d_mlp=512,
        act_fn="relu", normalization_type=None,
        d_vocab=p + 1, d_vocab_out=p, n_ctx=3,
        init_weights=True, device=device, seed=999,
    )
    model = HookedTransformer(cfg)
    for name, param in model.named_parameters():
        if "b_" in name:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0,
                                   betas=(0.9, 0.98))

    train_losses, test_losses = [], []
    for epoch in tqdm(range(n_epochs), desc=f"{operation}"):
        train_logits = model(train_data)[:, -1]
        train_loss = F.cross_entropy(train_logits, train_labels)
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        with torch.no_grad():
            test_logits = model(test_data)[:, -1]
            test_loss = F.cross_entropy(test_logits, test_labels)
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())

    final_test_loss = test_losses[-1]
    grokked = final_test_loss < 0.1
    print(f"[{utc_ts()}] {operation}: grokked={grokked}, test_loss={final_test_loss:.4f}")

    if not grokked:
        return {"operation": operation, "grokked": False, "final_test_loss": final_test_loss}

    model.eval()
    for p_param in model.parameters():
        p_param.requires_grad_(False)

    # ── Cache counterfactual pairs ──────────────────────────────────────

    pairs = []
    n = len(train_data)
    perm2 = torch.randperm(n)
    for step in range(0, n - 1, 2):
        i, j = perm2[step].item(), perm2[step + 1].item()
        if train_labels[i] != train_labels[j]:
            tok_i = train_data[i].unsqueeze(0)
            tok_j = train_data[j].unsqueeze(0)
            _, c_i = model.run_with_cache(tok_i, names_filter=[hook_name])
            _, c_j = model.run_with_cache(tok_j, names_filter=[hook_name])
            ba = c_i[hook_name][0, -1, :].clone()
            sa = c_j[hook_name][0, -1, :].clone()
            si = train_labels[j].item()
            pairs.append((tok_i, ba, sa, si))
            if len(pairs) >= N_PAIRS:
                break

    n_train_pairs = int(len(pairs) * 0.75)
    train_pairs = pairs[:n_train_pairs]
    eval_pairs = pairs[n_train_pairs:]
    print(f"[{utc_ts()}] Cached {len(pairs)} pairs ({len(train_pairs)} train, {len(eval_pairs)} eval)")

    # ── Eval helper ─────────────────────────────────────────────────────

    def eval_subspace_iia(Q_mat):
        """Evaluate IIA for a given orthonormal basis Q (d_model, k)."""
        proj = Q_mat @ Q_mat.T
        correct = 0
        total = 0
        with torch.no_grad():
            for bt, ba, sa, si in eval_pairs:
                iv = ba - ba @ proj + sa @ proj

                def hook_fn(act, hook, iv_vec=iv):
                    new = act.clone()
                    new[0, -1, :] = iv_vec
                    return new

                logits_iv = model.run_with_hooks(
                    bt, fwd_hooks=[(hook_name, hook_fn)]
                )[0, -1, :]
                if logits_iv.argmax().item() == si:
                    correct += 1
                total += 1
        return correct / max(total, 1)

    # ── Get SVD subspaces from weight matrices ──────────────────────────

    W_in = model.blocks[layer].mlp.W_in.detach()
    W_out = model.blocks[layer].mlp.W_out.detach()
    W_eff = W_in @ W_out

    U_in_full, S_in_full, _ = torch.linalg.svd(W_in, full_matrices=False)
    U_eff_full, S_eff_full, _ = torch.linalg.svd(W_eff, full_matrices=False)

    svd_results = {}
    for k in K_VALUES:
        Q_svd_win = U_in_full[:, :k]
        Q_svd_eff = U_eff_full[:, :k]

        iia_win = eval_subspace_iia(Q_svd_win)
        iia_eff = eval_subspace_iia(Q_svd_eff)

        svd_results[f"k{k}"] = {
            "svd_W_in_iia": iia_win,
            "svd_W_eff_iia": iia_eff,
            "top_svals_W_in": S_in_full[:k].tolist(),
            "top_svals_W_eff": S_eff_full[:k].tolist(),
            "Q_svd_win": Q_svd_win,
            "Q_svd_eff": Q_svd_eff,
        }
        print(f"[{utc_ts()}] SVD k={k}: W_in IIA={iia_win:.3f}, W_eff IIA={iia_eff:.3f}")

    # ── Train DAS at each k and compute principal angles ────────────────

    das_results = {}
    for k in K_VALUES:
        param = VanillaQ(d_model, k, device=device)
        opt = torch.optim.Adam([param.R], lr=DAS_LR)

        for step in range(DAS_STEPS):
            opt.zero_grad()
            Q = param()
            proj = Q @ Q.T
            batch_idx = torch.randint(0, len(train_pairs), (min(DAS_BATCH, len(train_pairs)),))
            loss = torch.tensor(0.0, device=device)
            for idx in batch_idx:
                bt, ba, sa, si = train_pairs[idx]
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

        Q_das = param().detach()
        iia_das = eval_subspace_iia(Q_das)

        Q_svd_win = svd_results[f"k{k}"]["Q_svd_win"]
        Q_svd_eff = svd_results[f"k{k}"]["Q_svd_eff"]

        M_win = Q_das.T @ Q_svd_win
        svals_win = torch.linalg.svdvals(M_win).clamp(-1, 1)
        angles_win = torch.acos(svals_win).tolist()

        M_eff = Q_das.T @ Q_svd_eff
        svals_eff = torch.linalg.svdvals(M_eff).clamp(-1, 1)
        angles_eff = torch.acos(svals_eff).tolist()

        das_results[f"k{k}"] = {
            "das_iia": iia_das,
            "principal_angles_vs_svd_win": [math.degrees(a) for a in angles_win],
            "principal_angles_vs_svd_eff": [math.degrees(a) for a in angles_eff],
        }
        print(f"[{utc_ts()}] DAS k={k}: IIA={iia_das:.3f}")
        print(f"[{utc_ts()}] k={k} angles DAS↔SVD(W_in): {[f'{math.degrees(a):.1f}°' for a in angles_win]}")
        print(f"[{utc_ts()}] k={k} angles DAS↔SVD(W_eff): {[f'{math.degrees(a):.1f}°' for a in angles_eff]}")

    # ── Random subspace baseline ────────────────────────────────────────

    random_results = {}
    n_random_trials = 10
    for k in K_VALUES:
        iias = []
        for trial in range(n_random_trials):
            Q_rand = torch.randn(d_model, k, device=device)
            Q_rand, _ = torch.linalg.qr(Q_rand)
            iias.append(eval_subspace_iia(Q_rand))
        random_results[f"k{k}"] = {
            "mean_iia": sum(iias) / len(iias),
            "max_iia": max(iias),
            "min_iia": min(iias),
        }
        print(f"[{utc_ts()}] Random k={k}: mean IIA={sum(iias)/len(iias):.3f}")

    # ── Compile results ─────────────────────────────────────────────────

    result = {
        "operation": operation,
        "grokked": True,
        "final_test_loss": final_test_loss,
        "p": op_cfg["p"],
        "n_epochs": op_cfg["n_epochs"],
        "d_model": d_model,
        "das_steps": DAS_STEPS,
        "k_values": K_VALUES,
    }

    for k in K_VALUES:
        result[f"k{k}"] = {
            "das_iia": das_results[f"k{k}"]["das_iia"],
            "svd_W_in_iia": svd_results[f"k{k}"]["svd_W_in_iia"],
            "svd_W_eff_iia": svd_results[f"k{k}"]["svd_W_eff_iia"],
            "random_mean_iia": random_results[f"k{k}"]["mean_iia"],
            "random_max_iia": random_results[f"k{k}"]["max_iia"],
            "principal_angles_das_vs_svd_win": das_results[f"k{k}"].get("principal_angles_vs_svd_win", []),
            "principal_angles_das_vs_svd_eff": das_results[f"k{k}"].get("principal_angles_vs_svd_eff", []),
            "top_svals_W_in": svd_results[f"k{k}"]["top_svals_W_in"],
            "top_svals_W_eff": svd_results[f"k{k}"]["top_svals_W_eff"],
        }

    # Save to volume
    save_dir = "/results/grassmannian_atlas/svd_vs_das_matched"
    os.makedirs(save_dir, exist_ok=True)
    out_path = f"{save_dir}/{operation}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    results_vol.commit()
    print(f"[{utc_ts()}] Saved {out_path}")

    return result


@app.local_entrypoint()
def main():
    t0 = time.time()
    ops = list(OPERATIONS.keys())
    print(f"[{utc_ts()}] SVD vs DAS dimension-matched comparison")
    print(f"  Operations: {ops}")
    print(f"  k values: {K_VALUES}")
    print(f"  DAS steps: {DAS_STEPS}")
    print()

    handles = []
    for op in ops:
        h = run_single_operation.spawn(operation=op)
        handles.append((op, h))
        print(f"  Spawned {op}")

    print(f"\n{len(handles)} containers spawned. Collecting results...\n")

    all_results = []
    for op, h in handles:
        try:
            result = h.get()
        except Exception as e:
            tb = traceback.format_exc()
            result = {"operation": op, "grokked": False, "error": str(e), "traceback": tb[-2000:]}
        all_results.append(result)

        if result.get("grokked"):
            print(f"\n  {op}: GROKKED (test_loss={result['final_test_loss']:.4f})")
            for k in K_VALUES:
                kr = result.get(f"k{k}", {})
                print(f"    k={k:2d}:  DAS={kr.get('das_iia', 0):.3f}  "
                      f"SVD(W_in)={kr.get('svd_W_in_iia', 0):.3f}  "
                      f"SVD(W_eff)={kr.get('svd_W_eff_iia', 0):.3f}  "
                      f"Random={kr.get('random_mean_iia', 0):.3f}")
        else:
            print(f"\n  {op}: NOT GROKKED ({result.get('error', result.get('final_test_loss', 'unknown'))})")

    # Save summary
    summary_path = "experiments/results/svd_vs_das_matched_summary.jsonl"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, default=str) + "\n")

    elapsed = time.time() - t0

    print(f"\n{'='*80}")
    print(f"SVD vs DAS Dimension-Matched Comparison — Summary")
    print(f"{'='*80}")
    print(f"\n{'Operation':>22s}  {'k':>3s}  {'DAS':>6s}  {'SVD(Win)':>8s}  {'SVD(Weff)':>9s}  {'Random':>6s}")
    print("-" * 70)

    for r in all_results:
        if not r.get("grokked"):
            print(f"{r['operation']:>22s}  --- NOT GROKKED ---")
            continue
        for k in K_VALUES:
            kr = r.get(f"k{k}", {})
            print(f"{r['operation']:>22s}  {k:3d}  "
                  f"{kr.get('das_iia', 0):6.3f}  "
                  f"{kr.get('svd_W_in_iia', 0):8.3f}  "
                  f"{kr.get('svd_W_eff_iia', 0):9.3f}  "
                  f"{kr.get('random_mean_iia', 0):6.3f}")

    print(f"\nTotal wall time: {elapsed:.0f}s")
    print(f"Results: {summary_path}")
