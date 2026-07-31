"""k=1 VAE vs DAS with full controls + pi-VAE/pi-SAE ablations.

Tests whether a single-dimensional VAE z_causal genuinely recovers the causal
variable, or whether the nonlinear encoder-decoder is just memorizing.

Controls:
  C1. Random labels: train VAE with shuffled labels.
  C2. Reconstruction-only: train VAE with alpha=0 (no classifier loss).
  C3. Untrained VAE: random weights, no training.
  C4. Reconstruction quality: MSE of decoded activations.
  C5. Held-out generalization: train on 70%, eval on 30%.
  C6. Nonlinear DAS (bijective featurizer): MLP featurizer before linear DAS.

Ablations (new):
  pi-VAE:  Structured VAE (causal/nuisance split) with label-conditional prior
           p(z_c|y) = N(mu(y), sigma^2(y)) instead of N(0,I).
           Gives identifiability guarantees (Zhou & Wei, NeurIPS 2020).
  pi-SAE:  pi-VAE with overcomplete causal z (8x expansion) + L1 sparsity.
           Combines SAE-style sparse dictionary with label-conditional prior.

Tasks: IOI (GPT-2-small) + addition + multiplication + squaring (grokking)
k values: 1, 2, 4

Usage:
    modal run --detach experiments/k1_vae_vs_das.py
"""
from __future__ import annotations

import ast
import json
import logging
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone

import modal

try:
    import einops
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from tqdm import tqdm
    from transformer_lens import HookedTransformer, HookedTransformerConfig
except (ImportError, AttributeError):
    pass

# The canonical DAS module. "/root" is where it lands inside the Modal image;
# the directory of this file is where it lives in the repo.
for _p in ("/root", os.path.dirname(os.path.abspath(__file__))):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    import das
except (ImportError, AttributeError):
    das = None

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
        "matplotlib",
        "tqdm",
        "pandas",
        "scipy",
    )
    # pyvene supplies LowRankRotateLayer, which MIB's SubspaceFeaturizer wraps.
    .pip_install("pyvene==0.1.8")
    .add_local_dir(
        "./data/mib",
        remote_path="/mib_data",
    )
    # MIB's causal-variable track, imported by experiments/das.py at /mib_track.
    .add_local_dir(
        "./reference/MIB/MIB-causal-variable-track",
        remote_path="/mib_track",
    )
    .add_local_file("experiments/das.py", remote_path="/root/das.py")
)

app = modal.App("k1-vae-vs-das-pi-ablations", image=image)
results_vol = modal.Volume.from_name("fc-results", create_if_missing=True)

P = 113
FRAC_TRAIN = 0.3
DATA_SEED = 598

IOI_NAMES = [
    " Mary", " John", " Alice", " Bob", " Tom", " Claire",
    " Dave", " Sarah", " James", " Emma", " Mike", " Kate",
    " Jack", " Anna", " Dan", " Amy", " Sam", " Lisa",
]
IOI_PLACES = ["store", "park", "office", "restaurant", "library", "gym"]
IOI_OBJECTS = ["book", "drink", "ball", "pen", "bag", "phone"]
IOI_TEMPLATES = [
    "Then,{A} and{B} went to the {PLACE}.{B} gave a {OBJ} to",
    "Then,{A} and{B} had a lot of fun at the {PLACE}.{B} gave a {OBJ} to",
    "Then,{A} and{B} were working at the {PLACE}.{B} decided to give a {OBJ} to",
    "Then,{B} and{A} went to the {PLACE}.{B} gave a {OBJ} to",
    "Then,{B} and{A} had a lot of fun at the {PLACE}.{B} gave a {OBJ} to",
]


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ===================================================================
# VAE
# ===================================================================


def build_vae(d_input, z_causal_dim, z_nuisance_dim, hidden_dim, n_classes, device):
    class StructuredVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.z_causal_dim = z_causal_dim
            z_dim = z_causal_dim + z_nuisance_dim

            self.enc_trunk = nn.Sequential(
                nn.Linear(d_input, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            )
            self.enc_causal_mu = nn.Linear(hidden_dim, z_causal_dim)
            self.enc_causal_logvar = nn.Linear(hidden_dim, z_causal_dim)
            self.enc_nuisance_mu = nn.Linear(hidden_dim, z_nuisance_dim)
            self.enc_nuisance_logvar = nn.Linear(hidden_dim, z_nuisance_dim)

            self.decoder = nn.Sequential(
                nn.Linear(z_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, d_input),
            )
            self.classifier = nn.Linear(z_causal_dim, n_classes)

        def encode(self, x):
            h = self.enc_trunk(x)
            return (self.enc_causal_mu(h), self.enc_causal_logvar(h),
                    self.enc_nuisance_mu(h), self.enc_nuisance_logvar(h))

        def reparameterize(self, mu, logvar):
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

        def forward(self, x):
            mu_c, lv_c, mu_n, lv_n = self.encode(x)
            z_c = self.reparameterize(mu_c, lv_c)
            z_n = self.reparameterize(mu_n, lv_n)
            z = torch.cat([z_c, z_n], dim=-1)
            x_recon = self.decoder(z)
            logits = self.classifier(z_c)
            return x_recon, logits, mu_c, lv_c, mu_n, lv_n

    return StructuredVAE().to(device)


def build_unconstrained_vae(d_input, z_dim, hidden_dim, device):
    """Plain VAE with one big z — no causal/nuisance split, no classifier.

    If this gets high IIA via arbitrary dimension-swap, then ANY autoencoder
    with enough capacity can "cheat" and the structured VAE's advantage is
    meaningless. If it gets LOW IIA, then the structured separation matters.
    """

    class PlainVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.z_dim = z_dim

            self.encoder = nn.Sequential(
                nn.Linear(d_input, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            )
            self.enc_mu = nn.Linear(hidden_dim, z_dim)
            self.enc_logvar = nn.Linear(hidden_dim, z_dim)

            self.decoder = nn.Sequential(
                nn.Linear(z_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, d_input),
            )

        def encode(self, x):
            h = self.encoder(x)
            return self.enc_mu(h), self.enc_logvar(h)

        def reparameterize(self, mu, logvar):
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

        def forward(self, x):
            mu, lv = self.encode(x)
            z = self.reparameterize(mu, lv)
            return self.decoder(z), mu, lv

    return PlainVAE().to(device)


def train_unconstrained_vae(vae, acts, device, n_epochs=500, batch_size=256, lr=1e-3):
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    n = len(acts)
    for epoch in tqdm(range(n_epochs), desc="Plain VAE", leave=False):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x = acts[idx]
            x_r, mu, lv = vae(x)
            recon = F.mse_loss(x_r, x)
            kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()
            loss = recon + kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    vae.eval()
    return vae


def _iia_from_logits(logits, d):
    """Compute both standard and strict IIA for a single example.

    Standard: logit[src_id] > logit[base_id] (matches factorized DAS / MIB convention)
    Strict: argmax == src_id (src token must beat ALL 50k vocab tokens)
    """
    src_id, base_id = d["src_label"], d["base_label"]
    standard = int(logits[src_id].item() > logits[base_id].item())
    strict = int(logits.argmax().item() == src_id)
    return standard, strict


def _make_hook_replacement(h_iv):
    def hook_fn(act, hook=None, iv=h_iv):
        new = act.clone()
        new[0, -1, :] = iv
        return new
    return hook_fn


def _make_hook_additive(base_act, h_reconstructed, h_intervened):
    delta = h_intervened - h_reconstructed
    def hook_fn(act, hook=None, d=delta):
        new = act.clone()
        new[0, -1, :] += d
        return new
    return hook_fn


def _store_iia(results, prefix, iia_dict):
    """Unpack IIA dict (repl_std, repl_strict, add_std, add_strict) into results."""
    results[f"{prefix}_iia"] = iia_dict["add_std"]
    results[f"{prefix}_strict_iia"] = iia_dict["add_strict"]
    results[f"{prefix}_repl_iia"] = iia_dict["repl_std"]
    results[f"{prefix}_repl_strict_iia"] = iia_dict["repl_strict"]


def eval_unconstrained_vae_iia(vae, model_lm, pairs, hook_name, k, device):
    """Swap first k dims of z, keep rest — arbitrary split, no supervision."""
    vae.eval()
    counts = {"repl_std": 0, "repl_strict": 0, "add_std": 0, "add_strict": 0}
    with torch.inference_mode():
        for d in pairs:
            mu_b, _ = vae.encode(d["base_act"].unsqueeze(0))
            mu_s, _ = vae.encode(d["src_act"].unsqueeze(0))
            z_iv = mu_b.clone()
            z_iv[0, :k] = mu_s[0, :k]
            h_recon = vae.decoder(mu_b).squeeze(0)
            h_iv = vae.decoder(z_iv).squeeze(0)

            for mode in ("repl", "add"):
                if mode == "repl":
                    hook_fn = _make_hook_replacement(h_iv)
                else:
                    hook_fn = _make_hook_additive(d["base_act"], h_recon, h_iv)
                logits = model_lm.run_with_hooks(
                    d["base_toks"], fwd_hooks=[(hook_name, hook_fn)]
                )[0, -1, :]
                s, st = _iia_from_logits(logits, d)
                counts[f"{mode}_std"] += s
                counts[f"{mode}_strict"] += st
    n = len(pairs) if pairs else 1
    return {k_: v / n for k_, v in counts.items()}


def train_vae(vae, acts, labels, device, n_epochs=500, batch_size=256,
              lr=1e-3, alpha=10.0):
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    n = len(acts)
    for epoch in tqdm(range(n_epochs), desc="VAE", leave=False):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x, y = acts[idx], labels[idx]
            x_r, logits, mu_c, lv_c, mu_n, lv_n = vae(x)
            recon = F.mse_loss(x_r, x)
            kl_c = -0.5 * (1 + lv_c - mu_c.pow(2) - lv_c.exp()).mean()
            kl_n = -0.5 * (1 + lv_n - mu_n.pow(2) - lv_n.exp()).mean()
            ce = F.cross_entropy(logits, y)
            loss = recon + kl_c + kl_n + alpha * ce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    vae.eval()
    return vae


def eval_vae_metrics(vae, model_lm, pairs, hook_name, acts_for_recon, device):
    """Evaluate IIA (standard + strict) with both replacement and additive interventions."""
    vae.eval()
    counts = {"repl_std": 0, "repl_strict": 0, "add_std": 0, "add_strict": 0}
    with torch.inference_mode():
        for d in pairs:
            base_act = d["base_act"].unsqueeze(0)
            src_act = d["src_act"].unsqueeze(0)
            mu_c_b, _, mu_n_b, _ = vae.encode(base_act)
            mu_c_s, _, _, _ = vae.encode(src_act)
            z_base = torch.cat([mu_c_b, mu_n_b], dim=-1)
            z_iv = torch.cat([mu_c_s, mu_n_b], dim=-1)
            h_recon = vae.decoder(z_base).squeeze(0)
            h_iv = vae.decoder(z_iv).squeeze(0)

            for mode in ("repl", "add"):
                if mode == "repl":
                    hook_fn = _make_hook_replacement(h_iv)
                else:
                    hook_fn = _make_hook_additive(d["base_act"], h_recon, h_iv)
                logits = model_lm.run_with_hooks(
                    d["base_toks"], fwd_hooks=[(hook_name, hook_fn)]
                )[0, -1, :]
                s, st = _iia_from_logits(logits, d)
                counts[f"{mode}_std"] += s
                counts[f"{mode}_strict"] += st

    n = len(pairs) if pairs else 1
    with torch.inference_mode():
        x_r, _, mu_c, lv_c, mu_n, lv_n = vae(acts_for_recon)
        recon_mse = F.mse_loss(x_r, acts_for_recon).item()

    return {k: v / n for k, v in counts.items()}, recon_mse


# ===================================================================
# pi-plain: label-conditional prior, NO causal/nuisance split
# ===================================================================


def build_pi_plain_vae(d_input, z_dim, hidden_dim, n_classes, k_causal, device):
    class PiPlainVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.z_dim = z_dim
            self.k_causal = k_causal

            self.encoder = nn.Sequential(
                nn.Linear(d_input, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            )
            self.enc_mu = nn.Linear(hidden_dim, z_dim)
            self.enc_logvar = nn.Linear(hidden_dim, z_dim)

            self.decoder = nn.Sequential(
                nn.Linear(z_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, d_input),
            )
            self.classifier = nn.Linear(k_causal, n_classes)

            self.prior_mu = nn.Embedding(n_classes, z_dim)
            self.prior_logvar = nn.Embedding(n_classes, z_dim)

        def encode(self, x):
            h = self.encoder(x)
            return self.enc_mu(h), self.enc_logvar(h)

        def reparameterize(self, mu, logvar):
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

        def forward(self, x):
            mu, lv = self.encode(x)
            z = self.reparameterize(mu, lv)
            x_recon = self.decoder(z)
            logits = self.classifier(z[:, :self.k_causal])
            return x_recon, logits, mu, lv

    return PiPlainVAE().to(device)


def train_pi_plain_vae(vae, acts, labels, device, n_epochs=500, batch_size=256,
                       lr=1e-3, alpha=10.0):
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    n = len(acts)
    for epoch in tqdm(range(n_epochs), desc="pi-plain-VAE", leave=False):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x, y = acts[idx], labels[idx]
            x_r, logits, mu, lv = vae(x)
            recon = F.mse_loss(x_r, x)
            prior_mu = vae.prior_mu(y)
            prior_lv = vae.prior_logvar(y)
            kl = -0.5 * (1 + lv - prior_lv
                         - ((mu - prior_mu).pow(2) + lv.exp()) / prior_lv.exp()).mean()
            ce = F.cross_entropy(logits, y)
            loss = recon + kl + alpha * ce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    vae.eval()
    return vae


def build_pi_plain_sae(d_input, z_dim, hidden_dim, n_classes, k_causal,
                       expansion_factor, device):
    z_sparse = z_dim * expansion_factor

    class PiPlainSAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.z_dim = z_sparse
            self.k_causal = k_causal * expansion_factor

            self.encoder = nn.Sequential(
                nn.Linear(d_input, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            )
            self.enc_mu = nn.Linear(hidden_dim, z_sparse)
            self.enc_logvar = nn.Linear(hidden_dim, z_sparse)

            self.decoder = nn.Sequential(
                nn.Linear(z_sparse, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, d_input),
            )
            self.classifier = nn.Linear(self.k_causal, n_classes)

            self.prior_mu = nn.Embedding(n_classes, z_sparse)
            self.prior_logvar = nn.Embedding(n_classes, z_sparse)

        def encode(self, x):
            h = self.encoder(x)
            return self.enc_mu(h), self.enc_logvar(h)

        def reparameterize(self, mu, logvar):
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

        def forward(self, x):
            mu, lv = self.encode(x)
            z = self.reparameterize(mu, lv)
            x_recon = self.decoder(z)
            logits = self.classifier(z[:, :self.k_causal])
            return x_recon, logits, mu, lv

    return PiPlainSAE().to(device)


def train_pi_plain_sae(vae, acts, labels, device, n_epochs=500, batch_size=256,
                       lr=1e-3, alpha=10.0, l1_coeff=1e-3):
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    n = len(acts)
    for epoch in tqdm(range(n_epochs), desc="pi-plain-SAE", leave=False):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x, y = acts[idx], labels[idx]
            x_r, logits, mu, lv = vae(x)
            recon = F.mse_loss(x_r, x)
            prior_mu = vae.prior_mu(y)
            prior_lv = vae.prior_logvar(y)
            kl = -0.5 * (1 + lv - prior_lv
                         - ((mu - prior_mu).pow(2) + lv.exp()) / prior_lv.exp()).mean()
            ce = F.cross_entropy(logits, y)
            sparsity = mu.abs().mean()
            loss = recon + kl + alpha * ce + l1_coeff * sparsity
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    vae.eval()
    return vae


def eval_pi_plain_iia(vae, model_lm, pairs, hook_name, k, device):
    vae.eval()
    counts = {"repl_std": 0, "repl_strict": 0, "add_std": 0, "add_strict": 0}
    with torch.inference_mode():
        for d in pairs:
            mu_b, _ = vae.encode(d["base_act"].unsqueeze(0))
            mu_s, _ = vae.encode(d["src_act"].unsqueeze(0))
            z_iv = mu_b.clone()
            z_iv[0, :k] = mu_s[0, :k]
            h_recon = vae.decoder(mu_b).squeeze(0)
            h_iv = vae.decoder(z_iv).squeeze(0)

            for mode in ("repl", "add"):
                if mode == "repl":
                    hook_fn = _make_hook_replacement(h_iv)
                else:
                    hook_fn = _make_hook_additive(d["base_act"], h_recon, h_iv)
                logits = model_lm.run_with_hooks(
                    d["base_toks"], fwd_hooks=[(hook_name, hook_fn)]
                )[0, -1, :]
                s, st = _iia_from_logits(logits, d)
                counts[f"{mode}_std"] += s
                counts[f"{mode}_strict"] += st
    n = len(pairs) if pairs else 1
    return {k_: v / n for k_, v in counts.items()}


# ===================================================================
# pi-VAE: label-conditional prior (Zhou & Wei, NeurIPS 2020)
# ===================================================================


def build_pi_vae(d_input, z_causal_dim, z_nuisance_dim, hidden_dim, n_classes, device):
    class PiVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.z_causal_dim = z_causal_dim
            z_dim = z_causal_dim + z_nuisance_dim

            self.enc_trunk = nn.Sequential(
                nn.Linear(d_input, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            )
            self.enc_causal_mu = nn.Linear(hidden_dim, z_causal_dim)
            self.enc_causal_logvar = nn.Linear(hidden_dim, z_causal_dim)
            self.enc_nuisance_mu = nn.Linear(hidden_dim, z_nuisance_dim)
            self.enc_nuisance_logvar = nn.Linear(hidden_dim, z_nuisance_dim)

            self.decoder = nn.Sequential(
                nn.Linear(z_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, d_input),
            )
            self.classifier = nn.Linear(z_causal_dim, n_classes)

            self.prior_mu = nn.Embedding(n_classes, z_causal_dim)
            self.prior_logvar = nn.Embedding(n_classes, z_causal_dim)

        def encode(self, x):
            h = self.enc_trunk(x)
            return (self.enc_causal_mu(h), self.enc_causal_logvar(h),
                    self.enc_nuisance_mu(h), self.enc_nuisance_logvar(h))

        def reparameterize(self, mu, logvar):
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

        def forward(self, x):
            mu_c, lv_c, mu_n, lv_n = self.encode(x)
            z_c = self.reparameterize(mu_c, lv_c)
            z_n = self.reparameterize(mu_n, lv_n)
            z = torch.cat([z_c, z_n], dim=-1)
            x_recon = self.decoder(z)
            logits = self.classifier(z_c)
            return x_recon, logits, mu_c, lv_c, mu_n, lv_n

    return PiVAE().to(device)


def train_pi_vae(vae, acts, labels, device, n_epochs=500, batch_size=256,
                 lr=1e-3, alpha=10.0):
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    n = len(acts)
    for epoch in tqdm(range(n_epochs), desc="pi-VAE", leave=False):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x, y = acts[idx], labels[idx]
            x_r, logits, mu_c, lv_c, mu_n, lv_n = vae(x)
            recon = F.mse_loss(x_r, x)
            prior_mu = vae.prior_mu(y)
            prior_lv = vae.prior_logvar(y)
            kl_c = -0.5 * (1 + lv_c - prior_lv
                           - ((mu_c - prior_mu).pow(2) + lv_c.exp()) / prior_lv.exp()).mean()
            kl_n = -0.5 * (1 + lv_n - mu_n.pow(2) - lv_n.exp()).mean()
            ce = F.cross_entropy(logits, y)
            loss = recon + kl_c + kl_n + alpha * ce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    vae.eval()
    return vae


# ===================================================================
# pi-SAE: sparse overcomplete + label-conditional prior
# ===================================================================


def build_pi_sae(d_input, z_causal_dim, z_nuisance_dim, hidden_dim, n_classes,
                 expansion_factor, device):
    z_sparse_dim = z_causal_dim * expansion_factor

    class PiSAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.z_causal_dim = z_causal_dim
            self.z_sparse_dim = z_sparse_dim
            z_dim = z_sparse_dim + z_nuisance_dim

            self.enc_trunk = nn.Sequential(
                nn.Linear(d_input, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            )
            self.enc_causal_mu = nn.Linear(hidden_dim, z_sparse_dim)
            self.enc_causal_logvar = nn.Linear(hidden_dim, z_sparse_dim)
            self.enc_nuisance_mu = nn.Linear(hidden_dim, z_nuisance_dim)
            self.enc_nuisance_logvar = nn.Linear(hidden_dim, z_nuisance_dim)

            self.decoder = nn.Sequential(
                nn.Linear(z_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, d_input),
            )
            self.classifier = nn.Linear(z_sparse_dim, n_classes)

            self.prior_mu = nn.Embedding(n_classes, z_sparse_dim)
            self.prior_logvar = nn.Embedding(n_classes, z_sparse_dim)

        def encode(self, x):
            h = self.enc_trunk(x)
            return (self.enc_causal_mu(h), self.enc_causal_logvar(h),
                    self.enc_nuisance_mu(h), self.enc_nuisance_logvar(h))

        def reparameterize(self, mu, logvar):
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

        def forward(self, x):
            mu_c, lv_c, mu_n, lv_n = self.encode(x)
            z_c = self.reparameterize(mu_c, lv_c)
            z_n = self.reparameterize(mu_n, lv_n)
            z = torch.cat([z_c, z_n], dim=-1)
            x_recon = self.decoder(z)
            logits = self.classifier(z_c)
            return x_recon, logits, mu_c, lv_c, mu_n, lv_n

    return PiSAE().to(device)


def train_pi_sae(vae, acts, labels, device, n_epochs=500, batch_size=256,
                 lr=1e-3, alpha=10.0, l1_coeff=1e-3):
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    n = len(acts)
    for epoch in tqdm(range(n_epochs), desc="pi-SAE", leave=False):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x, y = acts[idx], labels[idx]
            x_r, logits, mu_c, lv_c, mu_n, lv_n = vae(x)
            recon = F.mse_loss(x_r, x)
            prior_mu = vae.prior_mu(y)
            prior_lv = vae.prior_logvar(y)
            kl_c = -0.5 * (1 + lv_c - prior_lv
                           - ((mu_c - prior_mu).pow(2) + lv_c.exp()) / prior_lv.exp()).mean()
            kl_n = -0.5 * (1 + lv_n - mu_n.pow(2) - lv_n.exp()).mean()
            ce = F.cross_entropy(logits, y)
            sparsity = mu_c.abs().mean()
            loss = recon + kl_c + kl_n + alpha * ce + l1_coeff * sparsity
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    vae.eval()
    return vae


def train_pi_sae_e2e(vae, acts, labels, train_pairs, model_lm, hook_name,
                     device, n_epochs=500, batch_size=256, lr=1e-3,
                     alpha=10.0, l1_coeff=1e-3, beta=1.0, interv_batch=8):
    """Train Structured pi-SAE with end-to-end intervention CE loss.

    Two-phase: 200 epochs reconstruction-only warmup, then add intervention loss.
    """
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    n = len(acts)
    warmup_epochs = min(200, n_epochs // 3)
    for epoch in tqdm(range(n_epochs), desc="pi-SAE-e2e", leave=False):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x, y = acts[idx], labels[idx]
            x_r, logits, mu_c, lv_c, mu_n, lv_n = vae(x)
            recon = F.mse_loss(x_r, x)
            prior_mu = vae.prior_mu(y)
            prior_lv = vae.prior_logvar(y)
            kl_c = -0.5 * (1 + lv_c - prior_lv
                           - ((mu_c - prior_mu).pow(2) + lv_c.exp()) / prior_lv.exp()).mean()
            kl_n = -0.5 * (1 + lv_n - mu_n.pow(2) - lv_n.exp()).mean()
            ce = F.cross_entropy(logits, y)
            sparsity = mu_c.abs().mean()
            loss = recon + kl_c + kl_n + alpha * ce + l1_coeff * sparsity

            if epoch >= warmup_epochs and train_pairs:
                batch_pairs = random.sample(train_pairs, min(interv_batch, len(train_pairs)))
                interv_loss = torch.tensor(0.0, device=device)
                for d in batch_pairs:
                    base_act = d["base_act"].unsqueeze(0)
                    src_act = d["src_act"].unsqueeze(0)
                    mu_c_b, _, mu_n_b, _ = vae.encode(base_act)
                    mu_c_s, _, _, _ = vae.encode(src_act)
                    z_base = torch.cat([mu_c_b, mu_n_b], dim=-1)
                    z_iv = torch.cat([mu_c_s, mu_n_b], dim=-1)
                    h_recon = vae.decoder(z_base)
                    h_iv = vae.decoder(z_iv)
                    delta = (h_iv - h_recon).squeeze(0)

                    def hook_fn(act, hook=None, d=delta):
                        new = act.clone()
                        new[0, -1, :] = new[0, -1, :] + d
                        return new

                    iv_logits = model_lm.run_with_hooks(
                        d["base_toks"], fwd_hooks=[(hook_name, hook_fn)]
                    )[0, -1, :]
                    interv_loss = interv_loss - F.log_softmax(iv_logits, dim=-1)[d["src_label"]]
                loss = loss + beta * (interv_loss / len(batch_pairs))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    vae.eval()
    return vae


def eval_pi_vae_metrics(vae, model_lm, pairs, hook_name, acts_for_recon,
                        labels_for_recon, device):
    """Evaluate IIA (standard + strict, replacement + additive) for pi-VAE/pi-SAE."""
    vae.eval()
    counts = {"repl_std": 0, "repl_strict": 0, "add_std": 0, "add_strict": 0}
    with torch.inference_mode():
        for d in pairs:
            base_act = d["base_act"].unsqueeze(0)
            src_act = d["src_act"].unsqueeze(0)
            mu_c_b, _, mu_n_b, _ = vae.encode(base_act)
            mu_c_s, _, _, _ = vae.encode(src_act)
            z_base = torch.cat([mu_c_b, mu_n_b], dim=-1)
            z_iv = torch.cat([mu_c_s, mu_n_b], dim=-1)
            h_recon = vae.decoder(z_base).squeeze(0)
            h_iv = vae.decoder(z_iv).squeeze(0)

            for mode in ("repl", "add"):
                if mode == "repl":
                    hook_fn = _make_hook_replacement(h_iv)
                else:
                    hook_fn = _make_hook_additive(d["base_act"], h_recon, h_iv)
                logits = model_lm.run_with_hooks(
                    d["base_toks"], fwd_hooks=[(hook_name, hook_fn)]
                )[0, -1, :]
                s, st = _iia_from_logits(logits, d)
                counts[f"{mode}_std"] += s
                counts[f"{mode}_strict"] += st

    n = len(pairs) if pairs else 1
    with torch.inference_mode():
        x_r, _, mu_c, lv_c, mu_n, lv_n = vae(acts_for_recon)
        recon_mse = F.mse_loss(x_r, acts_for_recon).item()

    return {k: v / n for k, v in counts.items()}, recon_mse


# ===================================================================
# DAS (linear + nonlinear featurizer)
# ===================================================================


def train_das(model_lm, pairs, hook_name, d_model, k, device,
              n_steps=300, lr=1e-3, batch_size=16):
    R = nn.Parameter(torch.randn(d_model, k, device=device) * 0.02)
    optimizer = torch.optim.Adam([R], lr=lr)

    for step in tqdm(range(n_steps), desc=f"DAS k={k}", leave=False):
        Q, _ = torch.linalg.qr(R)
        proj = Q @ Q.T
        batch = random.sample(pairs, min(batch_size, len(pairs)))
        loss = torch.tensor(0.0, device=device)
        for d in batch:
            iv = d["base_act"] - proj @ d["base_act"] + proj @ d["src_act"]

            def hook_fn(act, hook=None, iv_vec=iv):
                new = act.clone()
                new[0, -1, :] = iv_vec
                return new

            logits = model_lm.run_with_hooks(
                d["base_toks"], fwd_hooks=[(hook_name, hook_fn)]
            )[0, -1, :]
            loss = loss - F.log_softmax(logits, dim=-1)[d["src_label"]]
        (loss / len(batch)).backward()
        optimizer.step()
        optimizer.zero_grad()

    with torch.no_grad():
        Q, _ = torch.linalg.qr(R)
    return Q.detach()


# ===================================================================
# DAS matching the MIB / pyvene reference implementation
#
# MIB builds its subspace featurizer as
#     LowRankRotateLayer(d, k, init_orth=True)
#     -> torch.nn.utils.parametrizations.orthogonal(...)
# i.e. a random orthogonal initialisation kept on the Stiefel manifold
# throughout optimisation, rather than the Gaussian initialisation with per-step
# QR used by train_das above. Both are fitted in the same run so the discrepancy
# is measured under identical data and seeds (Experiment 0 of
# PREREGISTRATION_RECONSTRUCTION_CRITERION.md). pyvene is not a dependency here;
# this reproduces the same construction.
# ===================================================================

def train_das_reference(model_lm, pairs, hook_name, d_model, k, device,
                        n_steps=None, lr=None, batch_size=None, task=None):
    """Standard DAS, run with MIB's own code and MIB's hyperparameters.

    Delegates to experiments/das.py, whose standard arm is MIB's
    CausalAbstraction.neural.featurizers.SubspaceFeaturizer wrapping pyvene's
    LowRankRotateLayer. Budget is MIB's 3 epochs at batch 32 with AdamW at
    lr=1e-2, rather than this file's historical 300 steps at lr=1e-3.

    experiments/test_das_matches_mib.py verifies the equivalence.
    """
    if das is None:
        raise ImportError(
            "experiments/das.py did not import; the reference DAS arm needs it "
            "along with pyvene and the vendored MIB checkout.")
    if n_steps is None:
        return das.train_das_mib(
            model_lm, pairs, hook_name, device, k=k, task=task,
            key_base_act="base_act", key_src_act="src_act",
            key_base_toks="base_toks", key_target="src_label")
    return das.train_das(
        model_lm, pairs, hook_name, device, k=k, n_steps=n_steps,
        lr=lr or das.MIB_CONFIG["lr"],
        batch_size=batch_size or das.MIB_CONFIG["batch_size"],
        key_base_act="base_act", key_src_act="src_act",
        key_base_toks="base_toks", key_target="src_label")


# Student's t 95% critical values, keyed by sample count. Same table as
# experiments/aggregate_seeds.py so intervals are computed identically.
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
       9: 2.306, 10: 2.262}

# Independent random subspaces drawn per task to estimate the noise floor.
# Five matches the fit count fixed for Experiment 0.
FLOOR_DRAWS = 5


def mean_ci95(xs):
    """Mean and 95% Student-t half-width. Half-width is 0.0 for a single value."""
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    se = (var / n) ** 0.5
    return m, T95.get(n, 1.96) * se


def eval_das_iia(Q, model_lm, pairs, hook_name, device):
    proj = Q @ Q.T
    std_correct = strict_correct = 0
    with torch.inference_mode():
        for d in pairs:
            iv = d["base_act"] - proj @ d["base_act"] + proj @ d["src_act"]

            def hook_fn(act, hook=None, iv_vec=iv):
                new = act.clone()
                new[0, -1, :] = iv_vec
                return new

            logits = model_lm.run_with_hooks(
                d["base_toks"], fwd_hooks=[(hook_name, hook_fn)]
            )[0, -1, :]
            s, st = _iia_from_logits(logits, d)
            std_correct += s
            strict_correct += st
    n = len(pairs) if pairs else 1
    return std_correct / n, strict_correct / n


def train_nonlinear_das(model_lm, pairs, hook_name, d_model, k, device,
                        hidden_dim=256, n_steps=400, lr=1e-3, batch_size=16):
    """Nonlinear DAS: MLP featurizer -> linear projection -> MLP inverse.

    Same encoder capacity as the VAE but no generative structure (no ELBO).
    Tests whether the advantage comes from nonlinearity or from the VAE structure.
    """
    featurizer = nn.Sequential(
        nn.Linear(d_model, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, d_model),
    ).to(device)
    inv_featurizer = nn.Sequential(
        nn.Linear(d_model, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, d_model),
    ).to(device)
    R = nn.Parameter(torch.randn(d_model, k, device=device) * 0.02)

    params = list(featurizer.parameters()) + list(inv_featurizer.parameters()) + [R]
    optimizer = torch.optim.Adam(params, lr=lr)

    for step in tqdm(range(n_steps), desc=f"NL-DAS k={k}", leave=False):
        Q, _ = torch.linalg.qr(R)
        proj = Q @ Q.T

        batch = random.sample(pairs, min(batch_size, len(pairs)))
        loss = torch.tensor(0.0, device=device)
        for d in batch:
            feat_base = featurizer(d["base_act"])
            feat_src = featurizer(d["src_act"])
            feat_iv = feat_base - proj @ feat_base + proj @ feat_src
            iv = inv_featurizer(feat_iv)

            def hook_fn(act, hook=None, iv_vec=iv):
                new = act.clone()
                new[0, -1, :] = iv_vec
                return new

            logits = model_lm.run_with_hooks(
                d["base_toks"], fwd_hooks=[(hook_name, hook_fn)]
            )[0, -1, :]
            loss = loss - F.log_softmax(logits, dim=-1)[d["src_label"]]
        (loss / len(batch)).backward()
        optimizer.step()
        optimizer.zero_grad()

    featurizer.eval()
    inv_featurizer.eval()
    with torch.no_grad():
        Q, _ = torch.linalg.qr(R)
        Q = Q.detach()

    return featurizer, inv_featurizer, Q


def eval_nonlinear_das_iia(featurizer, inv_featurizer, Q, model_lm, pairs, hook_name, device):
    proj = Q @ Q.T
    std_correct = strict_correct = 0
    with torch.inference_mode():
        for d in pairs:
            feat_b = featurizer(d["base_act"])
            feat_s = featurizer(d["src_act"])
            feat_iv = feat_b - proj @ feat_b + proj @ feat_s
            iv = inv_featurizer(feat_iv)

            def hook_fn(act, hook=None, iv_vec=iv):
                new = act.clone()
                new[0, -1, :] = iv_vec
                return new

            logits = model_lm.run_with_hooks(
                d["base_toks"], fwd_hooks=[(hook_name, hook_fn)]
            )[0, -1, :]
            s, st = _iia_from_logits(logits, d)
            std_correct += s
            strict_correct += st
    n = len(pairs) if pairs else 1
    return std_correct / n, strict_correct / n


# ===================================================================
# Run all controls for a single (task, k) combo
# ===================================================================


def run_all_variants(model_lm, train_pairs, eval_pairs, act_t, lab_t,
                     n_classes, d_model, k, hook_name, device, log,
                     hidden_dim=256, vae_epochs=500, train_idx=None, task=None):
    """Run VAE + all controls + DAS + nonlinear DAS for one k value."""

    results = {}

    train_acts = act_t[train_idx] if train_idx is not None else act_t
    train_labs = lab_t[train_idx] if train_idx is not None else lab_t

    # --- Linear DAS ---
    log.info(f"[{utc_ts()}]   DAS k={k}")
    Q = train_das(model_lm, train_pairs[:400], hook_name, d_model, k, device, n_steps=300)
    das_std, das_strict = eval_das_iia(Q, model_lm, eval_pairs, hook_name, device)
    results["das_iia"] = das_std
    results["das_strict_iia"] = das_strict
    log.info(f"[{utc_ts()}]     IIA={das_std:.4f} (strict={das_strict:.4f})")

    # --- DAS under the reference orthogonal parametrisation, same run ---
    # Experiment 0: both parametrisations are fitted on the same data under the
    # same seed so the discrepancy is measured within a run rather than across
    # commits. The local implementation is removed only after this completes.
    log.info(f"[{utc_ts()}]   DAS (reference parametrisation) k={k}")
    # Which configuration this task actually got. MIB has baselines for some
    # tasks and not others; where it has none the library default is used, and
    # `is_mib_baseline` records that so results cannot claim MIB's settings for a
    # task MIB never configured.
    ref_cfg = das.mib_config(task) if das is not None else {}
    results["das_reference_config"] = ref_cfg
    log.info(f"[{utc_ts()}]   DAS reference config for task={task!r}: "
             f"lr={ref_cfg.get('lr')} epochs={ref_cfg.get('n_epochs')} "
             f"batch={ref_cfg.get('batch_size')} "
             f"mib_baseline={ref_cfg.get('is_mib_baseline')}")
    # The reference arm uses the FULL training split, as MIB does. The 400-pair
    # cap below is retained only for the historical arm, so its published number
    # stays reproducible. Capping here would be actively wrong: MIB's budget is
    # in epochs, and at batch 1024 a 400-pair cap yields two optimiser steps.
    results["das_reference_n_train"] = len(train_pairs)
    results["das_local_n_train"] = len(train_pairs[:400])
    Q_ref = train_das_reference(model_lm, train_pairs, hook_name, d_model,
                                k, device, task=task)  # MIB's per-task budget
    das_ref_std, das_ref_strict = eval_das_iia(Q_ref, model_lm, eval_pairs, hook_name, device)
    results["das_reference_iia"] = das_ref_std
    results["das_reference_strict_iia"] = das_ref_strict
    results["das_parametrisation_delta"] = das_ref_std - das_std
    results["das_parametrisation_delta_strict"] = das_ref_strict - das_strict
    log.info(f"[{utc_ts()}]     IIA={das_ref_std:.4f} (strict={das_ref_strict:.4f}), "
             f"delta vs local={das_ref_std - das_std:+.4f} "
             f"(strict {das_ref_strict - das_strict:+.4f})")

    # --- Step C of Experiment 0: convergence curve on a real task ---
    # The planted-direction toy converges within fifty steps at MIB's learning
    # rate; GPT-2 tasks are not assumed to. Snapshots come from ONE trajectory,
    # so the curve costs a single fit rather than one fit per budget. Step counts
    # are doublings around MIB's own budget so the pre-registered rule (within
    # 0.02 of 2x the steps, twice consecutively) has doublings to work with.
    das_train = train_pairs
    mib_budget = das.steps_for_epochs(len(das_train), ref_cfg.get("batch_size"),
                                      ref_cfg.get("n_epochs"))
    snap_steps = sorted({max(1, mib_budget // 4), max(1, mib_budget // 2),
                         mib_budget, 2 * mib_budget, 4 * mib_budget})
    log.info(f"[{utc_ts()}]   DAS convergence curve k={k}: MIB budget "
             f"{mib_budget} steps ({len(das_train)} pairs, 3 epochs @ batch "
             f"{das.MIB_CONFIG['batch_size']}); snapshots at {snap_steps}")
    snaps = das.train_das_snapshots(
        model_lm, das_train, hook_name, device, k=k, snapshot_steps=snap_steps,
        key_base_act="base_act", key_src_act="src_act",
        key_base_toks="base_toks", key_target="src_label")
    curve = {}
    for s in sorted(snaps):
        s_std, s_strict = eval_das_iia(snaps[s], model_lm, eval_pairs, hook_name, device)
        curve[s] = {"iia": s_std, "strict_iia": s_strict}
        log.info(f"[{utc_ts()}]     step {s:5d}: IIA={s_std:.4f} strict={s_strict:.4f}"
                 + ("  <- MIB budget" if s == mib_budget else ""))
    conv = das.converged_step({s: v["iia"] for s, v in curve.items()})
    results["das_convergence_curve"] = curve
    results["das_mib_budget_steps"] = mib_budget
    results["das_converged_step"] = conv
    results["das_converged_at_mib_budget"] = (conv is not None and conv <= mib_budget)
    log.info(f"[{utc_ts()}]     converged at step {conv} "
             f"(MIB budget {mib_budget}); "
             f"{'converged within budget' if conv is not None and conv <= mib_budget else 'NOT converged within budget tested'}")

    # --- Nonlinear DAS (C6: same encoder capacity, no ELBO) ---
    log.info(f"[{utc_ts()}]   Nonlinear DAS k={k}")
    feat, inv_feat, Q_nl = train_nonlinear_das(
        model_lm, train_pairs[:400], hook_name, d_model, k, device,
        hidden_dim=hidden_dim, n_steps=400,
    )
    nl_std, nl_strict = eval_nonlinear_das_iia(
        feat, inv_feat, Q_nl, model_lm, eval_pairs, hook_name, device,
    )
    results["nl_das_iia"] = nl_std
    results["nl_das_strict_iia"] = nl_strict
    log.info(f"[{utc_ts()}]     IIA={nl_std:.4f} (strict={nl_strict:.4f})")
    del feat, inv_feat
    torch.cuda.empty_cache()

    # --- VAE (main) ---
    log.info(f"[{utc_ts()}]   VAE k={k}")
    vae = build_vae(d_model, k, 16, hidden_dim, n_classes, device)
    vae = train_vae(vae, train_acts, train_labs, device, n_epochs=vae_epochs, alpha=10.0)
    vae_iia_d, vae_recon = eval_vae_metrics(vae, model_lm, eval_pairs, hook_name, act_t[:200], device)
    with torch.inference_mode():
        mu_c, _, _, _ = vae.encode(act_t[:200])
        cls_acc = (vae.classifier(mu_c).argmax(-1) == lab_t[:200]).float().mean().item()
    _store_iia(results, "vae", vae_iia_d)
    results["vae_recon_mse"] = vae_recon
    results["vae_cls_acc"] = cls_acc
    log.info(f"[{utc_ts()}]     add={vae_iia_d['add_std']:.4f} repl={vae_iia_d['repl_std']:.4f} (strict: add={vae_iia_d['add_strict']:.4f} repl={vae_iia_d['repl_strict']:.4f}), recon={vae_recon:.6f}, cls={cls_acc:.4f}")
    del vae
    torch.cuda.empty_cache()

    # --- C1: Random labels ---
    log.info(f"[{utc_ts()}]   C1: Random labels k={k}")
    rand_labels = train_labs[torch.randperm(len(train_labs), device=device)]
    vae_rand = build_vae(d_model, k, 16, hidden_dim, n_classes, device)
    vae_rand = train_vae(vae_rand, train_acts, rand_labels, device, n_epochs=vae_epochs, alpha=10.0)
    rand_iia_d, rand_recon = eval_vae_metrics(vae_rand, model_lm, eval_pairs, hook_name, act_t[:200], device)
    with torch.inference_mode():
        mu_c, _, _, _ = vae_rand.encode(act_t[:200])
        rand_cls = (vae_rand.classifier(mu_c).argmax(-1) == rand_labels[:200]).float().mean().item()
    _store_iia(results, "c1_random_labels", rand_iia_d)
    results["c1_random_labels_recon"] = rand_recon
    results["c1_random_labels_cls_acc"] = rand_cls
    log.info(f"[{utc_ts()}]     add={rand_iia_d['add_std']:.4f} repl={rand_iia_d['repl_std']:.4f}, recon={rand_recon:.6f}, cls={rand_cls:.4f}")
    del vae_rand
    torch.cuda.empty_cache()

    # --- C2: Reconstruction-only (alpha=0) ---
    log.info(f"[{utc_ts()}]   C2: Recon-only k={k}")
    vae_recon_only = build_vae(d_model, k, 16, hidden_dim, n_classes, device)
    vae_recon_only = train_vae(vae_recon_only, train_acts, train_labs, device, n_epochs=vae_epochs, alpha=0.0)
    recon_iia_d, recon_mse = eval_vae_metrics(vae_recon_only, model_lm, eval_pairs, hook_name, act_t[:200], device)
    _store_iia(results, "c2_recon_only", recon_iia_d)
    results["c2_recon_only_recon"] = recon_mse
    log.info(f"[{utc_ts()}]     add={recon_iia_d['add_std']:.4f} repl={recon_iia_d['repl_std']:.4f}, recon={recon_mse:.6f}")
    del vae_recon_only
    torch.cuda.empty_cache()

    # --- C3: Untrained VAE ---
    log.info(f"[{utc_ts()}]   C3: Untrained VAE k={k}")
    vae_untrained = build_vae(d_model, k, 16, hidden_dim, n_classes, device)
    vae_untrained.eval()
    untrained_iia_d, untrained_recon = eval_vae_metrics(
        vae_untrained, model_lm, eval_pairs, hook_name, act_t[:200], device,
    )
    _store_iia(results, "c3_untrained", untrained_iia_d)
    results["c3_untrained_recon"] = untrained_recon
    log.info(f"[{utc_ts()}]     add={untrained_iia_d['add_std']:.4f} repl={untrained_iia_d['repl_std']:.4f}, recon={untrained_recon:.6f}")
    del vae_untrained
    torch.cuda.empty_cache()

    # --- C4 is captured by vae_recon_mse above ---

    # --- C5: Held-out generalization is the default (train/eval split) ---

    # --- C7: Unconstrained VAE (no causal/nuisance split, no supervision) ---
    log.info(f"[{utc_ts()}]   C7: Unconstrained VAE k={k}")
    total_z = k + 16
    plain_vae = build_unconstrained_vae(d_model, total_z, hidden_dim, device)
    plain_vae = train_unconstrained_vae(plain_vae, act_t, device, n_epochs=vae_epochs)
    plain_iia_d = eval_unconstrained_vae_iia(plain_vae, model_lm, eval_pairs, hook_name, k, device)
    with torch.inference_mode():
        x_r, _, _ = plain_vae(act_t[:200])
        plain_recon = F.mse_loss(x_r, act_t[:200]).item()
    _store_iia(results, "c7_unconstrained", plain_iia_d)
    results["c7_unconstrained_recon"] = plain_recon
    log.info(f"[{utc_ts()}]     add={plain_iia_d['add_std']:.4f} repl={plain_iia_d['repl_std']:.4f}, recon={plain_recon:.6f}")
    del plain_vae
    torch.cuda.empty_cache()

    # --- Random subspace DAS (noise floor), estimated with an interval ---
    # Every vacuity threshold in PREREGISTRATION_RECONSTRUCTION_CRITERION.md is
    # defined as an excess over this floor's UPPER bound. A single draw gives a
    # point estimate with no sampling spread, which is why three tasks previously
    # reported a floor of exactly 0.000 against which a 0.05 margin is not a
    # detectable quantity. Estimated over FLOOR_DRAWS independent subspaces.
    floor_std_vals, floor_strict_vals = [], []
    for _ in range(FLOOR_DRAWS):
        Q_rand, _ = torch.linalg.qr(torch.randn(d_model, k, device=device))
        s, st = eval_das_iia(Q_rand, model_lm, eval_pairs, hook_name, device)
        floor_std_vals.append(s)
        floor_strict_vals.append(st)
    rand_das_std, rand_std_hw = mean_ci95(floor_std_vals)
    rand_das_strict, rand_strict_hw = mean_ci95(floor_strict_vals)
    results["random_das_iia"] = rand_das_std
    results["random_das_strict_iia"] = rand_das_strict
    results["random_das_iia_ci95_halfwidth"] = rand_std_hw
    results["random_das_strict_ci95_halfwidth"] = rand_strict_hw
    results["random_das_iia_upper"] = rand_das_std + rand_std_hw
    results["random_das_strict_upper"] = rand_das_strict + rand_strict_hw
    results["random_das_iia_draws"] = floor_std_vals
    results["random_das_strict_draws"] = floor_strict_vals
    log.info(f"[{utc_ts()}]   Random DAS floor over {FLOOR_DRAWS} draws: "
             f"IIA={rand_das_std:.4f}+/-{rand_std_hw:.4f} (upper={rand_das_std + rand_std_hw:.4f}), "
             f"strict={rand_das_strict:.4f}+/-{rand_strict_hw:.4f} "
             f"(upper={rand_das_strict + rand_strict_hw:.4f})")

    # --- pi-plain-VAE (label-conditional prior, NO causal/nuisance split) ---
    log.info(f"[{utc_ts()}]   pi-plain-VAE k={k}")
    total_z = k + 16
    pp_vae = build_pi_plain_vae(d_model, total_z, hidden_dim, n_classes, k, device)
    pp_vae = train_pi_plain_vae(pp_vae, act_t, lab_t, device, n_epochs=vae_epochs, alpha=10.0)
    pp_iia_d = eval_pi_plain_iia(pp_vae, model_lm, eval_pairs, hook_name, k, device)
    with torch.inference_mode():
        mu, _ = pp_vae.encode(act_t[:200])
        pp_recon = F.mse_loss(pp_vae.decoder(mu), act_t[:200]).item()
        pp_cls = (pp_vae.classifier(mu[:, :k]).argmax(-1) == lab_t[:200]).float().mean().item()
    _store_iia(results, "pi_plain_vae", pp_iia_d)
    results["pi_plain_vae_recon_mse"] = pp_recon
    results["pi_plain_vae_cls_acc"] = pp_cls
    log.info(f"[{utc_ts()}]     add={pp_iia_d['add_std']:.4f} repl={pp_iia_d['repl_std']:.4f}, recon={pp_recon:.6f}, cls={pp_cls:.4f}")
    del pp_vae
    torch.cuda.empty_cache()

    # --- pi-plain-SAE (label-conditional prior, overcomplete, NO split) ---
    log.info(f"[{utc_ts()}]   pi-plain-SAE k={k}")
    pp_sae = build_pi_plain_sae(d_model, total_z, hidden_dim, n_classes, k,
                                expansion_factor=8, device=device)
    pp_sae = train_pi_plain_sae(pp_sae, act_t, lab_t, device, n_epochs=vae_epochs,
                                alpha=10.0, l1_coeff=1e-3)
    pp_sae_k = k * 8
    pps_iia_d = eval_pi_plain_iia(pp_sae, model_lm, eval_pairs, hook_name, pp_sae_k, device)
    with torch.inference_mode():
        mu, _ = pp_sae.encode(act_t[:200])
        pp_sae_recon = F.mse_loss(pp_sae.decoder(mu), act_t[:200]).item()
        pp_sae_cls = (pp_sae.classifier(mu[:, :pp_sae_k]).argmax(-1) == lab_t[:200]).float().mean().item()
        pp_sae_l0 = (mu.abs() > 0.01).float().mean().item()
    _store_iia(results, "pi_plain_sae", pps_iia_d)
    results["pi_plain_sae_recon_mse"] = pp_sae_recon
    results["pi_plain_sae_cls_acc"] = pp_sae_cls
    results["pi_plain_sae_l0"] = pp_sae_l0
    log.info(f"[{utc_ts()}]     add={pps_iia_d['add_std']:.4f} repl={pps_iia_d['repl_std']:.4f}, recon={pp_sae_recon:.6f}, cls={pp_sae_cls:.4f}, L0={pp_sae_l0:.3f}")
    del pp_sae
    torch.cuda.empty_cache()

    # --- pi-VAE (structured, label-conditional prior) ---
    log.info(f"[{utc_ts()}]   pi-VAE k={k}")
    pi_vae = build_pi_vae(d_model, k, 16, hidden_dim, n_classes, device)
    pi_vae = train_pi_vae(pi_vae, act_t, lab_t, device, n_epochs=vae_epochs, alpha=10.0)
    pi_iia_d, pi_recon = eval_pi_vae_metrics(
        pi_vae, model_lm, eval_pairs, hook_name, act_t[:200], lab_t[:200], device)
    with torch.inference_mode():
        mu_c, _, _, _ = pi_vae.encode(act_t[:200])
        pi_cls = (pi_vae.classifier(mu_c).argmax(-1) == lab_t[:200]).float().mean().item()
    _store_iia(results, "pi_vae", pi_iia_d)
    results["pi_vae_recon_mse"] = pi_recon
    results["pi_vae_cls_acc"] = pi_cls
    log.info(f"[{utc_ts()}]     add={pi_iia_d['add_std']:.4f} repl={pi_iia_d['repl_std']:.4f}, recon={pi_recon:.6f}, cls={pi_cls:.4f}")
    del pi_vae
    torch.cuda.empty_cache()

    # --- Structured pi-SAE (sparse overcomplete + label-conditional prior + causal/nuisance split) ---
    log.info(f"[{utc_ts()}]   Structured pi-SAE k={k}")
    pi_sae = build_pi_sae(d_model, k, 16, hidden_dim, n_classes,
                          expansion_factor=8, device=device)
    pi_sae = train_pi_sae(pi_sae, act_t, lab_t, device, n_epochs=vae_epochs,
                          alpha=10.0, l1_coeff=1e-3)
    sae_iia_d, sae_recon = eval_pi_vae_metrics(
        pi_sae, model_lm, eval_pairs, hook_name, act_t[:200], lab_t[:200], device)
    with torch.inference_mode():
        mu_c, _, _, _ = pi_sae.encode(act_t[:200])
        sae_cls = (pi_sae.classifier(mu_c).argmax(-1) == lab_t[:200]).float().mean().item()
        sae_l0 = (mu_c.abs() > 0.01).float().mean().item()
    _store_iia(results, "pi_sae", sae_iia_d)
    results["pi_sae_recon_mse"] = sae_recon
    results["pi_sae_cls_acc"] = sae_cls
    results["pi_sae_l0"] = sae_l0
    log.info(f"[{utc_ts()}]     add={sae_iia_d['add_std']:.4f} repl={sae_iia_d['repl_std']:.4f}, recon={sae_recon:.6f}, cls={sae_cls:.4f}, L0={sae_l0:.3f}")
    del pi_sae
    torch.cuda.empty_cache()

    # --- Structured pi-SAE E2E (end-to-end intervention CE loss) ---
    log.info(f"[{utc_ts()}]   Structured pi-SAE E2E k={k}")
    pi_sae_e2e = build_pi_sae(d_model, k, 16, hidden_dim, n_classes,
                              expansion_factor=8, device=device)
    pi_sae_e2e = train_pi_sae_e2e(
        pi_sae_e2e, act_t, lab_t, train_pairs[:400], model_lm, hook_name,
        device, n_epochs=vae_epochs, alpha=10.0, l1_coeff=1e-3, beta=1.0,
    )
    e2e_iia_d, e2e_recon = eval_pi_vae_metrics(
        pi_sae_e2e, model_lm, eval_pairs, hook_name, act_t[:200], lab_t[:200], device)
    with torch.inference_mode():
        mu_c, _, _, _ = pi_sae_e2e.encode(act_t[:200])
        e2e_cls = (pi_sae_e2e.classifier(mu_c).argmax(-1) == lab_t[:200]).float().mean().item()
        e2e_l0 = (mu_c.abs() > 0.01).float().mean().item()
    _store_iia(results, "pi_sae_e2e", e2e_iia_d)
    results["pi_sae_e2e_recon_mse"] = e2e_recon
    results["pi_sae_e2e_cls_acc"] = e2e_cls
    results["pi_sae_e2e_l0"] = e2e_l0
    log.info(f"[{utc_ts()}]     add={e2e_iia_d['add_std']:.4f} repl={e2e_iia_d['repl_std']:.4f}, recon={e2e_recon:.6f}, cls={e2e_cls:.4f}, L0={e2e_l0:.3f}")
    del pi_sae_e2e
    torch.cuda.empty_cache()

    return results


# ===================================================================
# IOI
# ===================================================================


def run_ioi(device, log):
    log.info(f"[{utc_ts()}] === IOI ===")

    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    LAYER = 10
    D = 768
    hook_name = f"blocks.{LAYER}.hook_resid_post"

    valid_names = [n for n in IOI_NAMES if len(model.tokenizer.encode(n)) == 1]
    log.info(f"[{utc_ts()}] Single-token names: {len(valid_names)}")

    rng = random.Random(42)
    raw_pairs = []
    for _ in range(2000):
        t = rng.choice(IOI_TEMPLATES)
        a, b = rng.sample(valid_names, 2)
        base = t.format(A=a, B=b, PLACE=rng.choice(IOI_PLACES), OBJ=rng.choice(IOI_OBJECTS))
        source = t.format(A=b, B=a, PLACE=rng.choice(IOI_PLACES), OBJ=rng.choice(IOI_OBJECTS))
        raw_pairs.append((base, source, model.tokenizer.encode(a)[0], model.tokenizer.encode(b)[0]))

    log.info(f"[{utc_ts()}] Caching activations...")
    data = []
    for base_text, src_text, base_id, src_id in tqdm(raw_pairs, desc="caching"):
        bt = model.to_tokens(base_text)
        st = model.to_tokens(src_text)
        with torch.no_grad():
            _, bc = model.run_with_cache(bt, names_filter=hook_name)
            bl = model(bt)[0, -1]
            _, sc = model.run_with_cache(st, names_filter=hook_name)
            sl = model(st)[0, -1]

        bm = (bl[base_id] - bl[src_id]).item()
        sm = (sl[src_id] - sl[base_id]).item()
        if bm > 0 and sm > 0:
            data.append({
                "base_act": bc[hook_name][0, -1],
                "src_act": sc[hook_name][0, -1],
                "base_toks": bt,
                "src_label": src_id,
                "base_label": base_id,
                "margin": min(bm, sm),
            })

    log.info(f"[{utc_ts()}] Both-correct pairs: {len(data)}")

    rng2 = random.Random(42)
    indices = list(range(len(data)))
    rng2.shuffle(indices)
    n_train = int(0.7 * len(indices))
    train_pairs = [data[i] for i in indices[:n_train]]
    eval_pairs = [data[i] for i in indices[n_train:]]

    log.info(f"[{utc_ts()}] Train: {len(train_pairs)}, Eval: {len(eval_pairs)}")

    label_map = {}
    all_acts, all_labels = [], []
    for d in train_pairs:
        for rk, lk in [("base_act", "base_label"), ("src_act", "src_label")]:
            tid = d[lk]
            if tid not in label_map:
                label_map[tid] = len(label_map)
            all_acts.append(d[rk])
            all_labels.append(label_map[tid])

    act_t = torch.stack(all_acts)
    lab_t = torch.tensor(all_labels, device=device)
    n_classes = len(label_map)
    log.info(f"[{utc_ts()}] VAE data: {len(all_acts)} acts, {n_classes} classes")

    task_results = {"task": "ioi", "n_pairs": len(data), "n_classes": n_classes}
    for k in [1, 2, 4]:
        log.info(f"[{utc_ts()}] --- IOI k={k} ---")
        task_results[f"k{k}"] = run_all_variants(
            model, train_pairs, eval_pairs, act_t, lab_t,
            n_classes, D, k, hook_name, device, log,
            hidden_dim=256, vae_epochs=500, task="ioi",
        )

    del model
    torch.cuda.empty_cache()
    return task_results


# ===================================================================
# MIB tasks — loaded from Hanna et al. 2024 CSVs
# ===================================================================

MIB_DATA_DIR = "/mib_data"

MIB_TASKS = {
    "greater_than": {"csv": "greater-than/gpt2.csv", "layer": 8},
    "sva": {"csv": "sva/gpt2.csv", "layer": 8},
    "gender_bias": {"csv": "gender-bias/gpt2.csv", "layer": 8},
    "capitals": {"csv": "fact-retrieval-comma/gpt2.csv", "layer": 8},
    "hypernymy": {"csv": "hypernymy-comma/gpt2.csv", "layer": 8},
}


def run_mib_task(task_name, device, log, *, max_pairs=0, n_seeds=1,
                 k_values=(1, 2, 4), vae_epochs=500, layer_override=None,
                 seed_offset=0):
    import ast
    import pandas as pd
    from tqdm import tqdm

    task_cfg = MIB_TASKS[task_name]
    csv_path = f"{MIB_DATA_DIR}/{task_cfg['csv']}"
    log.info(f"[{utc_ts()}] === MIB: {task_name} (from {csv_path}) ===")

    df = pd.read_csv(csv_path)
    log.info(f"[{utc_ts()}] Loaded {len(df)} rows from CSV")

    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    LAYER = layer_override if layer_override is not None else task_cfg["layer"]
    D = 768
    hook_name = f"blocks.{LAYER}.hook_resid_post"

    pairs = []
    all_acts_list = []
    all_labels_list = []
    label_set = set()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"cache {task_name}"):
        clean_text = row["clean"]
        corrupted_text = row["corrupted"]

        clean_toks = model.to_tokens(clean_text)
        corr_toks = model.to_tokens(corrupted_text)
        if clean_toks.shape[1] != corr_toks.shape[1]:
            continue

        with torch.no_grad():
            _, clean_cache = model.run_with_cache(clean_toks, names_filter=hook_name)
            _, corr_cache = model.run_with_cache(corr_toks, names_filter=hook_name)

        clean_act = clean_cache[hook_name][0, -1]
        corr_act = corr_cache[hook_name][0, -1]

        if task_name == "sva":
            clean_label = int(row["plural"])
            corr_label = 1 - clean_label
        elif task_name == "greater_than":
            clean_label = int(row["correct_idx"])
            corr_label = 0
        elif task_name == "gender_bias":
            clean_label = int(row["clean_answer_idx"])
            corr_label = int(row["corrupted_answer_idx"])
        elif task_name == "capitals":
            clean_label = int(row["country_idx"])
            corr_label = int(row["corrupted_country_idx"])
        elif task_name == "hypernymy":
            clean_answers = ast.literal_eval(row["answers_idx"])
            corr_answers = ast.literal_eval(row["corrupted_answers_idx"])
            clean_label = clean_answers[0]
            corr_label = corr_answers[0]

        label_set.add(clean_label)
        label_set.add(corr_label)

        all_acts_list.append(clean_act)
        all_acts_list.append(corr_act)
        all_labels_list.append(clean_label)
        all_labels_list.append(corr_label)

        pairs.append({
            "base_act": clean_act,
            "src_act": corr_act,
            "base_toks": clean_toks,
            "src_label": corr_label,
            "base_label": clean_label,
        })

    log.info(f"[{utc_ts()}] Valid pairs: {len(pairs)}, unique labels: {len(label_set)}")

    label_remap = {old: new for new, old in enumerate(sorted(label_set))}
    n_classes = len(label_remap)
    act_t = torch.stack(all_acts_list)
    lab_t = torch.tensor([label_remap[l] for l in all_labels_list], device=device)

    n_pairs_orig = len(pairs)
    if max_pairs > 0 and len(pairs) > max_pairs:
        import random as _rand
        _rand.seed(42)
        pairs = _rand.sample(pairs, max_pairs)
        act_t = act_t[:max_pairs * 2]
        lab_t = lab_t[:max_pairs * 2]
        log.info(f"[{utc_ts()}] Subsampled to {max_pairs} pairs (from {n_pairs_orig})")

    task_results = {"task": task_name, "n_pairs": len(pairs),
                    "n_pairs_orig": n_pairs_orig, "n_classes": n_classes,
                    "layer": LAYER, "n_seeds": n_seeds, "max_pairs": max_pairs}

    for k in k_values:
        seed_runs = []
        for seed_idx in range(n_seeds):
            seed = seed_idx + seed_offset
            log.info(f"[{utc_ts()}] --- {task_name} k={k} seed={seed} ({seed_idx+1}/{n_seeds}) ---")
            rng = random.Random(seed)
            indices = list(range(len(pairs)))
            rng.shuffle(indices)
            n_train = int(0.7 * len(pairs))
            train_pairs = [pairs[i] for i in indices[:n_train]]
            eval_pairs = [pairs[i] for i in indices[n_train:]]

            run_result = run_all_variants(
                model, train_pairs, eval_pairs, act_t, lab_t,
                n_classes, D, k, hook_name, device, log,
                hidden_dim=256, vae_epochs=vae_epochs, task=task_name,
            )
            seed_runs.append(run_result)

        if n_seeds == 1:
            task_results[f"k{k}"] = seed_runs[0]
        else:
            import numpy as np
            agg = {"seeds": seed_runs}
            key_metrics = ["das_iia", "nl_das_iia", "vae_iia", "pi_vae_iia",
                           "pi_sae_iia", "pi_plain_vae_iia", "pi_plain_sae_iia",
                           "random_das_iia", "c1_random_labels_iia",
                           "c2_recon_only_iia", "c3_untrained_iia",
                           "c7_unconstrained_iia"]
            for m in key_metrics:
                vals = [r[m] for r in seed_runs if m in r]
                if vals:
                    agg[f"{m}_mean"] = float(np.mean(vals))
                    agg[f"{m}_std"] = float(np.std(vals))
            log.info(f"[{utc_ts()}] k={k} over {n_seeds} seeds:")
            for m in ["das_iia", "nl_das_iia", "pi_sae_iia"]:
                if f"{m}_mean" in agg:
                    log.info(f"  {m}: {agg[f'{m}_mean']:.3f} +/- {agg[f'{m}_std']:.3f}")
            task_results[f"k{k}"] = agg

    del model
    torch.cuda.empty_cache()
    return task_results


# ===================================================================
# Grokking
# ===================================================================


def compute_labels(a_vec, b_vec, operation, p):
    if operation == "addition":
        return (a_vec + b_vec) % p
    elif operation == "multiplication":
        return (a_vec * b_vec) % p
    elif operation == "squaring":
        return (a_vec * a_vec) % p
    elif operation == "quartic_sum":
        return (a_vec.pow(4) + b_vec.pow(4)) % p
    elif operation == "mixed_product":
        return (a_vec * b_vec * (a_vec + b_vec)) % p
    elif operation == "symmetric_power":
        return torch.tensor(
            [pow(int(a.item()), int(b.item()), p) + pow(int(b.item()), int(a.item()), p)
             for a, b in zip(a_vec, b_vec)]
        ).long() % p
    elif operation == "double_add_mult":
        return ((a_vec + b_vec).pow(2) + a_vec * b_vec) % p
    elif operation == "addition_mod35":
        return (a_vec + b_vec) % p
    elif operation == "addition_mod91":
        return (a_vec + b_vec) % p
    elif operation == "multiplication_mod91":
        return (a_vec * b_vec) % p
    elif operation == "addition_mod77":
        return (a_vec + b_vec) % p
    else:
        raise ValueError(operation)


def build_grok_data(operation, p, device):
    is_unary = operation == "squaring"
    excludes_zero = operation in ("multiplication", "symmetric_power", "multiplication_mod91")

    if is_unary:
        a_vec = torch.arange(p)
        b_vec = torch.zeros(p, dtype=torch.long)
    elif excludes_zero:
        a_vals = torch.arange(1, p)
        b_vals = torch.arange(1, p)
        a_vec = einops.repeat(a_vals, "i -> (i j)", j=len(b_vals))
        b_vec = einops.repeat(b_vals, "j -> (i j)", i=len(a_vals))
    else:
        a_vec = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
        b_vec = einops.repeat(torch.arange(p), "j -> (i j)", i=p)

    eq_vec = torch.full_like(a_vec, p)
    dataset = torch.stack([a_vec, b_vec, eq_vec], dim=1).to(device)
    labels = compute_labels(a_vec, b_vec, operation, p).to(device)

    torch.manual_seed(DATA_SEED)
    indices = torch.randperm(len(dataset))
    cutoff = int(len(dataset) * FRAC_TRAIN)
    return dataset, labels, indices[:cutoff], indices[cutoff:]


def run_grokking(operation, n_epochs, device, log, seed=999):
    log.info(f"[{utc_ts()}] === Grokking: {operation} (seed={seed}) ===")

    p = OP_MODULUS.get(operation, P)
    log.info(f"[{utc_ts()}] Using modulus p={p}")

    cfg = HookedTransformerConfig(
        n_layers=1, n_heads=4, d_model=128, d_head=32, d_mlp=512,
        act_fn="relu", normalization_type=None,
        d_vocab=p + 1, d_vocab_out=p, n_ctx=3,
        init_weights=True, device=device, seed=seed,
    )
    model = HookedTransformer(cfg)
    for name, param in model.named_parameters():
        if "b_" in name:
            param.requires_grad = False

    dataset, labels, train_idx, test_idx = build_grok_data(operation, p, device)
    train_data, train_labels = dataset[train_idx], labels[train_idx]
    test_data, test_labels = dataset[test_idx], labels[test_idx]

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
    for epoch in tqdm(range(n_epochs), desc=f"train {operation}"):
        logits = model(train_data)[:, -1]
        loss = F.cross_entropy(logits, train_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    with torch.inference_mode():
        test_logits = model(test_data)[:, -1]
        test_acc = (test_logits.argmax(-1) == test_labels).float().mean().item()
    grokked = test_acc > 0.95
    log.info(f"[{utc_ts()}] test_acc={test_acc:.4f}, grokked={grokked}")

    hook_name = "blocks.0.hook_resid_post"
    D = 128
    all_acts = []
    for i in range(0, len(dataset), 256):
        with torch.inference_mode():
            _, cache = model.run_with_cache(dataset[i:i + 256], names_filter=[hook_name])
        all_acts.append(cache[hook_name][:, -1, :].clone())
    activations = torch.cat(all_acts, dim=0)

    test_acts = activations[test_idx]
    test_labels_t = labels[test_idx]
    test_data_t = dataset[test_idx]

    n_test = len(test_idx)
    pairs = []
    for i in range(n_test):
        for j in range(i + 1, min(i + 50, n_test)):
            if test_labels_t[i] != test_labels_t[j]:
                pairs.append({
                    "base_act": test_acts[i],
                    "src_act": test_acts[j],
                    "base_toks": test_data_t[i:i + 1],
                    "base_label": test_labels_t[i].item(),
                    "src_label": test_labels_t[j].item(),
                })
                if len(pairs) >= 600:
                    break
        if len(pairs) >= 600:
            break

    n_train_p = int(0.7 * len(pairs))
    train_pairs = pairs[:n_train_p]
    eval_pairs = pairs[n_train_p:]

    log.info(f"[{utc_ts()}] Pairs: {len(train_pairs)} train, {len(eval_pairs)} eval")

    task_results = {"task": operation, "grokked": grokked, "test_accuracy": test_acc, "modulus": p}

    k_values = OP_K_VALUES.get(operation, [1, 2, 4])
    for k in k_values:
        log.info(f"[{utc_ts()}] --- {operation} k={k} ---")
        task_results[f"k{k}"] = run_all_variants(
            model, train_pairs, eval_pairs, activations, labels,
            p, D, k, hook_name, device, log,
            hidden_dim=128, vae_epochs=500, train_idx=train_idx,
            task=operation,
        )

    del model, activations
    torch.cuda.empty_cache()
    return task_results


# ===================================================================
# Modal function
# ===================================================================


GROK_EPOCHS = {
    "addition": 25000, "multiplication": 40000, "squaring": 60000,
    "quartic_sum": 60000, "mixed_product": 60000,
    "symmetric_power": 80000, "double_add_mult": 60000,
    "addition_mod35": 100000, "addition_mod91": 50000,
    "multiplication_mod91": 60000, "addition_mod77": 50000,
}

OP_MODULUS = {
    "addition_mod35": 35,
    "addition_mod91": 91,
    "multiplication_mod91": 91,
    "addition_mod77": 77,
}

OP_K_VALUES = {
    "addition_mod35": [1, 2, 3, 4, 6],
    "addition_mod91": [1, 2, 3, 4, 6, 8],
    "multiplication_mod91": [1, 2, 3, 4, 6, 8],
    "addition_mod77": [1, 2, 3, 4, 6, 8],
}


def _log_task_summary(task_results, task_name, log):
    log.info(f"\n{'=' * 130}")
    log.info(f"{'Task':>15s} {'k':>2s}  {'DAS':>6s}  {'NL-DAS':>6s}  "
             f"{'VAE':>6s}  {'piVAE':>6s}  {'piSAE':>6s}  "
             f"{'ppVAE':>6s}  {'ppSAE':>6s}  "
             f"{'C1:Rnd':>6s}  {'C7:Plain':>7s}  {'Rand':>6s}")
    log.info("-" * 120)
    k_values = OP_K_VALUES.get(task_name, [1, 2, 4])
    for k in k_values:
        kr = task_results.get(f"k{k}", {})
        log.info(
            f"{task_name:>15s} {k:2d}  "
            f"{kr.get('das_iia', 0):6.3f}  "
            f"{kr.get('nl_das_iia', 0):6.3f}  "
            f"{kr.get('vae_iia', 0):6.3f}  "
            f"{kr.get('pi_vae_iia', 0):6.3f}  "
            f"{kr.get('pi_sae_iia', 0):6.3f}  "
            f"{kr.get('pi_plain_vae_iia', 0):6.3f}  "
            f"{kr.get('pi_plain_sae_iia', 0):6.3f}  "
            f"{kr.get('c1_random_labels_iia', 0):6.3f}  "
            f"{kr.get('c7_unconstrained_iia', 0):7.3f}  "
            f"{kr.get('random_das_iia', 0):6.3f}"
        )


# ===================================================================
# Cross-task generalization
# ===================================================================


def load_mib_task_data(task_name, model_lm, device, max_pairs=1000, layer=8):
    """Load and cache activations for a MIB task. Returns (pairs, act_t, lab_t, n_classes, hook_name)."""
    import ast
    import pandas as pd

    task_cfg = MIB_TASKS[task_name]
    csv_path = f"{MIB_DATA_DIR}/{task_cfg['csv']}"
    df = pd.read_csv(csv_path)

    hook_name = f"blocks.{layer}.hook_resid_post"

    pairs = []
    all_acts_list = []
    all_labels_list = []
    label_set = set()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"cache {task_name}"):
        clean_toks = model_lm.to_tokens(row["clean"])
        corr_toks = model_lm.to_tokens(row["corrupted"])
        if clean_toks.shape[1] != corr_toks.shape[1]:
            continue

        with torch.no_grad():
            _, clean_cache = model_lm.run_with_cache(clean_toks, names_filter=hook_name)
            _, corr_cache = model_lm.run_with_cache(corr_toks, names_filter=hook_name)

        clean_act = clean_cache[hook_name][0, -1]
        corr_act = corr_cache[hook_name][0, -1]

        if task_name == "sva":
            clean_label = int(row["plural"])
            corr_label = 1 - clean_label
        elif task_name == "greater_than":
            clean_label = int(row["correct_idx"])
            corr_label = 0
        elif task_name == "gender_bias":
            clean_label = int(row["clean_answer_idx"])
            corr_label = int(row["corrupted_answer_idx"])
        elif task_name == "capitals":
            clean_label = int(row["country_idx"])
            corr_label = int(row["corrupted_country_idx"])
        elif task_name == "hypernymy":
            clean_answers = ast.literal_eval(row["answers_idx"])
            corr_answers = ast.literal_eval(row["corrupted_answers_idx"])
            clean_label = clean_answers[0]
            corr_label = corr_answers[0]

        label_set.add(clean_label)
        label_set.add(corr_label)
        all_acts_list.append(clean_act)
        all_acts_list.append(corr_act)
        all_labels_list.append(clean_label)
        all_labels_list.append(corr_label)

        pairs.append({
            "base_act": clean_act, "src_act": corr_act,
            "base_toks": clean_toks, "src_label": corr_label,
            "base_label": clean_label,
        })

    label_remap = {old: new for new, old in enumerate(sorted(label_set))}
    n_classes = len(label_remap)
    act_t = torch.stack(all_acts_list)
    lab_t = torch.tensor([label_remap[l] for l in all_labels_list], device=device)

    if max_pairs > 0 and len(pairs) > max_pairs:
        rng = random.Random(42)
        pairs = rng.sample(pairs, max_pairs)
        act_t = act_t[:max_pairs * 2]
        lab_t = lab_t[:max_pairs * 2]

    rng = random.Random(42)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)
    n_train = int(0.7 * len(pairs))
    train_pairs = [pairs[i] for i in indices[:n_train]]
    eval_pairs = [pairs[i] for i in indices[n_train:]]

    return train_pairs, eval_pairs, act_t, lab_t, n_classes, hook_name


def compute_subspace_overlap(models, task_names):
    """Compute principal angles between decoder causal subspaces across tasks.

    Returns dict with pairwise cosine similarities and principal angles.
    """
    import numpy as np

    decoder_weights = {}
    for name, model in zip(task_names, models):
        W = None
        for module in model.decoder:
            if isinstance(module, nn.Linear):
                W = module.weight.detach().cpu()
                break
        z_c_dim = model.z_causal_dim if hasattr(model, "z_causal_dim") else model.z_sparse_dim
        W_causal = W[:, :z_c_dim]
        U, _, _ = torch.linalg.svd(W_causal, full_matrices=False)
        decoder_weights[name] = U.numpy()

    results = {}
    for i, name_a in enumerate(task_names):
        for j, name_b in enumerate(task_names):
            if i >= j:
                continue
            U_a = decoder_weights[name_a]
            U_b = decoder_weights[name_b]
            k = min(U_a.shape[1], U_b.shape[1])
            cos_angles = np.linalg.svd(U_a[:, :k].T @ U_b[:, :k], compute_uv=False)
            cos_angles = np.clip(cos_angles, -1, 1)
            angles_deg = np.degrees(np.arccos(cos_angles))
            results[f"{name_a}_vs_{name_b}"] = {
                "principal_angles_deg": angles_deg.tolist(),
                "mean_cos_similarity": float(cos_angles.mean()),
                "max_cos_similarity": float(cos_angles.max()),
                "grassmann_distance": float(np.sqrt(np.sum(np.arccos(cos_angles) ** 2))),
            }
    return results


def compute_sparse_overlap(models, task_names, all_task_data):
    """Compute Jaccard overlap of active sparse features across tasks.

    For each task, encode its eval activations and find which causal features
    are active (|z_c| > threshold). Then compute pairwise Jaccard.
    """
    active_features = {}
    for name, model in zip(task_names, models):
        _, eval_pairs, act_t, _, _, _ = all_task_data[name]
        model.eval()
        with torch.inference_mode():
            mu_c, _, _, _ = model.encode(act_t[:500])
            mean_abs = mu_c.abs().mean(dim=0)
            active = set((mean_abs > 0.05).nonzero(as_tuple=True)[0].cpu().tolist())
        active_features[name] = active

    results = {}
    for i, name_a in enumerate(task_names):
        for j, name_b in enumerate(task_names):
            if i >= j:
                continue
            a, b = active_features[name_a], active_features[name_b]
            jaccard = len(a & b) / max(len(a | b), 1)
            results[f"{name_a}_vs_{name_b}"] = {
                "jaccard": jaccard,
                "shared_features": len(a & b),
                "only_a": len(a - b),
                "only_b": len(b - a),
                "total_a": len(a),
                "total_b": len(b),
            }
    return results


@app.function(gpu="A100", timeout=14400, volumes={"/results": results_vol})
def run_cross_task(tasks: str = "gender_bias,hypernymy,greater_than",
                   k: int = 4, layer: int = 8, max_pairs: int = 500,
                   vae_epochs: int = 500, use_e2e: bool = True) -> dict:
    """Train Structured pi-SAE on each task, then cross-evaluate."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger(__name__)
    DEVICE = "cuda"

    task_list = [t.strip() for t in tasks.split(",")]
    D = 768
    hook_name = f"blocks.{layer}.hook_resid_post"

    model_lm = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    model_lm.eval()
    for p in model_lm.parameters():
        p.requires_grad_(False)

    log.info(f"[{utc_ts()}] Loading data for {len(task_list)} tasks...")
    all_task_data = {}
    for task_name in task_list:
        log.info(f"[{utc_ts()}]   Loading {task_name}...")
        data = load_mib_task_data(task_name, model_lm, DEVICE,
                                 max_pairs=max_pairs, layer=layer)
        all_task_data[task_name] = data
        log.info(f"[{utc_ts()}]   {task_name}: {len(data[0])} train, {len(data[1])} eval, {data[4]} classes")

    log.info(f"[{utc_ts()}] Training models (k={k})...")
    trained_models = {}
    for task_name in task_list:
        train_pairs, eval_pairs, act_t, lab_t, n_classes, _ = all_task_data[task_name]
        log.info(f"[{utc_ts()}]   Training on {task_name} (n_classes={n_classes})...")

        sae = build_pi_sae(D, k, 16, 256, n_classes, expansion_factor=8, device=DEVICE)
        if use_e2e:
            sae = train_pi_sae_e2e(
                sae, act_t, lab_t, train_pairs[:400], model_lm, hook_name,
                DEVICE, n_epochs=vae_epochs, alpha=10.0, l1_coeff=1e-3, beta=1.0,
            )
        else:
            sae = train_pi_sae(sae, act_t, lab_t, DEVICE,
                               n_epochs=vae_epochs, alpha=10.0, l1_coeff=1e-3)
        trained_models[task_name] = sae

        same_iia_d, same_recon = eval_pi_vae_metrics(
            sae, model_lm, eval_pairs, hook_name, act_t[:200], lab_t[:200], DEVICE)
        log.info(f"[{utc_ts()}]   {task_name} same-task: add={same_iia_d['add_std']:.3f} repl={same_iia_d['repl_std']:.3f}")

    log.info(f"[{utc_ts()}] Cross-task evaluation...")
    cross_results = {}
    for train_task in task_list:
        sae = trained_models[train_task]
        for eval_task in task_list:
            _, eval_pairs, act_t, lab_t, _, _ = all_task_data[eval_task]
            iia_d, recon = eval_pi_vae_metrics(
                sae, model_lm, eval_pairs, hook_name, act_t[:200], lab_t[:200], DEVICE)
            key = f"train_{train_task}__eval_{eval_task}"
            cross_results[key] = {
                "add_std": iia_d["add_std"], "add_strict": iia_d["add_strict"],
                "repl_std": iia_d["repl_std"], "repl_strict": iia_d["repl_strict"],
                "recon_mse": recon,
            }
            same = "SAME" if train_task == eval_task else "CROSS"
            log.info(f"[{utc_ts()}]   [{same}] train={train_task} eval={eval_task}: "
                     f"add={iia_d['add_std']:.3f} repl={iia_d['repl_std']:.3f}")

    log.info(f"[{utc_ts()}] Computing subspace overlap...")
    subspace = compute_subspace_overlap(
        [trained_models[t] for t in task_list], task_list)

    log.info(f"[{utc_ts()}] Computing sparse feature overlap...")
    sparse = compute_sparse_overlap(
        [trained_models[t] for t in task_list], task_list, all_task_data)

    for pair_key, metrics in subspace.items():
        log.info(f"[{utc_ts()}]   Subspace {pair_key}: "
                 f"mean_cos={metrics['mean_cos_similarity']:.3f} "
                 f"grassmann_dist={metrics['grassmann_distance']:.3f}")
    for pair_key, metrics in sparse.items():
        log.info(f"[{utc_ts()}]   Sparse {pair_key}: "
                 f"jaccard={metrics['jaccard']:.3f} "
                 f"shared={metrics['shared_features']} "
                 f"a_only={metrics['only_a']} b_only={metrics['only_b']}")

    result = {
        "tasks": task_list, "k": k, "layer": layer, "max_pairs": max_pairs,
        "use_e2e": use_e2e, "vae_epochs": vae_epochs,
        "cross_task_iia": cross_results,
        "subspace_overlap": subspace,
        "sparse_feature_overlap": sparse,
        "timestamp": utc_ts(),
    }

    save_dir = "/results/grassmannian_atlas/cross_task"
    os.makedirs(save_dir, exist_ok=True)
    out_path = f"{save_dir}/results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    results_vol.commit()
    log.info(f"[{utc_ts()}] Saved to {out_path}")

    return result


# ===================================================================
# Feature analysis: max-activating, Fourier alignment, ablation
# ===================================================================


def analyze_max_activating(sae, acts, texts, k_top=10):
    """For each causal feature, find the top-k activating examples."""
    sae.eval()
    with torch.inference_mode():
        mu_c, _, _, _ = sae.encode(acts)

    n_features = mu_c.shape[1]
    results = {}
    for f_idx in range(n_features):
        col = mu_c[:, f_idx]
        mean_act = col.abs().mean().item()
        if mean_act < 0.01:
            continue
        topk_vals, topk_idx = col.abs().topk(min(k_top, len(col)))
        examples = []
        for rank, (val, idx) in enumerate(zip(topk_vals, topk_idx)):
            entry = {"rank": rank, "activation": col[idx.item()].item(),
                     "abs_activation": val.item(), "example_idx": idx.item()}
            if texts is not None and idx.item() < len(texts):
                entry["text"] = texts[idx.item()]
            examples.append(entry)
        results[f"feature_{f_idx}"] = {
            "mean_abs_activation": mean_act,
            "max_abs_activation": col.abs().max().item(),
            "frac_active": (col.abs() > 0.01).float().mean().item(),
            "top_examples": examples,
        }
    return results


def analyze_fourier_alignment(sae, dataset, labels, activations, p, device):
    """Correlate each causal feature with Fourier components sin(2*pi*k*a/p), cos(2*pi*k*a/p)."""
    import numpy as np
    sae.eval()
    with torch.inference_mode():
        mu_c, _, _, _ = sae.encode(activations)

    mu_np = mu_c.cpu().numpy()
    a_vals = dataset[:, 0].cpu().numpy().astype(float)
    b_vals = dataset[:, 1].cpu().numpy().astype(float)

    n_features = mu_np.shape[1]
    max_freq = min(p // 2, 20)
    results = {}

    for f_idx in range(n_features):
        col = mu_np[:, f_idx]
        if np.abs(col).mean() < 0.01:
            continue

        best_corr = 0.0
        best_freq = -1
        best_component = ""
        correlations = []

        for freq in range(1, max_freq + 1):
            for var_name, var_vals in [("a", a_vals), ("b", b_vals), ("a+b", a_vals + b_vals)]:
                sin_comp = np.sin(2 * np.pi * freq * var_vals / p)
                cos_comp = np.cos(2 * np.pi * freq * var_vals / p)
                for comp_name, comp in [("sin", sin_comp), ("cos", cos_comp)]:
                    corr = np.abs(np.corrcoef(col, comp)[0, 1])
                    if np.isnan(corr):
                        corr = 0.0
                    correlations.append({
                        "freq": freq, "variable": var_name,
                        "component": comp_name, "correlation": float(corr),
                    })
                    if corr > best_corr:
                        best_corr = corr
                        best_freq = freq
                        best_component = f"{comp_name}(2π·{freq}·{var_name}/{p})"

        top5 = sorted(correlations, key=lambda x: x["correlation"], reverse=True)[:5]
        results[f"feature_{f_idx}"] = {
            "best_correlation": best_corr,
            "best_frequency": best_freq,
            "best_component": best_component,
            "top5_correlations": top5,
            "mean_abs_activation": float(np.abs(col).mean()),
        }

    return results


def analyze_feature_ablation(sae, model_lm, eval_pairs, hook_name, device):
    """Zero out one causal feature at a time, measure IIA drop."""
    sae.eval()
    n_features = sae.z_sparse_dim if hasattr(sae, 'z_sparse_dim') else sae.z_causal_dim

    baseline_counts = {"std": 0, "strict": 0}
    with torch.inference_mode():
        for d in eval_pairs:
            base_act = d["base_act"].unsqueeze(0)
            src_act = d["src_act"].unsqueeze(0)
            mu_c_b, _, mu_n_b, _ = sae.encode(base_act)
            mu_c_s, _, _, _ = sae.encode(src_act)
            z_base = torch.cat([mu_c_b, mu_n_b], dim=-1)
            z_iv = torch.cat([mu_c_s, mu_n_b], dim=-1)
            h_recon = sae.decoder(z_base).squeeze(0)
            h_iv = sae.decoder(z_iv).squeeze(0)
            hook_fn = _make_hook_additive(d["base_act"], h_recon, h_iv)
            logits = model_lm.run_with_hooks(
                d["base_toks"], fwd_hooks=[(hook_name, hook_fn)]
            )[0, -1, :]
            s, st = _iia_from_logits(logits, d)
            baseline_counts["std"] += s
            baseline_counts["strict"] += st

    n = len(eval_pairs)
    baseline_iia = baseline_counts["std"] / max(n, 1)

    ablation_results = {"baseline_iia": baseline_iia, "n_eval": n, "features": {}}

    for f_idx in range(n_features):
        counts = {"std": 0, "strict": 0}
        with torch.inference_mode():
            for d in eval_pairs:
                base_act = d["base_act"].unsqueeze(0)
                src_act = d["src_act"].unsqueeze(0)
                mu_c_b, _, mu_n_b, _ = sae.encode(base_act)
                mu_c_s, _, _, _ = sae.encode(src_act)
                mu_c_s_ablated = mu_c_s.clone()
                mu_c_s_ablated[0, f_idx] = mu_c_b[0, f_idx]
                z_base = torch.cat([mu_c_b, mu_n_b], dim=-1)
                z_iv = torch.cat([mu_c_s_ablated, mu_n_b], dim=-1)
                h_recon = sae.decoder(z_base).squeeze(0)
                h_iv = sae.decoder(z_iv).squeeze(0)
                hook_fn = _make_hook_additive(d["base_act"], h_recon, h_iv)
                logits = model_lm.run_with_hooks(
                    d["base_toks"], fwd_hooks=[(hook_name, hook_fn)]
                )[0, -1, :]
                s, st = _iia_from_logits(logits, d)
                counts["std"] += s
                counts["strict"] += st

        ablated_iia = counts["std"] / max(n, 1)
        iia_drop = baseline_iia - ablated_iia
        ablation_results["features"][f"feature_{f_idx}"] = {
            "ablated_iia": ablated_iia,
            "iia_drop": iia_drop,
            "importance_rank": 0,
        }

    drops = [(k, v["iia_drop"]) for k, v in ablation_results["features"].items()]
    drops.sort(key=lambda x: x[1], reverse=True)
    for rank, (k, _) in enumerate(drops):
        ablation_results["features"][k]["importance_rank"] = rank

    return ablation_results


@app.function(gpu="A100", timeout=14400, volumes={"/results": results_vol})
def run_feature_analysis(tasks: str = "gender_bias,addition",
                         k: int = 4, layer: int = 8, max_pairs: int = 500,
                         vae_epochs: int = 500) -> dict:
    """Train Structured pi-SAE on each task, then analyze learned features."""
    import numpy as np
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger(__name__)
    DEVICE = "cuda"

    task_list = [t.strip() for t in tasks.split(",")]
    grok_tasks = [t for t in task_list if t in GROK_EPOCHS]
    mib_tasks = [t for t in task_list if t in MIB_TASKS]
    D_grok = 128
    D_mib = 768

    all_results = {"tasks": task_list, "k": k, "timestamp": utc_ts()}

    if mib_tasks:
        model_lm = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
        model_lm.eval()
        for p in model_lm.parameters():
            p.requires_grad_(False)

    for task_name in task_list:
        log.info(f"[{utc_ts()}] === Feature analysis: {task_name} ===")
        task_result = {}

        if task_name in MIB_TASKS:
            import pandas as pd
            hook_name = f"blocks.{layer}.hook_resid_post"
            train_pairs, eval_pairs, act_t, lab_t, n_classes, _ = load_mib_task_data(
                task_name, model_lm, DEVICE, max_pairs=max_pairs, layer=layer)

            df = pd.read_csv(f"{MIB_DATA_DIR}/{MIB_TASKS[task_name]['csv']}")
            texts = df["clean"].tolist()[:len(act_t)]

            log.info(f"[{utc_ts()}]   Training Structured pi-SAE (k={k})...")
            sae = build_pi_sae(D_mib, k, 16, 256, n_classes,
                               expansion_factor=8, device=DEVICE)
            sae = train_pi_sae(sae, act_t, lab_t, DEVICE,
                               n_epochs=vae_epochs, alpha=10.0, l1_coeff=1e-3)

            log.info(f"[{utc_ts()}]   Max-activating examples...")
            task_result["max_activating"] = analyze_max_activating(sae, act_t, texts)
            n_active = len(task_result["max_activating"])
            log.info(f"[{utc_ts()}]     {n_active} active features")

            log.info(f"[{utc_ts()}]   Feature ablation ({len(eval_pairs)} pairs)...")
            task_result["ablation"] = analyze_feature_ablation(
                sae, model_lm, eval_pairs[:200], hook_name, DEVICE)
            log.info(f"[{utc_ts()}]     baseline IIA={task_result['ablation']['baseline_iia']:.3f}")
            top3 = sorted(task_result["ablation"]["features"].items(),
                          key=lambda x: x[1]["iia_drop"], reverse=True)[:3]
            for fname, fdata in top3:
                log.info(f"[{utc_ts()}]     {fname}: drop={fdata['iia_drop']:.3f}")

        elif task_name in GROK_EPOCHS:
            p = OP_MODULUS.get(task_name, P)
            hook_name = "blocks.0.hook_resid_post"

            cfg = HookedTransformerConfig(
                n_layers=1, n_heads=4, d_model=128, d_head=32, d_mlp=512,
                act_fn="relu", normalization_type=None,
                d_vocab=p + 1, d_vocab_out=p, n_ctx=3,
                init_weights=True, device=DEVICE, seed=999,
            )
            grok_model = HookedTransformer(cfg)
            for name, param in grok_model.named_parameters():
                if "b_" in name:
                    param.requires_grad = False

            dataset, labels, train_idx, test_idx = build_grok_data(task_name, p, DEVICE)
            train_data, train_labels = dataset[train_idx], labels[train_idx]

            log.info(f"[{utc_ts()}]   Training grokking model ({GROK_EPOCHS[task_name]} epochs)...")
            optimizer = torch.optim.AdamW(grok_model.parameters(), lr=1e-3,
                                          weight_decay=1.0, betas=(0.9, 0.98))
            for epoch in tqdm(range(GROK_EPOCHS[task_name]), desc=f"grok {task_name}"):
                logits = grok_model(train_data)[:, -1]
                loss = F.cross_entropy(logits, train_labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            grok_model.eval()
            with torch.inference_mode():
                test_logits = grok_model(dataset[test_idx])[:, -1]
                test_acc = (test_logits.argmax(-1) == labels[test_idx]).float().mean().item()
            log.info(f"[{utc_ts()}]   test_acc={test_acc:.4f}")

            all_acts = []
            for i in range(0, len(dataset), 256):
                with torch.inference_mode():
                    _, cache = grok_model.run_with_cache(
                        dataset[i:i + 256], names_filter=[hook_name])
                all_acts.append(cache[hook_name][:, -1, :].clone())
            activations = torch.cat(all_acts, dim=0)

            test_acts = activations[test_idx]
            test_labels_t = labels[test_idx]
            test_data_t = dataset[test_idx]
            n_test = len(test_idx)
            pairs = []
            for i in range(n_test):
                for j in range(i + 1, min(i + 50, n_test)):
                    if test_labels_t[i] != test_labels_t[j]:
                        pairs.append({
                            "base_act": test_acts[i], "src_act": test_acts[j],
                            "base_toks": test_data_t[i:i + 1],
                            "src_label": test_labels_t[j].item(),
                            "base_label": test_labels_t[i].item(),
                        })
                        if len(pairs) >= 600:
                            break
                if len(pairs) >= 600:
                    break
            n_train_p = int(0.7 * len(pairs))
            train_pairs = pairs[:n_train_p]
            eval_pairs = pairs[n_train_p:]

            log.info(f"[{utc_ts()}]   Training Structured pi-SAE (k={k})...")
            sae = build_pi_sae(D_grok, k, 16, 128, p,
                               expansion_factor=8, device=DEVICE)
            sae = train_pi_sae(sae, activations, labels, DEVICE,
                               n_epochs=vae_epochs, alpha=10.0, l1_coeff=1e-3)

            log.info(f"[{utc_ts()}]   Max-activating examples...")
            input_texts = [f"a={dataset[i,0].item()}, b={dataset[i,1].item()}" for i in range(len(dataset))]
            task_result["max_activating"] = analyze_max_activating(sae, activations, input_texts)
            n_active = len(task_result["max_activating"])
            log.info(f"[{utc_ts()}]     {n_active} active features")

            log.info(f"[{utc_ts()}]   Fourier alignment...")
            task_result["fourier"] = analyze_fourier_alignment(
                sae, dataset, labels, activations, p, DEVICE)
            aligned = sum(1 for v in task_result["fourier"].values()
                          if v["best_correlation"] > 0.5)
            log.info(f"[{utc_ts()}]     {aligned}/{len(task_result['fourier'])} features with Fourier corr > 0.5")
            for fname, fdata in sorted(task_result["fourier"].items(),
                                        key=lambda x: x[1]["best_correlation"], reverse=True)[:5]:
                log.info(f"[{utc_ts()}]     {fname}: r={fdata['best_correlation']:.3f} -> {fdata['best_component']}")

            log.info(f"[{utc_ts()}]   Feature ablation ({len(eval_pairs)} pairs)...")
            task_result["ablation"] = analyze_feature_ablation(
                sae, grok_model, eval_pairs[:200], hook_name, DEVICE)
            log.info(f"[{utc_ts()}]     baseline IIA={task_result['ablation']['baseline_iia']:.3f}")
            top3 = sorted(task_result["ablation"]["features"].items(),
                          key=lambda x: x[1]["iia_drop"], reverse=True)[:3]
            for fname, fdata in top3:
                log.info(f"[{utc_ts()}]     {fname}: drop={fdata['iia_drop']:.3f}")

            task_result["test_accuracy"] = test_acc
            del grok_model
            torch.cuda.empty_cache()

        all_results[task_name] = task_result

    save_dir = "/results/grassmannian_atlas/feature_analysis"
    os.makedirs(save_dir, exist_ok=True)
    out_path = f"{save_dir}/results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    results_vol.commit()
    log.info(f"[{utc_ts()}] Saved to {out_path}")
    return all_results


@app.function(gpu="A100", timeout=14400, volumes={"/results": results_vol})
def run_multi_seed(operation: str, n_seeds: int = 10) -> dict:
    """Run grokking with multiple seeds to test stochastic stability."""
    import numpy as np
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger(__name__)
    DEVICE = "cuda"

    seeds = list(range(1, n_seeds + 1))
    n_epochs = GROK_EPOCHS.get(operation, 60000)
    results = {"operation": operation, "n_seeds": n_seeds, "seeds": seeds,
               "n_epochs": n_epochs, "timestamp": utc_ts()}

    for seed in seeds:
        log.info(f"[{utc_ts()}] === {operation} seed={seed} ({seeds.index(seed)+1}/{n_seeds}) ===")
        try:
            seed_result = run_grokking(operation, n_epochs, DEVICE, log, seed=seed)
            results[f"seed_{seed}"] = seed_result
            grokked = seed_result.get("grokked", False)
            test_acc = seed_result.get("test_accuracy", 0)
            log.info(f"[{utc_ts()}]   seed={seed}: grokked={grokked}, test_acc={test_acc:.4f}")
        except Exception as e:
            log.error(f"[{utc_ts()}]   seed={seed} failed: {e}")
            results[f"seed_{seed}"] = {"error": str(e)}
        torch.cuda.empty_cache()

    grok_count = sum(1 for s in seeds
                     if results.get(f"seed_{s}", {}).get("grokked", False))
    results["summary"] = {
        "grok_rate": grok_count / n_seeds,
        "grokked_count": grok_count,
        "total_seeds": n_seeds,
    }
    log.info(f"[{utc_ts()}] === Summary: {grok_count}/{n_seeds} grokked ({grok_count/n_seeds:.0%}) ===")

    save_dir = f"/results/grassmannian_atlas/multi_seed/{operation}"
    os.makedirs(save_dir, exist_ok=True)
    out_path = f"{save_dir}/results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    results_vol.commit()
    log.info(f"[{utc_ts()}] Saved to {out_path}")
    return results


@app.function(gpu="A100", timeout=7200, volumes={"/results": results_vol})
def run_s1_visualization(operation: str = "addition", n_epochs: int = 25000) -> dict:
    """Save PCA projections of grokked model activations to show S¹ structure."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger(__name__)
    DEVICE = "cuda"

    p = OP_MODULUS.get(operation, P)
    cfg = HookedTransformerConfig(
        n_layers=1, n_heads=4, d_model=128, d_head=32, d_mlp=512,
        act_fn="relu", normalization_type=None,
        d_vocab=p + 1, d_vocab_out=p, n_ctx=3,
        init_weights=True, device=DEVICE, seed=999,
    )
    model = HookedTransformer(cfg)
    for name, param in model.named_parameters():
        if "b_" in name:
            param.requires_grad = False

    dataset, labels, train_idx, test_idx = build_grok_data(operation, p, DEVICE)
    train_data, train_labels = dataset[train_idx], labels[train_idx]

    log.info(f"[{utc_ts()}] Training {operation} for {n_epochs} epochs...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
    for epoch in tqdm(range(n_epochs), desc=f"train {operation}"):
        logits = model(train_data)[:, -1]
        loss = F.cross_entropy(logits, train_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    hook_name = "blocks.0.hook_resid_post"
    all_acts = []
    for i in range(0, len(dataset), 256):
        with torch.inference_mode():
            _, cache = model.run_with_cache(dataset[i:i + 256], names_filter=[hook_name])
        all_acts.append(cache[hook_name][:, -1, :].clone())
    activations = torch.cat(all_acts, dim=0).cpu().numpy()

    a_vals = dataset[:, 0].cpu().numpy()
    b_vals = dataset[:, 1].cpu().numpy()
    lab_np = labels.cpu().numpy()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=6)
    proj = pca.fit_transform(activations)

    save_dir = "/results/grassmannian_atlas/s1_visualization"
    os.makedirs(save_dir, exist_ok=True)

    np.save(f"{save_dir}/activations.npy", activations)
    np.save(f"{save_dir}/pca_projection.npy", proj)
    np.save(f"{save_dir}/labels.npy", lab_np)
    np.save(f"{save_dir}/a_vals.npy", a_vals)
    np.save(f"{save_dir}/b_vals.npy", b_vals)
    np.save(f"{save_dir}/pca_explained_variance.npy", pca.explained_variance_ratio_)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sc = axes[0].scatter(proj[:, 0], proj[:, 1], c=lab_np, cmap="hsv", s=1, alpha=0.5)
    axes[0].set_title(f"PCA 1-2 (colored by (a+b) mod {p})")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.colorbar(sc, ax=axes[0])

    sc = axes[1].scatter(proj[:, 2], proj[:, 3], c=lab_np, cmap="hsv", s=1, alpha=0.5)
    axes[1].set_title("PCA 3-4")
    axes[1].set_xlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
    axes[1].set_ylabel(f"PC4 ({pca.explained_variance_ratio_[3]:.1%})")
    plt.colorbar(sc, ax=axes[1])

    test_mask = np.zeros(len(dataset), dtype=bool)
    test_mask[test_idx.cpu().numpy()] = True
    fix_b = b_vals == 0
    mask = test_mask & fix_b
    if mask.sum() > 5:
        sc = axes[2].scatter(proj[mask, 0], proj[mask, 1], c=a_vals[mask], cmap="hsv", s=20)
        axes[2].set_title(f"b=0 only: a colored by value (circle = S¹)")
        axes[2].set_xlabel("PC1")
        axes[2].set_ylabel("PC2")
        plt.colorbar(sc, ax=axes[2])
    else:
        axes[2].text(0.5, 0.5, "Not enough b=0 points", transform=axes[2].transAxes, ha="center")

    plt.tight_layout()
    fig_path = f"{save_dir}/pca_circle_structure.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    log.info(f"[{utc_ts()}] Saved PCA plot to {fig_path}")
    log.info(f"[{utc_ts()}] Explained variance: {pca.explained_variance_ratio_[:6]}")

    results_vol.commit()
    return {"explained_variance": pca.explained_variance_ratio_.tolist(),
            "n_points": len(activations), "operation": operation}


def measure_identifiability(model_variant, acts, labels, n_classes, device):
    """Measure iVAE identifiability metrics for a trained model.

    Returns dict with:
      - max_spearman: max |Spearman ρ| between any z_causal component and true label
      - mean_spearman: mean |Spearman ρ| across components
      - rank_condition: whether per-label prior means satisfy the rank condition
      - decoder_injectivity: ratio of decoder output distances to z distances
      - mcc: mean correlation coefficient (standard disentanglement metric)
    """
    import numpy as np
    from scipy.stats import spearmanr

    model_variant.eval()
    with torch.inference_mode():
        if hasattr(model_variant, 'encode'):
            enc_out = model_variant.encode(acts)
            mu_c = enc_out[0]
        else:
            return {"error": "model has no encode method"}

    z = mu_c.cpu().numpy()
    y = labels.cpu().numpy()
    n_components = z.shape[1]

    spearman_per_dim = []
    for j in range(n_components):
        rho, _ = spearmanr(z[:, j], y)
        spearman_per_dim.append(abs(rho) if not np.isnan(rho) else 0.0)

    corr_matrix = np.zeros((n_components, n_classes))
    for c in range(n_classes):
        mask = y == c
        if mask.sum() < 2:
            continue
        for j in range(n_components):
            corr_matrix[j, c] = z[mask, j].mean()

    _, s, _ = np.linalg.svd(corr_matrix, full_matrices=False)
    rank_k = min(n_components, n_classes)
    effective_rank = (s > s[0] * 1e-5).sum()

    rank_condition_met = effective_rank >= n_components

    has_prior = hasattr(model_variant, 'prior_mu')
    rank_condition_prior = False
    prior_rank = 0
    if has_prior:
        prior_means = model_variant.prior_mu.weight.detach().cpu().numpy()
        n_labels = prior_means.shape[0]
        if n_labels > n_components:
            diffs = prior_means[1:n_components + 1] - prior_means[0:1]
            _, s_prior, _ = np.linalg.svd(diffs, full_matrices=False)
            prior_rank = int((s_prior > s_prior[0] * 1e-5).sum())
            rank_condition_prior = prior_rank >= n_components

    decoder_injectivity = 0.0
    with torch.inference_mode():
        z_sample = mu_c[:200]
        n_pairs = min(500, len(z_sample) * (len(z_sample) - 1) // 2)
        idx_a = torch.randint(0, len(z_sample), (n_pairs,))
        idx_b = torch.randint(0, len(z_sample), (n_pairs,))
        z_a, z_b = z_sample[idx_a], z_sample[idx_b]
        z_nuisance = torch.zeros(n_pairs, 16, device=device)
        za_full = torch.cat([z_a, z_nuisance], dim=-1)
        zb_full = torch.cat([z_b, z_nuisance], dim=-1)
        dec_a = model_variant.decoder(za_full)
        dec_b = model_variant.decoder(zb_full)
        z_dist = (z_a - z_b).norm(dim=-1)
        dec_dist = (dec_a - dec_b).norm(dim=-1)
        valid = z_dist > 1e-6
        if valid.sum() > 10:
            ratios = dec_dist[valid] / z_dist[valid]
            decoder_injectivity = ratios.mean().item()

    abs_corr = np.abs(corr_matrix)
    row_max = abs_corr.max(axis=1)
    mcc = float(np.mean(row_max))

    return {
        "max_spearman": float(max(spearman_per_dim)),
        "mean_spearman": float(np.mean(spearman_per_dim)),
        "spearman_per_dim": [round(x, 4) for x in spearman_per_dim[:16]],
        "rank_condition_empirical": bool(rank_condition_met),
        "effective_rank": int(effective_rank),
        "has_label_conditional_prior": has_prior,
        "rank_condition_prior": bool(rank_condition_prior),
        "prior_rank": int(prior_rank) if has_prior else None,
        "decoder_injectivity_ratio": round(decoder_injectivity, 4),
        "mcc": round(mcc, 4),
        "n_components": n_components,
    }


@app.function(gpu="A100", timeout=14400, volumes={"/results": results_vol})
def run_ivae_verification(operations: str = "addition,multiplication",
                          k: int = 2, vae_epochs: int = 500) -> dict:
    """Verify iVAE identifiability conditions empirically across all 4 ablation variants."""
    import torch
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger(__name__)
    DEVICE = "cuda"
    t0 = time.time()

    op_list = [o.strip() for o in operations.split(",")]
    all_results = {}

    for operation in op_list:
        log.info(f"[{utc_ts()}] === iVAE verification: {operation} ===")
        p = OP_MODULUS.get(operation, P)
        n_epochs = GROK_EPOCHS.get(operation, 25000)

        cfg = HookedTransformerConfig(
            n_layers=1, n_heads=4, d_model=128, d_head=32, d_mlp=512,
            act_fn="relu", normalization_type=None,
            d_vocab=p + 1, d_vocab_out=p, n_ctx=3,
            init_weights=True, device=DEVICE, seed=999,
        )
        model = HookedTransformer(cfg)
        for name, param in model.named_parameters():
            if "b_" in name:
                param.requires_grad = False

        dataset, labels, train_idx, test_idx = build_grok_data(operation, p, DEVICE)
        train_data, train_labels = dataset[train_idx], labels[train_idx]
        test_data, test_labels = dataset[test_idx], labels[test_idx]

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                       weight_decay=1.0, betas=(0.9, 0.98))
        for epoch in tqdm(range(n_epochs), desc=f"train {operation}"):
            logits = model(train_data)[:, -1]
            loss = F.cross_entropy(logits, train_labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.inference_mode():
            test_logits = model(test_data)[:, -1]
            test_acc = (test_logits.argmax(-1) == test_labels).float().mean().item()
        grokked = test_acc > 0.95
        log.info(f"[{utc_ts()}] test_acc={test_acc:.4f}, grokked={grokked}")

        hook_name = "blocks.0.hook_resid_post"
        D = 128
        all_acts = []
        for i in range(0, len(dataset), 256):
            with torch.inference_mode():
                _, cache = model.run_with_cache(dataset[i:i + 256],
                                                 names_filter=[hook_name])
            all_acts.append(cache[hook_name][:, -1, :].clone())
        activations = torch.cat(all_acts, dim=0)

        op_results = {"grokked": grokked, "test_accuracy": test_acc, "modulus": p}

        # Train all 4 ablation variants and measure identifiability
        hidden_dim = 128
        n_classes = p

        # 1. Plain VAE (no label prior, no sparsity)
        log.info(f"[{utc_ts()}]   Plain VAE")
        plain_vae = build_vae(D, k, 16, hidden_dim, n_classes, DEVICE)
        plain_vae = train_vae(plain_vae, activations, labels, DEVICE,
                              n_epochs=vae_epochs, alpha=10.0)
        op_results["plain_vae"] = measure_identifiability(
            plain_vae, activations, labels, n_classes, DEVICE)
        log.info(f"[{utc_ts()}]     max_spearman={op_results['plain_vae']['max_spearman']:.4f}")
        del plain_vae
        torch.cuda.empty_cache()

        # 2. Plain SAE (no label prior, overcomplete + L1)
        log.info(f"[{utc_ts()}]   Plain SAE")
        pp_sae = build_pi_plain_sae(D, k + 16, hidden_dim, n_classes, k,
                                     expansion_factor=8, device=DEVICE)
        pp_sae = train_pi_plain_sae(pp_sae, activations, labels, DEVICE,
                                     n_epochs=vae_epochs, alpha=10.0, l1_coeff=1e-3)
        op_results["plain_sae"] = measure_identifiability(
            pp_sae, activations, labels, n_classes, DEVICE)
        log.info(f"[{utc_ts()}]     max_spearman={op_results['plain_sae']['max_spearman']:.4f}")
        del pp_sae
        torch.cuda.empty_cache()

        # 3. pi-VAE (label prior, no sparsity)
        log.info(f"[{utc_ts()}]   pi-VAE")
        pi_vae = build_pi_vae(D, k, 16, hidden_dim, n_classes, DEVICE)
        pi_vae = train_pi_vae(pi_vae, activations, labels, DEVICE,
                               n_epochs=vae_epochs, alpha=10.0)
        op_results["pi_vae"] = measure_identifiability(
            pi_vae, activations, labels, n_classes, DEVICE)
        log.info(f"[{utc_ts()}]     max_spearman={op_results['pi_vae']['max_spearman']:.4f}")
        del pi_vae
        torch.cuda.empty_cache()

        # 4. Structured pi-SAE (label prior + sparsity + causal/nuisance split)
        log.info(f"[{utc_ts()}]   Structured pi-SAE")
        pi_sae = build_pi_sae(D, k, 16, hidden_dim, n_classes,
                               expansion_factor=8, device=DEVICE)
        pi_sae = train_pi_sae(pi_sae, activations, labels, DEVICE,
                               n_epochs=vae_epochs, alpha=10.0, l1_coeff=1e-3)
        op_results["pi_sae"] = measure_identifiability(
            pi_sae, activations, labels, n_classes, DEVICE)
        log.info(f"[{utc_ts()}]     max_spearman={op_results['pi_sae']['max_spearman']:.4f}")
        del pi_sae
        torch.cuda.empty_cache()

        # Summary table
        log.info(f"\n{'=' * 80}")
        log.info(f"iVAE Identifiability: {operation} (grokked={grokked})")
        log.info(f"{'Variant':>20s} {'max_ρ':>8s} {'mean_ρ':>8s} {'rank':>6s} "
                 f"{'prior_rank':>10s} {'dec_inj':>8s} {'MCC':>6s}")
        log.info(f"{'-' * 80}")
        for variant in ["plain_vae", "plain_sae", "pi_vae", "pi_sae"]:
            v = op_results[variant]
            pr = str(v.get('prior_rank', '-'))
            log.info(f"{variant:>20s} {v['max_spearman']:>8.4f} {v['mean_spearman']:>8.4f} "
                     f"{v['effective_rank']:>6d} {pr:>10s} "
                     f"{v['decoder_injectivity_ratio']:>8.4f} {v['mcc']:>6.4f}")
        log.info(f"{'=' * 80}\n")

        all_results[operation] = op_results
        del model, activations
        torch.cuda.empty_cache()

    save_dir = "/results/grassmannian_atlas/ivae_verification"
    os.makedirs(save_dir, exist_ok=True)
    out_path = f"{save_dir}/results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    results_vol.commit()
    log.info(f"[{utc_ts()}] Saved to {out_path}")
    log.info(f"Done in {time.time() - t0:.0f}s")
    return all_results


@app.function(gpu="A100", timeout=86400, volumes={"/results": results_vol})
def run_task(task_name: str, n_seeds: int = 1, max_pairs: int = 0,
             k_values: str = "1,2,4", layer: int = -1,
             vae_epochs: int = 500, output_dir: str = "",
             seed_offset: int = 0) -> dict:
    import torch
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger(__name__)
    DEVICE = "cuda"
    t0 = time.time()

    k_list = [int(x) for x in k_values.split(",")]
    layer_override = layer if layer >= 0 else None
    mib_kwargs = dict(max_pairs=max_pairs, n_seeds=n_seeds,
                      k_values=k_list, vae_epochs=vae_epochs,
                      layer_override=layer_override,
                      seed_offset=seed_offset)

    try:
        if task_name == "ioi":
            result = run_ioi(DEVICE, log)
        elif task_name in MIB_TASKS:
            result = run_mib_task(task_name, DEVICE, log, **mib_kwargs)
        else:
            result = run_grokking(task_name, GROK_EPOCHS[task_name], DEVICE, log)
    except Exception as e:
        log.error(f"[{utc_ts()}] {task_name} failed: {e}\n{traceback.format_exc()[-1500:]}")
        result = {"error": str(e)}

    result["elapsed_seconds"] = round(time.time() - t0, 1)
    result["timestamp"] = utc_ts()

    if not output_dir:
        output_dir = f"/results/grassmannian_atlas/k1_pi_ablations/{task_name}"
    save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)
    out_path = f"{save_dir}/results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    results_vol.commit()
    log.info(f"[{utc_ts()}] Saved to {out_path}")

    if "error" not in result:
        _log_task_summary(result, task_name, log)

    log.info(f"\n{task_name} done in {result['elapsed_seconds']:.0f}s")
    return result


@app.local_entrypoint()
def main(tasks: str = "ioi,addition,multiplication,squaring",
         n_seeds: int = 1, max_pairs: int = 0,
         k_values: str = "1,2,4", layer: int = -1,
         vae_epochs: int = 500, output_dir: str = "",
         mode: str = "single", seed_offset: int = 0):
    if mode == "cross-task":
        print(f"Cross-task mode: tasks={tasks}, k={k_values.split(',')[0]}, "
              f"layer={layer if layer >= 0 else 8}, max_pairs={max_pairs or 500}")
        k_val = int(k_values.split(",")[0])
        lyr = layer if layer >= 0 else 8
        mp = max_pairs if max_pairs > 0 else 500
        h = run_cross_task.spawn(tasks=tasks, k=k_val, layer=lyr,
                                 max_pairs=mp, vae_epochs=vae_epochs)
        print(f"[{utc_ts()}] Spawned cross-task pod: {h.object_id}")
        print(f"\nResults: /results/grassmannian_atlas/cross_task/results.json")
        return

    if mode == "feature_analysis":
        k_val = int(k_values.split(",")[0])
        lyr = layer if layer >= 0 else 8
        mp = max_pairs if max_pairs > 0 else 500
        print(f"Feature analysis mode: tasks={tasks}, k={k_val}, "
              f"layer={lyr}, max_pairs={mp}")
        h = run_feature_analysis.spawn(tasks=tasks, k=k_val, layer=lyr,
                                       max_pairs=mp, vae_epochs=vae_epochs)
        print(f"[{utc_ts()}] Spawned feature analysis pod: {h.object_id}")
        print(f"\nResults: /results/grassmannian_atlas/feature_analysis/results.json")
        return

    if mode == "multi_seed":
        ops = [t.strip() for t in tasks.split(",")]
        ns = n_seeds if n_seeds > 1 else 10
        print(f"Multi-seed mode: operations={ops}, n_seeds={ns}")
        handles = []
        for op in ops:
            h = run_multi_seed.spawn(operation=op, n_seeds=ns)
            handles.append((op, h))
            print(f"[{utc_ts()}] Spawned {op} multi-seed pod: {h.object_id}")
        print(f"\nResults: /results/grassmannian_atlas/multi_seed/<op>/results.json")
        return

    if mode == "s1_viz":
        print(f"S¹ visualization mode: operation={tasks.split(',')[0]}")
        h = run_s1_visualization.spawn(operation=tasks.split(",")[0])
        print(f"[{utc_ts()}] Spawned S¹ visualization pod: {h.object_id}")
        print(f"\nResults: /results/grassmannian_atlas/s1_visualization/")
        return

    if mode == "ivae":
        k_val = int(k_values.split(",")[0])
        print(f"iVAE verification mode: operations={tasks}, k={k_val}, "
              f"vae_epochs={vae_epochs}")
        h = run_ivae_verification.spawn(operations=tasks, k=k_val,
                                         vae_epochs=vae_epochs)
        print(f"[{utc_ts()}] Spawned iVAE verification pod: {h.object_id}")
        print(f"\nResults: /results/grassmannian_atlas/ivae_verification/results.json")
        return

    task_list = [t.strip() for t in tasks.split(",")]
    print(f"Config: n_seeds={n_seeds}, seed_offset={seed_offset}, "
          f"max_pairs={max_pairs or 'all'}, k_values={k_values}, "
          f"layer={layer if layer >= 0 else 'default'}, vae_epochs={vae_epochs}")
    handles = []
    for task in task_list:
        # output_dir is a PREFIX, not a full path. Previously it was used
        # verbatim for every task, so a multi-task run with a custom output_dir
        # had all six tasks writing to one file and overwriting each other,
        # while omitting it overwrote the published results in place.
        out = (f"{output_dir.rstrip('/')}/{task}" if output_dir
               else f"/results/grassmannian_atlas/k1_pi_ablations/{task}")
        h = run_task.spawn(task, n_seeds=n_seeds, max_pairs=max_pairs,
                           k_values=k_values, layer=layer,
                           vae_epochs=vae_epochs, output_dir=out,
                           seed_offset=seed_offset)
        handles.append((task, h))
        print(f"[{utc_ts()}] Spawned {task} pod: {h.object_id}")

    print(f"\n{len(task_list)} pods launched in parallel:")
    for task, h in handles:
        print(f"  {task:>20s}: {h.object_id}")
    print(f"\nResults: /results/grassmannian_atlas/k1_pi_ablations/<task>/results.json")
