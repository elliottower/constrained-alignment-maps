"""Hard-mode VAE vs DAS with vacuousness diagnostics + continuous metrics + CV.

Addresses three concerns from k1_vae_vs_das.py results:
  1. NL-DAS getting perfect IIA might be vacuous (lookup table, not genuine
     nonlinear coordinate system). We add reconstruction and diversity checks.
  2. Binary IIA saturates too easily. We measure KL, JS, prob_diff,
     normalized_logit_diff, and logit_diff — continuous metrics that don't
     saturate at 1.0.
  3. 70/30 random split is weak. We do 5-fold CV + a strict held-out split
     where train/eval use DISJOINT names (IOI) or disjoint input ranges
     (grokking).

Tasks:
  - IOI standard (18 names, 5 templates)
  - IOI hard mode: disjoint-name CV (train on names A, eval on names B)
  - IOI hard mode: disjoint-template CV (train on templates 1-3, eval on 4-5)
  - Greater-than task: "The war lasted from 17XX to 17" → YY > XX
  - Gendered pronouns: swap gender of subject
  - Grokking addition (for comparison)

Methods compared:
  - Linear DAS
  - NL-DAS (unconstrained encoder-decoder)
  - NL-DAS + reconstruction penalty (encoder-decoder must reconstruct)
  - Structured VAE
  - Unconstrained VAE (C7 from prior run)

Usage:
    modal run --detach experiments/batch6_atlas/06_21_2026_UPDATE/k1_hard_mode.py
"""
from __future__ import annotations

import json
import logging
import os
import random
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

app = modal.App("hard-iia-ksweep", image=image)

# Subspace dimensions swept under hard IIA. Set here rather than by env:
# Modal does not forward local environment variables into the container.
K_VALUES = [8, 16, 32]
results_vol = modal.Volume.from_name("fc-results", create_if_missing=True)


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ===================================================================
# Continuous metrics (from sub_edge.py / eap_metrics.py)
# ===================================================================


def compute_continuous_metrics(intervened_logits, clean_logits, source_label, base_label):
    """Compute all continuous metrics for a single intervention.

    Returns dict with: iia (binary), kl_div, js_div, prob_diff,
    normalized_logit_diff, logit_diff, source_prob, base_prob.
    """
    iv_probs = F.softmax(intervened_logits, dim=-1)
    cl_probs = F.softmax(clean_logits, dim=-1)

    iia = int(intervened_logits.argmax().item() == source_label)

    cl_lp = F.log_softmax(clean_logits, dim=-1)
    iv_lp = F.log_softmax(intervened_logits, dim=-1)
    kl = (cl_probs * (cl_lp - iv_lp)).sum().item()

    m = 0.5 * (cl_probs + iv_probs)
    m_lp = m.clamp(min=1e-12).log()
    js = 0.5 * (cl_probs * (cl_lp - m_lp)).sum() + 0.5 * (iv_probs * (iv_lp - m_lp)).sum()
    js = js.item()

    prob_diff = (iv_probs[source_label] - iv_probs[base_label]).item()

    logit_diff = (intervened_logits[source_label] - intervened_logits[base_label]).item()

    clean_logit_diff = (clean_logits[source_label] - clean_logits[base_label]).item()
    denom = max(abs(clean_logit_diff), 1e-6)
    normalized_logit_diff = logit_diff / denom

    source_prob = iv_probs[source_label].item()
    base_prob = iv_probs[base_label].item()

    return {
        "iia": iia,
        "kl_div": kl,
        "js_div": js,
        "prob_diff": prob_diff,
        "logit_diff": logit_diff,
        "normalized_logit_diff": normalized_logit_diff,
        "source_prob": source_prob,
        "base_prob": base_prob,
    }


def aggregate_metrics(metric_dicts):
    """Aggregate list of per-example metric dicts into means."""
    if not metric_dicts:
        return {}
    keys = metric_dicts[0].keys()
    result = {}
    for k in keys:
        vals = [d[k] for d in metric_dicts]
        result[f"{k}_mean"] = sum(vals) / len(vals)
        if len(vals) > 1:
            mean = result[f"{k}_mean"]
            result[f"{k}_std"] = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    return result


# ===================================================================
# Models: VAE, NL-DAS, NL-DAS+recon
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


def train_das(model_lm, train_pairs, hook_name, d_model, k, device,
              n_steps=300, lr=1e-3, batch_size=16):
    R = nn.Parameter(torch.randn(d_model, k, device=device) * 0.02)
    optimizer = torch.optim.Adam([R], lr=lr)
    for step in tqdm(range(n_steps), desc=f"DAS k={k}", leave=False):
        Q, _ = torch.linalg.qr(R)
        proj = Q @ Q.T
        batch = random.sample(train_pairs, min(batch_size, len(train_pairs)))
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


def train_nldas(model_lm, train_pairs, hook_name, d_model, k, device,
                hidden_dim=256, n_steps=400, lr=1e-3, batch_size=16,
                recon_weight=0.0, recon_acts=None):
    """NL-DAS with optional reconstruction penalty.

    If recon_weight > 0, adds ||inv_feat(feat(x)) - x||^2 loss to prevent
    degenerate encoder-decoder solutions.
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

    label = f"NL-DAS k={k}" + (" +recon" if recon_weight > 0 else "")
    for step in tqdm(range(n_steps), desc=label, leave=False):
        Q, _ = torch.linalg.qr(R)
        proj = Q @ Q.T
        batch = random.sample(train_pairs, min(batch_size, len(train_pairs)))
        iia_loss = torch.tensor(0.0, device=device)
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
            iia_loss = iia_loss - F.log_softmax(logits, dim=-1)[d["src_label"]]

        loss = iia_loss / len(batch)

        if recon_weight > 0 and recon_acts is not None:
            idx = torch.randint(0, len(recon_acts), (64,), device=device)
            x = recon_acts[idx]
            x_hat = inv_featurizer(featurizer(x))
            loss = loss + recon_weight * F.mse_loss(x_hat, x)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    featurizer.eval()
    inv_featurizer.eval()
    with torch.no_grad():
        Q, _ = torch.linalg.qr(R)
        Q = Q.detach()

    return featurizer, inv_featurizer, Q


# ===================================================================
# Evaluation with continuous metrics + vacuousness diagnostics
# ===================================================================


def eval_das_full(Q, model_lm, eval_pairs, hook_name, device):
    """Eval linear DAS with all continuous metrics."""
    proj = Q @ Q.T
    metrics = []
    with torch.inference_mode():
        for d in eval_pairs:
            iv = d["base_act"] - proj @ d["base_act"] + proj @ d["src_act"]

            def hook_fn(act, hook=None, iv_vec=iv):
                new = act.clone()
                new[0, -1, :] = iv_vec
                return new

            logits = model_lm.run_with_hooks(
                d["base_toks"], fwd_hooks=[(hook_name, hook_fn)]
            )[0, -1, :]
            clean_logits = model_lm(d["base_toks"])[0, -1, :]
            m = compute_continuous_metrics(logits, clean_logits, d["src_label"], d["base_label"])
            metrics.append(m)
    return aggregate_metrics(metrics)


def eval_nldas_full(feat, inv_feat, Q, model_lm, eval_pairs, hook_name, all_acts, device):
    """Eval NL-DAS with continuous metrics + vacuousness diagnostics."""
    proj = Q @ Q.T
    metrics = []
    intervened_acts = []

    with torch.inference_mode():
        for d in eval_pairs:
            feat_b = feat(d["base_act"])
            feat_s = feat(d["src_act"])
            feat_iv = feat_b - proj @ feat_b + proj @ feat_s
            iv = inv_feat(feat_iv)
            intervened_acts.append(iv)

            def hook_fn(act, hook=None, iv_vec=iv):
                new = act.clone()
                new[0, -1, :] = iv_vec
                return new

            logits = model_lm.run_with_hooks(
                d["base_toks"], fwd_hooks=[(hook_name, hook_fn)]
            )[0, -1, :]
            clean_logits = model_lm(d["base_toks"])[0, -1, :]
            m = compute_continuous_metrics(logits, clean_logits, d["src_label"], d["base_label"])
            metrics.append(m)

    result = aggregate_metrics(metrics)

    # Vacuousness diagnostic 1: reconstruction quality
    with torch.inference_mode():
        sample = all_acts[:200]
        recon = inv_feat(feat(sample))
        result["recon_mse"] = F.mse_loss(recon, sample).item()
        result["recon_cos_sim"] = F.cosine_similarity(recon, sample, dim=-1).mean().item()

    # Vacuousness diagnostic 2: intervention diversity
    # For each unique source label, compute std of intervened activations
    # If std ≈ 0, the decoder outputs the same thing regardless of base → lookup table
    iv_stack = torch.stack(intervened_acts)
    label_to_ivs = {}
    for i, d in enumerate(eval_pairs):
        sl = d["src_label"]
        if sl not in label_to_ivs:
            label_to_ivs[sl] = []
        label_to_ivs[sl].append(iv_stack[i])
    diversities = []
    for sl, ivs in label_to_ivs.items():
        if len(ivs) >= 3:
            stacked = torch.stack(ivs)
            div = stacked.std(dim=0).mean().item()
            diversities.append(div)
    result["intervention_diversity_mean"] = sum(diversities) / len(diversities) if diversities else 0.0

    # Compare to natural diversity of source activations
    src_acts_by_label = {}
    for d in eval_pairs:
        sl = d["src_label"]
        if sl not in src_acts_by_label:
            src_acts_by_label[sl] = []
        src_acts_by_label[sl].append(d["src_act"])
    nat_divs = []
    for sl, acts in src_acts_by_label.items():
        if len(acts) >= 3:
            stacked = torch.stack(acts)
            nat_divs.append(stacked.std(dim=0).mean().item())
    result["natural_diversity_mean"] = sum(nat_divs) / len(nat_divs) if nat_divs else 0.0

    # Ratio: if intervention_diversity / natural_diversity ≈ 0, it's a lookup table
    if result["natural_diversity_mean"] > 1e-8:
        result["diversity_ratio"] = result["intervention_diversity_mean"] / result["natural_diversity_mean"]
    else:
        result["diversity_ratio"] = float("nan")

    return result


def eval_vae_full(vae, model_lm, eval_pairs, hook_name, all_acts, device):
    """Eval structured VAE with continuous metrics."""
    metrics = []
    vae.eval()
    with torch.inference_mode():
        for d in eval_pairs:
            base_act = d["base_act"].unsqueeze(0)
            src_act = d["src_act"].unsqueeze(0)
            mu_c_b, _, mu_n_b, _ = vae.encode(base_act)
            mu_c_s, _, _, _ = vae.encode(src_act)
            z_iv = torch.cat([mu_c_s, mu_n_b], dim=-1)
            h_iv = vae.decoder(z_iv).squeeze(0)

            def hook_fn(act, hook=None, iv=h_iv):
                new = act.clone()
                new[0, -1, :] = iv
                return new

            logits = model_lm.run_with_hooks(
                d["base_toks"], fwd_hooks=[(hook_name, hook_fn)]
            )[0, -1, :]
            clean_logits = model_lm(d["base_toks"])[0, -1, :]
            m = compute_continuous_metrics(logits, clean_logits, d["src_label"], d["base_label"])
            metrics.append(m)

    result = aggregate_metrics(metrics)

    # VAE reconstruction
    with torch.inference_mode():
        sample = all_acts[:200]
        x_r, _, mu_c, lv_c, mu_n, lv_n = vae(sample)
        result["recon_mse"] = F.mse_loss(x_r, sample).item()
        result["recon_cos_sim"] = F.cosine_similarity(x_r, sample, dim=-1).mean().item()

    return result


# ===================================================================
# IOI data generation
# ===================================================================

IOI_NAMES_ALL = [
    " Mary", " John", " Alice", " Bob", " Tom", " Claire",
    " Dave", " Sarah", " James", " Emma", " Mike", " Kate",
    " Jack", " Anna", " Dan", " Amy", " Sam", " Lisa",
    " Paul", " Jane", " Mark", " Lucy", " Nick", " Helen",
    " Adam", " Laura", " Ben", " Sophie", " Chris", " Diana",
]

IOI_TEMPLATES = [
    "Then,{A} and{B} went to the {PLACE}.{B} gave a {OBJ} to",
    "Then,{A} and{B} had a lot of fun at the {PLACE}.{B} gave a {OBJ} to",
    "Then,{A} and{B} were working at the {PLACE}.{B} decided to give a {OBJ} to",
    "Then,{B} and{A} went to the {PLACE}.{B} gave a {OBJ} to",
    "Then,{B} and{A} had a lot of fun at the {PLACE}.{B} gave a {OBJ} to",
]

IOI_TEMPLATES_HARD = [
    "Then,{A} and{B} went to the {PLACE} together.{B} picked up a {OBJ} and gave it to",
    "After arriving at the {PLACE},{A} met{B} there.{B} handed a {OBJ} to",
    "At the {PLACE},{B} saw{A} and they chatted for a while. Then{B} gave a {OBJ} to",
    "{A} was waiting at the {PLACE} when{B} arrived with a {OBJ}. Then{B} gave the {OBJ} to",
]

IOI_PLACES = ["store", "park", "office", "restaurant", "library", "gym",
              "museum", "school", "hospital", "beach"]
IOI_OBJECTS = ["book", "drink", "ball", "pen", "bag", "phone",
               "key", "letter", "toy", "card"]


def generate_ioi_pairs(model, names, templates, rng, n_pairs=1500):
    """Generate IOI base/source pairs, return only both-correct ones."""
    valid_names = [n for n in names if len(model.tokenizer.encode(n)) == 1]
    LAYER = 10
    hook_name = f"blocks.{LAYER}.hook_resid_post"

    raw_pairs = []
    for _ in range(n_pairs):
        t = rng.choice(templates)
        a, b = rng.sample(valid_names, 2)
        base = t.format(A=a, B=b, PLACE=rng.choice(IOI_PLACES), OBJ=rng.choice(IOI_OBJECTS))
        source = t.format(A=b, B=a, PLACE=rng.choice(IOI_PLACES), OBJ=rng.choice(IOI_OBJECTS))
        raw_pairs.append((base, source, model.tokenizer.encode(a)[0], model.tokenizer.encode(b)[0]))

    data = []
    for base_text, src_text, base_id, src_id in tqdm(raw_pairs, desc="cache IOI"):
        bt = model.to_tokens(base_text)
        st = model.to_tokens(src_text)
        with torch.no_grad():
            _, bc = model.run_with_cache(bt, names_filter=hook_name)
            bl = model(bt)[0, -1]
            _, sc = model.run_with_cache(st, names_filter=hook_name)

        bm = (bl[base_id] - bl[src_id]).item()
        if bm > 0:
            data.append({
                "base_act": bc[hook_name][0, -1],
                "src_act": sc[hook_name][0, -1],
                "base_toks": bt,
                "src_label": src_id,
                "base_label": base_id,
            })

    return data, hook_name


def run_methods_on_split(model, train_pairs, eval_pairs, act_t, lab_t,
                         n_classes, d_model, k, hook_name, device, log,
                         hidden_dim=256, vae_epochs=500):
    """Run all methods on a given train/eval split. Returns results dict."""
    results = {}

    # Linear DAS
    log.info(f"[{utc_ts()}]     DAS k={k}")
    Q = train_das(model, train_pairs[:400], hook_name, d_model, k, device)
    results["das"] = eval_das_full(Q, model, eval_pairs, hook_name, device)

    # NL-DAS (unconstrained — the suspect)
    log.info(f"[{utc_ts()}]     NL-DAS k={k}")
    feat, inv_feat, Q_nl = train_nldas(
        model, train_pairs[:400], hook_name, d_model, k, device,
        hidden_dim=hidden_dim, recon_weight=0.0,
    )
    results["nldas"] = eval_nldas_full(feat, inv_feat, Q_nl, model, eval_pairs, hook_name, act_t, device)
    del feat, inv_feat
    torch.cuda.empty_cache()

    # NL-DAS + reconstruction penalty (non-degenerate encoder-decoder)
    log.info(f"[{utc_ts()}]     NL-DAS+recon k={k}")
    feat_r, inv_feat_r, Q_nlr = train_nldas(
        model, train_pairs[:400], hook_name, d_model, k, device,
        hidden_dim=hidden_dim, recon_weight=1.0, recon_acts=act_t,
    )
    results["nldas_recon"] = eval_nldas_full(feat_r, inv_feat_r, Q_nlr, model, eval_pairs, hook_name, act_t, device)
    del feat_r, inv_feat_r
    torch.cuda.empty_cache()

    # Structured VAE
    log.info(f"[{utc_ts()}]     VAE k={k}")
    vae = build_vae(d_model, k, 16, hidden_dim, n_classes, device)
    vae = train_vae(vae, act_t, lab_t, device, n_epochs=vae_epochs, alpha=10.0)
    results["vae"] = eval_vae_full(vae, model, eval_pairs, hook_name, act_t, device)
    del vae
    torch.cuda.empty_cache()

    # Random DAS baseline
    R_rand = torch.randn(d_model, k, device=device)
    Q_rand, _ = torch.linalg.qr(R_rand)
    results["random"] = eval_das_full(Q_rand, model, eval_pairs, hook_name, device)

    return results


# ===================================================================
# IOI experiments
# ===================================================================


def run_ioi_standard(model, device, log):
    """Standard IOI with continuous metrics and 5-fold CV."""
    log.info(f"[{utc_ts()}] === IOI Standard (5-fold CV) ===")

    data, hook_name = generate_ioi_pairs(
        model, IOI_NAMES_ALL[:18], IOI_TEMPLATES, random.Random(42), n_pairs=2000,
    )
    log.info(f"[{utc_ts()}] Valid pairs: {len(data)}")

    rng = random.Random(42)
    indices = list(range(len(data)))
    rng.shuffle(indices)

    n_folds = 5
    fold_size = len(indices) // n_folds
    all_fold_results = {}

    for fold in range(n_folds):
        log.info(f"[{utc_ts()}]   Fold {fold+1}/{n_folds}")
        eval_idx = indices[fold * fold_size: (fold + 1) * fold_size]
        train_idx = indices[:fold * fold_size] + indices[(fold + 1) * fold_size:]

        train_pairs = [data[i] for i in train_idx]
        eval_pairs = [data[i] for i in eval_idx]

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

        for k in K_VALUES:
            key = f"fold{fold}_k{k}"
            all_fold_results[key] = run_methods_on_split(
                model, train_pairs, eval_pairs, act_t, lab_t,
                n_classes, 768, k, hook_name, device, log,
            )

    # Average across folds
    averaged = {}
    for k in K_VALUES:
        averaged[f"k{k}"] = {}
        methods = all_fold_results[f"fold0_k{k}"].keys()
        for method in methods:
            metric_keys = all_fold_results[f"fold0_k{k}"][method].keys()
            averaged[f"k{k}"][method] = {}
            for mk in metric_keys:
                vals = [all_fold_results[f"fold{f}_k{k}"][method][mk] for f in range(n_folds)]
                averaged[f"k{k}"][method][mk] = sum(vals) / len(vals)

    return {"task": "ioi_standard_5fold", "n_pairs": len(data), "averaged": averaged,
            "per_fold": all_fold_results}


def run_ioi_disjoint_names(model, device, log):
    """Hard mode: train on names A-set, eval on names B-set. No name overlap."""
    log.info(f"[{utc_ts()}] === IOI Disjoint Names ===")

    valid_names = [n for n in IOI_NAMES_ALL if len(model.tokenizer.encode(n)) == 1]
    log.info(f"[{utc_ts()}] Valid names: {len(valid_names)}")

    rng = random.Random(99)
    rng.shuffle(valid_names)
    mid = len(valid_names) // 2
    train_names = valid_names[:mid]
    eval_names = valid_names[mid:]
    log.info(f"[{utc_ts()}] Train names: {train_names}")
    log.info(f"[{utc_ts()}] Eval names:  {eval_names}")

    hook_name = f"blocks.10.hook_resid_post"

    train_data, _ = generate_ioi_pairs(model, train_names, IOI_TEMPLATES, random.Random(42), n_pairs=1500)
    eval_data, _ = generate_ioi_pairs(model, eval_names, IOI_TEMPLATES, random.Random(43), n_pairs=1500)
    log.info(f"[{utc_ts()}] Train pairs: {len(train_data)}, Eval pairs: {len(eval_data)}")

    label_map = {}
    all_acts, all_labels = [], []
    for d in train_data:
        for rk, lk in [("base_act", "base_label"), ("src_act", "src_label")]:
            tid = d[lk]
            if tid not in label_map:
                label_map[tid] = len(label_map)
            all_acts.append(d[rk])
            all_labels.append(label_map[tid])

    act_t = torch.stack(all_acts)
    lab_t = torch.tensor(all_labels, device=device)
    n_classes = len(label_map)

    results = {"task": "ioi_disjoint_names", "train_names": train_names, "eval_names": eval_names}
    for k in K_VALUES:
        log.info(f"[{utc_ts()}]   k={k}")
        results[f"k{k}"] = run_methods_on_split(
            model, train_data, eval_data, act_t, lab_t,
            n_classes, 768, k, hook_name, device, log,
        )

    return results


def run_ioi_disjoint_templates(model, device, log):
    """Hard mode: train on templates 1-3, eval on templates 4-5 + hard templates."""
    log.info(f"[{utc_ts()}] === IOI Disjoint Templates ===")

    train_templates = IOI_TEMPLATES[:3]
    eval_templates = IOI_TEMPLATES[3:] + IOI_TEMPLATES_HARD

    hook_name = f"blocks.10.hook_resid_post"
    names = IOI_NAMES_ALL[:18]

    train_data, _ = generate_ioi_pairs(model, names, train_templates, random.Random(42), n_pairs=1500)
    eval_data, _ = generate_ioi_pairs(model, names, eval_templates, random.Random(43), n_pairs=1500)
    log.info(f"[{utc_ts()}] Train: {len(train_data)}, Eval: {len(eval_data)}")

    label_map = {}
    all_acts, all_labels = [], []
    for d in train_data:
        for rk, lk in [("base_act", "base_label"), ("src_act", "src_label")]:
            tid = d[lk]
            if tid not in label_map:
                label_map[tid] = len(label_map)
            all_acts.append(d[rk])
            all_labels.append(label_map[tid])

    act_t = torch.stack(all_acts)
    lab_t = torch.tensor(all_labels, device=device)
    n_classes = len(label_map)

    results = {"task": "ioi_disjoint_templates", "n_train_templates": len(train_templates),
               "n_eval_templates": len(eval_templates)}
    for k in K_VALUES:
        log.info(f"[{utc_ts()}]   k={k}")
        results[f"k{k}"] = run_methods_on_split(
            model, train_data, eval_data, act_t, lab_t,
            n_classes, 768, k, hook_name, device, log,
        )

    return results


# ===================================================================
# Greater-than task
# ===================================================================


def run_greater_than(model, device, log):
    """Greater-than: 'The war lasted from 17XX to 17' → model outputs YY > XX."""
    log.info(f"[{utc_ts()}] === Greater-Than ===")

    hook_name = "blocks.10.hook_resid_post"
    LAYER = 10
    prefixes = [
        "The war lasted from 17{:02d} to 17",
        "The bridge was built from 17{:02d} to 17",
        "The dynasty ruled from 17{:02d} to 17",
        "The expedition ran from 17{:02d} to 17",
    ]

    year_tokens = {}
    for y in range(100):
        tok = model.tokenizer.encode(f"{y:02d}")
        if len(tok) == 1:
            year_tokens[y] = tok[0]

    data = []
    for prefix_template in prefixes:
        for start_year in range(10, 80):
            if start_year not in year_tokens:
                continue
            prompt = prefix_template.format(start_year)
            toks = model.to_tokens(prompt)
            with torch.no_grad():
                _, cache = model.run_with_cache(toks, names_filter=hook_name)
                logits = model(toks)[0, -1]

            correct_years = [y for y in range(start_year + 1, 100) if y in year_tokens]
            if not correct_years:
                continue

            correct_token_ids = [year_tokens[y] for y in correct_years]
            correct_prob = sum(F.softmax(logits, dim=-1)[tid].item() for tid in correct_token_ids)

            if correct_prob > 0.3:
                data.append({
                    "act": cache[hook_name][0, -1],
                    "toks": toks,
                    "start_year": start_year,
                    "correct_years": correct_years,
                    "template_idx": prefixes.index(prefix_template),
                })

    log.info(f"[{utc_ts()}] Valid examples: {len(data)}")

    pairs = []
    rng = random.Random(42)
    for i in range(len(data)):
        for _ in range(3):
            j = rng.randint(0, len(data) - 1)
            if data[i]["start_year"] != data[j]["start_year"]:
                gt_years_j = [y for y in data[j]["correct_years"] if y in year_tokens]
                if gt_years_j:
                    pairs.append({
                        "base_act": data[i]["act"],
                        "src_act": data[j]["act"],
                        "base_toks": data[i]["toks"],
                        "src_label": year_tokens[gt_years_j[0]],
                        "base_label": year_tokens[data[i]["correct_years"][0]] if data[i]["correct_years"] else 0,
                        "src_year": data[j]["start_year"],
                        "base_year": data[i]["start_year"],
                    })

    rng2 = random.Random(42)
    rng2.shuffle(pairs)
    n_train = int(0.7 * len(pairs))
    train_pairs = pairs[:n_train]
    eval_pairs = pairs[n_train:]
    log.info(f"[{utc_ts()}] Train: {len(train_pairs)}, Eval: {len(eval_pairs)}")

    all_acts = torch.stack([d["act"] for d in data])
    all_labels = torch.tensor([d["start_year"] for d in data], device=device)
    n_classes = 100

    results = {"task": "greater_than", "n_examples": len(data), "n_pairs": len(pairs)}
    for k in K_VALUES:
        log.info(f"[{utc_ts()}]   k={k}")
        results[f"k{k}"] = run_methods_on_split(
            model, train_pairs, eval_pairs, all_acts, all_labels,
            n_classes, 768, k, hook_name, device, log,
        )

    return results


# ===================================================================
# Gender pronoun task
# ===================================================================


def run_gender_pronouns(model, device, log):
    """Gender: swap gender variable via pronoun prediction."""
    log.info(f"[{utc_ts()}] === Gender Pronouns ===")

    hook_name = "blocks.10.hook_resid_post"

    male_names = [" John", " James", " Mike", " Tom", " Bob", " Dave", " Jack", " Dan",
                  " Sam", " Paul", " Mark", " Nick", " Adam", " Ben", " Chris"]
    female_names = [" Mary", " Sarah", " Emma", " Kate", " Anna", " Amy", " Lisa",
                    " Jane", " Lucy", " Helen", " Laura", " Sophie", " Diana", " Claire", " Alice"]

    he_id = model.tokenizer.encode(" he")[0]
    she_id = model.tokenizer.encode(" she")[0]

    templates = [
        "{name} went to the store and",
        "{name} was at the park when",
        "{name} picked up the phone and",
        "{name} opened the door and then",
        "{name} looked at the menu and",
    ]

    valid_male = [n for n in male_names if len(model.tokenizer.encode(n)) == 1]
    valid_female = [n for n in female_names if len(model.tokenizer.encode(n)) == 1]

    data = []
    for names, gender_label, correct_id, wrong_id in [
        (valid_male, 0, he_id, she_id),
        (valid_female, 1, she_id, he_id),
    ]:
        for name in names:
            for template in templates:
                prompt = template.format(name=name)
                toks = model.to_tokens(prompt)
                with torch.no_grad():
                    _, cache = model.run_with_cache(toks, names_filter=hook_name)
                    logits = model(toks)[0, -1]

                if logits[correct_id] > logits[wrong_id]:
                    data.append({
                        "act": cache[hook_name][0, -1],
                        "toks": toks,
                        "gender": gender_label,
                        "correct_id": correct_id,
                        "wrong_id": wrong_id,
                        "name": name,
                    })

    log.info(f"[{utc_ts()}] Valid examples: {len(data)}")

    pairs = []
    rng = random.Random(42)
    for i in range(len(data)):
        for _ in range(3):
            j = rng.randint(0, len(data) - 1)
            if data[i]["gender"] != data[j]["gender"]:
                pairs.append({
                    "base_act": data[i]["act"],
                    "src_act": data[j]["act"],
                    "base_toks": data[i]["toks"],
                    "src_label": data[j]["correct_id"],
                    "base_label": data[i]["correct_id"],
                })

    rng2 = random.Random(42)
    rng2.shuffle(pairs)
    n_train = int(0.7 * len(pairs))
    train_pairs = pairs[:n_train]
    eval_pairs = pairs[n_train:]

    all_acts = torch.stack([d["act"] for d in data])
    all_labels = torch.tensor([d["gender"] for d in data], device=device)

    log.info(f"[{utc_ts()}] Train: {len(train_pairs)}, Eval: {len(eval_pairs)}")

    results = {"task": "gender_pronouns", "n_examples": len(data), "n_pairs": len(pairs)}
    for k in K_VALUES:
        log.info(f"[{utc_ts()}]   k={k}")
        results[f"k{k}"] = run_methods_on_split(
            model, train_pairs, eval_pairs, all_acts, all_labels,
            2, 768, k, hook_name, device, log,
        )

    return results


# ===================================================================
# Main
# ===================================================================


def print_summary(all_results, log):
    """Print summary table with continuous metrics + vacuousness diagnostics."""
    log.info(f"\n{'=' * 130}")
    log.info(f"{'Task':>22s} {'k':>2s}  {'Method':>12s}  "
             f"{'IIA':>6s}  {'KL':>7s}  {'JS':>7s}  {'ProbD':>7s}  "
             f"{'NormLD':>7s}  {'ReconMSE':>9s}  {'DivRatio':>8s}")
    log.info("-" * 120)

    for task_key in ["ioi_standard", "ioi_disjoint_names", "ioi_disjoint_templates",
                     "greater_than", "gender_pronouns"]:
        r = all_results.get(task_key, {})
        if "error" in r:
            log.info(f"{task_key:>22s} ERROR: {r['error'][:60]}")
            continue

        src = r.get("averaged", r)
        for k in K_VALUES:
            kr = src.get(f"k{k}", {})
            if not kr:
                continue
            for method in ["das", "nldas", "nldas_recon", "vae", "random"]:
                mr = kr.get(method, {})
                if not mr:
                    continue
                log.info(
                    f"{task_key:>22s} {k:2d}  {method:>12s}  "
                    f"{mr.get('iia_mean', 0):6.3f}  "
                    f"{mr.get('kl_div_mean', 0):7.4f}  "
                    f"{mr.get('js_div_mean', 0):7.4f}  "
                    f"{mr.get('prob_diff_mean', 0):7.4f}  "
                    f"{mr.get('normalized_logit_diff_mean', 0):7.4f}  "
                    f"{mr.get('recon_mse', 0):9.5f}  "
                    f"{mr.get('diversity_ratio', float('nan')):8.4f}"
                )

    log.info(f"\nVacuousness interpretation:")
    log.info(f"  diversity_ratio ≈ 0  => lookup table (intervened acts don't vary with base)")
    log.info(f"  diversity_ratio ≈ 1  => genuine (intervention preserves base-dependent structure)")
    log.info(f"  recon_mse high       => encoder-decoder is degenerate (can't reconstruct)")
    log.info(f"  nldas_recon vs nldas => does forcing reconstruction kill IIA?")
    log.info(f"  disjoint_names       => does it generalize to UNSEEN names?")




@app.function(gpu="A100", timeout=86400, volumes={"/results": results_vol})
def run_hard_mode() -> dict:
    import torch
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger(__name__)
    DEVICE = "cuda"
    t0 = time.time()

    all_results = {"timestamp": utc_ts()}

    log.info(f"[{utc_ts()}] Loading GPT-2...")
    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tasks = [
        ("ioi_standard", lambda: run_ioi_standard(model, DEVICE, log)),
        ("ioi_disjoint_names", lambda: run_ioi_disjoint_names(model, DEVICE, log)),
        ("ioi_disjoint_templates", lambda: run_ioi_disjoint_templates(model, DEVICE, log)),
        ("greater_than", lambda: run_greater_than(model, DEVICE, log)),
        ("gender_pronouns", lambda: run_gender_pronouns(model, DEVICE, log)),
    ]

    for task_name, task_fn in tasks:
        try:
            all_results[task_name] = task_fn()
        except Exception as e:
            log.error(f"[{utc_ts()}] {task_name} failed: {e}\n{traceback.format_exc()[-2000:]}")
            all_results[task_name] = {"error": str(e)}

    all_results["elapsed_seconds"] = round(time.time() - t0, 1)

    save_dir = "/results/hard_iia_ksweep/k" + "-".join(str(x) for x in K_VALUES)
    os.makedirs(save_dir, exist_ok=True)
    out_path = f"{save_dir}/results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    results_vol.commit()
    log.info(f"[{utc_ts()}] Saved to {out_path}")

    print_summary(all_results, log)
    log.info(f"\nTotal: {all_results['elapsed_seconds']:.0f}s")

    return all_results


@app.local_entrypoint()
def main():
    handle = run_hard_mode.spawn()
    print(f"[{utc_ts()}] Spawned hard-mode VAE vs DAS diagnostics")
    print(f"  Handle: {handle.object_id}")
    print(f"  Tasks: IOI 5-fold CV, disjoint names, disjoint templates, greater-than, gender pronouns")
    print(f"  Methods: DAS, NL-DAS, NL-DAS+recon, VAE, random")
    print(f"  Metrics: IIA, KL, JS, prob_diff, normalized_logit_diff + vacuousness diagnostics")
    print(f"  Results: /results/grassmannian_atlas/k1_hard_mode/results.json")
