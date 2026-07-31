"""Unified 6-method comparison on grokking + GPT-2 tasks.

Mirrors phonetic-circuits/lib/run_method_comparison.py exactly:
same 6 methods, same IIA + diversity_ratio output format, same summary table.

Methods:
  1. Delta-PCA (training-free SVD baseline)
  2. DAS (trained rotation, PCA-initialized)
  3. Nonlinear DAS (MLP featurizer + linear proj + MLP inverse)
  4. Structured VAE (causal/nuisance split, classifier)
  5. pi-VAE (label-conditional prior, identifiability)
  6. pi-SAE (overcomplete sparse + label-conditional prior)

Tasks:
  - Grokking: addition, multiplication, squaring (+ optional extras)
  - GPT-2: IOI (layer 10)

Usage:
    modal run --detach experiments/method_comparison_unified.py
    modal run --detach experiments/method_comparison_unified.py --tasks addition squaring ioi
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
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
    .pip_install("torch==2.5.1", "numpy==1.26.4", "setuptools<71")
    .pip_install(
        "transformer-lens==2.11.0", "transformers==4.46.3",
        "einops>=0.8", "matplotlib", "tqdm",
    )
    # pyvene supplies LowRankRotateLayer, which MIB's SubspaceFeaturizer wraps.
    .pip_install("pyvene==0.1.8")
    # MIB's causal-variable track, imported by experiments/das.py at /mib_track.
    .add_local_dir(
        "./reference/MIB/MIB-causal-variable-track",
        remote_path="/mib_track",
    )
    .add_local_file("experiments/das.py", remote_path="/root/das.py")
)

app = modal.App("random-network-control", image=image)
results_vol = modal.Volume.from_name("fc-results", create_if_missing=True)

P = 113
FRAC_TRAIN = 0.3
DATA_SEED = 598

GROK_EPOCHS = {
    "addition": 25000, "multiplication": 40000, "squaring": 80000,
}

OP_MODULUS = {}


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_msg(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ===================================================================
# Grokking data + model
# ===================================================================

def compute_labels(a_vec, b_vec, operation, p):
    if operation == "addition":
        return (a_vec + b_vec) % p
    elif operation == "multiplication":
        return (a_vec * b_vec) % p
    elif operation == "squaring":
        return (a_vec * a_vec) % p
    else:
        raise ValueError(f"Unknown operation: {operation}")


UNARY_OPS = {"squaring", "cubing"}


def build_grok_data(operation, p, device):
    is_unary = operation in UNARY_OPS
    excludes_zero = operation == "multiplication"
    frac_train = 0.5 if is_unary else FRAC_TRAIN

    if is_unary:
        a_vec = torch.arange(1, p)
        b_vec = torch.zeros(len(a_vec), dtype=torch.long)
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

    split_seed = 0 if is_unary else DATA_SEED
    rng = torch.Generator()
    rng.manual_seed(split_seed)
    indices = torch.randperm(len(dataset), generator=rng)
    cutoff = int(len(dataset) * frac_train)
    return dataset, labels, indices[:cutoff], indices[cutoff:]


def train_grokking_model(operation, p, device, n_epochs, seed=999):
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

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98)
    )
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

    return model, dataset, labels, test_idx, test_acc


# ===================================================================
# IOI data
# ===================================================================

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


# ===================================================================
# Pair generation
# ===================================================================

def generate_grok_pairs(model, dataset, labels, test_idx, hook_name, device, max_pairs=600):
    all_acts = []
    for i in range(0, len(dataset), 256):
        with torch.inference_mode():
            _, cache = model.run_with_cache(dataset[i:i + 256], names_filter=[hook_name])
        all_acts.append(cache[hook_name][:, -1, :].clone())
    activations = torch.cat(all_acts, dim=0)

    test_acts = activations[test_idx]
    test_labels = labels[test_idx]
    test_data = dataset[test_idx]

    # Both-correct filter: model predicts correct label for both base and source
    with torch.inference_mode():
        test_logits = model(test_data)[:, -1]
        correct_mask = test_logits.argmax(-1) == test_labels

    correct_indices = correct_mask.nonzero(as_tuple=True)[0]
    n_correct = len(correct_indices)

    pairs = []
    for ii in range(n_correct):
        i = correct_indices[ii].item()
        for jj in range(ii + 1, min(ii + 50, n_correct)):
            j = correct_indices[jj].item()
            if test_labels[i] != test_labels[j]:
                pairs.append({
                    "base_act": test_acts[i],
                    "source_act": test_acts[j],
                    "base_toks": test_data[i:i + 1],
                    "src_id": test_labels[j].item(),
                    "base_id": test_labels[i].item(),
                })
                if len(pairs) >= max_pairs:
                    break
        if len(pairs) >= max_pairs:
            break

    return pairs, activations


def generate_ioi_pairs(model, device, max_pairs=600, require_correct=True):
    hook_name = "blocks.10.hook_resid_post"
    valid_names = [n for n in IOI_NAMES if len(model.tokenizer.encode(n)) == 1]

    rng = random.Random(42)
    data = []
    for _ in range(2000):
        t = rng.choice(IOI_TEMPLATES)
        a, b = rng.sample(valid_names, 2)
        base = t.format(A=a, B=b, PLACE=rng.choice(IOI_PLACES), OBJ=rng.choice(IOI_OBJECTS))
        source = t.format(A=b, B=a, PLACE=rng.choice(IOI_PLACES), OBJ=rng.choice(IOI_OBJECTS))

        bt = model.to_tokens(base)
        st = model.to_tokens(source)
        base_id = model.tokenizer.encode(a)[0]
        src_id = model.tokenizer.encode(b)[0]

        with torch.no_grad():
            _, bc = model.run_with_cache(bt, names_filter=hook_name)
            bl = model(bt)[0, -1]
            _, sc = model.run_with_cache(st, names_filter=hook_name)
            sl = model(st)[0, -1]

        bm = (bl[base_id] - bl[src_id]).item()
        sm = (sl[src_id] - sl[base_id]).item()
        # On a random network the model is correct on nothing, so the
        # counterfactual target is defined by the IOI algorithm rather than
        # by the model's own behaviour. Sutter et al.'s claim is exactly that
        # an alignment map can make such a network track the algorithm.
        if (bm > 0 and sm > 0) or not require_correct:
            data.append({
                "base_act": bc[hook_name][0, -1],
                "source_act": sc[hook_name][0, -1],
                "base_toks": bt,
                "src_id": src_id,
                "base_id": base_id,
            })
        if len(data) >= max_pairs:
            break

    return data, hook_name


# ===================================================================
# Diversity ratio
# ===================================================================

_ACT_DUMP_DIR = os.environ.get("ACT_DUMP_DIR", "/results/activation_dumps")
_ACT_DUMP_N = [0]


def _persist_acts(intervened_acts, natural_acts, labels):
    """Save raw activations + labels so any future metric is a re-score, not a re-run."""
    try:
        os.makedirs(_ACT_DUMP_DIR, exist_ok=True)
        _ACT_DUMP_N[0] += 1
        torch.save({"intervened": intervened_acts.cpu(),
                    "natural": natural_acts.cpu(),
                    "labels": list(map(int, labels))},
                   os.path.join(_ACT_DUMP_DIR, f"acts_{_ACT_DUMP_N[0]:03d}.pt"))
    except Exception as e:
        log_msg(f"  [warn] activation dump failed: {e}")


def compute_diversity_ratio(intervened_acts, natural_acts):
    """Global dispersion ratio: std over ALL examples, no label grouping.

    Retained as the secondary metric. It is conservative for detecting a
    lookup table, because a lookup table preserves across-label variation and
    that variation inflates this numerator. See compute_diversity_grouped for
    the within-label metric the paper's Eq. (diversity_ratio) defines.
    """
    if len(intervened_acts) < 3:
        return float("nan")
    iv_std = intervened_acts.std(dim=0).mean().item()
    nat_std = natural_acts.std(dim=0).mean().item()
    if nat_std < 1e-8:
        return float("nan")
    return iv_std / nat_std


def compute_diversity_grouped(intervened_acts, natural_acts, labels, min_group=5):
    """Within-label diversity ratio, matching Eq. (diversity_ratio).

    rho = E_{y_s}[ std(h'_{y_s}) ] / E_{y_s}[ std(h_{y_s}) ]

    A lookup table maps every base sharing a source label to one activation, so
    its numerator collapses while the denominator does not. Numerator and
    denominator are returned separately: on a model with no structure the
    per-label denominator can itself become degenerate, which would inflate the
    ratio for reasons unrelated to the intervention.
    """
    from collections import defaultdict
    idx = defaultdict(list)
    for i, y in enumerate(labels):
        idx[int(y)].append(i)
    iv_stds, nat_stds, used = [], [], 0
    for _, ids in idx.items():
        if len(ids) < min_group:
            continue
        used += 1
        iv_stds.append(intervened_acts[ids].std(dim=0).mean().item())
        nat_stds.append(natural_acts[ids].std(dim=0).mean().item())
    dropped = len(idx) - used
    if not iv_stds:
        return {"rho_within": float("nan"), "iv_std_within": float("nan"),
                "nat_std_within": float("nan"), "n_groups_kept": 0,
                "n_groups_dropped": dropped, "min_group": min_group}
    num = sum(iv_stds) / len(iv_stds)
    den = sum(nat_stds) / len(nat_stds)
    return {"rho_within": (num / den) if den > 1e-8 else float("nan"),
            "iv_std_within": num, "nat_std_within": den,
            "n_groups_kept": used, "n_groups_dropped": dropped,
            "min_group": min_group}


# ===================================================================
# Method 1: Delta-PCA (training-free)
# ===================================================================

def run_delta_pca(train_data, eval_data, model, hook_name, device, k):
    deltas = torch.stack([d["source_act"] - d["base_act"] for d in train_data])
    _, _, Vh = torch.linalg.svd(deltas, full_matrices=False)
    U = Vh[:k].T.to(device)
    proj = U @ U.T
    return _eval_linear(model, eval_data, proj, hook_name, device, "delta_pca")


# ===================================================================
# Method 2: DAS (trained rotation)
# ===================================================================

def train_das(model, train_data, hook_name, device, k=1, n_steps=300):
    d_model = train_data[0]["base_act"].shape[0]

    deltas = torch.stack([d["source_act"] - d["base_act"] for d in train_data])
    _, _, Vh = torch.linalg.svd(deltas, full_matrices=False)
    A = nn.Parameter(Vh[:k].T.clone().to(device))
    optimizer = torch.optim.Adam([A], lr=1e-3)

    batch_size = 16
    for step in range(n_steps):
        optimizer.zero_grad()
        Q, _ = torch.linalg.qr(A)
        proj = Q @ Q.T
        batch = random.sample(train_data, min(batch_size, len(train_data)))
        loss = torch.tensor(0.0, device=device)
        for d in batch:
            intervention = proj @ (d["source_act"] - d["base_act"])

            def make_hook(_iv):
                def hk(act, hook):
                    new = act.clone()
                    new[0, -1, :] += _iv
                    return new
                return hk

            logits = model.run_with_hooks(
                d["base_toks"], fwd_hooks=[(hook_name, make_hook(intervention))]
            )
            loss -= logits[0, -1, :].log_softmax(dim=-1)[d["src_id"]]
        (loss / len(batch)).backward()
        optimizer.step()

    with torch.no_grad():
        Q, _ = torch.linalg.qr(A)
    return (Q @ Q.T).detach()


def run_das(train_data, eval_data, model, hook_name, device, k, n_steps=300):
    proj = train_das(model, train_data, hook_name, device, k=k, n_steps=n_steps)
    return _eval_linear(model, eval_data, proj, hook_name, device, "das")


# ===================================================================
# DAS matching the MIB / pyvene reference implementation
#
# NOTE ON WHAT train_das ABOVE ACTUALLY IS. It warm-starts from the top-k right
# singular vectors of the intervention deltas, which multiseed_ksweep.py
# describes as "a different (warm-started) method and is not standard DAS" and
# which the manuscript reports separately as the delta-PCA variant. The arm below
# is standard DAS: random orthogonal initialisation kept on the Stiefel manifold
# by torch.nn.utils.parametrizations.orthogonal, matching pyvene's
# LowRankRotateLayer(init_orth=True) as used by MIB.
#
# Both are fitted in the same run so the discrepancy is measured under identical
# data and seeds (Experiment 0 of PREREGISTRATION_RECONSTRUCTION_CRITERION.md).
# ===================================================================

def train_das_reference(model, train_data, hook_name, device, k=1, n_steps=None,
                        lr=None, task=None):
    """Standard DAS, run with MIB's own code and MIB's hyperparameters.

    Delegates to experiments/das.py, whose standard arm is MIB's
    CausalAbstraction.neural.featurizers.SubspaceFeaturizer wrapping pyvene's
    LowRankRotateLayer. Budget is MIB's 3 epochs at batch 32 with AdamW at
    lr=1e-2, rather than this file's historical 300 steps at lr=1e-3.

    Returns the projector, matching train_das above.
    """
    if das is None:
        raise ImportError(
            "experiments/das.py did not import; the reference DAS arm needs it "
            "along with pyvene and the vendored MIB checkout.")
    keys = dict(key_base_act="base_act", key_src_act="source_act",
                key_base_toks="base_toks", key_target="src_id")
    if n_steps is None:
        Q = das.train_das_mib(model, train_data, hook_name, device, k=k,
                              task=task, **keys)
    else:
        Q = das.train_das(model, train_data, hook_name, device, k=k,
                          n_steps=n_steps, lr=lr or das.MIB_CONFIG["lr"], **keys)
    return (Q @ Q.T).detach()


def run_das_reference(train_data, eval_data, model, hook_name, device, k,
                      n_steps=None, task=None):
    """n_steps=None uses MIB's per-task epoch budget, not a fixed step count."""
    proj = train_das_reference(model, train_data, hook_name, device, k=k,
                               n_steps=n_steps, task=task)
    return _eval_linear(model, eval_data, proj, hook_name, device, "das_reference")


# Student's t 95% critical values, keyed by sample count. Same table as
# experiments/aggregate_seeds.py so intervals are computed identically.
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
       9: 2.306, 10: 2.262}

# Independent random subspaces drawn per task to estimate the noise floor.
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


# ===================================================================
# Method 3: Nonlinear DAS
# ===================================================================

def train_nldas(model, train_data, hook_name, device, k=1, hidden_dim=256, n_steps=200,
                recon_weight=0.0, recon_acts=None):
    d_model = train_data[0]["base_act"].shape[0]

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
    optimizer = torch.optim.Adam(params, lr=1e-3)
    batch_size = 16

    label = f"NL-DAS k={k}" + (" +recon" if recon_weight > 0 else "")
    for step in tqdm(range(n_steps), desc=label, leave=False):
        Q, _ = torch.linalg.qr(R)
        proj = Q @ Q.T
        batch = random.sample(train_data, min(batch_size, len(train_data)))
        loss = torch.tensor(0.0, device=device)
        for d in batch:
            feat_b = featurizer(d["base_act"])
            feat_s = featurizer(d["source_act"])
            feat_iv = feat_b - proj @ feat_b + proj @ feat_s
            iv = inv_featurizer(feat_iv)

            def make_hook(_iv):
                def hk(act, hook):
                    new = act.clone()
                    new[0, -1, :] = _iv
                    return new
                return hk

            logits = model.run_with_hooks(
                d["base_toks"], fwd_hooks=[(hook_name, make_hook(iv))]
            )
            loss -= logits[0, -1, :].log_softmax(dim=-1)[d["src_id"]]
        loss = loss / len(batch)

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
    return featurizer, inv_featurizer, Q.detach()


def run_nldas(train_data, eval_data, model, hook_name, device, k, n_steps=200,
              recon_weight=0.0, recon_acts=None):
    featurizer, inv_featurizer, Q = train_nldas(
        model, train_data, hook_name, device, k=k, n_steps=n_steps,
        recon_weight=recon_weight, recon_acts=recon_acts,
    )
    proj = Q @ Q.T
    correct = 0
    total = 0
    intervened_acts_list = []
    prob_diffs = []
    logit_diffs = []

    with torch.inference_mode():
        for d in eval_data:
            feat_b = featurizer(d["base_act"])
            feat_s = featurizer(d["source_act"])
            feat_iv = feat_b - proj @ feat_b + proj @ feat_s
            iv = inv_featurizer(feat_iv)
            intervened_acts_list.append(iv.clone())

            def make_hook(_iv):
                def hk(act, hook):
                    new = act.clone()
                    new[0, -1, :] = _iv
                    return new
                return hk

            logits = model.run_with_hooks(
                d["base_toks"], fwd_hooks=[(hook_name, make_hook(iv))]
            )
            iv_logits = logits[0, -1]
            correct += int(iv_logits.argmax().item() == d["src_id"])
            total += 1
            s_l = iv_logits[d["src_id"]].item()
            b_l = iv_logits[d["base_id"]].item()
            logit_diffs.append(s_l - b_l)
            probs = torch.softmax(iv_logits, dim=-1)
            prob_diffs.append(probs[d["src_id"]].item() - probs[d["base_id"]].item())

    natural_acts = torch.stack([d["source_act"] for d in eval_data])
    intervened_acts = torch.stack(intervened_acts_list)
    div_ratio = compute_diversity_ratio(intervened_acts, natural_acts)
    _labels = [d["src_id"] for d in eval_data]
    grouped = compute_diversity_grouped(intervened_acts, natural_acts, _labels)
    _persist_acts(intervened_acts, natural_acts, _labels)

    return {
        "method": "nldas",
        "iia": correct / total if total else 0.0,
        "n_eval": total,
        "mean_prob_diff": sum(prob_diffs) / len(prob_diffs) if prob_diffs else None,
        "mean_logit_diff": sum(logit_diffs) / len(logit_diffs) if logit_diffs else None,
        "diversity_ratio": div_ratio,
        **grouped,
    }


# ===================================================================
# VAE family
# ===================================================================

def _build_structured_vae(d_input, z_causal_dim, z_nuisance_dim, hidden_dim, n_classes, device):
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

        def forward(self, x):
            mu_c, lv_c, mu_n, lv_n = self.encode(x)
            z_c = mu_c + torch.exp(0.5 * lv_c) * torch.randn_like(lv_c)
            z_n = mu_n + torch.exp(0.5 * lv_n) * torch.randn_like(lv_n)
            z = torch.cat([z_c, z_n], dim=-1)
            return self.decoder(z), self.classifier(z_c), mu_c, lv_c, mu_n, lv_n

    return StructuredVAE().to(device)


def _build_pi_vae(d_input, z_causal_dim, z_nuisance_dim, hidden_dim, n_classes, device):
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

        def forward(self, x):
            mu_c, lv_c, mu_n, lv_n = self.encode(x)
            z_c = mu_c + torch.exp(0.5 * lv_c) * torch.randn_like(lv_c)
            z_n = mu_n + torch.exp(0.5 * lv_n) * torch.randn_like(lv_n)
            z = torch.cat([z_c, z_n], dim=-1)
            return self.decoder(z), self.classifier(z_c), mu_c, lv_c, mu_n, lv_n

    return PiVAE().to(device)


def _build_pi_sae(d_input, z_causal_dim, z_nuisance_dim, hidden_dim, n_classes, device, expansion_factor=8):
    z_sparse = z_causal_dim * expansion_factor

    class PiSAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.z_causal_dim = z_causal_dim
            self.z_sparse_dim = z_sparse
            z_dim = z_sparse + z_nuisance_dim
            self.enc_trunk = nn.Sequential(
                nn.Linear(d_input, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            )
            self.enc_causal_mu = nn.Linear(hidden_dim, z_sparse)
            self.enc_causal_logvar = nn.Linear(hidden_dim, z_sparse)
            self.enc_nuisance_mu = nn.Linear(hidden_dim, z_nuisance_dim)
            self.enc_nuisance_logvar = nn.Linear(hidden_dim, z_nuisance_dim)
            self.decoder = nn.Sequential(
                nn.Linear(z_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, d_input),
            )
            self.classifier = nn.Linear(z_sparse, n_classes)
            self.prior_mu = nn.Embedding(n_classes, z_sparse)
            self.prior_logvar = nn.Embedding(n_classes, z_sparse)

        def encode(self, x):
            h = self.enc_trunk(x)
            return (self.enc_causal_mu(h), self.enc_causal_logvar(h),
                    self.enc_nuisance_mu(h), self.enc_nuisance_logvar(h))

        def forward(self, x):
            mu_c, lv_c, mu_n, lv_n = self.encode(x)
            z_c = mu_c + torch.exp(0.5 * lv_c) * torch.randn_like(lv_c)
            z_n = mu_n + torch.exp(0.5 * lv_n) * torch.randn_like(lv_n)
            z = torch.cat([z_c, z_n], dim=-1)
            return self.decoder(z), self.classifier(z_c), mu_c, lv_c, mu_n, lv_n

    return PiSAE().to(device)


def train_vae_family(vae, acts, labels, device, n_epochs=300, batch_size=128,
                     lr=1e-3, alpha=10.0, l1_coeff=0.0, use_pi_prior=False):
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    n = len(acts)
    for epoch in tqdm(range(n_epochs), desc=type(vae).__name__, leave=False):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x, y = acts[idx], labels[idx]
            x_r, logits, mu_c, lv_c, mu_n, lv_n = vae(x)
            recon = F.mse_loss(x_r, x)
            if use_pi_prior and hasattr(vae, "prior_mu"):
                p_mu = vae.prior_mu(y)
                p_lv = vae.prior_logvar(y)
                kl_c = -0.5 * (1 + lv_c - p_lv - ((mu_c - p_mu).pow(2) + lv_c.exp()) / p_lv.exp()).mean()
            else:
                kl_c = -0.5 * (1 + lv_c - mu_c.pow(2) - lv_c.exp()).mean()
            kl_n = -0.5 * (1 + lv_n - mu_n.pow(2) - lv_n.exp()).mean()
            ce = F.cross_entropy(logits, y)
            loss = recon + kl_c + kl_n + alpha * ce
            if l1_coeff > 0:
                loss += l1_coeff * mu_c.abs().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    vae.eval()
    return vae


def train_pi_sae_e2e(vae, acts, labels, train_pairs, model, hook_name, device,
                     n_epochs=500, batch_size=256, lr=1e-3, alpha=10.0,
                     l1_coeff=1e-3, beta=1.0, interv_batch=8):
    """Structured pi-SAE with an end-to-end intervention cross-entropy term.

    Ported verbatim from k1_vae_vs_das.py::train_pi_sae_e2e, with this file's key
    names (`source_act`, `src_id` rather than `src_act`, `src_label`). Two
    phases: reconstruction-only warmup, then the intervention term is added to
    the ELBO rather than substituted for it, so the generative terms stay active.

    The intervention is additive, h' = h_base + dec(z_swap) - dec(z_orig), which
    cancels reconstruction error to first order.
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
                    src_act = d["source_act"].unsqueeze(0)
                    mu_c_b, _, mu_n_b, _ = vae.encode(base_act)
                    mu_c_s, _, _, _ = vae.encode(src_act)
                    z_base = torch.cat([mu_c_b, mu_n_b], dim=-1)
                    z_iv = torch.cat([mu_c_s, mu_n_b], dim=-1)
                    h_recon = vae.decoder(z_base)
                    h_iv = vae.decoder(z_iv)
                    delta = (h_iv - h_recon).squeeze(0)

                    def hook_fn(act, hook=None, _d=delta):
                        new = act.clone()
                        new[0, -1, :] = new[0, -1, :] + _d
                        return new

                    iv_logits = model.run_with_hooks(
                        d["base_toks"], fwd_hooks=[(hook_name, hook_fn)]
                    )[0, -1, :]
                    interv_loss = interv_loss - F.log_softmax(iv_logits, dim=-1)[d["src_id"]]
                loss = loss + beta * (interv_loss / len(batch_pairs))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    vae.eval()
    return vae


def eval_vae_iia(vae, model, eval_data, hook_name, device):
    correct = 0
    total = 0
    intervened_acts_list = []
    prob_diffs = []
    logit_diffs = []

    vae.eval()
    with torch.inference_mode():
        for d in eval_data:
            base_act = d["base_act"].unsqueeze(0)
            src_act = d["source_act"].unsqueeze(0)
            mu_c_b, _, mu_n_b, _ = vae.encode(base_act)
            mu_c_s, _, _, _ = vae.encode(src_act)
            z_base = torch.cat([mu_c_b, mu_n_b], dim=-1)
            z_iv = torch.cat([mu_c_s, mu_n_b], dim=-1)
            h_recon = vae.decoder(z_base).squeeze(0)
            h_iv = vae.decoder(z_iv).squeeze(0)
            intervened_acts_list.append(h_iv.clone())
            delta = h_iv - h_recon

            def make_hook(_d):
                def hk(act, hook):
                    new = act.clone()
                    new[0, -1, :] += _d
                    return new
                return hk

            logits = model.run_with_hooks(
                d["base_toks"], fwd_hooks=[(hook_name, make_hook(delta))]
            )
            iv_logits = logits[0, -1]
            correct += int(iv_logits.argmax().item() == d["src_id"])
            total += 1
            s_l = iv_logits[d["src_id"]].item()
            b_l = iv_logits[d["base_id"]].item()
            logit_diffs.append(s_l - b_l)
            probs = torch.softmax(iv_logits, dim=-1)
            prob_diffs.append(probs[d["src_id"]].item() - probs[d["base_id"]].item())

    natural_acts = torch.stack([d["source_act"] for d in eval_data])
    intervened_acts = torch.stack(intervened_acts_list)
    div_ratio = compute_diversity_ratio(intervened_acts, natural_acts)
    _labels = [d["src_id"] for d in eval_data]
    grouped = compute_diversity_grouped(intervened_acts, natural_acts, _labels)
    _persist_acts(intervened_acts, natural_acts, _labels)

    return {
        "iia": correct / total if total else 0.0,
        "n_eval": total,
        "mean_prob_diff": sum(prob_diffs) / len(prob_diffs) if prob_diffs else None,
        "mean_logit_diff": sum(logit_diffs) / len(logit_diffs) if logit_diffs else None,
        "diversity_ratio": div_ratio,
        **grouped,
    }


# ===================================================================
# Shared linear intervention eval
# ===================================================================

def _eval_linear(model, eval_data, proj, hook_name, device, method_name):
    if not eval_data:
        return {"method": method_name, "iia": None, "n_eval": 0, "diversity_ratio": None}

    correct = 0
    total = 0
    intervened_acts_list = []
    prob_diffs = []
    logit_diffs = []

    for d in eval_data:
        intervention = proj @ (d["source_act"] - d["base_act"])
        iv_act = d["base_act"] + intervention
        intervened_acts_list.append(iv_act.clone())

        def make_hook(_iv):
            def hk(act, hook):
                new = act.clone()
                new[0, -1, :] += _iv
                return new
            return hk

        with torch.no_grad():
            logits = model.run_with_hooks(
                d["base_toks"], fwd_hooks=[(hook_name, make_hook(intervention))]
            )
        iv_logits = logits[0, -1]
        correct += int(iv_logits.argmax().item() == d["src_id"])
        total += 1
        s_l = iv_logits[d["src_id"]].item()
        b_l = iv_logits[d["base_id"]].item()
        logit_diffs.append(s_l - b_l)
        probs = torch.softmax(iv_logits, dim=-1)
        prob_diffs.append(probs[d["src_id"]].item() - probs[d["base_id"]].item())

    natural_acts = torch.stack([d["source_act"] for d in eval_data])
    intervened_acts = torch.stack(intervened_acts_list)
    div_ratio = compute_diversity_ratio(intervened_acts, natural_acts)
    _labels = [d["src_id"] for d in eval_data]
    grouped = compute_diversity_grouped(intervened_acts, natural_acts, _labels)
    _persist_acts(intervened_acts, natural_acts, _labels)

    return {
        "method": method_name,
        "iia": correct / total if total else 0.0,
        "n_eval": total,
        "mean_prob_diff": sum(prob_diffs) / len(prob_diffs) if prob_diffs else None,
        "mean_logit_diff": sum(logit_diffs) / len(logit_diffs) if logit_diffs else None,
        "diversity_ratio": div_ratio,
        **grouped,
    }


# ===================================================================
# Main comparison runner
# ===================================================================

def run_comparison(model, pairs, all_acts, labels_for_vae, d_model, n_classes,
                   hook_name, device, k=1, vae_epochs=300, das_steps=300,
                   nldas_steps=200, task=None):
    if not pairs:
        log_msg("  No pairs — skipping all methods")
        return {"methods": {}, "skipped": True}

    n_train = int(0.7 * len(pairs))
    train_data = pairs[:n_train]
    eval_data = pairs[n_train:]
    log_msg(f"  n_train={len(train_data)}  n_eval={len(eval_data)}")

    if not eval_data:
        log_msg("  No eval pairs — skipping all methods")
        return {"methods": {}, "skipped": True}

    results = {"methods": {}}

    # Random baseline (noise floor), estimated with an interval.
    # Every vacuity threshold in PREREGISTRATION_RECONSTRUCTION_CRITERION.md is
    # defined as an excess over this floor's UPPER bound, so a single draw is not
    # enough: it gives a point estimate with no sampling spread behind it.
    floor_draws = []
    for _ in range(FLOOR_DRAWS):
        random_U, _ = torch.linalg.qr(torch.randn(d_model, k, device=device))
        rd = _eval_linear(model, eval_data, random_U @ random_U.T, hook_name,
                          device, "random")
        if rd["iia"] is not None:
            floor_draws.append(rd["iia"])
    floor_mean, floor_hw = mean_ci95(floor_draws) if floor_draws else (None, None)
    r = dict(rd)
    r["iia"] = floor_mean
    r["iia_ci95_halfwidth"] = floor_hw
    r["iia_upper"] = None if floor_mean is None else floor_mean + floor_hw
    r["iia_draws"] = floor_draws
    r["n_draws"] = len(floor_draws)
    results["methods"]["random"] = r
    _log_result("random", r)
    if floor_mean is not None:
        log_msg(f"    floor over {len(floor_draws)} draws: {floor_mean:.4f} "
                f"+/-{floor_hw:.4f} (upper={floor_mean + floor_hw:.4f})")

    # 1. Delta-PCA
    log_msg("  Running delta-PCA...")
    r = run_delta_pca(train_data, eval_data, model, hook_name, device, k)
    results["methods"]["delta_pca"] = r
    _log_result("delta_pca", r)

    # 2. DAS (delta-PCA warm start -- see the note above train_das_reference;
    #    this is the warm-started variant, not standard DAS)
    log_msg(f"  Running DAS/delta-PCA-init ({das_steps} steps)...")
    r = run_das(train_data, eval_data, model, hook_name, device, k, n_steps=das_steps)
    r["init"] = "delta_pca_warm_start"
    results["methods"]["das"] = r
    _log_result("das", r)

    # 2b. DAS under the reference implementation (random orthogonal init,
    #     orthogonal parametrisation), same run and same data as 2.
    log_msg(f"  Running DAS/reference ({das_steps} steps)...")
    r_ref = run_das_reference(train_data, eval_data, model, hook_name, device, k,
                              task=task)
    results["das_reference_config"] = das.mib_config(task) if das is not None else {}
    r_ref["init"] = "random_orthogonal"
    results["methods"]["das_reference"] = r_ref
    _log_result("das_reference", r_ref)
    if r_ref["iia"] is not None and r["iia"] is not None:
        results["das_init_delta"] = r_ref["iia"] - r["iia"]
        log_msg(f"    delta vs delta-PCA-init DAS: {r_ref['iia'] - r['iia']:+.4f}")

    # 3. NL-DAS
    log_msg(f"  Running NL-DAS ({nldas_steps} steps)...")
    r = run_nldas(train_data, eval_data, model, hook_name, device, k, n_steps=nldas_steps)
    results["methods"]["nldas"] = r
    _log_result("nldas", r)

    # 3b. NL-DAS + reconstruction penalty
    log_msg(f"  Running NL-DAS+recon ({nldas_steps} steps)...")
    r = run_nldas(train_data, eval_data, model, hook_name, device, k,
                  n_steps=nldas_steps, recon_weight=1.0, recon_acts=all_acts)
    results["methods"]["nldas_recon"] = r
    _log_result("nldas_recon", r)

    # VAE family setup
    z_causal = max(k, 1)
    z_nuisance = max(k * 4, 4)
    hidden = min(256, max(128, d_model))

    # 4. Structured VAE
    log_msg(f"  Running Structured VAE ({vae_epochs} epochs)...")
    vae = _build_structured_vae(d_model, z_causal, z_nuisance, hidden, n_classes, device)
    train_vae_family(vae, all_acts, labels_for_vae, device, n_epochs=vae_epochs)
    r = eval_vae_iia(vae, model, eval_data, hook_name, device)
    r["method"] = "structured_vae"
    results["methods"]["structured_vae"] = r
    _log_result("structured_vae", r)
    del vae

    # 5. pi-VAE
    log_msg(f"  Running pi-VAE ({vae_epochs} epochs)...")
    pv = _build_pi_vae(d_model, z_causal, z_nuisance, hidden, n_classes, device)
    train_vae_family(pv, all_acts, labels_for_vae, device, n_epochs=vae_epochs, use_pi_prior=True)
    r = eval_vae_iia(pv, model, eval_data, hook_name, device)
    r["method"] = "pi_vae"
    results["methods"]["pi_vae"] = r
    _log_result("pi_vae", r)
    del pv

    # 6. pi-SAE
    log_msg(f"  Running pi-SAE ({vae_epochs} epochs)...")
    ps = _build_pi_sae(d_model, z_causal, z_nuisance, hidden, n_classes, device)
    train_vae_family(ps, all_acts, labels_for_vae, device, n_epochs=vae_epochs,
                     use_pi_prior=True, l1_coeff=1e-3)
    r = eval_vae_iia(ps, model, eval_data, hook_name, device)
    r["method"] = "pi_sae"
    results["methods"]["pi_sae"] = r
    _log_result("pi_sae", r)
    del ps

    # 7. pi-SAE with end-to-end intervention training.
    #
    # This is the arm that produces every headline number in the manuscript's
    # six-task table, and until now it was absent from this control: no script
    # in the repository contained both `e2e` and `random_init`. The control
    # therefore tested a different method from the one the paper reports.
    #
    # It matters most here because the end-to-end objective is the same one
    # unconstrained nonlinear DAS optimises. The paper's argument is that
    # constraints rather than objectives separate the two, and that argument is
    # untested precisely where it carries weight.
    log_msg(f"  Running pi-SAE-e2e ({vae_epochs} epochs)...")
    pse = _build_pi_sae(d_model, z_causal, z_nuisance, hidden, n_classes, device)
    train_pi_sae_e2e(pse, all_acts, labels_for_vae, train_data, model, hook_name,
                     device, n_epochs=vae_epochs)
    r = eval_vae_iia(pse, model, eval_data, hook_name, device)
    r["method"] = "pi_sae_e2e"
    results["methods"]["pi_sae_e2e"] = r
    _log_result("pi_sae_e2e", r)
    del pse

    torch.cuda.empty_cache()
    return results


def _log_result(name, r):
    iia_s = f"{r['iia']:.3f}" if r.get("iia") is not None else "N/A"
    div_s = f"{r['diversity_ratio']:.3f}" if r.get("diversity_ratio") is not None else "N/A"
    pd_s = f"{r['mean_prob_diff']:.3f}" if r.get("mean_prob_diff") is not None else "N/A"
    ld_s = f"{r['mean_logit_diff']:.2f}" if r.get("mean_logit_diff") is not None else "N/A"
    log_msg(f"    {name}: IIA={iia_s}  prob_diff={pd_s}  logit_diff={ld_s}  div_ratio={div_s}  n={r.get('n_eval', 0)}")


# ===================================================================
# Task runners
# ===================================================================

def run_grokking_task(operation, device, k=1):
    p = OP_MODULUS.get(operation, P)
    n_epochs = GROK_EPOCHS.get(operation, 25000)
    log_msg(f"=== Grokking: {operation} (p={p}, epochs={n_epochs}) ===")

    model, dataset, labels, test_idx, test_acc = train_grokking_model(
        operation, p, device, n_epochs
    )
    grokked = test_acc > 0.95
    log_msg(f"  test_acc={test_acc:.4f}  grokked={grokked}")

    if not grokked:
        log_msg(f"  Model did not grok — skipping {operation}")
        del model
        torch.cuda.empty_cache()
        return {"task": operation, "grokked": False, "test_accuracy": test_acc, "k": k,
                "methods": {}, "skipped": True}

    hook_name = "blocks.0.hook_resid_post"
    d_model = 128

    pairs, activations = generate_grok_pairs(model, dataset, labels, test_idx, hook_name, device)
    log_msg(f"  {len(pairs)} intervention pairs")

    # Build VAE training data from all test activations
    test_acts = activations[test_idx]
    test_labels = labels[test_idx]
    label_map = {}
    vae_acts, vae_labels = [], []
    for i in range(len(test_idx)):
        lid = test_labels[i].item()
        if lid not in label_map:
            label_map[lid] = len(label_map)
        vae_acts.append(test_acts[i])
        vae_labels.append(label_map[lid])
    act_t = torch.stack(vae_acts)
    lab_t = torch.tensor(vae_labels, device=device)
    n_classes = len(label_map)

    results = run_comparison(
        model, pairs, act_t, lab_t, d_model, n_classes,
        hook_name, device, k=k, vae_epochs=500, das_steps=300,
        nldas_steps=nldas_steps, task=operation,
    )
    results["task"] = operation
    results["grokked"] = grokked
    results["test_accuracy"] = test_acc
    results["k"] = k

    del model, activations
    torch.cuda.empty_cache()
    return results


def randomize_weights_(model):
    """Re-initialise every parameter in place, preserving config and tokenizer."""
    std = model.cfg.initializer_range
    with torch.no_grad():
        for name, prm in model.named_parameters():
            if prm.dim() > 1:
                nn.init.normal_(prm, mean=0.0, std=std)
            else:
                nn.init.zeros_(prm)
    return model


def run_ioi_task(device, k=1, random_init=False, nldas_steps=200, map_seed=0):
    log_msg(f"=== IOI (GPT-2) | random_init={random_init} | map_seed={map_seed} ===")
    # Seeds alignment-map init/training only. Pair generation keeps its own
    # fixed rng (random.Random(42)), so replicates differ in the map, not the data.
    torch.manual_seed(map_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(map_seed)
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    if random_init:
        randomize_weights_(model)
        log_msg("  weights re-initialised: this model cannot do the task")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    hook_name = "blocks.10.hook_resid_post"
    d_model = 768

    pairs, _ = generate_ioi_pairs(model, device, max_pairs=600,
                                  require_correct=not random_init)
    log_msg(f"  {len(pairs)} both-correct pairs")

    # Build VAE training data
    n_train = int(0.7 * len(pairs))
    train_pairs = pairs[:n_train]
    label_map = {}
    vae_acts, vae_labels = [], []
    for d in train_pairs:
        for rk, lk in [("base_act", "base_id"), ("source_act", "src_id")]:
            tid = d[lk]
            if tid not in label_map:
                label_map[tid] = len(label_map)
            vae_acts.append(d[rk])
            vae_labels.append(label_map[tid])
    act_t = torch.stack(vae_acts)
    lab_t = torch.tensor(vae_labels, device=device)
    n_classes = len(label_map)
    log_msg(f"  VAE: {len(vae_acts)} acts, {n_classes} classes")

    results = run_comparison(
        model, pairs, act_t, lab_t, d_model, n_classes,
        hook_name, device, k=k, vae_epochs=500, das_steps=300,
        nldas_steps=nldas_steps, task="ioi",
    )
    results["task"] = "ioi"
    results["k"] = k

    del model
    torch.cuda.empty_cache()
    return results


# ===================================================================
# Modal entry point
# ===================================================================

@app.function(
    gpu="A100",
    timeout=86400,
    volumes={"/results": results_vol},
)
def main(tasks: list[str] = None, k: int = 1, random_init: bool = False, nldas_steps: int = 200, map_seed: int = 0):
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_msg(f"Device: {device}")

    if tasks is None:
        tasks = ["ioi"]

    all_results = {}
    tag = ("random_init" if random_init else "pretrained") + f"_nldas{nldas_steps}_seed{map_seed}"
    out_dir = f"/results/random_network_control/{tag}/k{k}"
    os.makedirs(out_dir, exist_ok=True)

    for task in tasks:
        t0 = time.time()
        if task == "ioi":
            result = run_ioi_task(device, k=k, random_init=random_init, nldas_steps=nldas_steps, map_seed=map_seed)
        elif task in GROK_EPOCHS:
            result = run_grokking_task(task, device, k=k)
        else:
            log_msg(f"Unknown task: {task}, skipping")
            continue

        elapsed = time.time() - t0
        log_msg(f"  {task} done in {elapsed:.1f}s")
        all_results[task] = result

        # Save incremental
        out_path = os.path.join(out_dir, f"{task}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        results_vol.commit()
        log_msg(f"  Saved: {out_path}")

    # Print summary table (same format as phonetic-circuits)
    log_msg(f"\n{'='*80}")
    log_msg("=== METHOD COMPARISON SUMMARY ===")
    log_msg(f"k={k}")
    log_msg(f"{'='*80}")
    for task, r in all_results.items():
        methods_str = []
        for m in ["delta_pca", "das", "das_reference", "nldas", "nldas_recon",
                  "structured_vae", "pi_vae", "pi_sae", "pi_sae_e2e"]:
            v = r.get("methods", {}).get(m, {})
            iia = f"{v['iia']:.2f}" if v.get("iia") is not None else "N/A"
            pd = f"{v['mean_prob_diff']:.2f}" if v.get("mean_prob_diff") is not None else "?"
            ld = f"{v['mean_logit_diff']:.1f}" if v.get("mean_logit_diff") is not None else "?"
            div = f"{v.get('diversity_ratio', 0):.2f}" if v.get("diversity_ratio") is not None else "?"
            methods_str.append(f"{m}={iia}(pd={pd},ld={ld},dr={div})")
        grok_info = ""
        if "grokked" in r:
            grok_info = f"  grokked={r['grokked']}  acc={r.get('test_accuracy', 0):.3f}"
        log_msg(f"  {task}: {' | '.join(methods_str)}{grok_info}")

    return all_results


@app.local_entrypoint()
def cli(tasks: str = "ioi", k: int = 1, random_init: bool = False, nldas_steps: int = 200, map_seed: int = 0):
    task_list = [t.strip() for t in tasks.split(",")]
    result = main.remote(tasks=task_list, k=k, random_init=random_init, nldas_steps=nldas_steps, map_seed=map_seed)
    print(json.dumps(result, indent=2, default=str))
