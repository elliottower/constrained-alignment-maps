"""2x2x2 ablation on RAVEL: structure x pi x sparsity.

8 methods testing which components of Structured pi-SAE matter:
  {flat, structured} x {standard prior, pi (label-conditional)} x {VAE, SAE}

Each method trains with E2E intervention loss (after warmup) and is evaluated
on RAVEL cause/isolate/disentangle scores.

Methods:
  vae                 flat VAE, standard N(0,I) prior
  sae                 flat overcomplete + L1, standard prior
  pi_vae              flat VAE, label-conditional prior + classifier
  pi_sae              flat overcomplete + L1, label-conditional prior
  structured_vae      causal/nuisance split, standard prior
  structured_sae      causal/nuisance + overcomplete L1, standard prior
  structured_pi_vae   causal/nuisance split, label-conditional prior
  structured_pi_sae   full model (already running separately)

Usage:
    modal run -d experiments/ravel_pi_sae.py --method vae
    modal run -d experiments/ravel_pi_sae.py --method structured_sae --entity-types city
"""
from __future__ import annotations

import json
import logging
import os
import random
from collections import defaultdict
from datetime import datetime, timezone

import modal

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from tqdm import tqdm
    from transformer_lens import HookedTransformer
    _TORCH_AVAILABLE = True
except (ImportError, AttributeError):
    _TORCH_AVAILABLE = False

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformer_lens", "transformers", "einops",
        "tqdm", "datasets", "matplotlib",
    )
    .add_local_dir(
        "./data/ravel",
        remote_path="/ravel_data",
    )
)
app = modal.App("ravel-ablation")
volume = modal.Volume.from_name("fc-results", create_if_missing=True)

log = logging.getLogger("ravel")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

RAVEL_DATA_DIR = "/ravel_data"

METHODS = {
    "vae":                {"structured": False, "pi": False, "sparse": False},
    "sae":                {"structured": False, "pi": False, "sparse": True},
    "pi_vae":             {"structured": False, "pi": True,  "sparse": False},
    "pi_sae":             {"structured": False, "pi": True,  "sparse": True},
    "structured_vae":     {"structured": True,  "pi": False, "sparse": False},
    "structured_sae":     {"structured": True,  "pi": False, "sparse": True},
    "structured_pi_vae":  {"structured": True,  "pi": True,  "sparse": False},
    "structured_pi_sae":  {"structured": True,  "pi": True,  "sparse": True},
}


def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ===================================================================
# Data loading
# ===================================================================

def load_ravel_entity(entity_type):
    prefix = f"ravel_{entity_type}"
    attrs = json.load(open(os.path.join(RAVEL_DATA_DIR, f"{prefix}_entity_attributes.json")))
    prompts = json.load(open(os.path.join(RAVEL_DATA_DIR, f"{prefix}_attribute_to_prompts.json")))
    splits = json.load(open(os.path.join(RAVEL_DATA_DIR, f"{prefix}_entity_to_split.json")))
    train = [e for e, s in splits.items() if s == "train"]
    test = [e for e, s in splits.items() if s == "test"]
    return attrs, prompts, train, test


def build_pairs(attrs, entities, attribute, prompt_template, tokenizer,
                n_pairs=500, seed=42):
    rng = random.Random(seed)
    by_value = {}
    for ent in entities:
        if attribute not in attrs.get(ent, {}):
            continue
        by_value.setdefault(attrs[ent][attribute], []).append(ent)
    values_ok = [v for v, es in by_value.items() if len(es) >= 2]
    if len(values_ok) < 2:
        return []
    pairs = []
    for _ in range(n_pairs * 20):
        if len(pairs) >= n_pairs:
            break
        v1, v2 = rng.sample(values_ok, 2)
        base_ent, src_ent = rng.choice(by_value[v1]), rng.choice(by_value[v2])
        base_val = attrs[base_ent][attribute]
        src_val = attrs[src_ent][attribute]
        base_tids = tokenizer.encode(" " + base_val, add_special_tokens=False)
        src_tids = tokenizer.encode(" " + src_val, add_special_tokens=False)
        if not base_tids or not src_tids:
            continue
        pairs.append({
            "base_text": prompt_template % base_ent,
            "source_text": prompt_template % src_ent,
            "base_answer_id": base_tids[0],
            "source_answer_id": src_tids[0],
        })
    return pairs


def collect_resids(model, pairs, layer, device):
    hook_name = f"blocks.{layer}.hook_resid_post"
    data = []
    for p in tqdm(pairs, desc="  resids", leave=False):
        base_toks = model.to_tokens(p["base_text"]).to(device)
        src_toks = model.to_tokens(p["source_text"]).to(device)
        with torch.no_grad():
            _, bc = model.run_with_cache(base_toks, names_filter=[hook_name])
            _, sc = model.run_with_cache(src_toks, names_filter=[hook_name])
        data.append({
            "base_resid": bc[hook_name][0, -1, :].clone(),
            "source_resid": sc[hook_name][0, -1, :].clone(),
            "base_toks": base_toks,
            "src_id": p["source_answer_id"],
            "base_id": p["base_answer_id"],
        })
    return data


# ===================================================================
# Unified model: {structured, flat} x {pi, standard} x {SAE, VAE}
# ===================================================================

_BaseModule = nn.Module if _TORCH_AVAILABLE else object


class UnifiedModel(_BaseModule):
    def __init__(self, d_input, k, n_nuisance, hidden_dim, n_classes,
                 expansion, structured, pi, sparse):
        super().__init__()
        self.structured = structured
        self.pi = pi
        self.sparse = sparse

        self.z_c_dim = k * expansion if sparse else k
        self.z_n_dim = n_nuisance if structured else 0
        z_total = self.z_c_dim + self.z_n_dim

        self.enc_trunk = nn.Sequential(
            nn.Linear(d_input, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.enc_c_mu = nn.Linear(hidden_dim, self.z_c_dim)
        self.enc_c_logvar = nn.Linear(hidden_dim, self.z_c_dim)
        if structured:
            self.enc_n_mu = nn.Linear(hidden_dim, self.z_n_dim)
            self.enc_n_logvar = nn.Linear(hidden_dim, self.z_n_dim)

        self.decoder = nn.Sequential(
            nn.Linear(z_total, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, d_input),
        )

        if pi:
            self.classifier = nn.Linear(self.z_c_dim, n_classes)
            self.prior_mu = nn.Embedding(n_classes, self.z_c_dim)
            self.prior_logvar = nn.Embedding(n_classes, self.z_c_dim)

    def encode(self, x):
        h = self.enc_trunk(x)
        mu_c = self.enc_c_mu(h)
        lv_c = self.enc_c_logvar(h)
        if self.structured:
            mu_n = self.enc_n_mu(h)
            lv_n = self.enc_n_logvar(h)
        else:
            mu_n = lv_n = None
        return mu_c, lv_c, mu_n, lv_n

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

    def decode_z(self, z_c, z_n):
        if self.structured:
            z = torch.cat([z_c, z_n], dim=-1)
        else:
            z = z_c
        return self.decoder(z)

    def forward(self, x):
        mu_c, lv_c, mu_n, lv_n = self.encode(x)
        z_c = self.reparameterize(mu_c, lv_c)
        z_n = self.reparameterize(mu_n, lv_n) if self.structured else None
        x_recon = self.decode_z(z_c, z_n)
        logits = self.classifier(z_c) if self.pi else None
        return x_recon, logits, mu_c, lv_c, mu_n, lv_n

    def get_causal_basis(self):
        """Extract orthonormal basis for causal subspace from encoder.

        Computes SVD of the encoder's causal head Jacobian (through the trunk)
        evaluated at a batch of points, then returns the top-k right singular vectors
        as the subspace basis in input space.
        """
        # Use encoder weight matrices to get a linear approximation
        # For a linear encoder this would be exact; for ReLU it's an approximation
        # that captures the average subspace the encoder projects onto
        W1 = self.enc_trunk[0].weight  # (hidden, d_input)
        W2 = self.enc_trunk[2].weight  # (hidden, hidden)
        W_c = self.enc_c_mu.weight     # (z_c_dim, hidden)
        # Jacobian of mu_c w.r.t. input (ignoring ReLU): W_c @ W2 @ W1
        J = W_c @ W2 @ W1  # (z_c_dim, d_input)
        # SVD to get orthonormal basis in input space
        U, S, Vh = torch.linalg.svd(J, full_matrices=False)
        return Vh  # (z_c_dim, d_input) — rows are basis vectors

    def intervene(self, base_act, src_act):
        """DAS-style intervention using learned causal subspace."""
        diff = (src_act - base_act).squeeze(0)
        V = self.get_causal_basis()  # (z_c_dim, d_input)
        proj = V.T @ V  # (d_input, d_input)
        return (proj @ diff).squeeze(0)

    def resize_classifier(self, n_classes, device):
        if self.pi and n_classes > self.classifier.out_features:
            self.classifier = nn.Linear(self.z_c_dim, n_classes).to(device)
            self.prior_mu = nn.Embedding(n_classes, self.z_c_dim).to(device)
            self.prior_logvar = nn.Embedding(n_classes, self.z_c_dim).to(device)


def build_model(d_input, k, n_nuisance, hidden_dim, n_classes, expansion,
                method_cfg, device):
    return UnifiedModel(
        d_input, k, n_nuisance, hidden_dim, n_classes, expansion,
        structured=method_cfg["structured"],
        pi=method_cfg["pi"],
        sparse=method_cfg["sparse"],
    ).to(device)


# ===================================================================
# Unified training with E2E
# ===================================================================

def _contrastive_loss(mu_c, labels, temperature=0.1):
    """Supervised contrastive on z_c: same-label pairs attract, different repel."""
    z = F.normalize(mu_c, dim=-1)
    sim = z @ z.T / temperature  # (B, B)
    B = sim.size(0)
    self_mask = torch.eye(B, device=sim.device, dtype=torch.bool)
    mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
    if mask_pos.sum() == 0:
        return torch.tensor(0.0, device=mu_c.device)
    exp_sim = torch.exp(sim) * (~self_mask).float()
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    loss = -(log_prob * mask_pos).sum() / mask_pos.sum()
    return loss


def train_model_e2e(mdl, resid_data, model_lm, layer, device,
                    n_epochs=300, lr=1e-3, alpha=10.0, l1_coeff=1e-3,
                    beta=1.0, contrastive_weight=0.5, interv_batch=8):
    hook_name = f"blocks.{layer}.hook_resid_post"

    acts = torch.stack([d["base_resid"] for d in resid_data])
    n = len(acts)

    # Build class labels from answer token IDs
    token_to_class = {}
    class_labels = []
    for d in resid_data:
        for tid in [d["base_id"], d["src_id"]]:
            if tid not in token_to_class:
                token_to_class[tid] = len(token_to_class)
        class_labels.append(token_to_class[d["base_id"]])
    class_labels_t = torch.tensor(class_labels, device=device)
    n_classes = len(token_to_class)

    if mdl.pi:
        mdl.resize_classifier(n_classes, device)

    optimizer = torch.optim.Adam(mdl.parameters(), lr=lr)
    warmup_epochs = min(100, n_epochs // 3)

    for epoch in tqdm(range(n_epochs), desc="train", leave=False):
        perm = torch.randperm(n, device=device)
        bs = min(256, n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            x, y = acts[idx], class_labels_t[idx]
            x_r, logits, mu_c, lv_c, mu_n, lv_n = mdl(x)

            recon = F.mse_loss(x_r, x)

            # KL for causal z
            if mdl.pi:
                prior_mu = mdl.prior_mu(y)
                prior_lv = mdl.prior_logvar(y)
                kl_c = -0.5 * (1 + lv_c - prior_lv
                               - ((mu_c - prior_mu).pow(2) + lv_c.exp()) / prior_lv.exp()).mean()
            else:
                kl_c = -0.5 * (1 + lv_c - mu_c.pow(2) - lv_c.exp()).mean()

            # KL for nuisance z (always standard prior)
            if mdl.structured:
                kl_n = -0.5 * (1 + lv_n - mu_n.pow(2) - lv_n.exp()).mean()
            else:
                kl_n = torch.tensor(0.0, device=device)

            # Classifier loss
            ce = F.cross_entropy(logits, y) if mdl.pi else torch.tensor(0.0, device=device)

            # L1 sparsity
            sparsity = mu_c.abs().mean() if mdl.sparse else torch.tensor(0.0, device=device)

            # Contrastive on causal z
            con = _contrastive_loss(mu_c, y) * contrastive_weight

            loss = recon + kl_c + kl_n + alpha * ce + l1_coeff * sparsity + con

            # E2E intervention loss after warmup (accumulate in small chunks to save VRAM)
            if epoch >= warmup_epochs:
                mb_idx = idx[:interv_batch].cpu().tolist()
                batch_data = [resid_data[j] for j in mb_idx]
                e2e_chunk = 4
                for ci in range(0, len(batch_data), e2e_chunk):
                    chunk = batch_data[ci:ci + e2e_chunk]
                    iv_loss = torch.tensor(0.0, device=device)
                    for d in chunk:
                        delta = mdl.intervene(
                            d["base_resid"].unsqueeze(0),
                            d["source_resid"].unsqueeze(0))

                        def hook_fn(act, hook=None, d=delta):
                            new = act.clone()
                            new[0, -1, :] = new[0, -1, :] + d
                            return new

                        iv_logits = model_lm.run_with_hooks(
                            d["base_toks"], fwd_hooks=[(hook_name, hook_fn)]
                        )[0, -1, :]
                        iv_loss -= F.log_softmax(iv_logits, dim=-1)[d["src_id"]]
                    loss = loss + beta * (iv_loss / len(batch_data))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    mdl.eval()
    return mdl


# ===================================================================
# DAS baseline
# ===================================================================

def train_das_ravel(model, data, layer, device, k=1, n_steps=200, lr=1e-3):
    hook_name = f"blocks.{layer}.hook_resid_post"
    deltas = torch.stack([d["source_resid"] - d["base_resid"] for d in data])
    _, _, Vh = torch.linalg.svd(deltas, full_matrices=False)
    A = torch.nn.Parameter(Vh[:k].T.clone().to(device))
    optimizer = torch.optim.Adam([A], lr=lr)
    n_train = min(len(data), 400)
    accum_bs = 16

    for step in tqdm(range(n_steps), desc=f"DAS k={k}", leave=False):
        optimizer.zero_grad()
        for bi in range(0, n_train, accum_bs):
            chunk = data[bi:min(bi + accum_bs, n_train)]
            chunk_loss = torch.tensor(0.0, device=device)
            Q, _ = torch.linalg.qr(A)
            proj = Q @ Q.T
            for d in chunk:
                iv = proj @ (d["source_resid"] - d["base_resid"])

                def make_hook(_iv):
                    def hk(act, hook):
                        new = act.clone()
                        new[0, -1, :] += _iv
                        return new
                    return hk

                logits = model.run_with_hooks(
                    d["base_toks"], fwd_hooks=[(hook_name, make_hook(iv))])
                chunk_loss -= logits[0, -1, :].log_softmax(dim=-1)[d["src_id"]]
            (chunk_loss / n_train).backward()
        optimizer.step()

    Q, _ = torch.linalg.qr(A.detach())
    return Q


# ===================================================================
# Evaluation
# ===================================================================

def _model_intervene_logits(mdl, model_lm, d, layer):
    hook_name = f"blocks.{layer}.hook_resid_post"
    delta = mdl.intervene(
        d["base_resid"].unsqueeze(0), d["source_resid"].unsqueeze(0))

    def hook_fn(act, hook=None, d=delta):
        new = act.clone()
        new[0, -1, :] = new[0, -1, :] + d
        return new

    return model_lm.run_with_hooks(
        d["base_toks"], fwd_hooks=[(hook_name, hook_fn)])


def _das_intervene_logits(model_lm, d, Q, layer):
    hook_name = f"blocks.{layer}.hook_resid_post"
    proj = Q @ Q.T
    iv = proj @ (d["source_resid"] - d["base_resid"])

    def make_hook(_iv):
        def hk(act, hook):
            new = act.clone()
            new[0, -1, :] += _iv
            return new
        return hk

    return model_lm.run_with_hooks(
        d["base_toks"], fwd_hooks=[(hook_name, make_hook(iv))])


def eval_cause(intervene_fn, data):
    correct = 0
    for d in data:
        with torch.no_grad():
            logits = intervene_fn(d)
        if logits[0, -1, d["src_id"]].item() > logits[0, -1, d["base_id"]].item():
            correct += 1
    return correct / len(data) if data else 0.0


def eval_isolate(intervene_fn, data):
    preserved = 0
    for d in data:
        with torch.no_grad():
            logits = intervene_fn(d)
        if logits[0, -1, d["base_id"]].item() >= logits[0, -1, d["src_id"]].item():
            preserved += 1
    return preserved / len(data) if data else 0.0


def eval_full(intervene_fn, test_target, test_siblings):
    cause = eval_cause(intervene_fn, test_target)
    iso = {}
    for attr, sib_data in test_siblings.items():
        iso[attr] = eval_isolate(intervene_fn, sib_data)
    mean_iso = float(sum(iso.values()) / len(iso)) if iso else 0.0
    return {
        "cause": cause,
        "isolate_per_attr": iso,
        "mean_isolate": mean_iso,
        "disentangle": (cause + mean_iso) / 2,
    }


# ===================================================================
# Per-entity-type runner
# ===================================================================

def run_entity_type(entity_type, method_name, method_cfg, model, tokenizer,
                    layer, device, k=4, n_epochs=300, max_attrs=None,
                    n_train_pairs=500, n_test_pairs=200,
                    contrastive_weight=0.5):
    attrs, prompt_templates, train_ents, test_ents = load_ravel_entity(entity_type)
    all_attrs = list(prompt_templates.keys())
    if max_attrs:
        all_attrs = all_attrs[:max_attrs]
    log.info(f"[{utc_ts()}] {entity_type}: {len(attrs)} entities, "
             f"attrs={all_attrs}, train={len(train_ents)}, test={len(test_ents)}")

    results = {}
    for target_attr in all_attrs:
        log.info(f"[{utc_ts()}] === {method_name} | {entity_type}/{target_attr} ===")
        templates = prompt_templates[target_attr]
        if not templates:
            continue

        train_pairs = build_pairs(attrs, train_ents, target_attr, templates[0],
                                  tokenizer, n_pairs=n_train_pairs)
        test_pairs = build_pairs(attrs, test_ents, target_attr, templates[0],
                                 tokenizer, n_pairs=n_test_pairs, seed=99)
        if len(train_pairs) < 10:
            log.info(f"[{utc_ts()}]   Too few pairs ({len(train_pairs)}), skipping")
            continue

        train_data = collect_resids(model, train_pairs, layer, device)
        test_data = collect_resids(model, test_pairs, layer, device)

        test_siblings = {}
        for sib_attr in all_attrs:
            if sib_attr == target_attr:
                continue
            sib_templates = prompt_templates[sib_attr]
            if not sib_templates:
                continue
            sib_pairs = build_pairs(attrs, test_ents, sib_attr, sib_templates[0],
                                    tokenizer, n_pairs=100, seed=77)
            if sib_pairs:
                test_siblings[sib_attr] = collect_resids(model, sib_pairs, layer, device)

        torch.cuda.empty_cache()

        all_tids = set()
        for d in train_data:
            all_tids.add(d["base_id"])
            all_tids.add(d["src_id"])
        n_classes = max(len(all_tids), 2)

        d_model = model.cfg.d_model
        hidden_dim = min(512, d_model)
        expansion = 8

        # Sanity check: raw swap (full diff) should give high cause
        def _raw_swap_logits(model_lm, d, layer):
            hook_name = f"blocks.{layer}.hook_resid_post"
            delta = d["source_resid"] - d["base_resid"]
            def hook_fn(act, hook=None, _d=delta):
                new = act.clone()
                new[0, -1, :] = new[0, -1, :] + _d
                return new
            return model_lm.run_with_hooks(
                d["base_toks"], fwd_hooks=[(hook_name, hook_fn)])

        raw_fn = lambda d, _lm=model, _l=layer: _raw_swap_logits(_lm, d, _l)
        raw_cause = eval_cause(raw_fn, test_data[:20])
        log.info(f"[{utc_ts()}]   RAW SWAP cause={raw_cause:.3f} (should be ~1.0)")

        # Train the ablation method
        mdl = build_model(d_model, k, 16, hidden_dim, n_classes, expansion,
                          method_cfg, device)
        mdl = train_model_e2e(mdl, train_data, model, layer, device,
                              n_epochs=n_epochs,
                              contrastive_weight=contrastive_weight)

        mdl_fn = lambda d, _m=mdl, _lm=model, _l=layer: _model_intervene_logits(_m, _lm, d, _l)
        mdl_scores = eval_full(mdl_fn, test_data, test_siblings)
        log.info(f"[{utc_ts()}]   {method_name}: cause={mdl_scores['cause']:.3f}, "
                 f"iso={mdl_scores['mean_isolate']:.3f}, "
                 f"dis={mdl_scores['disentangle']:.3f}")

        results[target_attr] = {
            method_name: mdl_scores,
            "n_train": len(train_pairs),
            "n_test": len(test_pairs),
            "n_classes": n_classes,
        }

    if results:
        mdl_dis = [r[method_name]["disentangle"] for r in results.values()]
        return {
            "entity_type": entity_type,
            "method": method_name,
            "method_cfg": method_cfg,
            "n_attributes": len(results),
            "avg_disentangle": sum(mdl_dis) / len(mdl_dis),
            "per_attribute": results,
        }
    return None


# ===================================================================
# Modal remote function
# ===================================================================

@app.function(
    image=image, gpu="A100", timeout=86400,
    volumes={"/results": volume},
)
def run_ravel(method_name, entity_types, layer, k, n_epochs,
              max_attrs, n_train_pairs, n_test_pairs,
              contrastive_weight=0.5):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")

    method_cfg = METHODS[method_name]
    log.info(f"[{utc_ts()}] Method={method_name}, cfg={method_cfg}")
    log.info(f"[{utc_ts()}] Loading GPT-2-small")
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()
    tokenizer = model.tokenizer

    if layer < 0:
        layer = model.cfg.n_layers + layer

    suffix = "_no_con" if contrastive_weight == 0.0 else ""
    out_dir = f"/results/ravel_ablation/{method_name}{suffix}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.json")

    all_results = {}
    for etype in entity_types:
        log.info(f"[{utc_ts()}] ========= {etype} =========")
        result = run_entity_type(etype, method_name, method_cfg, model,
                                 tokenizer, layer, device, k=k, n_epochs=n_epochs,
                                 max_attrs=max_attrs, n_train_pairs=n_train_pairs,
                                 n_test_pairs=n_test_pairs,
                                 contrastive_weight=contrastive_weight)
        if result:
            all_results[etype] = result
            log.info(f"[{utc_ts()}] {etype}: {method_name}={result['avg_disentangle']:.3f}")
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            volume.commit()
            log.info(f"[{utc_ts()}] Incremental save to {out_path}")

    return all_results


@app.local_entrypoint()
def main(method: str = "structured_pi_sae",
         entity_types: str = "city",
         layer: int = 8, k: int = 4, n_epochs: int = 300,
         fast: bool = False, no_contrastive: bool = False):
    contrastive_weight = 0.0 if no_contrastive else 0.5
    if fast:
        n_epochs = 100
        max_attrs = 2
        n_train_pairs = 100
        n_test_pairs = 50
        log.info(f"[{utc_ts()}] FAST MODE: {n_epochs} epochs, {max_attrs} attrs, "
                 f"{n_train_pairs} train pairs, {n_test_pairs} test pairs")
    else:
        max_attrs = None
        n_train_pairs = 500
        n_test_pairs = 200

    log.info(f"[{utc_ts()}] contrastive_weight={contrastive_weight}")

    if method == "all":
        methods_to_run = list(METHODS.keys())
    else:
        methods_to_run = [m.strip() for m in method.split(",")]

    types = [t.strip() for t in entity_types.split(",")]
    for m in methods_to_run:
        if m not in METHODS:
            log.error(f"Unknown method: {m}. Choose from: {list(METHODS.keys())}")
            return
        log.info(f"[{utc_ts()}] Spawning {m} on {types}")
        handle = run_ravel.spawn(m, types, layer, k, n_epochs,
                                 max_attrs, n_train_pairs, n_test_pairs,
                                 contrastive_weight)
        log.info(f"[{utc_ts()}]   -> {handle.object_id}")
