"""Pilot gate for SPEC_VARY_ONE_GROKKING.md §3. Runs on CPU in minutes.

Three checks, all of which must pass before any compute is committed:

  1. design A recovers a subspace above the empirical null;
  2. design C also does, AND is a DIFFERENT subspace from A -- if they coincide,
     the designs are not separating anything and the result-subspace construction
     is void;
  3. (S_A intersect S_B) is non-trivial, so there is something for design C to be
     subtracted from.

Uses the cached grokked addition model rather than training one. All three
designs are different groupings of the SAME (a, b) activation grid, so the grid
is computed once.

Run: uv run python experiments/pilot_vary_one_grokking.py
"""

from __future__ import annotations

import json
import os

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

import vary_one as vo

P = 113
HOOK = "blocks.0.hook_resid_post"
CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "results", "holonomy_analysis_v4", "addition")
K_VALUES = (2, 4, 8, 16)


def load_grokked_addition(device="cpu"):
    """Load the cached grokked addition model and confirm it actually grokked."""
    meta = json.load(open(os.path.join(CKPT, "result.json")))
    assert meta["operation"] == "addition", meta["operation"]
    assert meta["grokked"] is True, "cached model did not grok"
    assert meta["p"] == P, (meta["p"], P)

    cfg = HookedTransformerConfig(
        n_layers=1, n_heads=4, d_model=128, d_head=32, d_mlp=512,
        act_fn="relu", normalization_type=None,
        d_vocab=P + 1, d_vocab_out=P, n_ctx=3,
        init_weights=True, device=device,
    )
    model = HookedTransformer(cfg)
    model.load_state_dict(torch.load(os.path.join(CKPT, "grokking_model.pt"),
                                     map_location=device))
    model.eval()
    return model, meta


def activation_grid(model, device="cpu", batch=512):
    """(P, P, d) activations at HOOK, indexed [a, b], for every input pair."""
    a_idx = torch.arange(P).repeat_interleave(P)
    b_idx = torch.arange(P).repeat(P)
    toks = torch.stack([a_idx, b_idx, torch.full_like(a_idx, P)], dim=1).to(device)

    outs, correct = [], 0
    for i in range(0, len(toks), batch):
        chunk = toks[i:i + batch]
        with torch.inference_mode():
            logits, cache = model.run_with_cache(chunk, names_filter=[HOOK])
        outs.append(cache[HOOK][:, -1, :].clone())
        target = (chunk[:, 0] + chunk[:, 1]) % P
        correct += (logits[:, -1].argmax(-1) == target).sum().item()
    acts = torch.cat(outs, dim=0).reshape(P, P, -1).double()
    return acts, correct / len(toks)


def configs_from_grid(acts, design):
    """Group the activation grid the way each design requires."""
    if design == "A":                      # fix b, vary a
        return [acts[:, b0, :] for b0 in range(P)]
    if design == "B":                      # fix a, vary b
        return [acts[a0, :, :] for a0 in range(P)]
    if design == "C":                      # fix a + b = s, vary a
        a = torch.arange(P)
        return [acts[a, (s - a) % P, :] for s in range(P)]
    raise ValueError(design)


def main():
    print("Loading cached grokked addition model ...")
    model, meta = load_grokked_addition()
    print(f"  grokked={meta['grokked']}  cached test acc={meta['test_accuracy']:.4f}")

    acts, acc = activation_grid(model)
    d = acts.shape[-1]
    print(f"  activation grid {tuple(acts.shape)} at {HOOK}")
    print(f"  full-grid accuracy recomputed here: {acc:.4f}")
    assert acc > 0.95, f"loaded model does not solve the task ({acc:.3f}); bad checkpoint?"

    print(f"\n{'k':>3}  {'null p97.5':>10}  {'ovl(A,C)':>9}  {'dGr(A,C)':>9}  "
          f"{'dim(A^B)':>9}  {'dim result':>10}  {'evr A (top4)':>26}")
    rows = []
    for k in K_VALUES:
        null = vo.overlap_null(d, k, n_pairs=800)

        Q_A, evr_A = vo.vary_one_pca(configs_from_grid(acts, "A"), k)
        Q_B, _ = vo.vary_one_pca(configs_from_grid(acts, "B"), k)
        Q_C, _ = vo.vary_one_pca(configs_from_grid(acts, "C"), k)

        ovl_AC = vo.subspace_overlap(Q_A, Q_C)
        dgr_AC = vo.grassmann_distance(Q_A, Q_C)
        inter = vo.subspace_intersection(Q_A, Q_B)
        S_r = vo.result_subspace(Q_A, Q_B, Q_C)
        evr_str = " ".join(f"{v:.3f}" for v in evr_A[:4].tolist())

        print(f"{k:>3}  {null['p97_5']:>10.4f}  {ovl_AC:>9.4f}  {dgr_AC:>9.4f}  "
              f"{inter.shape[1]:>9}  {S_r.shape[1]:>10}  {evr_str:>26}")
        rows.append({
            "k": k, "null_p97_5": null["p97_5"], "null_mean": null["mean"],
            "overlap_A_C": ovl_AC, "dgr_A_C": dgr_AC,
            "dim_intersection_AB": int(inter.shape[1]),
            "dim_result_subspace": int(S_r.shape[1]),
            "evr_A_top4": evr_A[:4].tolist(),
        })

    print("\n--- pilot gates (SPEC_VARY_ONE_GROKKING.md §3) ---")
    # Compared against isotropic noise, where top-4 EVR would be 4/d, rather than
    # against an absolute cut. An absolute cut would fail whenever the true
    # dimensionality exceeds four -- which the Fourier account predicts it does,
    # at roughly 2F dimensions for F key frequencies -- and that is not a reason
    # to abandon the method.
    iso = 4.0 / d
    g1 = all(sum(r["evr_A_top4"]) > 10 * iso for r in rows)
    print(f"  1. design A variance is anisotropic (top-4 EVR > 10x isotropic "
          f"{iso:.4f}): {'PASS' if g1 else 'FAIL'}")

    g2 = all(r["overlap_A_C"] > r["null_p97_5"] and r["overlap_A_C"] < 0.95 for r in rows)
    print(f"  2. A and C overlap above null but are not the same subspace: "
          f"{'PASS' if g2 else 'FAIL'}")

    g3 = any(r["dim_intersection_AB"] > 0 for r in rows)
    print(f"  3. S_A intersect S_B is non-trivial at some k: "
          f"{'PASS' if g3 else 'FAIL'}")

    g4 = any(r["dim_result_subspace"] > 0 for r in rows)
    print(f"  (bonus) result subspace non-empty at some k: "
          f"{'PASS' if g4 else 'FAIL'}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "pilot_vary_one_grokking.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"hook": HOOK, "p": P, "d_model": d, "grid_accuracy": acc,
               "gates": {"variance_concentrates": g1, "A_and_C_differ": g2,
                         "intersection_nonempty": g3, "result_nonempty": g4},
               "rows": rows}, open(out, "w"), indent=2)
    print(f"\nsaved -> {out}")

    print("\nVERDICT:", "PROCEED" if (g1 and g2 and g3) else
          "STOP - revise the spec before spending compute")


if __name__ == "__main__":
    main()
