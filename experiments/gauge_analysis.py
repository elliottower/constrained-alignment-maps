"""Does a sparse causal latent select a canonical basis, where a rotation cannot?

Distributed Alignment Search recovers a subspace, and a subspace has no
privileged basis: rotate the coordinates and rotate the weights and the model is
unchanged, so individual directions carry no meaning and only the span does. The
measured consequence in this project is that ten seeds of grokked multiplication
all reach interchange accuracy 0.94-1.00 while their recovered subspaces overlap
pairwise at 0.008, against 0.016 expected for random subspaces. Ten valid
answers, no shared directions, nothing to interpret.

An L1-penalized latent may break that invariance -- gauge fixing is the premise
the sparse-dictionary literature runs on. If it does, the same feature directions
recur across seeds and they are objects one can describe. If it does not, they are
as arbitrary as the rotation's and no feature-level story about them is founded.

Two numbers, because they answer different questions:

  subspace overlap   comparable to the 0.008 figure: do the seeds span the same
                     space at all, regardless of basis
  matched cosine     the gauge question proper: after optimally pairing features
                     between two seeds, do the individual directions agree

High subspace overlap with low matched cosine is the signature of a shared span
with an arbitrary basis, which is exactly DAS's situation and would mean sparsity
bought nothing.

    uv run python experiments/gauge_analysis.py --glob 'results/gauge/*.json'
"""

import argparse
import glob
import itertools
import json

import numpy as np
from scipy.optimize import linear_sum_assignment


def orthonormalize(directions):
    """Rows to an orthonormal basis for their span."""
    q, _ = np.linalg.qr(np.asarray(directions, dtype=np.float64).T)
    return q  # (d, r)


def subspace_overlap(a, b):
    """Normalized squared Frobenius norm of the projector product.

    1.0 when the spans coincide, 0.0 when they are orthogonal, and about r/d for
    two random r-dimensional subspaces of a d-dimensional space -- which is the
    floor any claim of agreement has to clear.
    """
    qa, qb = orthonormalize(a), orthonormalize(b)
    r = min(qa.shape[1], qb.shape[1])
    return float(np.linalg.norm(qa.T @ qb, "fro") ** 2 / r)


def matched_cosine(a, b):
    """Mean |cosine| after optimally pairing features between two seeds.

    Features carry no canonical order, so a permutation is not evidence of
    disagreement; the assignment is solved rather than assumed.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    sim = np.abs(a @ b.T)
    rows, cols = linear_sum_assignment(-sim)
    return float(sim[rows, cols].mean())


def within_span_baseline(directions, n_draws=20, rng=None):
    """Matched cosine for a random basis of the *same* span.

    This is the null the gauge question actually needs. Unrelated directions are
    the wrong control: if two seeds find the same subspace but pick arbitrary
    bases within it -- which is exactly what a rotation does -- matched cosine
    still lands far above the unrelated-directions floor. Only a score above
    *this* baseline shows that individual directions were selected rather than
    the span.
    """
    rng = rng or np.random.default_rng(0)
    a = np.asarray(directions, dtype=np.float64)
    q = orthonormalize(a)  # (d, r)
    r = q.shape[1]
    out = []
    for _ in range(n_draws):
        r1, _ = np.linalg.qr(rng.standard_normal((r, r)))
        r2, _ = np.linalg.qr(rng.standard_normal((r, r)))
        out.append(matched_cosine((q @ r1).T, (q @ r2).T))
    return float(np.mean(out))


def random_baseline(n_features, d_model, n_draws=20, rng=None):
    """What both numbers look like for directions with no structure at all."""
    rng = rng or np.random.default_rng(0)
    subs, cos = [], []
    for _ in range(n_draws):
        a = rng.standard_normal((n_features, d_model))
        b = rng.standard_normal((n_features, d_model))
        subs.append(subspace_overlap(a, b))
        cos.append(matched_cosine(a, b))
    return float(np.mean(subs)), float(np.mean(cos))


def load(paths, arm):
    runs = []
    for path in paths:
        with open(path) as f:
            d = json.load(f)
        m = d.get("methods", {}).get(arm, {})
        if "causal_directions" in m:
            runs.append({"seed": d.get("seed"), "model": d.get("model_key"),
                         "layer": d.get("layer"), "k": d.get("k"),
                         "condition": d.get("condition"),
                         "dirs": m["causal_directions"],
                         "write_rank": m.get("write_rank")})
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--arm", default="lcp_vae")
    args = ap.parse_args()

    runs = load(sorted(glob.glob(args.glob)), args.arm)
    if len(runs) < 2:
        raise SystemExit(f"need at least two seeds with directions, found {len(runs)}")

    groups = {}
    for r in runs:
        groups.setdefault((r["model"], r["layer"], r["k"], r["condition"]), []).append(r)

    for (model, layer, k, cond), rs in sorted(groups.items()):
        if len(rs) < 2:
            continue
        dirs = [np.asarray(r["dirs"], dtype=np.float64) for r in rs]
        n_features, d_model = dirs[0].shape
        subs = [subspace_overlap(a, b) for a, b in itertools.combinations(dirs, 2)]
        cos = [matched_cosine(a, b) for a, b in itertools.combinations(dirs, 2)]
        rand_sub, rand_cos = random_baseline(n_features, d_model)
        span_cos = float(np.mean([within_span_baseline(d) for d in dirs]))

        print(f"\n=== {model} L{layer} k={k} {cond} | {len(rs)} seeds, "
              f"{n_features} features of dim {d_model} ===")
        print(f"  subspace overlap  {np.mean(subs):.3f} +/- {np.std(subs):.3f}"
              f"   (random directions: {rand_sub:.3f})")
        print(f"  matched cosine    {np.mean(cos):.3f} +/- {np.std(cos):.3f}"
              f"   (random directions: {rand_cos:.3f}, "
              f"same span/random basis: {span_cos:.3f})")
        ranks = [r["write_rank"]["participation_ratio"] for r in rs
                 if r.get("write_rank")]
        if ranks:
            print(f"  effective write rank {np.mean(ranks):.2f} +/- {np.std(ranks):.2f}"
                  f"   (nominal k = {k})")
        # The span baseline, not the random one, is what a claim has to clear.
        verdict = ("directions recur across seeds: sparsity selects a basis"
                   if np.mean(cos) > span_cos + 2 * np.std(cos) + 0.05 else
                   "no basis selection beyond a shared span; as arbitrary as a rotation")
        print(f"  -> {verdict}")


if __name__ == "__main__":
    main()
