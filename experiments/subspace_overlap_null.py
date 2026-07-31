#!/usr/bin/env python3
"""Empirical null for the subspace-overlap statistic used in the degeneracy claim.

The claim is that DAS subspaces recovered from independently trained grokked
models have pairwise overlap ~0.008, *below* the chance value k/d = 2/128 =
0.0156, i.e. they are anti-aligned. That comparison uses an analytic expectation.

"Below chance" is exactly the kind of result that can come from bias in the
estimator rather than from real anti-alignment, so the analytic value is not
sufficient. This samples random orthonormal k-frames and pushes them through the
*same* subspace_overlap implementation the claim uses, giving an empirical null
with a confidence interval.

If the empirical null sits at ~0.008 rather than ~0.0156, the sub-chance finding
dissolves and the paper must not claim it.

Usage:  uv run python experiments/subspace_overlap_null.py
"""

import statistics

import torch


def subspace_overlap(Q1, Q2):
    """Mean squared cosine of principal angles. Identical to the version used
    in experiments/degeneracy_decomposition.py."""
    svals = torch.linalg.svdvals(Q1.T @ Q2).clamp(-1.0, 1.0)
    return (svals ** 2).mean().item()


def random_frame(d, k, generator=None):
    """Uniform random k-dimensional subspace of R^d (Haar, via QR of a Gaussian)."""
    A = torch.randn(d, k, generator=generator)
    Q, _ = torch.linalg.qr(A)
    return Q


def null_distribution(d, k, n_pairs=20000, seed=0):
    g = torch.Generator().manual_seed(seed)
    vals = [subspace_overlap(random_frame(d, k, g), random_frame(d, k, g))
            for _ in range(n_pairs)]
    m = statistics.mean(vals)
    sd = statistics.stdev(vals)
    se = sd / (n_pairs ** 0.5)
    vals.sort()
    return {
        "mean": m,
        "sd": sd,
        "ci95": (m - 1.96 * se, m + 1.96 * se),
        "p02_5": vals[int(0.025 * n_pairs)],
        "p97_5": vals[int(0.975 * n_pairs)],
        "min": vals[0],
        "analytic_k_over_d": k / d,
        "n_pairs": n_pairs,
    }


def main():
    observed = 0.008  # reported pairwise overlap across ten grokked models
    for d, k in ((128, 2), (128, 4), (768, 1)):
        r = null_distribution(d, k)
        print(f"d={d} k={k}  n={r['n_pairs']}")
        print(f"  analytic k/d            {r['analytic_k_over_d']:.5f}")
        print(f"  empirical mean          {r['mean']:.5f}  "
              f"(95% CI {r['ci95'][0]:.5f}, {r['ci95'][1]:.5f})")
        print(f"  empirical sd            {r['sd']:.5f}")
        print(f"  central 95% of nulls    [{r['p02_5']:.5f}, {r['p97_5']:.5f}]")
        print(f"  smallest null seen      {r['min']:.5f}")
        if (d, k) == (128, 2):
            below = "YES" if observed < r["p02_5"] else "NO"
            print(f"  observed {observed} below the 2.5th percentile? {below}")
            print(f"  -> sub-chance claim {'survives' if below == 'YES' else 'DISSOLVES'}")
        print()


if __name__ == "__main__":
    main()
