"""Correctness tests for vary-one PCA and the result-subspace construction.

The load-bearing test plants a known factored representation and checks the
construction from SPEC_VARY_ONE_GROKKING.md recovers the planted result
subspace. Synthetic activations are built as

    h(a, b) = U_a phi(a) + U_b phi(b) + U_r phi((a+b) mod p) + noise

with U_a, U_b, U_r mutually orthogonal and phi the Fourier pair
[cos(2 pi x / p), sin(2 pi x / p)], matching how grokked models are understood to
encode modular arithmetic. Under that construction the three designs have known
answers:

    A (fix b, vary a)          -> U_a (+) U_r    (result is a bijection of a)
    B (fix a, vary b)          -> U_b (+) U_r
    C (fix a+b, vary a)        -> U_a (+) U_b    (result is constant)
    (A cap B) minus C          -> U_r

so the test can only pass if the algebra and the centring are both right.

No seeds are set. Randomised checks run many trials and assert on statistics.

Run: uv run python experiments/test_vary_one.py
"""

from __future__ import annotations

import math

import torch

import vary_one as vo


P = 23
D = 64
DIMS = 2  # each factor occupies a Fourier pair


def _phi(x, p):
    ang = 2 * math.pi * x / p
    return torch.tensor([math.cos(ang), math.sin(ang)], dtype=torch.float64)


def build_synthetic(p=P, d=D, noise=0.02, include_result=True):
    """Planted factored representation with mutually orthogonal factor subspaces."""
    basis = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))[0]
    U_a, U_b, U_r = basis[:, 0:2], basis[:, 2:4], basis[:, 4:6]

    def act(a, b):
        h = U_a @ _phi(a, p) + U_b @ _phi(b, p)
        if include_result:
            h = h + U_r @ _phi((a + b) % p, p)
        return h + noise * torch.randn(d, dtype=torch.float64)

    return act, U_a, U_b, U_r


def run_design(act, design, p, k):
    configs = vo.design_configs(design, p)
    acts_by_config = [torch.stack([act(a, b) for (a, b) in cfg]) for cfg in configs]
    return vo.vary_one_pca(acts_by_config, k)


def test_designs_recover_their_planted_spans(n_trials=5):
    """Each design must recover exactly the union its construction implies."""
    got = {"A": [], "B": [], "C": []}
    leak = []
    for _ in range(n_trials):
        act, U_a, U_b, U_r = build_synthetic()
        Q_A, _ = run_design(act, "A", P, 2 * DIMS)
        Q_B, _ = run_design(act, "B", P, 2 * DIMS)
        Q_C, _ = run_design(act, "C", P, 2 * DIMS)
        got["A"].append(vo.subspace_overlap(Q_A, torch.cat([U_a, U_r], dim=1)))
        got["B"].append(vo.subspace_overlap(Q_B, torch.cat([U_b, U_r], dim=1)))
        got["C"].append(vo.subspace_overlap(Q_C, torch.cat([U_a, U_b], dim=1)))
        # design C pins the result, so it must NOT contain U_r
        leak.append(vo.subspace_overlap(Q_C, U_r))

    for name, expect in (("A", "U_a+U_r"), ("B", "U_b+U_r"), ("C", "U_a+U_b")):
        m = sum(got[name]) / len(got[name])
        print(f"  design {name} overlap with {expect:8s} = {m:.4f}")
        assert m > 0.95, f"design {name} did not recover {expect} (overlap {m:.3f})"

    m_leak = sum(leak) / len(leak)
    print(f"  design C overlap with U_r (result, should be ~0) = {m_leak:.4f}")
    assert m_leak < 0.15, f"design C leaked the result subspace (overlap {m_leak:.3f})"


def test_result_subspace_is_recovered(n_trials=5):
    """The headline construction: (A cap B) minus C must return U_r."""
    overlaps, dims = [], []
    for _ in range(n_trials):
        act, U_a, U_b, U_r = build_synthetic()
        Q_A, _ = run_design(act, "A", P, 2 * DIMS)
        Q_B, _ = run_design(act, "B", P, 2 * DIMS)
        Q_C, _ = run_design(act, "C", P, 2 * DIMS)
        S_r = vo.result_subspace(Q_A, Q_B, Q_C)
        dims.append(S_r.shape[1])
        overlaps.append(vo.subspace_overlap(S_r, U_r) if S_r.shape[1] else 0.0)
    m = sum(overlaps) / len(overlaps)
    print(f"  recovered dim {sum(dims)/len(dims):.1f} (planted {DIMS}), "
          f"overlap with U_r = {m:.4f}")
    assert m > 0.90, f"result subspace not recovered (overlap {m:.3f})"


def test_returns_nothing_when_no_result_is_encoded(n_trials=5):
    """Negative control: with no result term planted, the construction must be empty.

    This is the synthetic analogue of the random-network control (H3). If it
    returned a subspace here, the method would be finding structure that is not
    there and the whole design would be uninformative.
    """
    dims, overlaps = [], []
    for _ in range(n_trials):
        act, U_a, U_b, U_r = build_synthetic(include_result=False)
        Q_A, _ = run_design(act, "A", P, 2 * DIMS)
        Q_B, _ = run_design(act, "B", P, 2 * DIMS)
        Q_C, _ = run_design(act, "C", P, 2 * DIMS)
        S_r = vo.result_subspace(Q_A, Q_B, Q_C)
        dims.append(S_r.shape[1])
        overlaps.append(vo.subspace_overlap(S_r, U_r) if S_r.shape[1] else 0.0)
    md, mo = sum(dims) / len(dims), sum(overlaps) / len(overlaps)
    print(f"  no result planted -> recovered dim {md:.1f}, overlap with U_r {mo:.4f}")
    assert mo < 0.25, f"found a result subspace where none exists (overlap {mo:.3f})"


def test_global_centring_breaks_it(n_trials=5):
    """The within-configuration centring must be load-bearing, or it is not a method.

    Centring globally is the natural mistake. If it recovered the same answer,
    the emphasis the spec places on this step would be unjustified.
    """
    within, glob = [], []
    for _ in range(n_trials):
        act, U_a, U_b, U_r = build_synthetic()
        configs = vo.design_configs("C", P)
        acts = [torch.stack([act(a, b) for (a, b) in cfg]) for cfg in configs]

        Q_within, _ = vo.vary_one_pca(acts, 2 * DIMS)
        pooled = torch.cat(acts, dim=0)
        X = pooled - pooled.mean(dim=0, keepdim=True)
        Q_global = torch.linalg.svd(X.double(), full_matrices=False)[2][:2 * DIMS].T

        target = torch.cat([U_a, U_b], dim=1)
        within.append(vo.subspace_overlap(Q_within, target))
        glob.append(vo.subspace_overlap(Q_global, target))
    mw, mg = sum(within) / len(within), sum(glob) / len(glob)
    print(f"  within-config centring {mw:.4f} vs global centring {mg:.4f}")
    assert mw > 0.95, "within-configuration centring failed on its own design"
    assert mw - mg > 0.05, (
        f"global centring did just as well ({mg:.3f} vs {mw:.3f}); the spec's "
        "emphasis on within-configuration centring would be unwarranted")


def test_overlap_null_is_below_recovered_values():
    """Recovery must be interpretable against the empirical null, not asserted."""
    null = vo.overlap_null(D, DIMS, n_pairs=1000)
    print(f"  null for d={D} k={DIMS}: mean {null['mean']:.4f}, "
          f"97.5th pct {null['p97_5']:.4f} (analytic k/d = {null['analytic_k_over_d']:.4f})")
    assert null["p97_5"] < 0.5, "null is too wide for recovery claims to mean anything"


def test_intersection_and_minus_are_not_vacuous():
    """Guard: the algebra must return empty when it should."""
    d, k = 64, 3
    Q1, Q2 = vo.random_frame(d, k), vo.random_frame(d, k)
    assert vo.subspace_intersection(Q1, Q2).shape[1] == 0, \
        "two random subspaces reported a shared direction"
    assert vo.subspace_minus(Q1, Q1).shape[1] == 0, \
        "subtracting a subspace from itself left something behind"
    kept = vo.subspace_minus(Q1, Q2).shape[1]
    assert kept == k, f"subtracting an unrelated subspace dropped columns ({kept} of {k})"
    print("  random pair: intersection empty, self-minus empty, unrelated-minus full")


if __name__ == "__main__":
    print("test_intersection_and_minus_are_not_vacuous")
    test_intersection_and_minus_are_not_vacuous()
    print("test_overlap_null_is_below_recovered_values")
    test_overlap_null_is_below_recovered_values()
    print("test_designs_recover_their_planted_spans")
    test_designs_recover_their_planted_spans()
    print("test_result_subspace_is_recovered")
    test_result_subspace_is_recovered()
    print("test_returns_nothing_when_no_result_is_encoded")
    test_returns_nothing_when_no_result_is_encoded()
    print("test_global_centring_breaks_it")
    test_global_centring_breaks_it()
    print("\nALL PASS")
