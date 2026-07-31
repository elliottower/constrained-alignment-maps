"""Correctness tests for the canonical DAS implementation in experiments/das.py.

The load-bearing test plants a known causal direction in a toy model and checks
that DAS recovers it. A test that only asserts shapes would pass on an
implementation that optimises nothing.

Run: uv run python experiments/test_das.py
"""

from __future__ import annotations

import torch

import das


class PlantedDirectionModel:
    """Toy model whose output depends on the activation only through h . u.

    Logits are -(h.u - c_j)^2 over class centres c_j, so the predicted class is
    whichever centre the projection onto u is nearest. Swapping the u-component
    of a base activation for a source's must therefore flip the prediction to the
    source's label, and no other direction can do so. DAS at k=1 has exactly one
    correct answer, up to sign.
    """

    def __init__(self, u, centres):
        self.u = u
        self.centres = centres

    def run_with_hooks(self, acts, fwd_hooks):
        h = acts.clone()
        for _, hk in fwd_hooks:
            h = hk(h)
        proj = h[0, -1, :] @ self.u
        logits = -(proj - self.centres) ** 2
        return logits.view(1, 1, -1)


def build_toy(d_model=16, n_classes=4, n_pairs=120):
    u = torch.randn(d_model)
    u = u / u.norm()
    centres = torch.linspace(-3.0, 3.0, n_classes)
    model = PlantedDirectionModel(u, centres)

    # Orthogonal complement of u, so nuisance variation cannot carry the label.
    basis = torch.linalg.qr(torch.randn(d_model, d_model))[0]
    perp = basis - torch.outer(basis @ u, u)

    def make_act(label):
        nuisance = perp @ torch.randn(d_model) * 0.5
        return centres[label] * u + nuisance

    pairs = []
    for _ in range(n_pairs):
        yb, ys = torch.randint(0, n_classes, (2,)).tolist()
        base = make_act(yb)
        src = make_act(ys)
        pairs.append({
            "base_act": base,
            "src_act": src,
            "base_toks": base.view(1, 1, -1),
            "src_label": ys,
        })
    return model, pairs, u


def alignment(Q, u):
    """|cos| between the recovered 1-D subspace and the planted direction."""
    return (Q[:, 0] @ u).abs().item()


def _recovery_trials(init, param, n_steps, n_trials):
    """Mean |cos| with the planted direction over independent toy problems.

    No seeds are set. Each trial draws a fresh direction, fresh nuisance and a
    fresh initialisation, so the statistic is over the whole randomised setup.
    """
    scores, orth_err = [], 0.0
    for _ in range(n_trials):
        model, pairs, u = build_toy()
        Q = das.train_das(model, pairs, "hook", "cpu", k=1, n_steps=n_steps,
                          init=init, parametrization=param)
        scores.append(alignment(Q, u))
        orth_err = max(orth_err, (Q.T @ Q - torch.eye(1)).abs().max().item())
    mean, hw = das.mean_ci95(scores)
    return mean, hw, orth_err


def test_recovers_planted_direction(n_trials=10, n_steps=250):
    """The standard configuration must recover a planted 1-D causal direction."""
    results = {}
    for init in das.INIT_CHOICES:
        for param in das.PARAM_CHOICES:
            mean, hw, orth = _recovery_trials(init, param, n_steps, n_trials)
            results[(init, param)] = mean
            std = " <- STANDARD" if (init, param) == (das.STANDARD["init"],
                                                      das.STANDARD["parametrization"]) else ""
            print(f"  init={init:18s} param={param:11s} "
                  f"|cos|={mean:.4f} +/-{hw:.4f}  |Q^TQ-I|={orth:.1e}{std}")
            assert orth < 1e-4, f"{init}/{param} left the Stiefel manifold"

    std_mean = results[(das.STANDARD["init"], das.STANDARD["parametrization"])]
    assert std_mean > 0.85, (
        f"standard DAS failed to recover the planted direction "
        f"(mean |cos|={std_mean:.3f} over {n_trials} trials)")
    return results


def test_parametrisation_gap_is_convergence_not_capability():
    """Separate 'QR is slower' from 'QR is worse'.

    If the orthogonal parametrisation only converges faster, the gap should close
    as steps increase. If it persists, the parametrisation is not a footnote and
    H0.2 of PREREGISTRATION_RECONSTRUCTION_CRITERION.md is wrong.
    """
    print("  steps   orthogonal        qr               gap")
    gaps = {}
    for n_steps in (250, 500, 1000):
        o, o_hw, _ = _recovery_trials("random_orthogonal", "orthogonal", n_steps, 8)
        q, q_hw, _ = _recovery_trials("random_orthogonal", "qr", n_steps, 8)
        gaps[n_steps] = o - q
        print(f"  {n_steps:5d}   {o:.4f}+/-{o_hw:.4f}   {q:.4f}+/-{q_hw:.4f}   {o - q:+.4f}")
    print(f"  gap at 250 steps {gaps[250]:+.4f} -> at 1000 steps {gaps[1000]:+.4f}")
    return gaps


def test_random_subspace_is_not_aligned():
    """A random frame must not score like a fitted one, or the test above is vacuous."""
    _, _, u = build_toy()
    scores = [alignment(das.random_subspace(16, 1, "cpu"), u) for _ in range(200)]
    mean = sum(scores) / len(scores)
    print(f"  random 1-frame mean |cos| over 200 draws: {mean:.4f}")
    assert mean < 0.4, "random baseline is too aligned for the recovery test to mean anything"


def test_delta_pca_init_uses_the_deltas():
    """delta_pca must start from the top singular direction of the deltas."""
    model, pairs, u = build_toy()
    deltas = torch.stack([d["src_act"] - d["base_act"] for d in pairs])
    _, _, Vh = torch.linalg.svd(deltas, full_matrices=False)
    expected = Vh[0]
    got = das._init_weight(pairs, 16, 1, "cpu", "delta_pca", "base_act", "src_act")[:, 0]
    cos = (got @ expected).abs().item()
    print(f"  delta_pca init vs top right singular vector: |cos|={cos:.6f}")
    assert cos > 1 - 1e-5


def test_inits_differ():
    model, pairs, _ = build_toy()
    ws = {i: das._init_weight(pairs, 16, 1, "cpu", i, "base_act", "src_act")[:, 0]
          for i in das.INIT_CHOICES}
    ro_vs_gauss = (ws["random_orthogonal"] @ (ws["gaussian_qr"] / ws["gaussian_qr"].norm())).abs().item()
    print(f"  random_orthogonal vs gaussian_qr |cos|={ro_vs_gauss:.4f} (independent draws)")
    assert ro_vs_gauss < 0.95


def test_converged_step_rule():
    """The pre-registered rule: within 0.02 of 2x the steps, for two doublings."""
    steps = [100, 250, 500, 1000, 2000, 4000]

    plateau = dict(zip(steps, [0.50, 0.70, 0.90, 0.905, 0.910, 0.912]))
    assert das.converged_step(plateau) == 500, das.converged_step(plateau)

    climbing = dict(zip(steps, [0.10, 0.30, 0.50, 0.70, 0.85, 0.95]))
    assert das.converged_step(climbing) is None, das.converged_step(climbing)

    # plateaus only at the very end, so there is no room for two doublings
    late = dict(zip(steps, [0.10, 0.30, 0.50, 0.70, 0.90, 0.905]))
    assert das.converged_step(late) is None, das.converged_step(late)

    # converged from the start
    flat = dict(zip(steps, [0.90] * 6))
    assert das.converged_step(flat) == 100
    print("  converged_step: plateau=500, climbing=None, late=None, flat=100")


def test_snapshots_match_a_single_fit_trajectory():
    """Snapshots must come from one trajectory and improve along it."""
    model, pairs, u = build_toy()
    snaps = das.train_das_snapshots(model, pairs, "hook", "cpu", k=1,
                                    snapshot_steps=(50, 200, 800))
    assert sorted(snaps) == [50, 200, 800]
    aligns = {s: alignment(Q, u) for s, Q in snaps.items()}
    print("  |cos| along one trajectory: " +
          ", ".join(f"{s}={a:.4f}" for s, a in sorted(aligns.items())))
    for s, Q in snaps.items():
        assert (Q.T @ Q - torch.eye(1)).abs().max().item() < 1e-4, f"step {s} off-manifold"
    assert aligns[800] > aligns[50], "training did not improve alignment"


def test_mean_ci95():
    import math
    mu, hw = das.mean_ci95([0.10, 0.12, 0.08, 0.14, 0.11])
    sd = (sum((x - mu) ** 2 for x in [0.10, 0.12, 0.08, 0.14, 0.11]) / 4) ** 0.5
    assert abs(hw - 2.776 * sd / math.sqrt(5)) < 1e-12
    assert das.mean_ci95([0.3]) == (0.3, 0.0)
    assert das.mean_ci95([0.0] * 5) == (0.0, 0.0)
    print("  mean_ci95 matches hand-computed Student-t")


if __name__ == "__main__":
    print("test_mean_ci95");                   test_mean_ci95()
    print("test_converged_step_rule");         test_converged_step_rule()
    print("test_snapshots_match_a_single_fit_trajectory")
    test_snapshots_match_a_single_fit_trajectory()
    print("test_inits_differ");                test_inits_differ()
    print("test_delta_pca_init_uses_the_deltas"); test_delta_pca_init_uses_the_deltas()
    print("test_random_subspace_is_not_aligned"); test_random_subspace_is_not_aligned()
    print("test_recovers_planted_direction");  test_recovers_planted_direction()
    print("test_parametrisation_gap_is_convergence_not_capability")
    test_parametrisation_gap_is_convergence_not_capability()
    print("\nALL PASS")
