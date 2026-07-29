#!/usr/bin/env python3
"""Does the QR parametrisation behave differently from MIB's orthogonal one?

This codebase optimises an unconstrained (d, k) matrix A and calls
torch.linalg.qr(A) every step to obtain an orthonormal basis. MIB wraps the
parameter in torch.nn.utils.parametrizations.orthogonal, which maps an
unconstrained parameter to an orthogonal matrix via Householder reflections.

Both keep the *effective* subspace orthonormal, so the difference is not
constrained versus unconstrained. Whether the parametrisation matters for
optimisation is an empirical question, and this answers it on a controlled
subspace-recovery task where the target is known.

Two properties are worth separating:
  1. Do they reach the same solution, and at the same rate?
  2. Is QR stable? Q is unique only up to column signs, so a sign flip mid-run
     would make the map discontinuous and could disrupt the optimiser.

Usage:  uv run python experiments/parametrization_comparison.py
"""

import torch
import torch.nn as nn


def principal_angle(Q1, Q2):
    """Largest canonical angle between two subspaces, in radians."""
    svals = torch.linalg.svdvals(Q1.T @ Q2).clamp(-1.0, 1.0)
    return torch.acos(svals).max().item()


class RotateLayer(nn.Module):
    """Matches pyvene LowRankRotateLayer(init_orth=True)."""

    def __init__(self, d, k, init=None):
        super().__init__()
        if init is None:
            w = torch.empty(d, k)
            nn.init.orthogonal_(w)
        else:
            w = init.clone()
        self.weight = nn.Parameter(w)


def run(method, d, k, target_Q, X, steps, lr, init):
    """Optimise a projection to match the target subspace; return trace + result."""
    if method == "qr":
        A = nn.Parameter(init.clone())
        params = [A]
        get_Q = lambda: torch.linalg.qr(A)[0]
    else:
        layer = torch.nn.utils.parametrizations.orthogonal(RotateLayer(d, k, init))
        params = list(layer.parameters())
        get_Q = lambda: layer.weight

    opt = torch.optim.Adam(params, lr=lr)
    P_star = target_Q @ target_Q.T
    losses, sign_flips = [], 0
    prev_Q = get_Q().detach().clone()

    for _ in range(steps):
        opt.zero_grad()
        Q = get_Q()
        loss = ((X @ (Q @ Q.T) - X @ P_star) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        with torch.no_grad():
            cur = get_Q().detach()
            # a column sign flip shows up as a near-(-1) diagonal in prev^T cur
            diag = (prev_Q.T @ cur).diagonal()
            if (diag < -0.5).any():
                sign_flips += 1
            prev_Q = cur.clone()

    with torch.no_grad():
        Q = get_Q().detach()
        orth_err = (Q.T @ Q - torch.eye(k)).abs().max().item()
    return {
        "final_loss": losses[-1],
        "angle_to_truth": principal_angle(Q, target_Q),
        "orth_err": orth_err,
        "sign_flips": sign_flips,
        "steps_to_1pct": next((i for i, l in enumerate(losses)
                               if l < 0.01 * losses[0]), None),
    }


def main():
    d, k, n, steps, lr = 128, 2, 512, 400, 1e-2
    print(f"subspace recovery: d={d} k={k} n={n} steps={steps} lr={lr}")
    print(f"{'seed':>4}  {'method':>12}  {'final loss':>11}  {'angle':>7}"
          f"  {'orth err':>9}  {'flips':>6}  {'steps<1%':>9}")

    agg = {"qr": [], "orthogonal": []}
    for seed in range(5):
        g = torch.Generator().manual_seed(seed)
        target_Q, _ = torch.linalg.qr(torch.randn(d, k, generator=g))
        X = torch.randn(n, d, generator=g)
        init = torch.empty(d, k)
        nn.init.orthogonal_(init, generator=g) if "generator" in \
            nn.init.orthogonal_.__code__.co_varnames else nn.init.orthogonal_(init)

        for method in ("qr", "orthogonal"):
            torch.manual_seed(1000 + seed)
            r = run(method, d, k, target_Q, X, steps, lr, init)
            agg[method].append(r)
            print(f"{seed:>4}  {method:>12}  {r['final_loss']:>11.3e}"
                  f"  {r['angle_to_truth']:>7.4f}  {r['orth_err']:>9.2e}"
                  f"  {r['sign_flips']:>6}  {str(r['steps_to_1pct']):>9}")

    print()
    for method, rs in agg.items():
        fl = sum(r["final_loss"] for r in rs) / len(rs)
        an = sum(r["angle_to_truth"] for r in rs) / len(rs)
        oe = max(r["orth_err"] for r in rs)
        fp = sum(r["sign_flips"] for r in rs)
        conv = [r["steps_to_1pct"] for r in rs if r["steps_to_1pct"] is not None]
        print(f"{method:>12}: loss {fl:.3e} | angle {an:.4f} rad | "
              f"max orth err {oe:.1e} | total sign flips {fp} | "
              f"converged {len(conv)}/{len(rs)}"
              + (f" in {sum(conv)/len(conv):.0f} steps" if conv else ""))


if __name__ == "__main__":
    main()
