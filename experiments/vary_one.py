"""Vary-one PCA: training-free subspace identification for modular arithmetic.

Method from Shai et al., "Transformers learn factored representations", §H.1.1.
Spec and pre-registration in SPEC_VARY_ONE_GROKKING.md.

The idea: to find where a variable is represented, hold everything else fixed,
vary that variable, and look at where the activations move. No optimisation, so
nothing to overfit against a behavioural target -- which is what distinguishes it
from Distributed Alignment Search.

The result of `(a+b) mod p` is a function of the inputs, so it has no "everything
else" to hold fixed and strict vary-one does not apply to it. Addition being a
group operation supplies the fix: sweeping `a` with `b = s - a` moves both inputs
while pinning the result. Three designs then bracket the result subspace:

    A: fix b, vary a          -> S_a (+) S_result
    B: fix a, vary b          -> S_b (+) S_result
    C: fix a+b = s, vary a    -> S_a (+) S_b, result held constant

    S_result ~= (S_A intersect S_B) minus S_C

Torch is imported under a try/except and nn.Modules are defined inside functions,
matching the Modal scripts in this directory: they are imported locally to launch
on machines without torch installed.
"""

from __future__ import annotations

try:
    import torch
except (ImportError, AttributeError):
    pass


# --------------------------------------------------------------------------- #
#  Vary-one designs                                                           #
# --------------------------------------------------------------------------- #

def design_configs(design, p):
    """Return a list of configurations; each is a list of (a, b) input pairs.

    Every configuration holds the design's control variable fixed and sweeps the
    free one over all p values, so coverage is exhaustive rather than sampled.
    """
    if design == "A":            # fix b, vary a; the result moves with a
        return [[(a, b0) for a in range(p)] for b0 in range(p)]
    if design == "B":            # fix a, vary b; the result moves with b
        return [[(a0, b) for b in range(p)] for a0 in range(p)]
    if design == "C":            # fix a + b = s; inputs move, the result does not
        return [[(a, (s - a) % p) for a in range(p)] for s in range(p)]
    raise ValueError(f"design must be 'A', 'B' or 'C', got {design!r}")


def vary_one_pca(acts_by_config, k):
    """Top-k principal directions of within-configuration variation.

    Args:
        acts_by_config: sequence of (M, d) tensors, one per configuration, each
            row an activation for one realisation of the varied variable.
        k: number of components to return.

    Returns:
        (Q, explained_variance_ratio) where Q is (d, k) with orthonormal columns.

    Centring happens WITHIN each configuration and then the residuals are pooled.
    Centring globally instead removes the wrong variance and the method silently
    returns noise; this is the step most likely to be got wrong on a
    reimplementation, so it is done here rather than left to the caller.
    """
    centred = [A - A.mean(dim=0, keepdim=True) for A in acts_by_config]
    X = torch.cat(centred, dim=0).double()
    _, S, Vh = torch.linalg.svd(X, full_matrices=False)
    total = (S ** 2).sum()
    evr = (S ** 2) / total if total > 0 else S * 0
    return Vh[:k].T.contiguous(), evr


# --------------------------------------------------------------------------- #
#  Subspace algebra on the Grassmannian                                       #
# --------------------------------------------------------------------------- #

def principal_angles(Q1, Q2):
    """Principal angles in radians between the column spans, ascending."""
    s = torch.linalg.svdvals(Q1.T.double() @ Q2.double()).clamp(-1.0, 1.0)
    return torch.arccos(s)


def grassmann_distance(Q1, Q2):
    """Geodesic distance on Gr(k, d): the 2-norm of the principal angles."""
    return principal_angles(Q1, Q2).norm().item()


def subspace_overlap(Q1, Q2):
    """Mean squared cosine of principal angles. Matches subspace_overlap_null.py."""
    s = torch.linalg.svdvals(Q1.T.double() @ Q2.double()).clamp(-1.0, 1.0)
    return (s ** 2).mean().item()


def subspace_intersection(Q1, Q2, cos_tol=0.9):
    """Directions shared by both subspaces, as an orthonormal (d, m) frame.

    Principal vectors whose principal angle has cosine above `cos_tol` span the
    near-intersection. Returns an empty (d, 0) frame when nothing is shared,
    which is the expected result on a random network and must not be an error.
    """
    U, S, _ = torch.linalg.svd(Q1.T.double() @ Q2.double(), full_matrices=False)
    keep = S > cos_tol
    if keep.sum() == 0:
        return Q1.new_zeros((Q1.shape[0], 0))
    return (Q1.double() @ U[:, keep]).to(Q1.dtype).contiguous()


def subspace_minus(Q, Q_remove, min_norm=0.1):
    """Component of span(Q) orthogonal to span(Q_remove), re-orthonormalised.

    Columns whose residual norm falls below `min_norm` lay essentially inside
    Q_remove and are dropped; without that filter the result is padded with
    numerical noise that looks like real dimensions.
    """
    Q, Q_remove = Q.double(), Q_remove.double()
    if Q_remove.shape[1] == 0:
        return Q.contiguous()
    resid = Q - Q_remove @ (Q_remove.T @ Q)
    norms = resid.norm(dim=0)
    resid = resid[:, norms >= min_norm]
    if resid.shape[1] == 0:
        return Q.new_zeros((Q.shape[0], 0))
    Qr, _ = torch.linalg.qr(resid)
    return Qr.contiguous()


def result_subspace(Q_A, Q_B, Q_C, cos_tol=0.9, min_norm=0.1):
    """(S_A intersect S_B) minus S_C -- the construction from the spec.

    Returns an orthonormal frame, possibly with zero columns, which is the
    predicted outcome on a randomly initialised model.
    """
    shared = subspace_intersection(Q_A, Q_B, cos_tol=cos_tol)
    if shared.shape[1] == 0:
        return shared
    return subspace_minus(shared, Q_C, min_norm=min_norm)


# --------------------------------------------------------------------------- #
#  Empirical null                                                             #
# --------------------------------------------------------------------------- #

def random_frame(d, k, generator=None):
    """Uniformly random orthonormal k-frame in R^d (Haar, via QR of a Gaussian)."""
    A = torch.randn(d, k, generator=generator)
    return torch.linalg.qr(A)[0]


def overlap_null(d, k, n_pairs=2000, seed=0):
    """Empirical null for subspace_overlap between independent random k-frames.

    Empirical rather than the analytic k/d, because the analytic value is what
    produced this project's retracted below-chance claim.
    """
    g = torch.Generator().manual_seed(seed)
    vals = sorted(subspace_overlap(random_frame(d, k, g), random_frame(d, k, g))
                  for _ in range(n_pairs))
    n = len(vals)
    mean = sum(vals) / n
    return {
        "mean": mean,
        "p2_5": vals[int(0.025 * n)],
        "p97_5": vals[int(0.975 * n)],
        "n_pairs": n,
        "analytic_k_over_d": k / d,
    }
