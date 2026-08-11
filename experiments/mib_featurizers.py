"""Alignment maps for the random-network control, written as MIB Featurizers.

MIB's `Featurizer` is a pair of modules: `featurize(x) -> (features, error)` and
`inverse_featurize(features, error) -> x_hat`. The `error` term carries whatever
the map does not represent, so a round trip is exact and an interchange edits
only the feature coordinates. `SubspaceFeaturizer` gets this for free because an
orthogonal projection has an exact complement; a learned encoder/decoder pair
does not, which is why the error term must be `x - decode(featurize(x))` rather
than anything the decoder produces on its own.

Every map here follows the same contract, so the only thing that varies across
the arms is the parameterization and what the parameters are trained on. That is
the 2x2 the paper reports: nonlinearity crossed with interchange training.
"""

import torch
import torch.nn as nn

from CausalAbstraction.neural.featurizers import Featurizer


class _EncoderDecoderFeaturizerModule(nn.Module):
    """x -> (features, x - decode(features)).

    The encoder and decoder are shared with the inverse module, so the two
    directions stay parameter-tied. `nn.Module.parameters()` deduplicates shared
    submodules, so MIB's optimizer sees each parameter once.
    """

    def __init__(self, encode_fn, decoder):
        super().__init__()
        self.encode_fn = encode_fn
        self.decoder = decoder

    def forward(self, x):
        p = next(self.decoder.parameters())
        f = self.encode_fn(x.to(p.dtype))
        error = x - self.decoder(f).to(x.dtype)
        # pyvene writes the source features into this tensor in place. The
        # decoder above has already saved `f` for its backward, so the returned
        # tensor must not alias it or the graph is corrupted at the scatter.
        return f.clone(), error


class _EncoderDecoderInverseModule(nn.Module):
    """(features, error) -> decode(features) + error."""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, f, error):
        p = next(self.decoder.parameters())
        return self.decoder(f.to(p.dtype)).to(error.dtype) + error


class _MLPEncoder(nn.Module):
    def __init__(self, d_input, hidden_dim, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_features),
        )

    def forward(self, x):
        return self.net(x)


class _LCPVAECore(nn.Module):
    """Label-conditional partitioned VAE.

    The latent splits into a causal block and a nuisance block. Only the causal
    block is interchanged, which is what `feature_indices` selects downstream.
    Interventions use the posterior mean rather than a sample: sampling at
    evaluation time would add noise to a measurement of the map, not of the model.
    """

    def __init__(self, d_input, z_causal_dim, z_nuisance_dim, hidden_dim, n_classes,
                 expansion_factor=8, prototype_write=False):
        super().__init__()
        self.prototype_write = prototype_write
        # The causal block is `expansion_factor` times wider than k. Measured in
        # `results/sparsity_ablation_addition.json`: at k=1 an expansion of 1
        # scores 0.017 and an expansion of 8 scores 1.000, at every L1 weight
        # including zero. The expansion carries the low-k result, not the
        # sparsity penalty, whose measured L0 is 0.9975 (essentially dense).
        self.expansion_factor = expansion_factor
        self.z_sparse = z_causal_dim * expansion_factor
        self.z_causal_dim = self.z_sparse
        self.z_nuisance_dim = z_nuisance_dim
        z_causal_dim = self.z_sparse
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

    def encode_mean(self, x):
        mu_c, _, mu_n, _ = self.encode(x)
        if self.prototype_write:
            mu_c = self.to_prototype(mu_c)
        return torch.cat([mu_c, mu_n], dim=-1)

    def to_prototype(self, mu_c):
        """Replace the encoded causal latent with its class prototype.

        The label-conditional prior already learns one mean per class, so the
        codebook exists; the intervention simply writes a prototype instead of
        the source instance's own encoding. Two prompts of the same class then
        produce *identical* causal latents, the interchange delta is exactly
        zero, and the map is inert when the variable does not change — by
        construction rather than by training.

        The class is predicted by the model's own classifier, not read from a
        label, so nothing about the counterfactual leaks in at evaluation time.
        """
        y_hat = self.classifier(mu_c).argmax(dim=-1)
        proto = self.prior_mu(y_hat)
        # Straight-through: the forward pass uses the prototype, the backward
        # pass still reaches the encoder.
        return mu_c + (proto - mu_c).detach()

    def forward(self, x):
        mu_c, lv_c, mu_n, lv_n = self.encode(x)
        z_c = mu_c + torch.exp(0.5 * lv_c) * torch.randn_like(lv_c)
        z_n = mu_n + torch.exp(0.5 * lv_n) * torch.randn_like(lv_n)
        z = torch.cat([z_c, z_n], dim=-1)
        return self.decoder(z), self.classifier(z_c), mu_c, lv_c, mu_n, lv_n


class LCPVAEFeaturizer(Featurizer):
    """Label-conditional partitioned VAE as an alignment map.

    `core` is exposed so the reconstruction-only arm can be trained outside the
    interchange objective and then evaluated through the same code path as the
    interchange-trained arm. The two arms differ only in what the parameters saw.
    """

    def __init__(self, d_input, z_causal_dim, z_nuisance_dim, hidden_dim,
                 n_classes, expansion_factor=8, prototype_write=False,
                 id="lcp_vae"):
        self.core = _LCPVAECore(d_input, z_causal_dim, z_nuisance_dim,
                                hidden_dim, n_classes, expansion_factor,
                                prototype_write)
        featurizer = _EncoderDecoderFeaturizerModule(
            self.core.encode_mean, self.core.decoder)
        inverse = _EncoderDecoderInverseModule(self.core.decoder)
        # The featurizer module holds `encode_mean`, a bound method, so the
        # encoder parameters are not reachable from it by module traversal.
        # Registering the core makes them visible to MIB's optimizer.
        featurizer.add_module("core", self.core)
        super().__init__(featurizer, inverse,
                         n_features=self.core.z_sparse + z_nuisance_dim, id=id)
        # The interchange swaps the whole widened causal block, so a nominal
        # k = 1 writes `expansion_factor` latent coordinates, not one. That is
        # what the published k = 1 number means and it has to be reported.
        self.causal_indices = list(range(self.core.z_sparse))


class NonlinearFeaturizer(Featurizer):
    """Nonlinear DAS: a multilayer perceptron in place of the rotation."""

    def __init__(self, d_input, n_features, hidden_dim, id="nldas"):
        self.encoder = _MLPEncoder(d_input, hidden_dim, n_features)
        self.decoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, d_input),
        )
        featurizer = _EncoderDecoderFeaturizerModule(self.encoder, self.decoder)
        inverse = _EncoderDecoderInverseModule(self.decoder)
        featurizer.add_module("encoder", self.encoder)
        super().__init__(featurizer, inverse, n_features=n_features, id=id)
        self.causal_indices = list(range(n_features))


class _Coupling(nn.Module):
    """One affine coupling layer. Exactly invertible in closed form.

    Half the coordinates pass through untouched and condition an affine
    transform of the other half, so the inverse needs no optimization and no
    reconstruction error. `tanh` bounds the log-scale, which keeps the forward
    and inverse well conditioned when the residual stream has large norm.
    """

    def __init__(self, d_input, hidden_dim, flip, shift_scale=10.0):
        super().__init__()
        self.flip = flip
        # Both the log-scale and the translate are bounded. An unbounded shift on
        # a residual stream with norm in the hundreds diverges under the learning
        # rate MIB tuned for an orthogonal rotation, and a diverged flow destroys
        # the output — which reads as 0.000 on sensitivity *and* specificity.
        self.shift_scale = shift_scale
        self.d_a = d_input // 2
        self.d_b = d_input - self.d_a
        if flip:
            self.d_a, self.d_b = self.d_b, self.d_a
        self.net = nn.Sequential(
            nn.Linear(self.d_a, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * self.d_b),
        )
        # Start at the identity so an untrained flow is a no-op rather than noise.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _split(self, x):
        return (x[..., self.d_b:], x[..., :self.d_b]) if self.flip \
            else (x[..., :self.d_a], x[..., self.d_a:])

    def _join(self, a, b):
        return torch.cat([b, a], dim=-1) if self.flip else torch.cat([a, b], dim=-1)

    def forward(self, x):
        a, b = self._split(x)
        s, t = self.net(a).chunk(2, dim=-1)
        s, t = torch.tanh(s), self.shift_scale * torch.tanh(t)
        return self._join(a, b * torch.exp(s) + t)

    def inverse(self, y):
        a, b = self._split(y)
        s, t = self.net(a).chunk(2, dim=-1)
        s, t = torch.tanh(s), self.shift_scale * torch.tanh(t)
        return self._join(a, (b - t) * torch.exp(-s))


class _Flow(nn.Module):
    """A stack of coupling layers, alternating which half is transformed."""

    def __init__(self, d_input, hidden_dim, n_layers, shift_scale=10.0):
        super().__init__()
        self.layers = nn.ModuleList(
            _Coupling(d_input, hidden_dim, flip=(i % 2 == 1), shift_scale=shift_scale)
            for i in range(n_layers))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def inverse(self, y):
        for layer in reversed(self.layers):
            y = layer.inverse(y)
        return y


class _FlowFeaturizerModule(nn.Module):
    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def forward(self, x):
        p = next(self.flow.parameters())
        f = self.flow(x.to(p.dtype))
        # Exact by construction, so the residual is numerical noise rather than
        # the map's failure to reconstruct. Kept so the contract still holds.
        error = x - self.flow.inverse(f).to(x.dtype)
        # pyvene scatters the source features into this tensor in place, and the
        # inverse above has already saved `f`. Clone or the graph is corrupted.
        return f.to(x.dtype).clone(), error


class _FlowInverseModule(nn.Module):
    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def forward(self, f, error):
        p = next(self.flow.parameters())
        return self.flow.inverse(f.to(p.dtype)).to(error.dtype) + error


class FlowFeaturizer(Featurizer):
    """Nonlinear alignment map that is invertible by construction.

    Distributed Alignment Search succeeds at k = 1 because an orthogonal rotation
    preserves the complement exactly: swap one coordinate and the other d-1 pass
    through bit for bit. An autoencoder cannot do that — at k = 1 it reconstructs
    d dimensions from a couple of numbers, so the intervened write is a small
    noisy delta regardless of how well it is trained.

    A coupling flow keeps the exactness and drops the linearity. Restricted to a
    rotation it *is* DAS, so nonlinearity here is a strict generalization rather
    than a trade against write fidelity.
    """

    def __init__(self, d_input, k, hidden_dim=256, n_layers=4, shift_scale=10.0,
                 id="flow"):
        self.flow = _Flow(d_input, hidden_dim, n_layers, shift_scale)
        featurizer = _FlowFeaturizerModule(self.flow)
        inverse = _FlowInverseModule(self.flow)
        super().__init__(featurizer, inverse, n_features=d_input, id=id)
        self.causal_indices = list(range(k))


class _DirectionalCore(nn.Module):
    """Nonlinear readout, rank-k write.

    Distributed Alignment Search moves an activation along k fixed directions by
    an amount linear in the activation. This keeps the fixed directions and frees
    only the amount:

        h' = h + sum_j [alpha_j(z_src) - alpha_j(z_base)] d_j

    so the write stays inside span{d_1..d_k} and the effective write rank is k by
    construction, exactly as for a rotation. DAS is the special case where alpha
    is linear, which makes this a strict generalization rather than a trade
    against write fidelity.

    The directions are learned by the causal objective rather than fitted to
    reconstruct anything, so they are not a proxy basis; and being explicit
    vectors, they can be read off and compared across seeds without probing a
    nonlinear decoder.
    """

    def __init__(self, d_input, k, hidden_dim, n_classes, expansion_factor=8):
        super().__init__()
        self.k = k
        self.z_sparse = k * expansion_factor
        self.z_causal_dim = self.z_sparse
        self.z_nuisance_dim = max(4 * k, 4)
        self.enc_trunk = nn.Sequential(
            nn.Linear(d_input, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.enc_causal_mu = nn.Linear(hidden_dim, self.z_sparse)
        self.enc_causal_logvar = nn.Linear(hidden_dim, self.z_sparse)
        self.enc_nuisance_mu = nn.Linear(hidden_dim, self.z_nuisance_dim)
        self.enc_nuisance_logvar = nn.Linear(hidden_dim, self.z_nuisance_dim)
        # The whole write channel: k directions, and a nonlinear map from the
        # latent to how far to move along each.
        self.directions = nn.Parameter(torch.randn(k, d_input) / d_input ** 0.5)
        self.coeff = nn.Sequential(
            nn.Linear(self.z_sparse + self.z_nuisance_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, k),
        )
        self.classifier = nn.Linear(self.z_sparse, n_classes)
        self.prior_mu = nn.Embedding(n_classes, self.z_sparse)
        self.prior_logvar = nn.Embedding(n_classes, self.z_sparse)

    def unit_directions(self):
        return self.directions / (self.directions.norm(dim=-1, keepdim=True) + 1e-8)

    def encode(self, x):
        h = self.enc_trunk(x)
        return (self.enc_causal_mu(h), self.enc_causal_logvar(h),
                self.enc_nuisance_mu(h), self.enc_nuisance_logvar(h))

    def encode_mean(self, x):
        mu_c, _, mu_n, _ = self.encode(x)
        return torch.cat([mu_c, mu_n], dim=-1)

    def write(self, z):
        """The activation this latent would produce, as a point in the span."""
        return self.coeff(z) @ self.unit_directions()

    def forward(self, x):
        mu_c, lv_c, mu_n, lv_n = self.encode(x)
        z_c = mu_c + torch.exp(0.5 * lv_c) * torch.randn_like(lv_c)
        z_n = mu_n + torch.exp(0.5 * lv_n) * torch.randn_like(lv_n)
        z = torch.cat([z_c, z_n], dim=-1)
        # `x_r` is the reconstruction the ELBO scores. Only the in-span component
        # is the map's to explain, so the complement is passed through; otherwise
        # the reconstruction term would demand a rank-k map reproduce all of x.
        d = self.unit_directions()
        x_perp = x - (x @ d.T) @ d
        return x_perp + self.write(z), self.classifier(z_c), mu_c, lv_c, mu_n, lv_n


class _DirectionalFeaturizerModule(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, x):
        p = next(self.core.parameters())
        f = self.core.encode_mean(x.to(p.dtype))
        # The error carries everything outside the span, so a round trip is exact
        # and the interchange can only move within span{d_j}.
        error = x - self.core.write(f).to(x.dtype)
        return f.to(x.dtype).clone(), error


class _DirectionalInverseModule(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, f, error):
        p = next(self.core.parameters())
        return self.core.write(f.to(p.dtype)).to(error.dtype) + error


class DirectionalFeaturizer(Featurizer):
    """Rank-k write with a nonlinear readout. See `_DirectionalCore`."""

    def __init__(self, d_input, k, hidden_dim=256, n_classes=4,
                 expansion_factor=8, id="directional"):
        self.core = _DirectionalCore(d_input, k, hidden_dim, n_classes,
                                     expansion_factor)
        featurizer = _DirectionalFeaturizerModule(self.core)
        inverse = _DirectionalInverseModule(self.core)
        super().__init__(featurizer, inverse,
                         n_features=self.core.z_sparse + self.core.z_nuisance_dim,
                         id=id)
        self.causal_indices = list(range(self.core.z_sparse))
