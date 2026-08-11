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

    def __init__(self, d_input, z_causal_dim, z_nuisance_dim, hidden_dim, n_classes):
        super().__init__()
        self.z_causal_dim = z_causal_dim
        self.z_nuisance_dim = z_nuisance_dim
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
        return torch.cat([mu_c, mu_n], dim=-1)

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
                 n_classes, id="lcp_vae"):
        self.core = _LCPVAECore(d_input, z_causal_dim, z_nuisance_dim,
                                hidden_dim, n_classes)
        featurizer = _EncoderDecoderFeaturizerModule(
            self.core.encode_mean, self.core.decoder)
        inverse = _EncoderDecoderInverseModule(self.core.decoder)
        # The featurizer module holds `encode_mean`, a bound method, so the
        # encoder parameters are not reachable from it by module traversal.
        # Registering the core makes them visible to MIB's optimizer.
        featurizer.add_module("core", self.core)
        super().__init__(featurizer, inverse,
                         n_features=z_causal_dim + z_nuisance_dim, id=id)
        self.causal_indices = list(range(z_causal_dim))


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
