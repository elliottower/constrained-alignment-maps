#!/usr/bin/env bash
# Third-party dependencies that are not vendored in this repository.
set -e
cd "$(dirname "$0")/.."

# MIB causal-variable track: experiments/das.py imports its SubspaceFeaturizer,
# which lives in a submodule nested two levels down inside MIB.
if [ ! -d reference/MIB/MIB-causal-variable-track/CausalAbstraction/neural ]; then
  rm -rf reference/MIB
  git clone https://github.com/aaronmueller/MIB.git reference/MIB
  git -C reference/MIB submodule update --init MIB-causal-variable-track
  git -C reference/MIB/MIB-causal-variable-track submodule update --init CausalAbstraction
fi

# Grant et al. (ICLR 2026) reference implementation of the Counterfactual Latent loss.
[ -d reference/rep_divergence ] || \
  git clone --depth 1 https://github.com/grantsrb/rep_divergence.git reference/rep_divergence

# Papers cited in the revision spec (not redistributed here).
mkdir -p reference/papers
for id in 2511.04638 2602.22600 2502.20914 2507.08802 2303.02536 2504.13151; do
  [ -f "reference/papers/arxiv_${id}.pdf" ] || \
    curl -sL -A "Mozilla/5.0" "https://arxiv.org/pdf/${id}" -o "reference/papers/arxiv_${id}.pdf"
done
echo "dependencies ready"
