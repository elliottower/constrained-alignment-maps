#!/usr/bin/env python3
"""Aggregate multi-seed runs into mean +/- 95% CI, and emit a LaTeX table body.

Every number cited in the paper should carry an interval. This reads the
per-seed JSONs written by multiseed_ksweep.py and reports, for each (method, k)
cell, the mean across seeds with a 95% confidence interval from Student's t.

With three seeds the interval is wide by construction; that is the honest
consequence of three seeds and is preferable to a bare point estimate.

Usage:  uv run python experiments/aggregate_seeds.py results/multiseed/*.json
"""

import json
import statistics
import sys

# t_{0.975, n-1} for small n; 3 seeds -> 2 df -> 4.303
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}

METHODS = ["delta_pca", "das", "das_pca", "nldas", "nldas_recon",
           "structured_vae", "pi_vae", "pi_sae"]
DISPLAY = {"das": "DAS (random init)", "das_pca": "DAS (delta-PCA init)", "nldas": "NL-DAS",
           "nldas_recon": "NL-DAS + recon", "pi_sae": "Structured VAE",
           "delta_pca": "Delta-PCA", "structured_vae": "Structured VAE (plain prior)",
           "pi_vae": "Label-conditional VAE (no expansion)"}


def order_stats(xs):
    """Mean, CI half-width, and max. The max is an upper order statistic whose
    expectation grows with the number of restarts, so it is reported alongside
    the mean rather than in place of it."""
    m, h = ci95(xs)
    return m, h, max(xs)


def ci95(xs):
    """Mean and half-width of the 95% CI. Half-width is 0 when all values agree."""
    n = len(xs)
    m = statistics.mean(xs)
    if n < 2:
        return m, float("nan")
    sd = statistics.stdev(xs)
    if sd == 0.0:
        return m, 0.0
    return m, T95.get(n, 1.96) * sd / (n ** 0.5)


def load(paths):
    """path -> {k: {method: {metric: value}}} keyed by seed."""
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def main():
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        return
    runs = load(paths)
    n = len(runs)
    ks = [k for k in runs[0] if k.startswith("k")]
    ks.sort(key=lambda s: int(s[1:]))

    print(f"{n} seeds: {', '.join(paths)}\n")
    for metric in ("iia", "diversity_ratio"):
        print(f"=== {metric} ===")
        header = f"{'method':<34}" + "".join(f"{k:>18}" for k in ks)
        print(header)
        for meth in METHODS:
            cells, maxes = [], []
            any_present = False
            for k in ks:
                vals = []
                for r in runs:
                    v = r.get(k, {}).get("methods", {}).get(meth, {}).get(metric)
                    if isinstance(v, (int, float)):
                        vals.append(v)
                if len(vals) == n and n > 0:
                    any_present = True
                    m, h, mx = order_stats(vals)
                    cells.append(f"{m:.3f}+-{h:.3f}" if h == h else f"{m:.3f}")
                    maxes.append(f"{mx:.3f}")
                else:
                    cells.append("--")
                    maxes.append("--")
            if any_present:
                print(f"{DISPLAY.get(meth, meth):<34}" + "".join(f"{c:>18}" for c in cells))
                if metric == "iia":
                    print(f"{'  (max over seeds)':<34}" + "".join(f"{c:>18}" for c in maxes))
        print()

    # LaTeX body for the k-sweep table, IIA only
    print("=== LaTeX (IIA, k-sweep) ===")
    for meth in ("das", "nldas", "pi_sae"):
        row = [DISPLAY.get(meth, meth)]
        for k in ks:
            vals = [r.get(k, {}).get("methods", {}).get(meth, {}).get("iia") for r in runs]
            vals = [v for v in vals if isinstance(v, (int, float))]
            if len(vals) == n:
                m, h = ci95(vals)
                row.append(f"${m:.3f} \\pm {h:.3f}$")
            else:
                row.append("--")
        print(" & ".join(row) + r" \\")


if __name__ == "__main__":
    main()
