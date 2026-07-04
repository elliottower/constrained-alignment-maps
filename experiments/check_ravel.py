"""Check RAVEL ablation results on Modal volume.

Usage:
    modal run experiments/check_ravel.py
"""
import json
import os

import modal

app = modal.App("check-ravel")
volume = modal.Volume.from_name("fc-results")

METHODS = [
    "vae", "sae", "pi_vae", "pi_sae",
    "structured_vae", "structured_sae", "structured_pi_vae", "structured_pi_sae",
]


@app.function(volumes={"/results": volume})
def check():
    volume.reload()
    print(f"\n{'Method':<22} {'Cause':<8} {'Isolate':<8} {'Disent':<8} {'Status'}")
    print("-" * 62)

    found = 0
    for method in METHODS:
        path = f"/results/ravel_ablation/{method}/results.json"
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            for etype, result in data.items():
                if "avg_disentangle" in result:
                    cause_vals = [r[method]["cause"] for r in result["per_attribute"].values() if method in r]
                    iso_vals = [r[method]["mean_isolate"] for r in result["per_attribute"].values() if method in r]
                    avg_cause = sum(cause_vals) / len(cause_vals) if cause_vals else 0
                    avg_iso = sum(iso_vals) / len(iso_vals) if iso_vals else 0
                    print(f"{method:<22} {avg_cause:<8.3f} {avg_iso:<8.3f} "
                          f"{result['avg_disentangle']:<8.3f} done")
                    found += 1
        else:
            print(f"{method:<22} {'—':<8} {'—':<8} {'—':<8} running...")

    print(f"\n{found}/8 methods complete")

    for method in METHODS:
        path = f"/results/ravel_ablation/{method}/results.json"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for etype, result in data.items():
            if "per_attribute" not in result:
                continue
            print(f"\n  {method} / {etype}:")
            for attr, r in result["per_attribute"].items():
                if method in r:
                    s = r[method]
                    print(f"    {attr:<20} cause={s['cause']:.3f}  iso={s['mean_isolate']:.3f}  "
                          f"dis={s['disentangle']:.3f}")


@app.local_entrypoint()
def main():
    check.remote()
