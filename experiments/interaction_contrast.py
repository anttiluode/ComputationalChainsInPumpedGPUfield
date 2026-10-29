"""Non-additivity of the response to two bits, relative to trial-to-trial noise.

For a medium that is linear in its input, the mean response to the four patterns is a
parallelogram:  f(++) - f(+-) - f(-+) + f(--) = 0.  XOR (b1*b2) is decodable by a linear
readout exactly when this interaction contrast is non-zero. Reported: |contrast| divided by
the pooled within-pattern noise std (per feature, then RMS over features), and the same
number for the main effect of b1 for scale.

    python experiments/interaction_contrast.py runs/pm_mux_a02_n003
"""
import json
import sys

import numpy as np


def contrast(X, pats):
    key = [tuple(p) for p in pats]
    groups = {s: X[[k == s for k in key]] for s in [(1, 1), (1, -1), (-1, 1), (-1, -1)]}
    m = {s: g.mean(0) for s, g in groups.items()}
    noise = np.sqrt(np.mean([g.var(0, ddof=1) for g in groups.values()], axis=0))
    noise = np.where(noise < 1e-12, np.nan, noise)
    inter = (m[(1, 1)] - m[(1, -1)] - m[(-1, 1)] + m[(-1, -1)]) / 4
    main1 = (m[(1, 1)] + m[(1, -1)] - m[(-1, 1)] - m[(-1, -1)]) / 4
    n = len(groups[(1, 1)])
    se = 1 / np.sqrt(n)            # std error of a contrast of 4 means, in noise units, /4 scaling
    z_inter = np.nanmax(np.abs(inter) / noise) / se
    z_main = np.nanmax(np.abs(main1) / noise) / se
    return float(z_inter), float(z_main)


if __name__ == "__main__":
    d = np.load(sys.argv[1] + "/features.npz")
    pats = d["pats"]
    out = {}
    for k in sorted(d.files):
        if k == "pats" or not k.endswith("near") or "_0.001_" not in k:
            continue
        zi, zm = contrast(d[k], pats)
        out[k] = {"z_interaction": zi, "z_main_b1": zm}
        print(f"{k:28s} interaction z={zi:8.1f}   main effect of b1 z={zm:8.1f}")
    json.dump(out, open(sys.argv[1] + "/interaction_contrast.json", "w"), indent=1)
