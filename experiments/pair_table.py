"""Measure which letters attract which: two monomers pinned r cells apart, the
time-averaged field force on each, projected toward the other. Positive = attraction.

    python experiments/pair_table.py --mode receiver
    python experiments/pair_table.py --mode medium
"""
import argparse, itertools, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from chains import Config, World


def pair_force(cfg, a, b, r, steps, warm):
    w = World(cfg)
    c = cfg.N / 2
    w.add_monomers([w.letters.index(a), w.letters.index(b)], pos=[(c, c), (c + r, c)])
    w.pinned[:] = True
    acc = torch.zeros(2, 2, device=w.device)
    n = 0
    for t in range(steps):
        w.step()
        if t >= warm:
            acc += w.field_force
            n += 1
    f = (acc / n).cpu().numpy()
    # attraction: force on 0 along +x, force on 1 along -x
    return float(f[0, 0]), float(-f[1, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="receiver")
    ap.add_argument("--r", type=float, nargs="+", default=[2.0, 3.0, 4.0])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    letters = list(Config().alphabet)
    for x, y in itertools.combinations_with_replacement(letters, 2):
        for r in a.r:
            vals = []
            for s in range(a.seeds):
                cfg = Config(N=48, seed=s, mode=a.mode, kT=0.0)
                f0, f1 = pair_force(cfg, x, y, r, a.steps, a.steps // 2)
                vals.append(0.5 * (f0 + f1))
            rows.append({"pair": x + y, "r": r, "attraction": float(np.mean(vals)),
                         "sd": float(np.std(vals))})
            print(f"{a.mode:8s} {x}-{y} r={r:.1f}  attraction {np.mean(vals):+.2e}  (sd over seeds {np.std(vals):.1e})", flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
