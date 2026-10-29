"""Autonomous vs input-driven (conditional) divergence of a frozen grown soup.

Corrects soup_diagnosis.py in two ways:
  * phase distance is CIRCULAR (theta is wrapped to [0, 2pi) in world.py; |dtheta| near
    2pi is a tiny real separation),
  * besides the undriven test, twins receive the SAME input stream. Reservoir computing
    needs this conditional stability (echo-state / generalised synchronisation), not
    autonomous stability.

Inputs: none | hold-encoded random bits | BPSK on the four complement carriers
(0.03, 0.05, 0.09, 0.11 summed, independent random bits per carrier).
For each, the RMS circular phase distance over all beads is tracked; the growth rate is
fitted on the window where it lies in (1e-9, 1e-2).

    python experiments/driven_stability.py --gphases 0.02 0.005 --amps 0.05 0.3 1.0
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
torch.set_num_threads(1)

from chains.probing import source_kernel
from experiments.bond_ablation import fork_world
from experiments.grow_set_compute import freeze, grow, source_xy

CARRIERS = (0.03, 0.05, 0.09, 0.11)


def cdist(a, b):
    d = torch.remainder(a - b + math.pi, 2 * math.pi) - math.pi
    return float(torch.sqrt((d ** 2).mean()))


def drive(kind, amp, t, bits):
    s = t // 12
    if kind == "none":
        return 0.0
    if kind == "hold":
        return amp * bits[s % len(bits), 0]
    return amp * sum(bits[s % len(bits), j] * math.cos(om * t) for j, om in enumerate(CARRIERS)) / 2


def rate(ts, ds):
    ts, ds = np.array(ts), np.array(ds)
    ok = (ds > 1e-9) & (ds < 1e-2)
    return float(np.polyfit(ts[ok], np.log(ds[ok]), 1)[0]) if ok.sum() > 4 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=96)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-each", type=int, default=80)
    p.add_argument("--grow-steps", type=int, default=20000)
    p.add_argument("--grow-amplitude", type=float, default=0.3)
    p.add_argument("--gphases", type=float, nargs="+", default=[0.02, 0.005])
    p.add_argument("--amps", type=float, nargs="+", default=[0.05, 0.3, 1.0])
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--out", default="runs/driven_stability")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    g = freeze(grow(a, a.seed, driven=False))
    kernel = source_kernel(g, source_xy(a.N, g.cfg.d0), 1.0)
    bits = np.random.default_rng(5).choice([-1.0, 1.0], size=(a.steps // 12 + 2, 4))
    rows = []
    for gp in a.gphases:
        base = fork_world(g)
        base.cfg.g_phase = gp
        for _ in range(500):
            base.step()
        cases = [("none", 0.0)] + [(k, amp) for k in ("hold", "carriers") for amp in a.amps]
        for kind, amp in cases:
            t1, t2 = fork_world(base), fork_world(base)
            t2.theta[0] = torch.remainder(t2.theta[0] + 1e-6, 2 * math.pi)
            ts, ds = [], []
            for t in range(a.steps):
                e = drive(kind, amp, t, bits)
                ext = kernel * float(e) if e != 0.0 else None
                t1.step(external_source=ext); t2.step(external_source=ext)
                if t % 25 == 0:
                    ts.append(t); ds.append(cdist(t1.theta, t2.theta))
            r = rate(ts, ds)
            rows.append({"g_phase": gp, "input": kind, "amplitude": amp, "rate": r,
                         "final_rms_dtheta": ds[-1], "trace": [ts, ds]})
            print(f"g_phase={gp:<6g} input={kind:8s} amp={amp:<5g} rate={r:+.5f}/step "
                  f"final RMS dtheta={ds[-1]:.2e}", flush=True)
            json.dump({"parameters": vars(a), "rows": rows}, open(os.path.join(a.out, "summary.json"), "w"))


if __name__ == "__main__":
    main()
