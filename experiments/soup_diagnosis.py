"""Why is a grown oscillator soup opaque? Buried (low SNR) or scrambled (chaos)?

1. SNR: memory capacity of the oracle response (driven - undriven twin) vs the
   raw probes. If the response is decodable, the information is there but
   buried under the soup's own emission.
2. Chaos: two frozen twins differing by 1e-6 rad in ONE bead's phase, no input.
   Track the probe-field difference over time. Exponential growth = sensitive
   dependence (inputs get scrambled); flat or decaying = not chaotic.

    python experiments/soup_diagnosis.py --n-each 80 --grow-steps 20000
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
torch.set_num_threads(1)

from chains.probing import FieldProbes, source_kernel
from experiments.bond_ablation import fork_world
from experiments.grow_set_compute import GEOM_LEN, freeze, grow, source_xy
from experiments.memory_capacity import memory_curves
from experiments.temporal_memory import make_probes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=96)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-each", type=int, default=80)
    p.add_argument("--grow-steps", type=int, default=20000)
    p.add_argument("--grow-amplitude", type=float, default=0.3)
    p.add_argument("--symbols", type=int, default=600)
    p.add_argument("--amplitude", type=float, default=0.05)
    p.add_argument("--chaos-steps", type=int, default=3000)
    p.add_argument("--out", default="runs/soup_diagnosis")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    g = freeze(grow(a, a.seed, driven=False))
    for _ in range(500):
        g.step()
    probes = make_probes(a.N, g.cfg.d0, GEOM_LEN)
    sensor = FieldProbes(g, probes)

    # ---- chaos: 1e-6 rad phase kick on one bead, no input
    t1, t2 = fork_world(g), fork_world(g)
    t2.theta[0] += 1e-6
    sep = []
    for t in range(a.chaos_steps):
        t1.step(); t2.step()
        if t % 50 == 0:
            sep.append([t, float((sensor.field(t1) - sensor.field(t2)).abs().max()),
                        float((t1.theta - t2.theta).abs().max())])
    sep = np.array(sep)
    late = sep[sep[:, 2] > 0]
    ok = (late[:, 2] > 1e-12) & (late[:, 2] < 1e-1)
    lam = float(np.polyfit(late[ok, 0], np.log(late[ok, 2]), 1)[0]) if ok.sum() > 3 else float("nan")
    print(f"phase separation: start 1e-6, after {a.chaos_steps} steps {sep[-1,2]:.3e}; "
          f"fitted growth rate {lam:.4g} per step", flush=True)

    # ---- SNR: raw vs oracle response
    kernel = source_kernel(g, source_xy(a.N, g.cfg.d0), 1.0)
    bits = np.random.default_rng(48271 + a.seed).choice([-1.0, 1.0], size=a.symbols)
    drv, und = fork_world(g), fork_world(g)
    raw, resp = [], []
    for b in bits:
        r1, r2 = [], []
        for s in range(12):
            drv.step(external_source=kernel * float(a.amplitude * b)); und.step()
            if s in (3, 7, 11):
                f1, f2 = sensor.field(drv), sensor.field(und)
                r1.append(f1.numpy().copy()); r2.append((f1 - f2).numpy().copy())
        raw.append(np.concatenate(r1)); resp.append(np.concatenate(r2))
    raw, resp = np.array(raw), np.array(resp)
    mc = argparse.Namespace(max_lag=20, train_fraction=0.7, discard=40, ridge=0.01)
    m_raw, m_resp = memory_curves(raw, bits, mc), memory_curves(resp, bits, mc)
    out = {"phase_sep": sep.tolist(), "growth_rate_per_step": lam,
           "raw_rms": float(np.sqrt((raw ** 2).mean())), "response_rms": float(np.sqrt((resp ** 2).mean())),
           "raw": m_raw, "response": m_resp}
    print(f"probe RMS raw {out['raw_rms']:.3e}  response {out['response_rms']:.3e}", flush=True)
    print(f"MC_lin raw {m_raw['MC_lin']:.2f}  response {m_resp['MC_lin']:.2f}; "
          f"MC_nl raw {m_raw['MC_nl']:.3f} response {m_resp['MC_nl']:.3f}", flush=True)
    json.dump(out, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
