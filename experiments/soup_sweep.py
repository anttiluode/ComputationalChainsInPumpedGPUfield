"""Edge-of-chaos sweep: phase-coupling strength of a frozen grown soup.

The default grown soup (g_phase=0.02) is chaotic (1e-6 rad kick grows ~0.04/step)
and has zero memory. Here one soup is grown once, frozen, and its phase coupling
g_phase is swept. For each value: growth rate of a 1e-6 phase kick (no input),
then linear memory and XOR capacity of the raw probes and of the oracle response.
g_phase = 0 makes every oscillator a fixed clock: the medium becomes linear in u
(XOR must be 0), a built-in control. The bare field is the reference.

    python experiments/soup_sweep.py --gphases 0.02 0.005 0.002 0.0005 0
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
torch.set_num_threads(1)

from chains import Config, World
from chains.probing import FieldProbes, source_kernel
from experiments.bond_ablation import fork_world
from experiments.grow_set_compute import GEOM_LEN, freeze, grow, source_xy
from experiments.memory_capacity import memory_curves
from experiments.temporal_memory import make_probes


def growth_rate(w, steps):
    t1, t2 = fork_world(w), fork_world(w)
    t2.theta[0] += 1e-6
    ts, ds = [], []
    for t in range(steps):
        t1.step(); t2.step()
        if t % 25 == 0:
            ts.append(t); ds.append(float((t1.theta - t2.theta).abs().max()))
    ts, ds = np.array(ts), np.array(ds)
    ok = (ds > 1e-12) & (ds < 1e-1)
    lam = float(np.polyfit(ts[ok], np.log(ds[ok]), 1)[0]) if ok.sum() > 3 else float("nan")
    return lam, float(ds[-1])


def capacities(w, sensor, kernel, bits, amp, mc):
    drv, und = fork_world(w), fork_world(w)
    raw, resp = [], []
    for b in bits:
        r1, r2 = [], []
        for s in range(12):
            drv.step(external_source=kernel * float(amp * b)); und.step()
            if s in (3, 7, 11):
                f1, f2 = sensor.field(drv), sensor.field(und)
                r1.append(f1.numpy().copy()); r2.append((f1 - f2).numpy().copy())
        raw.append(np.concatenate(r1)); resp.append(np.concatenate(r2))
    raw, resp = np.array(raw), np.array(resp)
    return memory_curves(raw, bits, mc), memory_curves(resp, bits, mc), raw, resp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=96)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-each", type=int, default=80)
    p.add_argument("--grow-steps", type=int, default=20000)
    p.add_argument("--grow-amplitude", type=float, default=0.3)
    p.add_argument("--gphases", type=float, nargs="+", default=[0.02, 0.005, 0.002, 0.0005, 0.0001, 0.0])
    p.add_argument("--symbols", type=int, default=600)
    p.add_argument("--amplitude", type=float, default=0.05)
    p.add_argument("--chaos-steps", type=int, default=2000)
    p.add_argument("--out", default="runs/soup_sweep")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    g = freeze(grow(a, a.seed, driven=False))
    sensor = FieldProbes(g, make_probes(a.N, g.cfg.d0, GEOM_LEN))
    kernel = source_kernel(g, source_xy(a.N, g.cfg.d0), 1.0)
    bits = np.random.default_rng(48271 + a.seed).choice([-1.0, 1.0], size=a.symbols)
    mc = argparse.Namespace(max_lag=20, train_fraction=0.7, discard=40, ridge=0.01)
    rows = []
    for gp in a.gphases:
        w = fork_world(g)
        w.cfg.g_phase = gp
        for _ in range(500):
            w.step()
        lam, final = growth_rate(w, a.chaos_steps)
        m_raw, m_resp, X_raw, X_resp = capacities(w, sensor, kernel, bits, a.amplitude, mc)
        np.savez_compressed(os.path.join(a.out, f"features_g{gp:g}.npz"), raw=X_raw, resp=X_resp, bits=bits)
        rows.append({"g_phase": gp, "growth_rate": lam, "final_sep": final, "raw": m_raw, "response": m_resp})
        print(f"g_phase={gp:<7g} growth={lam:+.4f}/step final_sep={final:.2e} | raw MC_lin={m_raw['MC_lin']:.2f} "
              f"MC_nl={m_raw['MC_nl']:.3f} sig_nl={m_raw['MC_nl_sig']:.3f} | response MC_lin={m_resp['MC_lin']:.2f} "
              f"MC_nl={m_resp['MC_nl']:.3f} sig_nl={m_resp['MC_nl_sig']:.3f}",
              flush=True)
        json.dump({"parameters": vars(a), "rows": rows}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
