"""Diagnostic: how often does each template site have its complement sitting next to it,
versus some other letter, as a function of pairing strength (k_force) and temperature (kT)?
Ligation is off here; this measures binding only.

    python experiments/occupancy.py --kf 1 2 4 --kT 0.006 --steps 12000
"""
import argparse, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from chains import Config, World
from chains.analysis import complement_map

T = "GATTACAG"


def occupancy(kf, kT, steps, seed=0, N=96, n_each=60, pump=True, boost=1.0):
    cfg = Config(N=N, seed=seed, k_force=kf, kT=kT, pump_on=pump, p_lig=0.0, chain_boost=boost)
    w = World(cfg)
    tids = w.add_chain(T, (N / 2 - (len(T) - 1) * cfg.d0 / 2, N / 2))
    w.pinned[tids] = True
    free = np.repeat(np.arange(4), n_each); w.rng.shuffle(free); w.add_monomers(free)
    cm = complement_map(cfg)
    want = torch.tensor([w.letters.index(cm[ch]) for ch in T], device=w.device)
    fids = torch.arange(len(T), w.P, device=w.device)
    good, bad, n = 0.0, 0.0, 0
    for t in range(steps):
        w.step()
        if t > steps // 3 and t % 20 == 0:
            d = w._delta(w.pos[fids][:, None, :], w.pos[tids][None, :, :]).norm(dim=-1)  # (F, 8)
            r, k = d.min(1)
            close = r < 2.6
            ok = close & (w.typ[fids] == want[k])
            good += float(ok.sum()); bad += float((close & ~ok).sum()); n += 1
    return good / n, bad / n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kf", type=float, nargs="+", default=[1, 2, 4])
    ap.add_argument("--kT", type=float, nargs="+", default=[0.006])
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--pump_off", action="store_true")
    ap.add_argument("--boost", type=float, nargs="+", default=[1.0])
    a = ap.parse_args()
    for kT in a.kT:
        for kf in a.kf:
          for bo in a.boost:
            g, b = occupancy(kf, kT, a.steps, pump=not a.pump_off, boost=bo)
            print(f"boost={bo} k_force={kf} kT={kT} pump={'off' if a.pump_off else 'on'}: per 8 sites, complement {g:.2f}, other letters {b:.2f}", flush=True)
