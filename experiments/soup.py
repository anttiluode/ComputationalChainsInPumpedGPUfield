"""Stage 1: a soup with no template. Do bonded chains form, and what are they made of?

For every backbone bond we record the two letters it joins. With the pump off, pairing
forces vanish and only chance encounters that keep returning to the backbone spacing can link.
With the pump on, complementary pairs hold monomers near each other, so we look at
which letter pairs end up adjacent along chains.

    python experiments/soup.py --seeds 2 --steps 40000
"""
import argparse, json, os, sys, time
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from chains import Config, World
from chains.analysis import complement_map, duplex_report


def run(cfg, n_each, steps):
    w = World(cfg)
    free = np.repeat(np.arange(len(w.letters)), n_each)
    w.rng.shuffle(free)
    w.add_monomers(free)
    t0 = time.time()
    for t in range(steps):
        w.step()
        if (t + 1) % 10000 == 0:
            print(f"   t={t+1} ligations={w.ligations} ({time.time()-t0:.0f}s)", flush=True)
    chains = [w.sequence(p) for p in w.chains(min_len=2)]
    bonds = Counter()
    for s in chains:
        for a, b in zip(s[:-1], s[1:]):
            bonds["".join(sorted(a + b))] += 1
    w.duplex = duplex_report(w, complement_map(w.cfg))
    return chains, bonds, w.duplex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--n_each", type=int, default=60)
    ap.add_argument("--held_min", type=float, default=None)
    ap.add_argument("--arms", nargs="+", default=["pump_on", "pump_off"])
    ap.add_argument("--out", default="results/soup.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    res = {}
    for arm in a.arms:
        res[arm] = []
        for s in range(a.seeds):
            cfg = Config(N=a.N, seed=s, pump_on=(arm == "pump_on"))
            if a.held_min is not None:
                cfg.held_min = a.held_min
            chains, bonds, dup = run(cfg, a.n_each, a.steps)
            lens = [len(c) for c in chains]
            r = {"seed": s, "chains": len(chains), "longest": max(lens, default=0),
                 "monomers_in_chains": int(sum(lens)), "bond_letters": dict(bonds),
                 "duplex": dup, "sequences": sorted(chains, key=len, reverse=True)[:30]}
            res[arm].append(r)
            print(f"{arm} seed {s}: {len(chains)} chains, longest {r['longest']}, "
                  f"{r['monomers_in_chains']} monomers in chains, bonds {dict(bonds)}, duplex {dup}", flush=True)
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
