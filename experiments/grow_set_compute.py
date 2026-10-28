"""Grow -> set -> compute with a whole soup as the medium.

GROW  a soup of monomers (default physics: pump on, ligation on, thermal noise,
      chain_boost) for --grow-steps, either QUIET or DRIVEN by the same kind of
      random +/-1 hold input the benchmark later uses (structured experience).
SET   pin every monomer, stop ligation and noise.
COMPUTE  hold-encoded random bits at the source; 15 distal probes; linear memory
      and XOR capacity (memory_capacity.memory_curves), plus a probe-noise floor.

Controls built from the SAME frozen monomers:
  cut       same positions, all bonds removed (topology + its emission boost gone)
  shuffled  same monomers and bonds, positions redrawn uniformly (geometry gone)
  bare      no monomers
A positive result needs grown > cut, shuffled, bare on held-out capacity, and
driven-grown > quiet-grown for "experience shaped the computer".

    python experiments/grow_set_compute.py --n-each 60 --grow-steps 15000
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
from experiments.memory_capacity import memory_curves
from experiments.temporal_memory import make_probes

GEOM_LEN = 8  # probe geometry identical to the chain benchmarks


def source_xy(N, d0):
    return (N / 2 - (GEOM_LEN - 1) * d0 / 2 - 5.0, N / 2)


def grow(args, seed, driven):
    cfg = Config(N=args.N, seed=seed, device=args.device)
    w = World(cfg)
    rng = np.random.default_rng(seed)
    types = np.repeat(np.arange(4), args.n_each)
    rng.shuffle(types)
    w.add_monomers(types)
    kernel = source_kernel(w, source_xy(args.N, cfg.d0), 1.0)
    bits = np.random.default_rng(9000 + seed).choice([-1.0, 1.0], size=args.grow_steps // 12 + 1)
    for t in range(args.grow_steps):
        ext = kernel * float(args.grow_amplitude * bits[t // 12]) if driven else None
        w.step(external_source=ext)
    return w


def freeze(w):
    f = fork_world(w)
    f.pinned[:] = True
    f.cfg.p_lig = 0.0
    f.cfg.kT = 0.0
    f.cfg.phase_noise = 0.0
    return f


def variants(grown, seed, args):
    g = freeze(grown)
    cut = fork_world(g)
    cut.nb.fill_(-1)
    shuf = fork_world(g)
    rng = torch.Generator(device=shuf.device).manual_seed(seed + 55)
    shuf.pos = torch.rand(shuf.pos.shape, generator=rng, device=shuf.device) * args.N
    bare = World(Config(N=args.N, seed=seed, device=args.device, p_lig=0.0, kT=0.0, phase_noise=0.0))
    worlds = {"grown": g, "cut": cut, "shuffled": shuf, "bare": bare}
    for w in worlds.values():   # let each settle into its own frozen steady state
        for _ in range(args.settle):
            w.step()
    return worlds


def benchmark(worlds, args, seed):
    ref = worlds["grown"]
    probes = make_probes(args.N, ref.cfg.d0, GEOM_LEN)
    sensor = FieldProbes(ref, probes)
    kernel = source_kernel(ref, source_xy(args.N, ref.cfg.d0), 1.0)
    bits = np.random.default_rng(48271 + seed).choice([-1.0, 1.0], size=args.symbols)
    S = 12
    feats = {k: [] for k in worlds}
    for b in bits:
        rows = {k: [] for k in worlds}
        for s in range(S):
            ext = kernel * float(args.amplitude * b)
            for k, w in worlds.items():
                w.step(external_source=ext)
            if s in (3, 7, 11):
                for k, w in worlds.items():
                    rows[k].append(sensor.field(w).cpu().numpy().copy())
        for k in worlds:
            feats[k].append(np.concatenate(rows[k]))
    return bits, {k: np.stack(v) for k, v in feats.items()}


def describe(w):
    paths = w.chains(min_len=2)
    lens = [len(p) for p in paths]
    return {"monomers": int(w.P), "ligations": int(w.ligations), "chains": len(lens),
            "longest": max(lens, default=0), "bonded_fraction": float((w.nb >= 0).any(1).float().mean())}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N", type=int, default=96)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--n-each", type=int, default=60)
    p.add_argument("--grow-steps", type=int, default=15000)
    p.add_argument("--grow-amplitude", type=float, default=0.3)
    p.add_argument("--settle", type=int, default=500)
    p.add_argument("--symbols", type=int, default=900)
    p.add_argument("--amplitude", type=float, default=0.05)
    p.add_argument("--max-lag", type=int, default=20)
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 1e-3])
    p.add_argument("--out", default="runs/grow_set_compute")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mc_args = argparse.Namespace(max_lag=a.max_lag, train_fraction=0.7, discard=40, ridge=0.01)
    rows = []
    for seed in range(a.seeds):
        for arm in ("quiet", "driven"):
            grown = grow(a, seed, driven=(arm == "driven"))
            info = describe(grown)
            print(f"seed {seed} {arm}: {info}", flush=True)
            bits, feats = benchmark(variants(grown, seed, a), a, seed)
            for k, X in feats.items():
                for sigma in a.sigmas:
                    noise = np.random.default_rng(seed * 7 + 1).normal(0, sigma, X.shape) if sigma else 0
                    m = memory_curves(X + noise, bits, mc_args)
                    rows.append({"seed": seed, "arm": arm, "variant": k, "sigma": sigma,
                                 "growth": info, **m})
                    print(f"  {k:8s} sigma={sigma:<6g} MC_lin={m['MC_lin']:.2f} MC_nl={m['MC_nl']:.3f}",
                          flush=True)
            json.dump({"parameters": vars(a), "rows": rows},
                      open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
