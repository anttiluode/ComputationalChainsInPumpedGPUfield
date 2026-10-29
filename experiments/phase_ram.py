"""Phase RAM: does capacity scale with chain length? (frequency x position addressing)

K blocks "ACGT" along x, BLOCK_GAP cells apart, one write source above each block.
Sub-gap carriers are evanescent (decay length ~4-6 cells), so a source addresses only its
own block: POSITION is one address bit, FREQUENCY (0.03 -> A/T pair, 0.09 -> C/G pair)
the other. Each trial writes 2K random bits at once, waits, then reads passively.
Readout is local, like a RAM row select: bit (k, register) is decoded only from the probes
of block k (8 probes, quadrature demod at the 4 letter frequencies). Crosstalk is measured
by decoding block k's bits from the NEIGHBOUR block's probes. Same protocol on the bare
field. 8-fold CV over trials, probe noise sigma.

    python experiments/phase_ram.py --blocks 1 2 4 --trials 128
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

from chains import Config, World
from chains.probing import FieldProbes, source_kernel
from experiments.bond_ablation import fork_world
from experiments.phase_memory import cv_decode

LETTER_F = (0.03, 0.05, 0.09, 0.11)
BLOCK = "ACGT"


def layout(K, args):
    span = (K - 1) * args.block_gap
    N = int(max(64, span + 4 * len(BLOCK) + 48))
    N += N % 2
    x_first = N / 2 - span / 2
    blocks = []
    for k in range(K):
        cx = x_first + k * args.block_gap
        xs = [cx - 1.5 * args.spacing + args.spacing * i for i in range(len(BLOCK))]
        blocks.append((cx, xs))
    return N, blocks


def build(K, args, beads):
    N, blocks = layout(K, args)
    cfg = Config(N=N, seed=args.seed, device="cpu", p_lig=0.0, kT=0.0, phase_noise=0.0, chain_boost=0.0)
    w = World(cfg)
    if beads:
        types, pos = [], []
        for _, xs in blocks:
            types += [w.letters.index(c) for c in BLOCK]
            pos += [(x, N / 2) for x in xs]
        w.add_monomers(types, pos=pos)
        w.pinned[:] = True
    for _ in range(args.warmup):
        w.step()
    return w, N, blocks


def demod(rec, sigma, rng, t0):
    x = rec + (rng.normal(0, sigma, rec.shape) if sigma > 0 else 0)
    k = t0 + np.arange(rec.shape[0])
    out = []
    for om in LETTER_F:
        out.append((x * np.cos(om * k)[:, None]).mean(0))
        out.append((x * np.sin(om * k)[:, None]).mean(0))
    return np.concatenate(out)


def trial(args, base, N, blocks, bits, i):
    w = fork_world(base)
    w.cfg.phase_noise = args.phase_noise
    w.gen.manual_seed(20_000 + i)
    y = N / 2
    kernels = [source_kernel(w, (cx, y + args.source_dy), 1.0) for cx, _ in blocks]
    probes = [FieldProbes(w, [(x, y - 2.0) for x in xs] + [(x, y - 3.5) for x in xs]) for _, xs in blocks]
    for t in range(args.write):
        ext = None
        for k, ker in enumerate(kernels):
            a = args.amplitude * (bits[2 * k] * math.cos(0.03 * t) + bits[2 * k + 1] * math.cos(0.09 * t))
            ext = ker * a if ext is None else ext + ker * a
        w.step(external_source=ext)
    out = {}
    t = 0
    for D in sorted(args.delays):
        while t < D:
            w.step(); t += 1
        rec = [[] for _ in blocks]
        for _ in range(args.read):
            w.step(); t += 1
            for k, p in enumerate(probes):
                rec[k].append(p.field(w).numpy().copy())
        out[D] = [np.array(r) for r in rec]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blocks", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--block-gap", type=float, default=24.0)
    p.add_argument("--spacing", type=float, default=3.6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=400)
    p.add_argument("--write", type=int, default=600)
    p.add_argument("--amplitude", type=float, default=0.2)
    p.add_argument("--source-dy", type=float, default=2.5)
    p.add_argument("--delays", type=int, nargs="+", default=[2000, 6000])
    p.add_argument("--read", type=int, default=400)
    p.add_argument("--trials", type=int, default=128)
    p.add_argument("--phase-noise", type=float, default=0.003)
    p.add_argument("--sigma", type=float, default=1e-3)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--out", default="runs/phase_ram")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for K in a.blocks:
        bits_all = np.random.default_rng(300 + K).choice([-1.0, 1.0], size=(a.trials, 2 * K))
        for world_name, beads in (("chain", True), ("bare", False)):
            base, N, blocks = build(K, a, beads)
            recs = [trial(a, base, N, blocks, bits_all[i], i) for i in range(a.trials)]
            for D in a.delays:
                feats = [np.array([demod(recs[i][D][k], a.sigma, np.random.default_rng(9000 + 31 * i + k), D)
                                   for i in range(a.trials)]) for k in range(K)]
                acc_local, acc_cross = [], []
                for k in range(K):
                    for r in range(2):
                        y = bits_all[:, 2 * k + r]
                        acc_local.append(cv_decode(feats[k], y, a.ridge))
                        if K > 1:
                            acc_cross.append(cv_decode(feats[(k + 1) % K], y, a.ridge))
                row = {"blocks": K, "bits": 2 * K, "world": world_name, "delay": D, "N": N,
                       "acc_local": acc_local, "mean_local": float(np.mean(acc_local)),
                       "min_local": float(np.min(acc_local)),
                       "mean_cross": float(np.mean(acc_cross)) if acc_cross else None}
                rows.append(row)
                print(f"K={K} ({2*K} bits, N={N}) {world_name:5s} delay={D:5d} local mean={row['mean_local']:.3f} "
                      f"min={row['min_local']:.3f} cross={row['mean_cross']}", flush=True)
            json.dump({"parameters": vars(a), "rows": rows}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
